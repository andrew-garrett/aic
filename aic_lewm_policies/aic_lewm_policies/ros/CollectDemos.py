from __future__ import annotations

#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Recording timing follows ReferenceDataCollection (DataCollectionPolicyV3):
#  move_robot → sleep(step_dt) → sample Observation. Unlike v3's PNG logger, we do
#  **not** deduplicate by camera stamp: proprio/state change every control step even
#  when the synchronized camera triple is slower than the pose command rate; skipping
#  those steps breaks (action, state, pixels) alignment for HDF5.
#

import importlib
import inspect
import os
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
from geometry_msgs.msg import Pose, Vector3, Wrench
from rclpy.duration import Duration
from rclpy.time import Time
from std_msgs.msg import Header
from tf2_ros import TransformException

try:
    import cv2
except ImportError:
    cv2 = None

from aic_control_interfaces.msg import (
    JointMotionUpdate,
    MotionUpdate,
    TrajectoryGenerationMode,
)
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task


class CollectDemos(Policy):
    """Wrap an expert policy and record SWM-compatible HDF5 demonstrations."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._stablewm_home = Path(
            os.environ.get("STABLEWM_HOME", str(Path.home() / ".stable-wm"))
        )
        self._dataset_name = os.environ.get("AIC_DATASET_NAME", "aic_cable_train")
        self._delegate_policy = os.environ.get(
            "AIC_DEMO_DELEGATE_POLICY", "aic_lewm_policies.ros.OracleDualInsert"
        )
        self._max_steps = int(os.environ.get("AIC_DEMO_MAX_STEPS", "2000"))
        self._capture_pixel_max = int(os.environ.get("AIC_CAPTURE_PIXEL_MAX", "512"))
        self._obs_wait_timeout_sec = float(
            os.environ.get("AIC_WAIT_OBS_TIMEOUT_SEC", "10.0")
        )
        # After each sleep, brief poll if Observation is briefly None (executor catch-up).
        self._post_step_obs_poll_sec = float(
            os.environ.get("AIC_POST_STEP_OBS_POLL_SEC", "0.5")
        )
        self._h5_path = self._stablewm_home / f"{self._dataset_name}.h5"
        self._stablewm_home.mkdir(parents=True, exist_ok=True)

        self._delegate = self._load_delegate_policy(self._delegate_policy)
        self._reset_episode_buffers()
        self.get_logger().info(
            f"CollectDemos initialized. dataset={self._h5_path} delegate={self._delegate_policy} "
            f"capture_pixel_max={self._capture_pixel_max} "
            f"obs_wait_timeout_sec={self._obs_wait_timeout_sec} "
            f"post_step_obs_poll_sec={self._post_step_obs_poll_sec}"
        )

    def _load_delegate_policy(self, module_name: str) -> Policy:
        module = importlib.import_module(module_name)
        expected_class_name = module_name.split(".")[-1]
        classes = inspect.getmembers(module, inspect.isclass)
        for class_name, cls in classes:
            if class_name == expected_class_name:
                self.get_logger().info(f"Using delegate policy: {module_name}")
                return cls(self._parent_node)
        raise LookupError(f"Class {expected_class_name} not found in {module_name}")

    def _reset_episode_buffers(self):
        self._pixels = []
        self._pixels_left = []
        self._pixels_right = []
        self._actions = []
        self._proprio = []
        self._state = []
        self._rewards = []
        self._terminated = []
        self._obs_last = None

    def _wait_for_observation(
        self,
        get_observation: GetObservationCallback,
        timeout_sec: float,
    ) -> Optional[Observation]:
        start = self.time_now()
        limit = Duration(seconds=timeout_sec)
        while (self.time_now() - start) < limit:
            obs = get_observation()
            if obs is not None:
                return obs
            self.sleep_for(0.05)
        self.get_logger().warn(
            f"No Observation received after {timeout_sec}s (check aic_adapter / camera topics)."
        )
        return None

    def _make_pose_motion_update(
        self,
        pose: Pose,
        frame_id: str = "base_link",
        stiffness: Optional[list] = None,
        damping: Optional[list] = None,
    ) -> MotionUpdate:
        """Same MotionUpdate construction as Policy.set_pose_target (for action logging)."""
        if stiffness is None:
            stiffness = [90.0, 90.0, 90.0, 50.0, 50.0, 50.0]
        if damping is None:
            damping = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]
        return MotionUpdate(
            header=Header(
                frame_id=frame_id,
                stamp=self._parent_node.get_clock().now().to_msg(),
            ),
            pose=pose,
            target_stiffness=np.diag(stiffness).flatten(),
            target_damping=np.diag(damping).flatten(),
            feedforward_wrench_at_tip=Wrench(
                force=Vector3(x=0.0, y=0.0, z=0.0),
                torque=Vector3(x=0.0, y=0.0, z=0.0),
            ),
            wrench_feedback_gains_at_tip=[0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION,
            ),
        )

    def _record_after_command(
        self,
        get_observation: GetObservationCallback,
        motion_update: MotionUpdate,
        step_index: int,
        phase: str,
    ) -> None:
        """Sample Observation after move+sleep; always append one transition if possible."""
        obs = get_observation()
        if obs is None:
            obs = self._wait_for_observation(
                get_observation, self._post_step_obs_poll_sec
            )
        if obs is None:
            self.get_logger().warn(
                f"CollectDemos: no Observation after {phase} step {step_index} "
                f"(poll {self._post_step_obs_poll_sec}s); skipping row."
            )
            return
        action = self._extract_action(motion_update, None)
        self._record_step(obs, action)

    @staticmethod
    def _image_to_hwc(image_msg) -> np.ndarray:
        h, w = int(image_msg.height), int(image_msg.width)
        need = h * w * 3
        buf = image_msg.data
        if len(buf) != need:
            raise ValueError(
                f"Image size mismatch: got {len(buf)} bytes for {h}x{w}x3 (expected {need}); "
                f"encoding={getattr(image_msg, 'encoding', '')!r}"
            )
        return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).copy()

    def _resize_for_storage(self, img_hwc: np.ndarray) -> np.ndarray:
        max_side = self._capture_pixel_max
        if max_side <= 0:
            return img_hwc
        h, w = img_hwc.shape[:2]
        m = max(h, w)
        if m <= max_side:
            return img_hwc
        scale = max_side / float(m)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        if cv2 is not None:
            return cv2.resize(img_hwc, (new_w, new_h), interpolation=cv2.INTER_AREA)
        try:
            from PIL import Image

            pil = Image.fromarray(img_hwc)
            pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            return np.asarray(pil)
        except ImportError as ex:
            raise RuntimeError(
                "CollectDemos needs opencv-python or Pillow to resize images; "
                "install one of them, or set AIC_CAPTURE_PIXEL_MAX=0 for full resolution."
            ) from ex

    def _center_image_for_dataset(self, obs: Observation) -> np.ndarray:
        raw = self._image_to_hwc(obs.center_image)
        return self._resize_for_storage(raw)

    def _left_image_for_dataset(self, obs: Observation) -> np.ndarray:
        raw = self._image_to_hwc(obs.left_image)
        return self._resize_for_storage(raw)

    def _right_image_for_dataset(self, obs: Observation) -> np.ndarray:
        raw = self._image_to_hwc(obs.right_image)
        return self._resize_for_storage(raw)

    @staticmethod
    def _safe_list(values, count):
        arr = list(values)[:count]
        if len(arr) < count:
            arr += [0.0] * (count - len(arr))
        return arr

    def _extract_proprio(self, obs: Observation) -> np.ndarray:
        joints = self._safe_list(obs.joint_states.position, 7)
        wrench = [
            obs.wrist_wrench.wrench.force.x,
            obs.wrist_wrench.wrench.force.y,
            obs.wrist_wrench.wrench.force.z,
            obs.wrist_wrench.wrench.torque.x,
            obs.wrist_wrench.wrench.torque.y,
            obs.wrist_wrench.wrench.torque.z,
        ]
        return np.asarray(joints + wrench, dtype=np.float32)

    def _extract_state(self, obs: Observation) -> np.ndarray:
        tcp_pose = obs.controller_state.tcp_pose
        tcp_vel = obs.controller_state.tcp_velocity
        state = [
            tcp_pose.position.x,
            tcp_pose.position.y,
            tcp_pose.position.z,
            tcp_pose.orientation.x,
            tcp_pose.orientation.y,
            tcp_pose.orientation.z,
            tcp_pose.orientation.w,
            tcp_vel.linear.x,
            tcp_vel.linear.y,
            tcp_vel.linear.z,
            tcp_vel.angular.x,
            tcp_vel.angular.y,
            tcp_vel.angular.z,
            *obs.controller_state.tcp_error,
            *self._safe_list(obs.joint_states.position, 7),
        ]
        return np.asarray(state, dtype=np.float32)

    def _extract_action(
        self,
        motion_update: MotionUpdate = None,
        joint_motion_update: JointMotionUpdate = None,
    ) -> np.ndarray:
        action = np.zeros((14,), dtype=np.float32)
        if motion_update is not None:
            action[0] = 1.0
            action[1:8] = np.asarray(
                [
                    motion_update.pose.position.x,
                    motion_update.pose.position.y,
                    motion_update.pose.position.z,
                    motion_update.pose.orientation.x,
                    motion_update.pose.orientation.y,
                    motion_update.pose.orientation.z,
                    motion_update.pose.orientation.w,
                ],
                dtype=np.float32,
            )
            action[8:14] = np.asarray(
                [
                    motion_update.velocity.linear.x,
                    motion_update.velocity.linear.y,
                    motion_update.velocity.linear.z,
                    motion_update.velocity.angular.x,
                    motion_update.velocity.angular.y,
                    motion_update.velocity.angular.z,
                ],
                dtype=np.float32,
            )
        elif joint_motion_update is not None:
            action[0] = 2.0
            positions = self._safe_list(joint_motion_update.target_state.positions, 7)
            velocities = self._safe_list(joint_motion_update.target_state.velocities, 6)
            action[1:8] = np.asarray(positions, dtype=np.float32)
            action[8:14] = np.asarray(velocities, dtype=np.float32)
        return action

    def _record_step(self, obs: Observation, action: np.ndarray):
        if len(self._actions) >= self._max_steps:
            return
        self._pixels.append(self._center_image_for_dataset(obs))
        self._pixels_left.append(self._left_image_for_dataset(obs))
        self._pixels_right.append(self._right_image_for_dataset(obs))
        self._actions.append(action)
        self._proprio.append(self._extract_proprio(obs))
        self._state.append(self._extract_state(obs))
        self._rewards.append(0.0)
        self._terminated.append(False)

    @staticmethod
    def _append_dataset(h5_file, key, values: np.ndarray):
        if key not in h5_file:
            shape = values.shape
            maxshape = (None,) + shape[1:]
            h5_file.create_dataset(
                key,
                data=values,
                maxshape=maxshape,
                chunks=True,
                compression="gzip",
            )
            return
        ds = h5_file[key]
        if ds.shape[1:] != values.shape[1:]:
            raise ValueError(
                f"HDF5 dataset shape mismatch for '{key}': existing trailing shape "
                f"{ds.shape[1:]}, new trailing shape {values.shape[1:]}. "
                "Use a new dataset name or delete the existing .h5 file."
            )
        old = ds.shape[0]
        ds.resize(old + values.shape[0], axis=0)
        ds[old:] = values

    def _flush_episode(self):
        episode_len = len(self._actions)
        if episode_len == 0:
            self.get_logger().warn("No transitions recorded for this episode.")
            return

        self._terminated[-1] = True

        pixels = np.asarray(self._pixels, dtype=np.uint8)
        pixels_left = np.asarray(self._pixels_left, dtype=np.uint8)
        pixels_right = np.asarray(self._pixels_right, dtype=np.uint8)
        actions = np.asarray(self._actions, dtype=np.float32)
        proprio = np.asarray(self._proprio, dtype=np.float32)
        state = np.asarray(self._state, dtype=np.float32)
        rewards = np.asarray(self._rewards, dtype=np.float32)
        terminated = np.asarray(self._terminated, dtype=np.bool_)

        with h5py.File(self._h5_path, "a") as h5_file:
            next_ep_idx = (
                int(h5_file["ep_len"].shape[0]) if "ep_len" in h5_file else 0
            )
            episode_idx = np.full(episode_len, next_ep_idx, dtype=np.int64)
            step_idx = np.arange(episode_len, dtype=np.int64)

            self._append_dataset(h5_file, "pixels", pixels)
            self._append_dataset(h5_file, "pixels_left", pixels_left)
            self._append_dataset(h5_file, "pixels_right", pixels_right)
            self._append_dataset(h5_file, "action", actions)
            self._append_dataset(h5_file, "proprio", proprio)
            self._append_dataset(h5_file, "state", state)
            self._append_dataset(h5_file, "reward", rewards)
            self._append_dataset(h5_file, "terminated", terminated)
            self._append_dataset(h5_file, "episode_idx", episode_idx)
            self._append_dataset(h5_file, "step_idx", step_idx)

            if "ep_len" not in h5_file:
                h5_file.create_dataset(
                    "ep_len",
                    data=np.asarray([episode_len], dtype=np.int64),
                    maxshape=(None,),
                    chunks=True,
                )
                h5_file.create_dataset(
                    "ep_offset",
                    data=np.asarray([0], dtype=np.int64),
                    maxshape=(None,),
                    chunks=True,
                )
            else:
                ep_len = h5_file["ep_len"]
                ep_offset = h5_file["ep_offset"]
                old_episodes = ep_len.shape[0]
                last_offset = int(ep_offset[-1])
                last_len = int(ep_len[-1])
                next_offset = last_offset + last_len

                ep_len.resize(old_episodes + 1, axis=0)
                ep_offset.resize(old_episodes + 1, axis=0)
                ep_len[old_episodes] = episode_len
                ep_offset[old_episodes] = next_offset

        self.get_logger().info(
            f"Recorded episode len={episode_len} to {self._h5_path.name}"
        )

    def _insert_cable_oracle_dual_insert(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        oracle: Any,
    ) -> bool:
        """Oracle motion (same as OracleDualInsert.insert_cable) + v3-style HDF5 timing."""
        oracle._task = task
        profile = oracle._get_task_profile(task)
        self.get_logger().info(
            f"CollectDemos oracle loop task={task.id} profile={profile['name']}"
        )

        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        cable_tip_frame = f"{task.cable_name}/{task.plug_name}_link"

        for frame in [port_frame, cable_tip_frame]:
            if not oracle._wait_for_tf("base_link", frame):
                return False

        try:
            port_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                port_frame,
                Time(),
            )
        except TransformException as ex:
            self.get_logger().error(f"Could not look up port transform: {ex}")
            return False

        port_transform = port_tf_stamped.transform
        z_offset = profile["approach_z_offset"]
        send_feedback("oracle aligning")

        for t in range(profile["interp_steps"]):
            interp_fraction = t / float(profile["interp_steps"])
            try:
                pose = oracle.calc_gripper_pose(
                    port_transform,
                    slerp_fraction=interp_fraction,
                    position_fraction=interp_fraction,
                    z_offset=z_offset,
                    reset_xy_integrator=True,
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during interpolation: {ex}")
                self.sleep_for(profile["step_sleep"])
                continue

            motion = self._make_pose_motion_update(pose)
            try:
                move_robot(motion_update=motion)
            except Exception as ex:
                self.get_logger().info(f"move_robot exception: {ex}")
            self.sleep_for(profile["step_sleep"])
            self._record_after_command(get_observation, motion, t, "approach")

        send_feedback("oracle descending")
        step_idx = 0
        while z_offset > profile["min_z_offset"]:
            z_offset -= profile["descend_step"]
            dither_amp = profile["xy_dither_amp"]
            dither_period = profile["xy_dither_period_steps"]
            phase = 2.0 * np.pi * (step_idx % dither_period) / float(dither_period)
            xy_dither = (
                dither_amp * np.cos(phase),
                dither_amp * np.sin(phase),
            )
            step_idx += 1
            try:
                pose = oracle.calc_gripper_pose(
                    port_transform,
                    z_offset=z_offset,
                    xy_dither=xy_dither,
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during insertion: {ex}")
                self.sleep_for(profile["step_sleep"])
                continue

            motion = self._make_pose_motion_update(pose)
            try:
                move_robot(motion_update=motion)
            except Exception as ex:
                self.get_logger().info(f"move_robot exception: {ex}")
            self.sleep_for(profile["step_sleep"])
            self._record_after_command(
                get_observation, motion, step_idx, "insertion"
            )

        send_feedback("oracle settling")
        self.sleep_for(profile["settle_sec"])
        self.get_logger().info("CollectDemos oracle loop done")
        return True

    def _insert_cable_delegate_wrapped(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        """Non-oracle delegates (e.g. RunACT): pair obs from get_observation with move_robot."""
        def wrapped_get_observation():
            obs = self._wait_for_observation(
                get_observation, self._obs_wait_timeout_sec
            )
            self._obs_last = obs
            return obs

        def wrapped_move_robot(motion_update=None, joint_motion_update=None):
            obs = self._obs_last
            if obs is None:
                obs = self._wait_for_observation(
                    get_observation, self._obs_wait_timeout_sec
                )
            if obs is not None:
                action = self._extract_action(motion_update, joint_motion_update)
                self._record_step(obs, action)
            self._obs_last = None
            return move_robot(
                motion_update=motion_update, joint_motion_update=joint_motion_update
            )

        return bool(
            self._delegate.insert_cable(
                task=task,
                get_observation=wrapped_get_observation,
                move_robot=wrapped_move_robot,
                send_feedback=send_feedback,
            )
        )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self._reset_episode_buffers()
        self.get_logger().info(
            f"CollectDemos.insert_cable() start. task={task.id} dataset={self._dataset_name}"
        )

        obs0 = self._wait_for_observation(get_observation, self._obs_wait_timeout_sec)
        if obs0 is None:
            self.get_logger().error("CollectDemos: no initial Observation; abort.")
            self._flush_episode()
            return False

        success = False
        send_feedback("collecting demonstration")
        try:
            delegate_name = self._delegate.__class__.__name__
            if delegate_name == "OracleDualInsert":
                success = self._insert_cable_oracle_dual_insert(
                    task, get_observation, move_robot, send_feedback, self._delegate
                )
            else:
                self.get_logger().info(
                    f"CollectDemos: delegate {delegate_name} uses wrapped insert_cable "
                    "(not the inline oracle loop)."
                )
                success = self._insert_cable_delegate_wrapped(
                    task, get_observation, move_robot, send_feedback
                )
        except Exception as ex:
            self.get_logger().error(f"CollectDemos failed: {ex}")
            success = False

        self._flush_episode()
        self.get_logger().info(f"CollectDemos.insert_cable() done success={success}")
        return success
