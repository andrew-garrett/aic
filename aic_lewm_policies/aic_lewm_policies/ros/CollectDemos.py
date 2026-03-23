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

import importlib
import inspect
import os
from pathlib import Path

import h5py
import numpy as np

from aic_control_interfaces.msg import JointMotionUpdate, MotionUpdate
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
        self._h5_path = self._stablewm_home / f"{self._dataset_name}.h5"
        self._stablewm_home.mkdir(parents=True, exist_ok=True)

        self._delegate = self._load_delegate_policy(self._delegate_policy)
        self._reset_episode_buffers()
        self.get_logger().info(
            f"CollectDemos initialized. dataset={self._h5_path} delegate={self._delegate_policy}"
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
        self._actions = []
        self._proprio = []
        self._state = []
        self._rewards = []
        self._terminated = []
        self._latest_observation = None

    @staticmethod
    def _image_to_hwc(image_msg) -> np.ndarray:
        img = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(
            image_msg.height, image_msg.width, 3
        )
        return img

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
        # Fixed 14-dim action representation:
        # [mode, 7 pose-or-joint slots, 6 twist-or-joint-velocity slots]
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
        self._pixels.append(self._image_to_hwc(obs.center_image))
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
        actions = np.asarray(self._actions, dtype=np.float32)
        proprio = np.asarray(self._proprio, dtype=np.float32)
        state = np.asarray(self._state, dtype=np.float32)
        rewards = np.asarray(self._rewards, dtype=np.float32)
        terminated = np.asarray(self._terminated, dtype=np.bool_)

        with h5py.File(self._h5_path, "a") as h5_file:
            self._append_dataset(h5_file, "pixels", pixels)
            self._append_dataset(h5_file, "action", actions)
            self._append_dataset(h5_file, "proprio", proprio)
            self._append_dataset(h5_file, "state", state)
            self._append_dataset(h5_file, "reward", rewards)
            self._append_dataset(h5_file, "terminated", terminated)

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

        def wrapped_get_observation():
            obs = get_observation()
            if obs is not None:
                self._latest_observation = obs
            return obs

        def wrapped_move_robot(motion_update=None, joint_motion_update=None):
            obs = self._latest_observation
            if obs is None:
                obs = wrapped_get_observation()
            if obs is not None:
                action = self._extract_action(motion_update, joint_motion_update)
                self._record_step(obs, action)
            return move_robot(
                motion_update=motion_update, joint_motion_update=joint_motion_update
            )

        success = False
        send_feedback("collecting demonstration")
        try:
            success = bool(
                self._delegate.insert_cable(
                    task=task,
                    get_observation=wrapped_get_observation,
                    move_robot=wrapped_move_robot,
                    send_feedback=send_feedback,
                )
            )
        except Exception as ex:
            self.get_logger().error(f"Delegate policy failed: {ex}")
            success = False

        self._flush_episode()
        self.get_logger().info(f"CollectDemos.insert_cable() done success={success}")
        return success
