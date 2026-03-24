#!/usr/bin/env python3
"""Generate randomized AIC engine config for data collection.

Focuses randomization on:
- task board x/y/yaw
- one NIC card translation/yaw (exactly one NIC rendered)
- one SC port rail translation (exactly one SC port rendered)

The output keeps the same top-level structure as sample_config.yaml.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
from pathlib import Path

import yaml


def _uniform(rng: random.Random, lo: float, hi: float) -> float:
    return round(rng.uniform(lo, hi), 6)


def _trial_template() -> dict:
    # Minimal trial schema expected by aic_engine.
    return {
        "scene": {
            "task_board": {
                "pose": {
                    "x": 0.15,
                    "y": -0.2,
                    "z": 1.14,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": math.pi,
                },
                "nic_rail_0": {"entity_present": False},
                "nic_rail_1": {"entity_present": False},
                "nic_rail_2": {"entity_present": False},
                "nic_rail_3": {"entity_present": False},
                "nic_rail_4": {"entity_present": False},
                "sc_rail_0": {"entity_present": False},
                "sc_rail_1": {"entity_present": False},
                "lc_mount_rail_0": {"entity_present": False},
                "sfp_mount_rail_0": {"entity_present": False},
                "sc_mount_rail_0": {"entity_present": False},
                "lc_mount_rail_1": {"entity_present": False},
                "sfp_mount_rail_1": {"entity_present": False},
                "sc_mount_rail_1": {"entity_present": False},
            },
            "cables": {},
        },
        "tasks": {},
    }


def _build_trial(
    rng: random.Random,
    trial_idx: int,
    task_type: str,
    board_x_range: tuple[float, float],
    board_y_range: tuple[float, float],
    board_yaw_range: tuple[float, float],
    nic_translation_range: tuple[float, float],
    nic_yaw_range_deg: tuple[float, float],
    sc_translation_range: tuple[float, float],
) -> dict:
    trial = _trial_template()

    # Board pose randomization
    trial["scene"]["task_board"]["pose"]["x"] = _uniform(
        rng, board_x_range[0], board_x_range[1]
    )
    trial["scene"]["task_board"]["pose"]["y"] = _uniform(
        rng, board_y_range[0], board_y_range[1]
    )
    trial["scene"]["task_board"]["pose"]["yaw"] = _uniform(
        rng, board_yaw_range[0], board_yaw_range[1]
    )

    # Exactly one NIC rendered with randomized translation + yaw
    nic_idx = rng.randint(0, 4)
    nic_key = f"nic_rail_{nic_idx}"
    nic_yaw_deg = rng.uniform(nic_yaw_range_deg[0], nic_yaw_range_deg[1])
    trial["scene"]["task_board"][nic_key] = {
        "entity_present": True,
        "entity_name": f"nic_card_{nic_idx}",
        "entity_pose": {
            "translation": _uniform(
                rng, nic_translation_range[0], nic_translation_range[1]
            ),
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": round(math.radians(nic_yaw_deg), 6),
        },
    }

    # Exactly one SC rail entity rendered with randomized translation.
    # This ultimately controls sc_port_0/sc_port_1 placement in the board model.
    sc_idx = rng.randint(0, 1)
    sc_key = f"sc_rail_{sc_idx}"
    trial["scene"]["task_board"][sc_key] = {
        "entity_present": True,
        "entity_name": f"sc_mount_{sc_idx}",
        "entity_pose": {
            "translation": _uniform(
                rng, sc_translation_range[0], sc_translation_range[1]
            ),
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
        },
    }

    if task_type == "sfp":
        cable_name = f"cable_{trial_idx}"
        trial["scene"]["cables"][cable_name] = {
            "pose": {
                "gripper_offset": {"x": 0.0, "y": 0.015385, "z": 0.04245},
                "roll": 0.4432,
                "pitch": -0.4838,
                "yaw": 1.3303,
            },
            "attach_cable_to_gripper": True,
            "cable_type": "sfp_sc_cable",
        }
        trial["tasks"]["task_1"] = {
            "cable_type": "sfp_sc",
            "cable_name": cable_name,
            "plug_type": "sfp",
            "plug_name": "sfp_tip",
            "port_type": "sfp",
            "port_name": "sfp_port_0",
            "target_module_name": f"nic_card_mount_{nic_idx}",
            "time_limit": 180,
        }
    else:
        cable_name = f"cable_{trial_idx}"
        trial["scene"]["cables"][cable_name] = {
            "pose": {
                "gripper_offset": {"x": 0.0, "y": 0.015385, "z": 0.04045},
                "roll": 0.4432,
                "pitch": -0.4838,
                "yaw": 1.3303,
            },
            "attach_cable_to_gripper": True,
            "cable_type": "sfp_sc_cable_reversed",
        }
        trial["tasks"]["task_1"] = {
            "cable_type": "sfp_sc",
            "cable_name": cable_name,
            "plug_type": "sc",
            "plug_name": "sc_tip",
            "port_type": "sc",
            "port_name": "sc_port_base",
            "target_module_name": f"sc_port_{sc_idx}",
            "time_limit": 180,
        }

    return trial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-config", required=True, type=Path)
    parser.add_argument("--output-config", required=True, type=Path)
    parser.add_argument("--num-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--task-mode",
        choices=["mixed", "sfp_only", "sc_only"],
        default="mixed",
    )
    parser.add_argument("--board-x-min", type=float, default=0.15)
    parser.add_argument("--board-x-max", type=float, default=0.19)
    parser.add_argument("--board-y-min", type=float, default=-0.20)
    parser.add_argument("--board-y-max", type=float, default=0.02)
    parser.add_argument("--board-yaw-min", type=float, default=2.9)
    parser.add_argument("--board-yaw-max", type=float, default=3.35)
    parser.add_argument("--nic-translation-min", type=float, default=-0.0215)
    parser.add_argument("--nic-translation-max", type=float, default=0.0234)
    parser.add_argument("--nic-yaw-deg-min", type=float, default=-10.0)
    parser.add_argument("--nic-yaw-deg-max", type=float, default=10.0)
    parser.add_argument("--sc-translation-min", type=float, default=-0.06)
    parser.add_argument("--sc-translation-max", type=float, default=0.055)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    with args.sample_config.open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    out = {
        "scoring": copy.deepcopy(base["scoring"]),
        "task_board_limits": copy.deepcopy(base["task_board_limits"]),
        "trials": {},
        "robot": copy.deepcopy(base["robot"]),
    }

    for i in range(1, args.num_trials + 1):
        if args.task_mode == "sfp_only":
            task_type = "sfp"
        elif args.task_mode == "sc_only":
            task_type = "sc"
        else:
            task_type = "sfp" if i % 2 == 1 else "sc"

        out["trials"][f"trial_{i}"] = _build_trial(
            rng=rng,
            trial_idx=i - 1,
            task_type=task_type,
            board_x_range=(args.board_x_min, args.board_x_max),
            board_y_range=(args.board_y_min, args.board_y_max),
            board_yaw_range=(args.board_yaw_min, args.board_yaw_max),
            nic_translation_range=(args.nic_translation_min, args.nic_translation_max),
            nic_yaw_range_deg=(args.nic_yaw_deg_min, args.nic_yaw_deg_max),
            sc_translation_range=(args.sc_translation_min, args.sc_translation_max),
        )

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    with args.output_config.open("w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print(f"Wrote randomized engine config: {args.output_config}")
    print(f"Trials: {args.num_trials}  task_mode={args.task_mode}  seed={args.seed}")


if __name__ == "__main__":
    main()
