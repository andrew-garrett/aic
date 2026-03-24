#!/usr/bin/env python3
"""Split a full AIC engine YAML into smaller configs (batches of trials).

Preserves scoring, task_board_limits, robot, and trial key names (trial_1, trial_2, ...).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

_TRIAL_NUM = re.compile(r"^trial_(\d+)$")


def _trial_sort_key(trial_id: str) -> int:
    m = _TRIAL_NUM.match(trial_id)
    if not m:
        raise ValueError(f"Unexpected trial id (expected trial_N): {trial_id!r}")
    return int(m.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    with args.input.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "trials" not in data or not data["trials"]:
        raise SystemExit("Input config has no trials")

    trials = data["trials"]
    keys = sorted(trials.keys(), key=_trial_sort_key)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    batch_idx = 0
    for start in range(0, len(keys), args.batch_size):
        batch_idx += 1
        chunk = keys[start : start + args.batch_size]
        out = {k: v for k, v in data.items() if k != "trials"}
        out["trials"] = {k: trials[k] for k in chunk}
        out_path = args.out_dir / f"batch_{batch_idx:03d}.yaml"
        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(out, f, sort_keys=False)
        print(f"Wrote {out_path} ({len(chunk)} trial(s): {chunk[0]}..{chunk[-1]})")


if __name__ == "__main__":
    main()
