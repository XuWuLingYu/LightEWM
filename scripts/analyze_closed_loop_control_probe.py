#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightewm.eval.closed_loop.gates import (  # noqa: E402
    analyze_control_probe,
    analyze_gripper_toggle_probe,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate deterministic closed-loop control-probe evidence."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--qpos-tolerance", type=float, default=1e-3)
    parser.add_argument("--gripper-tolerance", type=float, default=0.05)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if payload.get("action_mode") == "gripper_toggle":
        result = analyze_gripper_toggle_probe(
            payload,
            gripper_tolerance=args.gripper_tolerance,
        )
    else:
        result = analyze_control_probe(
            payload,
            qpos_tolerance=args.qpos_tolerance,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
