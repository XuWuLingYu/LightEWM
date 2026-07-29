#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightewm.eval.closed_loop.gates import analyze_episode_gate  # noqa: E402
from lightewm.eval.closed_loop.records import read_episode_records  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a normalized closed-loop episode acceptance gate."
    )
    parser.add_argument("--episodes", required=True)
    parser.add_argument(
        "--expect",
        required=True,
        choices=("all-success", "all-failure", "at-least-one-success"),
    )
    parser.add_argument("--min-episodes", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = analyze_episode_gate(
        read_episode_records(args.episodes),
        expectation=args.expect,
        min_episodes=args.min_episodes,
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
