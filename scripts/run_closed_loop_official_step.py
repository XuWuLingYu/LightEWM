#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightewm.eval.closed_loop.harness import OfficialHarness  # noqa: E402
from lightewm.eval.closed_loop.protocols import (  # noqa: E402
    RoboLabProtocol,
    RoboTwinProtocol,
)


def _required(value: str | None, name: str) -> str:
    if not value:
        raise ValueError(f"{name} is required for this gate")
    return value


def _overrides(values: list[str]) -> tuple[str, ...]:
    pairs: list[str] = []
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key:
            raise ValueError(f"override must be KEY=VALUE, got {value!r}")
        pairs.extend((key, item))
    return tuple(pairs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one official closed-loop benchmark step with evidence."
    )
    parser.add_argument("--benchmark", required=True, choices=("robotwin", "robolab"))
    parser.add_argument("--gate", required=True)
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--task")
    parser.add_argument("--setting", default="default")
    parser.add_argument("--config")
    parser.add_argument("--policy")
    parser.add_argument("--checkpoint-tag", default="reference")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--mode")
    parser.add_argument("--probe-path")
    parser.add_argument("--recorded-data-folder")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--output-folder-name")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def robotwin_step(args: argparse.Namespace):
    protocol = RoboTwinProtocol(root=args.official_root, python=args.python)
    if args.gate == "render":
        return protocol.render()
    task = _required(args.task, "--task")
    if args.gate == "ground-truth":
        return protocol.collect(task, args.setting)
    if args.gate == "negative":
        return protocol.negative_control(
            config=_required(args.config, "--config"),
            task=task,
            setting=args.setting,
            seed=args.seed,
            mode=_required(args.mode, "--mode"),
            probe_path=_required(args.probe_path, "--probe-path"),
        )
    if args.gate == "reference":
        return protocol.evaluate(
            config=_required(args.config, "--config"),
            policy_name=_required(args.policy, "--policy"),
            task=task,
            setting=args.setting,
            checkpoint_tag=args.checkpoint_tag,
            seed=args.seed,
            episodes=args.episodes,
            overrides=_overrides(args.override),
        )
    raise ValueError(f"unsupported RoboTwin gate: {args.gate}")


def robolab_step(args: argparse.Namespace):
    protocol = RoboLabProtocol(root=args.official_root, python=args.python)
    task = _required(args.task, "--task")
    if args.gate == "recorded-replay":
        return protocol.recorded_replay(
            task,
            recorded_data_folder=args.recorded_data_folder,
            episode=args.episode,
        )
    if args.gate == "gripper-toggle":
        return protocol.gripper_toggle(task)
    if args.gate == "negative-hold":
        return protocol.deterministic_hold(task, steps=args.steps)
    if args.gate == "reference":
        return protocol.reference_policy(
            task,
            policy=args.policy or "pi05",
            host=args.host,
            port=args.port or 8000,
            num_runs=args.episodes,
            output_folder_name=args.output_folder_name,
        )
    raise ValueError(f"unsupported RoboLab gate: {args.gate}")


def main() -> None:
    args = parse_args()
    step = robotwin_step(args) if args.benchmark == "robotwin" else robolab_step(args)
    result = OfficialHarness(args.evidence_root).run(step)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
