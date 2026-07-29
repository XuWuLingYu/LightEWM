#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightewm.eval.closed_loop.normalize import normalize_native_results  # noqa: E402
from lightewm.eval.closed_loop.records import (  # noqa: E402
    ArtifactIndex,
    write_episode_records,
    write_summary,
)
from lightewm.eval.closed_loop.specs import BenchmarkSpec, PolicySpec  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize official closed-loop benchmark outputs."
    )
    parser.add_argument("--benchmark", required=True, choices=("libero", "robotwin", "robolab"))
    parser.add_argument("--input", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--task")
    parser.add_argument("--setting", default="default")
    parser.add_argument("--artifacts-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--policy-spec")
    parser.add_argument("--benchmark-spec")
    parser.add_argument("--command-log", default="")
    parser.add_argument("--native-output-dir")
    parser.add_argument(
        "--source-revision",
        action="append",
        default=[],
        metavar="NAME=REVISION",
        help="Additional source revision to record in the artifact index.",
    )
    return parser.parse_args()


def parse_source_revisions(values: list[str]) -> dict[str, str]:
    revisions = {}
    for value in values:
        name, separator, revision = value.partition("=")
        if not separator or not name.strip() or not revision.strip():
            raise ValueError(
                f"--source-revision must be NAME=REVISION, got {value!r}"
            )
        revisions[name.strip()] = revision.strip()
    return revisions


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = normalize_native_results(
        args.benchmark,
        args.input,
        revision=args.revision,
        policy=args.policy,
        protocol=args.protocol,
        task=args.task,
        setting=args.setting,
        artifacts_dir=args.artifacts_dir,
    )
    episodes_path = write_episode_records(output_dir / "episodes.jsonl", records)
    summary_path = write_summary(output_dir / "summary.json", records)

    if bool(args.policy_spec) != bool(args.benchmark_spec):
        raise ValueError("--policy-spec and --benchmark-spec must be provided together")
    if args.policy_spec:
        policy_spec = PolicySpec.from_yaml(args.policy_spec)
        benchmark_spec = BenchmarkSpec.from_yaml(args.benchmark_spec)
        source_revisions = {args.benchmark: args.revision}
        source_revisions.update(parse_source_revisions(args.source_revision))
        ArtifactIndex(
            run_id=args.run_id or output_dir.name,
            policy_spec=str(Path(args.policy_spec)),
            policy_spec_hash=policy_spec.content_hash(),
            benchmark_spec=str(Path(args.benchmark_spec)),
            benchmark_spec_hash=benchmark_spec.content_hash(),
            command_log=args.command_log,
            native_output_dir=args.native_output_dir or str(Path(args.input).parent),
            episodes_path=str(episodes_path),
            summary_path=str(summary_path),
            source_revisions=source_revisions,
        ).write(output_dir / "artifact_index.json")

    print(f"Wrote {len(records)} episodes to {episodes_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
