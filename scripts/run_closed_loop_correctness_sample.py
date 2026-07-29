#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightewm.eval.closed_loop.gates import analyze_control_probe  # noqa: E402
from lightewm.eval.closed_loop.harness import OfficialHarness  # noqa: E402
from lightewm.eval.closed_loop.sampling import (  # noqa: E402
    CorrectnessSample,
    SampleCase,
    build_sample_step,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_revision(root: str) -> str:
    completed = subprocess.run(
        ["git", "-C", root, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _snapshot_outputs(root: str, outputs: tuple[str, ...]) -> dict[str, object]:
    snapshot: dict[str, object] = {}
    for value in outputs:
        path = Path(root) / value
        if path.is_file():
            stat = path.stat()
            snapshot[value] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        else:
            snapshot[value] = None
    return snapshot


def _case_id(benchmark: str, case: SampleCase) -> str:
    return f"{benchmark}__{case.task}__{case.gate}"


def _root_and_python(
    args: argparse.Namespace,
    benchmark: str,
) -> tuple[str, str]:
    root = getattr(args, f"{benchmark}_root")
    python = getattr(args, f"{benchmark}_python")
    if not root or not python:
        raise ValueError(f"{benchmark} root and python are required")
    return root, python


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a checked, stratified closed-loop correctness sample."
    )
    parser.add_argument(
        "--manifest",
        default=str(
            REPO_ROOT
            / "examples"
            / "closed_loop"
            / "samples"
            / "correctness_10pct.yaml"
        ),
    )
    parser.add_argument(
        "--benchmark",
        choices=("all", "robotwin", "robolab"),
        default="all",
    )
    parser.add_argument("--robotwin-root")
    parser.add_argument("--robotwin-python")
    parser.add_argument("--robolab-root")
    parser.add_argument("--robolab-python")
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest_bytes = manifest_path.read_bytes()
    sample = CorrectnessSample.from_yaml(manifest_path)
    selected = (
        tuple(sample.benchmarks)
        if args.benchmark == "all"
        else (args.benchmark,)
    )
    evidence_root = Path(args.evidence_root).resolve()
    evidence_root.mkdir(parents=True, exist_ok=True)
    summary_path = evidence_root / "summary.json"
    case_root = evidence_root / "cases"

    if args.list_only:
        listing = {
            name: [asdict(case) for case in sample.benchmarks[name].cases]
            for name in selected
        }
        print(json.dumps(listing, indent=2, sort_keys=True))
        return

    run = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "fraction": sample.fraction,
        "started_at": _utc_now(),
        "finished_at": None,
        "status": "running",
        "benchmarks": {},
        "cases": [],
    }

    failures = 0
    for benchmark in selected:
        root, python = _root_and_python(args, benchmark)
        benchmark_sample = sample.benchmarks[benchmark]
        repository_revision = _git_revision(root)
        if (
            benchmark_sample.repository_revision
            and repository_revision != benchmark_sample.repository_revision
        ):
            raise RuntimeError(
                f"{benchmark} revision mismatch: expected "
                f"{benchmark_sample.repository_revision}, got {repository_revision}"
            )
        run["benchmarks"][benchmark] = {
            "population": benchmark_sample.population,
            "sample_size": len(benchmark_sample.cases),
            "official_root": root,
            "repository_revision": repository_revision,
            "python": python,
        }
        harness = OfficialHarness(evidence_root / "commands" / benchmark)
        for case in benchmark_sample.cases:
            case_id = _case_id(benchmark, case)
            case_path = case_root / f"{case_id}.json"
            if args.resume and case_path.is_file():
                previous = json.loads(case_path.read_text(encoding="utf-8"))
                if previous.get("status") == "passed":
                    previous["resumed"] = True
                    run["cases"].append(previous)
                    _write_json(summary_path, run)
                    continue

            record = {
                "case_id": case_id,
                "benchmark": benchmark,
                **asdict(case),
                "started_at": _utc_now(),
                "finished_at": None,
                "status": "running",
                "error": None,
            }
            _write_json(case_path, record)
            try:
                step = build_sample_step(
                    benchmark,
                    case,
                    official_root=root,
                    python=python,
                )
                outputs_before = _snapshot_outputs(root, step.expected_output)
                result = harness.run(step)
                record["command_result"] = asdict(result)
                outputs_after = _snapshot_outputs(root, step.expected_output)
                record["expected_output_snapshots"] = {
                    "before": outputs_before,
                    "after": outputs_after,
                }
                unchanged = [
                    value
                    for value in step.expected_output
                    if outputs_before[value] is not None
                    and outputs_before[value] == outputs_after[value]
                ]
                if unchanged:
                    raise RuntimeError(
                        f"expected outputs were not refreshed: {unchanged}"
                    )
                if benchmark == "robolab" and case.gate == "deterministic-hold":
                    probe_path = Path(root) / result.expected_output[0]
                    probe = json.loads(probe_path.read_text(encoding="utf-8"))
                    analysis = analyze_control_probe(probe)
                    record["control_probe_analysis"] = analysis
                    if not analysis["passed"]:
                        raise RuntimeError(
                            f"control probe rejected: {analysis['failures']}"
                        )
                record["status"] = "passed"
            except Exception as error:  # preserve all cases in one sampled run
                failures += 1
                record["status"] = "failed"
                record["error"] = f"{type(error).__name__}: {error}"
            record["finished_at"] = _utc_now()
            record["duration_seconds"] = round(
                (
                    datetime.fromisoformat(record["finished_at"])
                    - datetime.fromisoformat(record["started_at"])
                ).total_seconds(),
                3,
            )
            _write_json(case_path, record)
            run["cases"].append(record)
            _write_json(summary_path, run)

    run["finished_at"] = _utc_now()
    run["status"] = "passed" if failures == 0 else "failed"
    run["passed"] = sum(case["status"] == "passed" for case in run["cases"])
    run["failed"] = sum(case["status"] == "failed" for case in run["cases"])
    _write_json(summary_path, run)
    print(json.dumps(run, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
