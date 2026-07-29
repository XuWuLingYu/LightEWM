from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from .records import ArtifactRef, EpisodeRecord


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
ROBOTWIN_PROGRESS = re.compile(
    r"Success rate:\s*(?P<successes>\d+)/(?P<trials>\d+).*?"
    r"current seed:\s*(?P<seed>\d+)"
)
ROBOTWIN_COLLECTION_SUCCESS = re.compile(
    r"simulate data episode\s+(?P<episode>\d+)\s+success!\s+"
    r"\(seed\s*=\s*(?P<seed>\d+)\)"
)


def _robotwin_episode_artifacts(
    artifact_root: Path | None,
    episode: int,
) -> tuple[ArtifactRef, ...]:
    if artifact_root is None or not artifact_root.exists():
        return ()
    kinds = {
        ".h5": "hdf5",
        ".hdf5": "hdf5",
        ".json": "event_log",
        ".mp4": "video",
    }
    artifacts = []
    candidates = set(artifact_root.rglob(f"episode{episode}.*"))
    candidates.update(artifact_root.rglob(f"run_{episode}.*"))
    for candidate in sorted(candidates):
        kind = kinds.get(candidate.suffix.lower())
        if candidate.is_file() and kind:
            artifacts.append(ArtifactRef(kind=kind, path=str(candidate)))
    return tuple(artifacts)


def parse_robotwin_collection_log(
    text: str,
    *,
    revision: str,
    policy: str,
    task: str,
    setting: str,
    protocol: str = "ground-truth-v1",
    artifacts_dir: str | Path | None = None,
) -> list[EpisodeRecord]:
    cleaned = ANSI_ESCAPE.sub("", text)
    artifact_root = Path(artifacts_dir) if artifacts_dir else None
    records = []
    for match in ROBOTWIN_COLLECTION_SUCCESS.finditer(cleaned):
        episode = int(match.group("episode"))
        seed = int(match.group("seed"))
        records.append(
            EpisodeRecord(
                benchmark="robotwin",
                benchmark_revision=revision,
                protocol=protocol,
                policy=policy,
                task=task,
                setting=setting,
                seed=seed,
                episode_id=str(episode),
                success=True,
                status="completed",
                artifacts=_robotwin_episode_artifacts(artifact_root, episode),
                source={
                    "official_entrypoint": "script/collect_data.py",
                    "check_success": True,
                    "plan_success": True,
                },
            )
        )
    if not records:
        raise ValueError("no successful RoboTwin collected trajectory found")
    return records


def parse_robotwin_progress_log(
    text: str,
    *,
    revision: str,
    policy: str,
    task: str,
    setting: str,
    protocol: str = "reference-smoke-v1",
    artifacts_dir: str | Path | None = None,
) -> list[EpisodeRecord]:
    cleaned = ANSI_ESCAPE.sub("", text)
    records: list[EpisodeRecord] = []
    previous_successes = 0
    previous_trials = 0
    artifact_root = Path(artifacts_dir) if artifacts_dir else None
    for match in ROBOTWIN_PROGRESS.finditer(cleaned):
        successes = int(match.group("successes"))
        trials = int(match.group("trials"))
        seed = int(match.group("seed"))
        if trials != previous_trials + 1:
            raise ValueError(
                f"non-contiguous RoboTwin trial count: {previous_trials} -> {trials}"
            )
        success_delta = successes - previous_successes
        if success_delta not in {0, 1}:
            raise ValueError(
                f"invalid RoboTwin success count delta: {previous_successes} -> {successes}"
            )
        artifacts = _robotwin_episode_artifacts(artifact_root, trials - 1)
        records.append(
            EpisodeRecord(
                benchmark="robotwin",
                benchmark_revision=revision,
                protocol=protocol,
                policy=policy,
                task=task,
                setting=setting,
                seed=seed,
                episode_id=str(trials - 1),
                success=bool(success_delta),
                status="completed",
                artifacts=artifacts,
                source={
                    "cumulative_successes": successes,
                    "cumulative_trials": trials,
                },
            )
        )
        previous_successes = successes
        previous_trials = trials
    if not records:
        raise ValueError("no RoboTwin episode progress found")
    return records


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def parse_robolab_episode_results(
    path: str | Path,
    *,
    revision: str,
    policy: str,
    protocol: str,
    setting: str = "default",
) -> list[EpisodeRecord]:
    result_path = Path(path)
    rows = _load_jsonl(result_path)
    records = []
    for index, row in enumerate(rows):
        run = int(row.get("run", row.get("episode", index)))
        env_id = int(row.get("env_id", 0))
        task = str(row.get("env_name", row.get("task", "unknown")))
        task_dir = result_path.parent / task
        artifact_candidates = [
            ("event_log", task_dir / f"log_{run}_env{env_id}.json"),
            ("hdf5", task_dir / f"run_{run}.hdf5"),
        ]
        instruction = str(row.get("instruction", ""))
        if instruction:
            cleaned_instruction = re.sub(r"[^\w\s]", "", instruction).replace(" ", "_")
            for suffix in (f"_{run}", f"_{run}_env{env_id}"):
                artifact_candidates.extend(
                    (
                        ("video", task_dir / f"{cleaned_instruction}{suffix}.mp4"),
                        (
                            "viewport_video",
                            task_dir
                            / f"{cleaned_instruction}{suffix}_viewport.mp4",
                        ),
                    )
                )
        artifacts = tuple(
            ArtifactRef(kind=kind, path=str(candidate))
            for kind, candidate in artifact_candidates
            if candidate.exists()
        )
        metrics = {
            str(key): float(value)
            for key, value in dict(row.get("metrics") or {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if row.get("score") is not None:
            metrics["subtask_score"] = float(row["score"])
        steps = row.get("episode_step", row.get("step"))
        records.append(
            EpisodeRecord(
                benchmark="robolab",
                benchmark_revision=revision,
                protocol=protocol,
                policy=policy,
                task=task,
                setting=setting,
                seed=int(row.get("seed", run)),
                episode_id=str(row.get("episode", run)),
                success=bool(row["success"]) if row.get("success") is not None else None,
                status=str(row.get("status", "completed")),
                steps=int(steps) if steps is not None else None,
                metrics=metrics,
                artifacts=artifacts,
                failure_reason=row.get("reason", row.get("failure_reason")),
                source=row,
            )
        )
    return records


def parse_libero_results(
    path: str | Path,
    *,
    revision: str,
    policy: str,
    protocol: str,
) -> list[EpisodeRecord]:
    result_path = Path(path)
    if result_path.suffix.lower() == ".csv":
        return _parse_libero_episode_csv(
            result_path,
            revision=revision,
            policy=policy,
            protocol=protocol,
        )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    default_setting = str(
        payload.get("task_suite_name", payload.get("suite_name", "default"))
    )
    task_rows = payload.get("task_results", payload.get("per_task", []))
    suite_stats = dict(payload.get("suite_stats", {}))
    if isinstance(task_rows, dict):
        task_items = [
            (
                task_key,
                row,
                _libero_suite_for_task_key(task_key, suite_stats) or default_setting,
            )
            for task_key, row in task_rows.items()
        ]
    else:
        task_items = [
            (str(task_index), row, default_setting)
            for task_index, row in enumerate(task_rows)
        ]
    records = []
    for task_index, (task_key, row, setting) in enumerate(task_items):
        successes = int(row.get("successes", row.get("num_successes", 0)))
        trials = int(
            row.get(
                "trials",
                row.get("num_trials", row.get("total_episodes", 0)),
            )
        )
        task = str(
            row.get(
                "task_description",
                row.get(
                    "task_name",
                    row.get("task", row.get("task_id", task_index)),
                ),
            )
        )
        for episode in range(trials):
            records.append(
                EpisodeRecord(
                    benchmark="libero",
                    benchmark_revision=revision,
                    protocol=protocol,
                    policy=policy,
                    task=task,
                    setting=setting,
                    seed=int(row.get("seed_start", 0)) + episode,
                    episode_id=f"{task_index}:{episode}",
                    success=episode < successes,
                    status="completed",
                    source={
                        "aggregate_expansion": True,
                        "native_result": str(result_path),
                        "native_task_key": task_key,
                        "native_task_result": row,
                    },
                )
            )
    total_trials = int(
        payload.get(
            "total_trials",
            sum(int(row.get("total_trials", 0)) for row in suite_stats.values())
            if suite_stats
            else len(records),
        )
    )
    total_successes = int(
        payload.get(
            "total_successes",
            sum(
                int(row.get("total_successes", 0))
                for row in suite_stats.values()
            )
            if suite_stats
            else sum(record.success for record in records),
        )
    )
    if len(records) != total_trials:
        raise ValueError(
            f"LIBERO task totals ({len(records)}) do not match suite total ({total_trials})"
        )
    if sum(record.success for record in records) != total_successes:
        raise ValueError("LIBERO task success counts do not match suite total")
    for suite, stats in suite_stats.items():
        suite_records = [record for record in records if record.setting == suite]
        expected_trials = int(stats.get("total_trials", len(suite_records)))
        expected_successes = int(
            stats.get(
                "total_successes",
                sum(bool(record.success) for record in suite_records),
            )
        )
        if len(suite_records) != expected_trials:
            raise ValueError(
                f"LIBERO suite {suite} task totals ({len(suite_records)}) "
                f"do not match suite total ({expected_trials})"
            )
        if sum(bool(record.success) for record in suite_records) != expected_successes:
            raise ValueError(f"LIBERO suite {suite} success counts do not match total")
    return records


def _libero_suite_for_task_key(
    task_key: str, suite_stats: dict[str, Any]
) -> str | None:
    matches = [
        suite
        for suite in suite_stats
        if task_key == suite or task_key.startswith(f"{suite}_")
    ]
    return max(matches, key=len) if matches else None


def _parse_libero_episode_csv(
    result_path: Path,
    *,
    revision: str,
    policy: str,
    protocol: str,
) -> list[EpisodeRecord]:
    summary_path = result_path.parent / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )
    with result_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    setting = str(summary.get("suite", summary.get("task_suite_name", "default")))
    default_task = str(summary.get("task", summary.get("task_description", "unknown")))
    records = []
    for index, row in enumerate(rows):
        episode_id = str(row.get("case_id", row.get("episode", index)))
        success_value = str(
            row.get("success", row.get("final_success", ""))
        ).strip().lower()
        if success_value not in {"0", "1", "false", "true"}:
            raise ValueError(
                f"invalid LIBERO success value at {result_path}:{index + 2}: "
                f"{success_value!r}"
            )
        artifacts = []
        for key in ("video_path", "rollout_video", "video"):
            value = row.get(key)
            if value and Path(value).exists():
                artifacts.append(ArtifactRef(kind="video", path=value))
                break
        steps_value = row.get("steps", row.get("num_replay_images"))
        records.append(
            EpisodeRecord(
                benchmark="libero",
                benchmark_revision=revision,
                protocol=protocol,
                policy=policy,
                task=str(row.get("task", default_task)),
                setting=setting,
                seed=int(row.get("seed", row.get("case_id", index))),
                episode_id=episode_id,
                success=success_value in {"1", "true"},
                status=str(row.get("status", "completed")),
                steps=int(steps_value) if steps_value else None,
                artifacts=tuple(artifacts),
                source={
                    "native_result": str(result_path),
                    "native_summary": str(summary_path) if summary_path.exists() else None,
                    "native_episode": row,
                },
            )
        )

    expected_cases = int(summary.get("num_cases", len(records)))
    expected_successes = int(
        summary.get("successes", sum(bool(record.success) for record in records))
    )
    if len(records) != expected_cases:
        raise ValueError(
            f"LIBERO CSV rows ({len(records)}) do not match summary num_cases "
            f"({expected_cases})"
        )
    if sum(bool(record.success) for record in records) != expected_successes:
        raise ValueError("LIBERO CSV successes do not match summary successes")
    return records
