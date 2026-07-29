from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ArtifactRef:
    kind: str
    path: str
    media_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactRef":
        return cls(
            kind=str(data["kind"]),
            path=str(data["path"]),
            media_type=data.get("media_type"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class EpisodeRecord:
    benchmark: str
    benchmark_revision: str
    protocol: str
    policy: str
    task: str
    setting: str
    seed: int
    episode_id: str
    success: bool | None
    status: str
    steps: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: tuple[ArtifactRef, ...] = ()
    failure_reason: str | None = None
    source: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"completed", "failed", "error"}:
            raise ValueError(f"unsupported episode status: {self.status}")
        if self.steps is not None and self.steps < 0:
            raise ValueError("episode steps cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["artifacts"] = [asdict(artifact) for artifact in self.artifacts]
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodeRecord":
        payload = dict(data)
        payload["artifacts"] = tuple(
            ArtifactRef.from_dict(value) for value in payload.get("artifacts", ())
        )
        return cls(**payload)


@dataclass(frozen=True)
class ArtifactIndex:
    run_id: str
    policy_spec: str
    policy_spec_hash: str
    benchmark_spec: str
    benchmark_spec_hash: str
    command_log: str
    native_output_dir: str
    episodes_path: str
    summary_path: str
    source_revisions: dict[str, str]
    runtime_versions: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination


def write_episode_records(
    path: str | Path, records: Iterable[EpisodeRecord]
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json() + "\n")
    return destination


def read_episode_records(path: str | Path) -> list[EpisodeRecord]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(EpisodeRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid episode record at {path}:{line_number}: {exc}"
                ) from exc
    return records


def _group_summary(records: list[EpisodeRecord]) -> dict[str, Any]:
    evaluated = [record for record in records if record.success is not None]
    successes = sum(record.success is True for record in evaluated)
    return {
        "total_episodes": len(records),
        "evaluated_episodes": len(evaluated),
        "successes": successes,
        "failures": len(evaluated) - successes,
        "success_rate": successes / len(evaluated) if evaluated else None,
        "errors": sum(record.status == "error" for record in records),
    }


def summarize_episodes(records: Iterable[EpisodeRecord]) -> dict[str, Any]:
    materialized = list(records)
    summary = _group_summary(materialized)
    by_task: dict[str, list[EpisodeRecord]] = defaultdict(list)
    by_setting: dict[str, list[EpisodeRecord]] = defaultdict(list)
    for record in materialized:
        by_task[record.task].append(record)
        by_setting[record.setting].append(record)
    summary["by_task"] = {
        key: _group_summary(value) for key, value in sorted(by_task.items())
    }
    summary["by_setting"] = {
        key: _group_summary(value) for key, value in sorted(by_setting.items())
    }
    return summary


def write_summary(path: str | Path, records: Iterable[EpisodeRecord]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summarize_episodes(records), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination
