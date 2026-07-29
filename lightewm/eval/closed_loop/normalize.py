from __future__ import annotations

from pathlib import Path

from .parsers import (
    parse_libero_results,
    parse_robolab_episode_results,
    parse_robotwin_collection_log,
    parse_robotwin_progress_log,
)
from .records import EpisodeRecord


def normalize_native_results(
    benchmark: str,
    input_path: str | Path,
    *,
    revision: str,
    policy: str,
    protocol: str,
    task: str | None = None,
    setting: str = "default",
    artifacts_dir: str | Path | None = None,
) -> list[EpisodeRecord]:
    benchmark = benchmark.lower()
    path = Path(input_path)
    if benchmark == "libero":
        return parse_libero_results(
            path, revision=revision, policy=policy, protocol=protocol
        )
    if benchmark == "robolab":
        return parse_robolab_episode_results(
            path,
            revision=revision,
            policy=policy,
            protocol=protocol,
            setting=setting,
        )
    if benchmark == "robotwin":
        if not task:
            raise ValueError("RoboTwin normalization requires task")
        text = path.read_text(encoding="utf-8", errors="replace")
        parser = (
            parse_robotwin_progress_log
            if "Success rate:" in text
            else parse_robotwin_collection_log
        )
        return parser(
            text,
            revision=revision,
            policy=policy,
            task=task,
            setting=setting,
            protocol=protocol,
            artifacts_dir=artifacts_dir,
        )
    raise ValueError(f"unsupported closed-loop benchmark: {benchmark}")
