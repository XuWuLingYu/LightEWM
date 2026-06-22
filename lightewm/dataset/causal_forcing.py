import csv
import json
import os
from pathlib import Path
from typing import Iterable


class CausalForcingJsonlAdapter:
    """Metadata adapter config for Causal-Forcing JSONL generation."""

    def __init__(
        self,
        base_path: str,
        metadata_path: str,
        video_key: str = "video",
        prompt_key: str = "prompt",
        fallback_prompt_keys: list[str] | None = None,
        max_samples: int | None = None,
        jsonl_base_path: str | None = None,
        filter_key: str | None = None,
        filter_value: str | None = None,
        proprio_stats_path: str | None = None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.video_key = video_key
        self.prompt_key = prompt_key
        self.fallback_prompt_keys = fallback_prompt_keys
        self.max_samples = max_samples
        self.jsonl_base_path = jsonl_base_path
        self.filter_key = filter_key
        self.filter_value = filter_value
        self.proprio_stats_path = proprio_stats_path

    def write_jsonl(self, output_path: str) -> int:
        return write_causal_forcing_jsonl(
            base_path=self.base_path,
            metadata_path=self.metadata_path,
            output_path=output_path,
            video_key=self.video_key,
            prompt_key=self.prompt_key,
            fallback_prompt_keys=self.fallback_prompt_keys,
            max_samples=self.max_samples,
            jsonl_base_path=self.jsonl_base_path,
            filter_key=self.filter_key,
            filter_value=self.filter_value,
            proprio_stats_path=self.proprio_stats_path,
        )


def _load_metadata_rows(metadata_path: str) -> list[dict]:
    path = Path(metadata_path)
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected list metadata in {metadata_path}")
        return data
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _first_present(row: dict, keys: Iterable[str]):
    for key in keys:
        value = row.get(key)
        if value is not None and str(value) != "":
            return value
    return None


def resolve_video_path(base_path: str, value: str) -> str:
    if os.path.isabs(value):
        return value
    return str(Path(base_path) / value)


def _resolve_metadata_path(
    *,
    base_path: str,
    metadata_path: str,
    output_path: str,
    value: str,
) -> str:
    path = Path(value)
    if path.is_absolute():
        if path.exists():
            return os.path.relpath(path.resolve(), start=Path(output_path).parent.resolve())
        try:
            return os.path.relpath(path, start=Path(output_path).parent.resolve())
        except ValueError:
            return str(path)

    candidates = [
        Path(base_path) / path,
        Path(metadata_path).parent / path,
        path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return os.path.relpath(candidate.resolve(), start=Path(output_path).parent.resolve())
    return os.path.relpath((Path(base_path) / path).resolve(), start=Path(output_path).parent.resolve())


def write_causal_forcing_jsonl(
    *,
    base_path: str,
    metadata_path: str,
    output_path: str,
    video_key: str = "video",
    prompt_key: str = "prompt",
    fallback_prompt_keys: list[str] | None = None,
    max_samples: int | None = None,
    jsonl_base_path: str | None = None,
    filter_key: str | None = None,
    filter_value: str | None = None,
    proprio_stats_path: str | None = None,
) -> int:
    rows = _load_metadata_rows(metadata_path)
    if filter_key is not None:
        rows = [row for row in rows if str(row.get(filter_key, "")) == str(filter_value)]
    if max_samples is not None:
        rows = rows[: int(max_samples)]

    prompt_keys = [prompt_key] + list(fallback_prompt_keys or ["dense_prompt", "sparse_prompt", "caption"])
    video_keys = [video_key, "video_path", "path"]
    passthrough_path_keys = (
        "action_path",
        "action_stats_path",
        "proprio_stats_path",
        "source_file",
        "video_latent_cache_path",
        "preencoded_cache_path",
    )
    passthrough_scalar_keys = (
        "action_shape",
        "demo_id",
        "camera_key",
        "num_frames",
        "dense_prompt",
        "sparse_prompt",
        "joint_action_stats_path",
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            video_value = _first_present(row, video_keys)
            prompt_value = _first_present(row, prompt_keys)
            if video_value is None:
                raise KeyError(f"Metadata row {idx} has no video path in keys: {video_keys}")
            if prompt_value is None:
                raise KeyError(f"Metadata row {idx} has no prompt in keys: {prompt_keys}")
            item = {
                "prompt": str(prompt_value),
                "video_path": _resolve_metadata_path(
                    base_path=jsonl_base_path or base_path,
                    metadata_path=metadata_path,
                    output_path=output_path,
                    value=str(video_value),
                ),
            }
            for key in passthrough_path_keys:
                value = row.get(key)
                if value is not None and str(value) != "":
                    item[key] = _resolve_metadata_path(
                        base_path=base_path,
                        metadata_path=metadata_path,
                        output_path=output_path,
                        value=str(value),
                    )
            if "proprio_stats_path" not in item and proprio_stats_path:
                item["proprio_stats_path"] = _resolve_metadata_path(
                    base_path=base_path,
                    metadata_path=metadata_path,
                    output_path=output_path,
                    value=str(proprio_stats_path),
                )
            for key in passthrough_scalar_keys:
                value = row.get(key)
                if value is not None and str(value) != "":
                    item[key] = value
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    return count
