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
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.video_key = video_key
        self.prompt_key = prompt_key
        self.fallback_prompt_keys = fallback_prompt_keys
        self.max_samples = max_samples
        self.jsonl_base_path = jsonl_base_path

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
) -> int:
    rows = _load_metadata_rows(metadata_path)
    if max_samples is not None:
        rows = rows[: int(max_samples)]

    prompt_keys = [prompt_key] + list(fallback_prompt_keys or ["dense_prompt", "sparse_prompt", "caption"])
    video_keys = [video_key, "video_path", "path"]

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
                "video_path": resolve_video_path(jsonl_base_path or base_path, str(video_value)),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    return count
