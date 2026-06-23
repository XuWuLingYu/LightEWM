#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a fixed evaluation split CSV and Causal-Forcing JSONL from LightEWM metadata."
    )
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--dataset-base-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--video-key", default="video")
    parser.add_argument("--prompt-key", default="prompt")
    return parser.parse_args()


def resolve_video_path(base_path: Path, value: str) -> str:
    path = Path(str(value))
    if path.is_absolute():
        return str(path)
    return str(base_path / path)


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata_path)
    dataset_base_path = Path(args.dataset_base_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with metadata_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError(f"No rows selected from {metadata_path}")

    split_csv = output_dir / "metadata.csv"
    with split_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    jsonl_path = output_dir / "causal_forcing_dataset.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            item = {
                "prompt": str(row[args.prompt_key]),
                "video_path": resolve_video_path(dataset_base_path, str(row[args.video_key])),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[Split] rows={len(rows)}")
    print(f"[Split] metadata={split_csv}")
    print(f"[Split] causal_jsonl={jsonl_path}")


if __name__ == "__main__":
    main()
