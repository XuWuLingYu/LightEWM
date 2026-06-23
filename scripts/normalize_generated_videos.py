#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".wmv", ".mkv", ".flv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize generated videos to LightEWM evaluator names "
            "(ROWID__DEMOID__CAMERA.mp4), optionally using a canonical metadata CSV "
            "to preserve row ids across shards."
        )
    )
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--canonical-metadata-path", default=None)
    parser.add_argument("--video-key", default="video")
    parser.add_argument("--demo-id-key", default="demo_id")
    parser.add_argument("--camera-key-col", default="camera_key")
    parser.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--recursive", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metadata_key(row: dict, video_key: str, demo_id_key: str, camera_key_col: str) -> tuple[str, str, str]:
    return (
        str(row.get(video_key, "")),
        str(row.get(demo_id_key, "")),
        str(row.get(camera_key_col, "")),
    )


def generated_video_name(row_id: int, demo_id: str, camera_key: str) -> str:
    return f"{int(row_id):06d}__{demo_id}__{camera_key}.mp4"


def list_videos(generated_dir: Path, recursive: bool) -> list[Path]:
    iterator = generated_dir.rglob("*") if recursive else generated_dir.glob("*")
    return sorted(
        path for path in iterator
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def link_or_copy(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata_path)
    generated_dir = Path(args.generated_dir)
    output_dir = Path(args.output_dir)
    rows = read_rows(metadata_path)
    if not rows:
        raise ValueError(f"No metadata rows: {metadata_path}")

    canonical_row_ids: dict[tuple[str, str, str], int] = {}
    if args.canonical_metadata_path:
        canonical_rows = read_rows(Path(args.canonical_metadata_path))
        canonical_row_ids = {
            metadata_key(row, args.video_key, args.demo_id_key, args.camera_key_col): idx
            for idx, row in enumerate(canonical_rows)
        }

    videos = list_videos(generated_dir, recursive=args.recursive)
    if len(videos) < len(rows):
        raise ValueError(
            f"Not enough generated videos in {generated_dir}: found {len(videos)}, need {len(rows)}"
        )

    written = 0
    missing_canonical = []
    for local_row_id, (row, src) in enumerate(zip(rows, videos)):
        key = metadata_key(row, args.video_key, args.demo_id_key, args.camera_key_col)
        if canonical_row_ids:
            if key not in canonical_row_ids:
                missing_canonical.append({"local_row_id": local_row_id, "key": key})
                continue
            row_id = canonical_row_ids[key]
        else:
            row_id = local_row_id
        demo_id = str(row.get(args.demo_id_key, row_id))
        camera_key = str(row.get(args.camera_key_col, "unknown"))
        dst = output_dir / generated_video_name(row_id, demo_id, camera_key)
        link_or_copy(src, dst, args.mode, args.overwrite)
        written += 1

    if missing_canonical:
        raise ValueError(f"{len(missing_canonical)} rows missing from canonical metadata: {missing_canonical[:5]}")
    print(f"[Normalize] source={generated_dir}")
    print(f"[Normalize] output={output_dir}")
    print(f"[Normalize] rows={len(rows)} videos_found={len(videos)} written={written}")


if __name__ == "__main__":
    main()
