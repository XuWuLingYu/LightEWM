#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import h5py
import numpy as np


def fps_indices(num_frames: int, source_fps: float, target_fps: float) -> np.ndarray:
    if num_frames <= 0:
        return np.zeros([0], dtype=np.int64)
    if target_fps >= source_fps:
        return np.arange(num_frames, dtype=np.int64)
    step = float(source_fps) / float(target_fps)
    return np.unique(np.floor(np.arange(0, num_frames, step)).astype(np.int64).clip(0, num_frames - 1))


def resolve_existing(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base / path


def relpath_for_csv(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), start=base.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-metadata", default="data/libero_i2v_train/metadata_dense_prompt.csv")
    parser.add_argument("--output-metadata", default="data/libero_i2v_train/metadata_dense_prompt_hdr_video_action_joint.csv")
    parser.add_argument("--stats-path", default="data/libero_i2v_train/hdr_video_action_joint_action_stats.json")
    parser.add_argument("--source-fps", type=float, default=16.0)
    parser.add_argument("--target-fps", type=float, default=16.0)
    parser.add_argument("--camera-key", default="agentview_rgb")
    parser.add_argument(
        "--include-source-contains",
        default="",
        help="Comma-separated substrings; keep rows whose source_file contains any of them.",
    )
    parser.add_argument(
        "--exclude-source-contains",
        default="",
        help="Comma-separated substrings; drop rows whose source_file contains any of them.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_metadata)
    output_path = Path(args.output_metadata)
    stats_path = Path(args.stats_path)
    base_dir = input_path.parent
    include_source = [part.strip() for part in args.include_source_contains.split(",") if part.strip()]
    exclude_source = [part.strip() for part in args.exclude_source_contains.split(",") if part.strip()]

    rows = []
    action_chunks = []
    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("camera_key", "")) != args.camera_key:
                continue
            source_value = str(row.get("source_file", ""))
            if include_source and not any(part in source_value for part in include_source):
                continue
            if exclude_source and any(part in source_value for part in exclude_source):
                continue
            source_file = resolve_existing(base_dir, row["source_file"])
            demo_id = row["demo_id"]
            with h5py.File(source_file, "r") as h5:
                demo = h5["data"][demo_id]
                frame_count = int(min(demo["actions"].shape[0], demo["obs"][args.camera_key].shape[0]))
                indices = fps_indices(frame_count, args.source_fps, args.target_fps)
                if indices.size == 0:
                    continue
                action_chunks.append(np.asarray(demo["actions"][indices], dtype=np.float32))
            out_row = dict(row)
            out_row["source_file"] = relpath_for_csv(source_file, base_dir)
            out_row["action_stats_path"] = str(stats_path)
            out_row.pop("action_path", None)
            out_row.pop("action_shape", None)
            rows.append(out_row)

    if not action_chunks:
        raise RuntimeError(f"No agentview action rows found in {input_path}.")
    actions = np.concatenate(action_chunks, axis=0).astype(np.float32)
    stats = {
        "source": "raw_libero_controller_actions_fps16",
        "source_fps": float(args.source_fps),
        "target_fps": float(args.target_fps),
        "camera_key": args.camera_key,
        "include_source_contains": include_source,
        "exclude_source_contains": exclude_source,
        "num_rows": len(rows),
        "num_actions": int(actions.shape[0]),
        "min": actions.min(axis=0).tolist(),
        "max": actions.max(axis=0).tolist(),
        "mean": actions.mean(axis=0).tolist(),
        "std": actions.std(axis=0).tolist(),
        "eps": 1.0e-6,
        "normalization": "minmax_to_minus1_plus1_then_clip",
    }

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    if "action_stats_path" not in fieldnames:
        fieldnames.append("action_stats_path")
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"Wrote action stats for {actions.shape[0]} actions to {stats_path}")


if __name__ == "__main__":
    main()
