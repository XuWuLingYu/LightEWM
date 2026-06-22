#!/usr/bin/env python3
"""Prepare fixed-size LIBERO action targets for HDR-ActionMoT."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-path", default="data/libero_i2v_train/metadata_dense_prompt.csv")
    parser.add_argument("--libero-root", default="data/LIBERO-datasets")
    parser.add_argument("--output-dir", default="data/libero_i2v_train/hdr_actions_leaf8")
    parser.add_argument("--output-metadata-path", default="data/libero_i2v_train/metadata_dense_prompt_hdr_actions_leaf8.csv")
    parser.add_argument("--stats-path", default="data/libero_i2v_train/hdr_actions_leaf8_stats.json")
    parser.add_argument("--latent-leaf-frames", type=int, default=13)
    parser.add_argument("--actions-per-leaf", type=int, default=8)
    parser.add_argument("--source-fps", type=float, default=16.0)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--video-frames", type=int, default=49)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def resolve_h5_path(source_file: str, libero_root: Path) -> Path:
    source = Path(source_file)
    parts = source.parts
    if "LIBERO-datasets" in parts:
        rel = Path(*parts[parts.index("LIBERO-datasets") + 1 :])
        candidate = libero_root / rel
        if candidate.exists():
            return candidate
    if source.exists():
        return source
    candidate = libero_root / source.name
    if candidate.exists():
        return candidate
    matches = list(libero_root.rglob(source.name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Cannot resolve hdf5 source: {source_file}")


def converted_frame_count(raw_frames: int, source_fps: float, target_fps: float) -> int:
    if target_fps >= source_fps:
        return raw_frames
    step = source_fps / target_fps
    indices = np.floor(np.arange(0, raw_frames, step)).astype(np.int64)
    indices = np.clip(indices, 0, raw_frames - 1)
    return int(np.unique(indices).shape[0])


def model_frame_raw_times(raw_frames: int, source_fps: float, target_fps: float, video_frames: int) -> np.ndarray:
    if target_fps >= source_fps:
        converted = np.arange(raw_frames, dtype=np.float64)
    else:
        step = source_fps / target_fps
        converted = np.unique(np.clip(np.floor(np.arange(0, raw_frames, step)).astype(np.int64), 0, raw_frames - 1))
        converted = converted.astype(np.float64)
    if converted.shape[0] == 1:
        return np.zeros(video_frames, dtype=np.float64)
    frame_ids = np.linspace(0, converted.shape[0] - 1, video_frames)
    return np.interp(frame_ids, np.arange(converted.shape[0], dtype=np.float64), converted)


def interpolate_sequence(values: np.ndarray, sample_times: np.ndarray) -> np.ndarray:
    raw_t = np.arange(values.shape[0], dtype=np.float64)
    interp = np.stack(
        [np.interp(sample_times.reshape(-1), raw_t, values[:, i]) for i in range(values.shape[1])],
        axis=-1,
    )
    return interp.reshape(*sample_times.shape, values.shape[1]).astype(np.float32)


def make_uniform_action_times(
    model_times: np.ndarray,
    latent_leaf_frames: int,
    actions_per_leaf: int,
) -> np.ndarray:
    total_actions = latent_leaf_frames * actions_per_leaf
    if total_actions <= 0:
        raise ValueError("latent_leaf_frames * actions_per_leaf must be positive.")
    action_times = np.linspace(model_times[0], model_times[-1], total_actions, dtype=np.float64)
    return action_times.reshape(latent_leaf_frames, actions_per_leaf)


def process_demo(
    h5_path: Path,
    demo_id: str,
    *,
    source_fps: float,
    target_fps: float,
    video_frames: int,
    latent_leaf_frames: int,
    actions_per_leaf: int,
) -> dict[str, np.ndarray | int]:
    with h5py.File(h5_path, "r") as f:
        demo = f["data"][demo_id]
        raw_actions = demo["actions"][...].astype(np.float64)

    if raw_actions.ndim != 2 or raw_actions.shape[1] != 7:
        raise ValueError(f"Expected LIBERO actions shape [T, 7], got {raw_actions.shape} in {h5_path}:{demo_id}")

    raw_frames = int(raw_actions.shape[0])
    model_times = model_frame_raw_times(raw_frames, source_fps, target_fps, video_frames)
    leaf_action_times = make_uniform_action_times(model_times, latent_leaf_frames, actions_per_leaf)
    controller_interp = interpolate_sequence(raw_actions, leaf_action_times).reshape(-1, 7).astype(np.float32)
    actions = controller_interp.reshape(
        latent_leaf_frames,
        actions_per_leaf,
        7,
    )
    return {
        "action": actions.astype(np.float32),
        "raw_controller_action": controller_interp.astype(np.float32).reshape(latent_leaf_frames, actions_per_leaf, 7),
        "leaf_action_times": leaf_action_times.astype(np.float32),
        "raw_frames": raw_frames,
        "converted_frames": converted_frame_count(raw_frames, source_fps, target_fps),
    }


def safe_stem(row: dict[str, str]) -> str:
    source = Path(row["source_file"]).stem
    demo_id = str(row["demo_id"])
    return f"{source}__{demo_id}"


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata_path)
    libero_root = Path(args.libero_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = Path(args.stats_path)
    output_metadata_path = Path(args.output_metadata_path)

    rows = load_rows(metadata_path)
    cache: dict[tuple[str, str], Path] = {}
    all_actions = []
    out_rows = []
    missing = 0

    for idx, row in enumerate(rows):
        key = (row["source_file"], row["demo_id"])
        if key not in cache:
            h5_path = resolve_h5_path(row["source_file"], libero_root)
            rel_suite = h5_path.parent.name
            out_path = output_dir / rel_suite / f"{safe_stem(row)}.npz"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if args.overwrite or not out_path.exists():
                processed = process_demo(
                    h5_path,
                    row["demo_id"],
                    source_fps=args.source_fps,
                    target_fps=args.target_fps,
                    video_frames=args.video_frames,
                    latent_leaf_frames=args.latent_leaf_frames,
                    actions_per_leaf=args.actions_per_leaf,
                )
                np.savez_compressed(out_path, **processed)
            loaded = np.load(out_path)
            all_actions.append(loaded["action"].reshape(-1, loaded["action"].shape[-1]))
            cache[key] = out_path
        action_path = cache[key]
        out_row = dict(row)
        out_row["action_path"] = action_path.as_posix()
        out_row["action_shape"] = f"{args.latent_leaf_frames},{args.actions_per_leaf},7"
        out_row["action_stats_path"] = stats_path.as_posix()
        out_rows.append(out_row)
        if (idx + 1) % 1000 == 0:
            print(f"[HDRAction] processed metadata rows {idx + 1}/{len(rows)} unique_demos={len(cache)}")

    if not all_actions:
        raise RuntimeError("No actions were generated.")
    action_values = np.concatenate(all_actions, axis=0)
    action_min = action_values.min(axis=0)
    action_max = action_values.max(axis=0)
    stats = {
        "normalization": "minmax",
        "action_layout": "raw_libero_controller_xyz_axisangle_plus_raw_gripper",
        "action": "interpolated_raw_libero_controller_actions_no_delta",
        "gripper": "raw_libero_action_dim6_interpolated_no_remap",
        "latent_leaf_frames": args.latent_leaf_frames,
        "actions_per_leaf": args.actions_per_leaf,
        "action_dim": 7,
        "source_fps": args.source_fps,
        "target_fps": args.target_fps,
        "video_frames": args.video_frames,
        "min": action_min.tolist(),
        "max": action_max.tolist(),
        "eps": 1e-6,
        "unique_demos": len(cache),
        "metadata_rows": len(rows),
        "missing": missing,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    fieldnames = list(out_rows[0].keys())
    output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with output_metadata_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"[HDRAction] wrote metadata: {output_metadata_path}")
    print(f"[HDRAction] wrote stats: {stats_path}")
    print(f"[HDRAction] unique demos: {len(cache)} rows: {len(rows)}")
    print(f"[HDRAction] action min: {action_min.tolist()}")
    print(f"[HDRAction] action max: {action_max.tolist()}")


if __name__ == "__main__":
    main()
