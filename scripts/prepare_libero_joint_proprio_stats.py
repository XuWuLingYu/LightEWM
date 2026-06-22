#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(0.0, 1.0 - float(quat[3] * quat[3])))
    if den < 1e-6:
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * np.arccos(quat[3]) / den)).astype(np.float32)


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path


def read_rows(metadata_path: Path, camera_key: str | None) -> list[dict]:
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if camera_key is not None:
        rows = [row for row in rows if row.get("camera_key") == camera_key]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/libero_i2v_train/metadata_dense_prompt_hdr_video_action_joint.csv")
    parser.add_argument("--output", default="data/libero_i2v_train/hdr_video_action_joint_proprio_stats.json")
    parser.add_argument("--camera-key", default="agentview_rgb")
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    rows = read_rows(metadata_path, args.camera_key)
    base_dir = metadata_path.parent
    values = []
    seen = set()
    for row in rows:
        source_file = row.get("source_file")
        demo_id = row.get("demo_id")
        if not source_file or not demo_id:
            continue
        source_path = resolve_path(base_dir, source_file).resolve()
        key = (str(source_path), str(demo_id))
        if key in seen:
            continue
        seen.add(key)
        with h5py.File(source_path, "r") as f:
            obs = f["data"][str(demo_id)]["obs"]
            if "ee_pos" in obs and "ee_ori" in obs and "gripper_states" in obs:
                eef_pos = np.asarray(obs["ee_pos"], dtype=np.float32)
                axis_angle = np.asarray(obs["ee_ori"], dtype=np.float32)
                gripper_qpos = np.asarray(obs["gripper_states"], dtype=np.float32)
            else:
                eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
                eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
                gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
                axis_angle = np.stack([quat_to_axis_angle(quat) for quat in eef_quat], axis=0)
            values.append(np.concatenate([eef_pos, axis_angle, gripper_qpos], axis=1))

    if not values:
        raise RuntimeError(f"No proprio values found in {metadata_path}.")
    data = np.concatenate(values, axis=0).astype(np.float32)
    stats = {
        "type": "libero_proprio_minmax",
        "dims": ["eef_x", "eef_y", "eef_z", "axis_x", "axis_y", "axis_z", "gripper_qpos0", "gripper_qpos1"],
        "count": int(data.shape[0]),
        "min": data.min(axis=0).tolist(),
        "max": data.max(axis=0).tolist(),
        "mean": data.mean(axis=0).tolist(),
        "std": data.std(axis=0).tolist(),
        "eps": 1e-6,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")
    print(f"Wrote {output} from {len(seen)} demos, {data.shape[0]} states.")


if __name__ == "__main__":
    main()
