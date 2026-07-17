#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_TASK = "A robot creates art painting with a brush."
TWO_PI = np.float32(2.0 * np.pi)
CAMERAS = (
    ("observation.images.image", "right"),
    ("observation.images.wrist_image", "wrist"),
    ("observation.images.front_image", "front"),
)


def list_array(values: np.ndarray) -> pa.Array:
    return pa.array([np.asarray(value, dtype=np.float32).tolist() for value in values], type=pa.list_(pa.float32()))


def stats(values: np.ndarray) -> dict:
    values = np.asarray(values)
    flat = values.reshape(values.shape[0], -1) if values.ndim > 1 else values[:, None]
    return {
        "min": flat.min(axis=0).astype(float).tolist(),
        "max": flat.max(axis=0).astype(float).tolist(),
        "mean": flat.mean(axis=0).astype(float).tolist(),
        "std": flat.std(axis=0).astype(float).tolist(),
        "count": [int(flat.shape[0])],
    }


def video_feature(height: int, width: int, fps: int) -> dict:
    return {
        "dtype": "video",
        "shape": [height, width, 3],
        "names": ["height", "width", "rgb"],
        "info": {
            "video.height": height,
            "video.width": width,
            "video.codec": "mpeg4",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
        },
    }


def info(total_episodes: int, total_frames: int, fps: int, height: int, width: int, robot_type: str) -> dict:
    features = {key: video_feature(height, width, fps) for key, _ in CAMERAS}
    features.update({
        "observation.state": {"dtype": "float32", "shape": [8], "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper", "gripper_aux"]}},
        "observation.states.ee_state": {"dtype": "float32", "shape": [6], "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw"]}},
        "observation.states.gripper_state": {"dtype": "float32", "shape": [2], "names": {"motors": ["gripper", "gripper_aux"]}},
        "action": {"dtype": "float32", "shape": [7], "names": {"motors": ["ee_delta_x", "ee_delta_y", "ee_delta_z", "ee_delta_roll", "ee_delta_pitch", "ee_delta_yaw", "gripper_abs"]}},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    })
    return {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_episodes * len(CAMERAS),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def complete_episodes(root: Path) -> list[Path]:
    episodes = []
    for episode in sorted(path for path in root.iterdir() if path.is_dir()):
        if not (episode / "replay.hdf5").is_file():
            continue
        if all((episode / "videos" / f"observation.images.{name}.mp4").is_file() for _, name in CAMERAS):
            episodes.append(episode)
    return episodes


def read_episode(episode: Path, pause_phase: str) -> dict:
    with h5py.File(episode / "replay.hdf5", "r") as handle:
        pose = np.asarray(handle["state/ee_pose_euler"][:], dtype=np.float32)
        grip = np.asarray(handle["state/gripper_width"][:], dtype=np.float32)
        aux = np.asarray(handle["state/gripper_target_width"][:], dtype=np.float32)
        phases = [value.decode() if isinstance(value, bytes) else str(value) for value in handle["meta/phase"][:]]
        fps = int(round(float(handle.attrs.get("fps", handle.attrs.get("record_fps", 10.0)))))
    keep = np.asarray([phase != pause_phase for phase in phases], dtype=bool)
    if int(keep.sum()) < 2:
        raise ValueError(f"{episode} has fewer than two non-pause frames")
    pose = pose[keep]
    grip = grip[keep]
    aux = aux[keep]
    delta = np.zeros_like(pose)
    delta[:-1, :3] = pose[1:, :3] - pose[:-1, :3]
    delta[:-1, 3:6] = ((pose[1:, 3:6] - pose[:-1, 3:6] + np.pi) % TWO_PI - np.pi).astype(np.float32)
    delta[-1] = delta[-2]
    grip_action = np.empty((len(grip),), dtype=np.float32)
    grip_action[:-1] = grip[1:]
    grip_action[-1] = grip[-1]
    return {
        "keep": keep,
        "state": np.concatenate([pose, grip[:, None], aux[:, None]], axis=1),
        "ee": pose,
        "gripper_state": np.stack([grip, aux], axis=1),
        "action": np.concatenate([delta, grip_action[:, None]], axis=1),
        "fps": fps,
    }


def write_filtered_video(source: Path, destination: Path, keep: np.ndarray, fps: int) -> tuple[int, int, int]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_frames != len(keep):
        raise ValueError(f"{source}: {source_frames} video frames != {len(keep)} metadata frames")
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer {destination}")
    written = 0
    for index, retain in enumerate(keep):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"{source}: decode failed at frame {index}")
        if retain:
            writer.write(frame)
            written += 1
    capture.release()
    writer.release()
    if written != int(keep.sum()):
        raise RuntimeError(f"{source}: wrote {written}, expected {int(keep.sum())}")
    return written, height, width


def write_split(episodes: list[Path], root: Path, pause_phase: str, task: str, robot_type: str) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    for key, _ in CAMERAS:
        (root / "videos/chunk-000" / key).mkdir(parents=True)
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": task}) + "\n")
    total_frames = 0
    episode_rows = []
    stats_rows = []
    first_video = None
    for episode_index, episode in enumerate(episodes):
        record = read_episode(episode, pause_phase)
        length = int(record["action"].shape[0])
        frame_index = np.arange(length, dtype=np.int64)
        global_index = np.arange(total_frames, total_frames + length, dtype=np.int64)
        table = pa.table({
            "observation.state": list_array(record["state"]),
            "observation.states.ee_state": list_array(record["ee"]),
            "observation.states.gripper_state": list_array(record["gripper_state"]),
            "action": list_array(record["action"]),
            "timestamp": pa.array(np.arange(length, dtype=np.float32) / record["fps"], type=pa.float32()),
            "frame_index": pa.array(frame_index, type=pa.int64()),
            "episode_index": pa.array(np.full(length, episode_index, dtype=np.int64), type=pa.int64()),
            "index": pa.array(global_index, type=pa.int64()),
            "task_index": pa.array(np.zeros(length, dtype=np.int64), type=pa.int64()),
        })
        pq.write_table(table, root / "data/chunk-000" / f"episode_{episode_index:06d}.parquet")
        for key, camera in CAMERAS:
            source = episode / "videos" / f"observation.images.{camera}.mp4"
            destination = root / "videos/chunk-000" / key / f"episode_{episode_index:06d}.mp4"
            written, height, width = write_filtered_video(source, destination, record["keep"], record["fps"])
            if first_video is None:
                first_video = (height, width, record["fps"])
            if written != length:
                raise AssertionError(f"{episode}: {camera} length mismatch")
        episode_rows.append({"episode_index": episode_index, "tasks": [task], "length": length, "source_episode": episode.name})
        stats_rows.append({"episode_index": episode_index, "stats": {
            "observation.state": stats(record["state"]),
            "observation.states.ee_state": stats(record["ee"]),
            "observation.states.gripper_state": stats(record["gripper_state"]),
            "action": stats(record["action"]),
            "timestamp": stats(np.arange(length, dtype=np.float32) / record["fps"]),
            "frame_index": stats(frame_index),
            "episode_index": stats(np.full(length, episode_index, dtype=np.int64)),
            "index": stats(global_index),
            "task_index": stats(np.zeros(length, dtype=np.int64)),
        }})
        total_frames += length
        print(f"[{episode_index + 1}/{len(episodes)}] {episode.name}: {len(record['keep'])} -> {length} frames", flush=True)
    if first_video is None:
        raise ValueError("No episodes written")
    height, width, fps = first_video
    (root / "meta/info.json").write_text(json.dumps(info(len(episodes), total_frames, fps, height, width, robot_type), indent=2) + "\n")
    with (root / "meta/episodes.jsonl").open("w") as handle:
        for row in episode_rows:
            handle.write(json.dumps(row) + "\n")
    with (root / "meta/episodes_stats.jsonl").open("w") as handle:
        for row in stats_rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--heldout-episode", default=None)
    parser.add_argument("--pause-phase", default="demo_empty_action_pause")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--robot-type", default="realbot_painting_new100_3cam")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    episodes = complete_episodes(args.input_root)
    if len(episodes) < 2:
        raise ValueError(f"Need at least two complete episodes, found {len(episodes)}")
    heldout = next((episode for episode in episodes if episode.name == args.heldout_episode), None) if args.heldout_episode else episodes[-1]
    if heldout is None:
        raise ValueError(f"Heldout episode not found: {args.heldout_episode}")
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {args.output_root}; pass --overwrite to replace it")
        shutil.rmtree(args.output_root)
    train = [episode for episode in episodes if episode != heldout]
    args.output_root.mkdir(parents=True)
    (args.output_root / "heldout_episode.txt").write_text(heldout.name + "\n")
    write_split(train, args.output_root / "train", args.pause_phase, args.task, args.robot_type)
    write_split([heldout], args.output_root / "heldout", args.pause_phase, args.task, args.robot_type)
    print(json.dumps({"train_episodes": len(train), "heldout": heldout.name, "output": str(args.output_root)}, indent=2))


if __name__ == "__main__":
    main()
