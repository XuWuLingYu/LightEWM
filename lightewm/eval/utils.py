from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas
from PIL import Image


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".wmv", ".mkv", ".flv", ".webm"}

@dataclass
class VideoPair:
    row_id: int
    demo_id: str
    camera_key: str
    real_path: str
    generated_path: str


def load_metadata_records(metadata_path: str) -> list[dict]:
    if metadata_path.endswith(".jsonl"):
        records = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    if metadata_path.endswith(".json"):
        with open(metadata_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError(f"Expected a list in JSON metadata: {metadata_path}")
        return records
    metadata = pandas.read_csv(metadata_path)
    return [metadata.iloc[i].to_dict() for i in range(len(metadata))]


def _metadata_video_value(value):
    if isinstance(value, dict):
        return value.get("path") or value.get("video") or value.get("file")
    return value


def resolve_dataset_path(base_path: str, value) -> str:
    value = _metadata_video_value(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError("metadata video field is empty")
    path = str(value)
    if os.path.isabs(path):
        return path
    return os.path.join(base_path, path)


def generated_video_name(row_id: int, demo_id: str, camera_key: str) -> str:
    return f"{int(row_id):06d}__{demo_id}__{camera_key}.mp4"


def find_generated_video(generated_dir: str, row_id: int, demo_id: str, camera_key: str) -> str | None:
    generated_root = Path(generated_dir)
    exact_path = generated_root / generated_video_name(row_id, demo_id, camera_key)
    if exact_path.exists():
        return str(exact_path)

    candidates = sorted(generated_root.glob(f"{int(row_id):06d}__*.mp4"))
    if len(candidates) == 1:
        return str(candidates[0])
    return None


def collect_video_pairs(
    metadata_path: str,
    dataset_base_path: str,
    generated_dir: str,
    *,
    max_samples: int | None = None,
    video_key: str = "video",
    demo_id_key: str = "demo_id",
    camera_key_col: str = "camera_key",
) -> tuple[list[VideoPair], list[dict]]:
    records = load_metadata_records(metadata_path)
    if max_samples is not None:
        records = records[: int(max_samples)]

    pairs: list[VideoPair] = []
    missing: list[dict] = []
    for row_id, record in enumerate(records):
        try:
            real_path = resolve_dataset_path(dataset_base_path, record.get(video_key))
        except Exception as exc:
            missing.append({"row_id": row_id, "reason": f"bad_real_path: {exc}"})
            continue

        demo_id = str(record.get(demo_id_key, row_id))
        camera_key = str(record.get(camera_key_col, "unknown"))
        generated_path = find_generated_video(generated_dir, row_id, demo_id, camera_key)
        if generated_path is None:
            missing.append(
                {
                    "row_id": row_id,
                    "demo_id": demo_id,
                    "camera_key": camera_key,
                    "reason": "generated_video_missing",
                }
            )
            continue
        if not os.path.exists(real_path):
            missing.append(
                {
                    "row_id": row_id,
                    "real_path": real_path,
                    "reason": "real_video_missing",
                }
            )
            continue
        pairs.append(
            VideoPair(
                row_id=row_id,
                demo_id=demo_id,
                camera_key=camera_key,
                real_path=real_path,
                generated_path=generated_path,
            )
        )
    return pairs, missing


def read_video_frames(path: str, *, max_decode_frames: int | None = None) -> np.ndarray:
    frames = []
    for frame_id, frame in enumerate(iio.imiter(path)):
        if max_decode_frames is not None and frame_id >= max_decode_frames:
            break
        frame = np.asarray(frame)
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=2)
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        frames.append(frame.astype(np.uint8, copy=False))
    if not frames:
        raise ValueError(f"No frames decoded from video: {path}")
    return np.stack(frames, axis=0)


def sample_frames(frames: np.ndarray, num_frames: int | None) -> np.ndarray:
    if num_frames is None:
        return frames
    num_frames = int(num_frames)
    if num_frames <= 0:
        return frames
    if len(frames) == num_frames:
        return frames
    if len(frames) > num_frames:
        indices = np.linspace(0, len(frames) - 1, num_frames).round().astype(np.int64)
        return frames[indices]
    pad_count = num_frames - len(frames)
    padding = np.repeat(frames[-1:], pad_count, axis=0)
    return np.concatenate([frames, padding], axis=0)


def resize_frames(frames: np.ndarray, size: tuple[int, int] | None) -> np.ndarray:
    if size is None:
        return frames
    height, width = int(size[0]), int(size[1])
    if frames.shape[1] == height and frames.shape[2] == width:
        return frames
    resized = []
    for frame in frames:
        image = Image.fromarray(frame)
        image = image.resize((width, height), Image.BICUBIC)
        resized.append(np.asarray(image, dtype=np.uint8))
    return np.stack(resized, axis=0)


def align_pair_frames(
    real_frames: np.ndarray,
    generated_frames: np.ndarray,
    *,
    num_frames: int | None,
    pair_resize: tuple[int, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if num_frames is None:
        num_frames = min(len(real_frames), len(generated_frames))
    real_frames = sample_frames(real_frames, num_frames)
    generated_frames = sample_frames(generated_frames, num_frames)

    if pair_resize is None:
        pair_resize = (int(generated_frames.shape[1]), int(generated_frames.shape[2]))
    real_frames = resize_frames(real_frames, pair_resize)
    generated_frames = resize_frames(generated_frames, pair_resize)
    return real_frames, generated_frames
