#!/usr/bin/env python3
"""Create dataset / GT action replay / HDR cache leaf decode comparison video."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-path", default="data/libero_i2v_train/metadata_dense_prompt_hdr_actions_leaf8_agentview_video_cache.csv")
    parser.add_argument("--base-path", default="data/libero_i2v_train")
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--gt-replay", default="logs/debug/libero_hdr_gt_controller_replay_row0.mp4")
    parser.add_argument("--output-dir", default="logs/debug/libero_hdr_cache_three_column_row0")
    parser.add_argument("--backend-root", default="../HiDiT/Causal-Forcing")
    parser.add_argument("--model-root", default="checkpoints")
    parser.add_argument("--model-name", default="Wan2.2-TI2V-5B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    return parser.parse_args()


def resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def read_video(path: Path) -> list[np.ndarray]:
    reader = imageio.get_reader(path)
    try:
        frames = [np.asarray(frame) for frame in reader]
    finally:
        reader.close()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return frames


def resize(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    import cv2

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def resample(frames: list[np.ndarray], count: int, height: int, width: int) -> list[np.ndarray]:
    if len(frames) == 1:
        ids = np.zeros(count, dtype=np.int64)
    else:
        ids = np.linspace(0, len(frames) - 1, count).round().astype(np.int64)
    return [resize(frames[int(i)], height, width) for i in ids]


def tensor_video_to_uint8(video: torch.Tensor) -> list[np.ndarray]:
    if video.ndim != 5:
        raise ValueError(f"Expected decoded video [B,T,C,H,W], got {tuple(video.shape)}")
    frames = []
    for frame in video[0].detach().float().cpu().permute(0, 2, 3, 1):
        arr = ((frame.clamp(-1, 1) + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8).numpy()
        frames.append(arr)
    return frames


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    base = resolve_path(repo_root, args.base_path)
    metadata_path = resolve_path(repo_root, args.metadata_path)
    rows = list(csv.DictReader(metadata_path.open("r", encoding="utf-8", newline="")))
    row = rows[int(args.row_index)]

    dataset_video_path = resolve_path(base, row["video"])
    cache_path = resolve_path(base, row["video_latent_cache_path"])
    gt_replay_path = resolve_path(repo_root, args.gt_replay)
    output_dir = resolve_path(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    backend_root = resolve_path(repo_root, args.backend_root)
    sys.path.insert(0, str(backend_root))
    from utils.wan_wrapper import WanVAEWrapper

    loaded = np.load(cache_path)
    leaf_latents = torch.from_numpy(loaded["vertical_latents"][-13:].astype(np.float32)).unsqueeze(0)
    device = torch.device(args.device)
    vae = WanVAEWrapper(model_name=args.model_name, model_root=str(resolve_path(repo_root, args.model_root))).to(device)
    vae.eval().requires_grad_(False)
    with torch.no_grad():
        decoded = vae.decode_to_pixel(leaf_latents.to(device=device, dtype=torch.float32))
    cache_frames = tensor_video_to_uint8(decoded)

    dataset_frames = read_video(dataset_video_path)
    replay_frames = read_video(gt_replay_path)
    count = max(len(dataset_frames), len(replay_frames), len(cache_frames))
    dataset_frames = resample(dataset_frames, count, args.height, args.width)
    replay_frames = resample(replay_frames, count, args.height, args.width)
    cache_frames = resample(cache_frames, count, args.height, args.width)

    imageio.mimsave(output_dir / "cache_leaf_decode.mp4", cache_frames, fps=args.fps)
    three = [
        np.concatenate([dataset, replay, cache], axis=1)
        for dataset, replay, cache in zip(dataset_frames, replay_frames, cache_frames)
    ]
    imageio.mimsave(output_dir / "dataset_gt_replay_cache_decode_three_column.mp4", three, fps=args.fps)
    print(f"dataset_video={dataset_video_path}")
    print(f"gt_replay={gt_replay_path}")
    print(f"cache_path={cache_path}")
    print(f"frames={count}")
    print(f"wrote={output_dir / 'cache_leaf_decode.mp4'}")
    print(f"wrote={output_dir / 'dataset_gt_replay_cache_decode_three_column.mp4'}")


if __name__ == "__main__":
    main()
