#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import cv2
import pyarrow.parquet as pq

DEFAULT_PROMPT = "A robot creates art painting with a brush."
DEFAULT_SOURCE_NAME = "new100_3cam"
DEFAULT_SPARSE_PROMPT = "robot art painting"
OUT_W, OUT_H = 320, 384
TOP_H = 224
BOTTOM_H = 160
LEFT_W = 160
RIGHT_W = 160
TARGET_FRAMES = 61
CLIP_STRIDE = 10
DEFAULT_INPUT = None
DEFAULT_OUTPUT = REPO_ROOT / "data/realbot_hdr_video_320x384_new100_3cam_61f_suffix10"
CAMERA_KEYS = {
    "right": "observation.images.image",
    "wrist": "observation.images.wrist_image",
    "front": "observation.images.front_image",
}


def resize_exact(img, w: int, h: int):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def read_video_meta(path: Path) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {path}")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if fps <= 0:
        fps = 10.0
    return frames, fps


def clip_starts(n: int, stride: int, min_frames: int) -> list[int]:
    if n < min_frames:
        return []
    starts = list(range(0, n - min_frames + 1, stride))
    last = n - min_frames
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def episode_rows(input_root: Path, split: str):
    split_root = input_root / split
    episodes_path = split_root / "meta/episodes.jsonl"
    source_names: dict[int, str] = {}
    if episodes_path.exists():
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            source_names[int(item["episode_index"])] = str(item.get("source_episode", f"episode_{int(item['episode_index']):06d}"))
    for pq_path in sorted((split_root / "data/chunk-000").glob("episode_*.parquet")):
        episode_index = int(pq_path.stem.split("_")[-1])
        videos = {
            name: split_root / "videos/chunk-000" / key / f"{pq_path.stem}.mp4"
            for name, key in CAMERA_KEYS.items()
        }
        missing = [str(path) for path in videos.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing videos for {split}/{pq_path.stem}: {missing}")
        frames = int(pq.read_metadata(pq_path).num_rows)
        yield {
            "split": split,
            "episode_index": episode_index,
            "episode_id": pq_path.stem,
            "source_episode": source_names.get(episode_index, pq_path.stem),
            "parquet": pq_path,
            "videos": videos,
            "frames": frames,
        }


def write_full_stitched_video(ep: dict, destination: Path, force: bool = False) -> dict:
    video_counts = {}
    fps_values = []
    for name, path in ep["videos"].items():
        count, fps = read_video_meta(path)
        video_counts[name] = count
        fps_values.append(fps)
    n = min([ep["frames"], *video_counts.values()])
    if n <= 0:
        raise RuntimeError(f"No frames for {ep['split']}/{ep['episode_id']}")
    fps = fps_values[0] if fps_values else 10.0
    if destination.exists() and not force:
        return {"source_frames": n, "target_frames": n, "fps": fps, "skipped_existing": True}
    destination.parent.mkdir(parents=True, exist_ok=True)
    caps = {name: cv2.VideoCapture(str(path)) for name, path in ep["videos"].items()}
    try:
        for name, cap in caps.items():
            if not cap.isOpened():
                raise RuntimeError(f"Could not open {name} video for {ep['episode_id']}")
        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (OUT_W, OUT_H))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open writer {destination}")
        try:
            for frame_idx in range(n):
                ok_r, right = caps["right"].read()
                ok_w, wrist = caps["wrist"].read()
                ok_f, front = caps["front"].read()
                if not (ok_r and ok_w and ok_f):
                    raise RuntimeError(f"Decode failed for {ep['episode_id']} at frame {frame_idx}")
                canvas = resize_exact(right, OUT_W, TOP_H)
                bottom = cv2.hconcat([
                    resize_exact(wrist, LEFT_W, BOTTOM_H),
                    resize_exact(front, RIGHT_W, BOTTOM_H),
                ])
                writer.write(cv2.vconcat([canvas, bottom]))
        finally:
            writer.release()
    finally:
        for cap in caps.values():
            cap.release()
    return {"source_frames": n, "target_frames": n, "fps": fps, "skipped_existing": False}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "video", "full_video", "prompt", "source", "episode_id", "parent_episode_id", "split",
        "num_frames", "source_frame_count", "clip_start", "clip_end", "clip_source_frames", "full_video_fps",
        "dense_prompt", "sparse_prompt", "source_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--clip-stride", type=int, default=CLIP_STRIDE)
    parser.add_argument("--target-frames", type=int, default=TARGET_FRAMES)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--source-name", default=DEFAULT_SOURCE_NAME)
    parser.add_argument("--sparse-prompt", default=DEFAULT_SPARSE_PROMPT)
    parser.add_argument("--full-video-subdir", default=DEFAULT_SOURCE_NAME)
    parser.add_argument("--augment-heldout", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    train_rows: list[dict] = []
    heldout_rows: list[dict] = []
    summary_rows: list[dict] = []
    episodes = list(episode_rows(args.input_root, "train")) + list(episode_rows(args.input_root, "heldout"))
    if not episodes:
        raise RuntimeError(f"No episodes found under {args.input_root}")

    for idx, ep in enumerate(episodes, start=1):
        full_rel = Path("full_videos") / args.full_video_subdir / ep["split"] / f"{ep['source_episode']}_realbot_hdr_320x384_full.mp4"
        full_path = args.output_root / full_rel
        full_stats = write_full_stitched_video(ep, full_path, force=args.force)
        n = int(full_stats["source_frames"])
        starts = clip_starts(n, args.clip_stride, args.target_frames)
        if ep["split"] == "heldout" and not args.augment_heldout:
            starts = starts[:1]
        if not starts:
            print(f"[{idx}/{len(episodes)}] skip short {ep['split']} {ep['source_episode']} n={n}", flush=True)
            continue
        for start in starts:
            clip_id = f"{ep['source_episode']}_s{start:06d}"
            row = {
                "video": str(full_rel),
                "full_video": str(full_rel),
                "prompt": args.prompt,
                "dense_prompt": args.prompt,
                "sparse_prompt": args.sparse_prompt,
                "source": args.source_name,
                "episode_id": clip_id,
                "parent_episode_id": ep["source_episode"],
                "split": ep["split"],
                "num_frames": int(args.target_frames),
                "source_frame_count": n,
                "clip_start": int(start),
                "clip_end": n - 1,
                "clip_source_frames": n - int(start),
                "full_video_fps": float(full_stats["fps"]),
                "source_file": str(ep["parquet"]),
            }
            (heldout_rows if ep["split"] == "heldout" else train_rows).append(row)
            summary_rows.append({**row, **full_stats})
        print(
            f"[{idx}/{len(episodes)}] {ep['split']} {ep['source_episode']} n={n} clips={len(starts)} first={starts[0]} last={starts[-1]}",
            flush=True,
        )

    write_csv(args.output_root / "metadata_train.csv", train_rows)
    write_csv(args.output_root / "metadata_heldout.csv", heldout_rows)
    (args.output_root / "summary.json").write_text(json.dumps({
        "source_dataset": str(args.input_root),
        "target_frames": int(args.target_frames),
        "clip_stride": int(args.clip_stride),
        "layout": "right_full_top_wrist_lower_left_front_lower_right_320x384",
        "prompt": args.prompt,
        "source_name": args.source_name,
        "sparse_prompt": args.sparse_prompt,
        "train_count": len(train_rows),
        "heldout_count": len(heldout_rows),
        "train_parent_episodes": len({row["parent_episode_id"] for row in train_rows}),
        "heldout_parent_episodes": len({row["parent_episode_id"] for row in heldout_rows}),
        "rows": summary_rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_root": str(args.output_root),
        "train": len(train_rows),
        "heldout": len(heldout_rows),
        "train_parent_episodes": len({row["parent_episode_id"] for row in train_rows}),
        "heldout_parent_episodes": len({row["parent_episode_id"] for row in heldout_rows}),
    }, indent=2))


if __name__ == "__main__":
    main()
