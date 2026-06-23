#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw


def read_row(metadata_path: Path, camera_key: str, ordinal: int) -> dict[str, str]:
    with metadata_path.open("r", newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row.get("camera_key") == camera_key]
    if not rows:
        raise RuntimeError(f"No rows for camera_key={camera_key} in {metadata_path}")
    if ordinal >= len(rows):
        raise IndexError(f"ordinal={ordinal} out of range for {len(rows)} rows")
    return rows[ordinal]


def matching_camera_row(metadata_path: Path, source_file: str, demo_id: str, camera_key: str) -> dict[str, str]:
    with metadata_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (
                row.get("source_file") == source_file
                and row.get("demo_id") == demo_id
                and row.get("camera_key") == camera_key
            ):
                return row
    raise RuntimeError(f"No matching row for demo_id={demo_id} camera_key={camera_key}")


def resize_frame(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    image = Image.fromarray(frame)
    return np.asarray(image.resize((width, height), Image.Resampling.BICUBIC))


def make_fastwam_frame(agent_frame: np.ndarray, hand_frame: np.ndarray, height: int) -> np.ndarray:
    agent = resize_frame(agent_frame, height, height)
    hand = resize_frame(hand_frame, height, height)
    return np.concatenate([agent, hand], axis=1)


def label_frame(frame: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 24), fill=(0, 0, 0))
    draw.text((6, 5), label, fill=(255, 255, 255))
    return np.asarray(image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("data/libero_i2v_train/metadata_dense_prompt.csv"))
    parser.add_argument("--base-path", type=Path, default=Path("data/libero_i2v_train"))
    parser.add_argument("--agent-camera-key", default="agentview_rgb")
    parser.add_argument("--hand-camera-key", default="eye_in_hand_rgb")
    parser.add_argument("--ordinal", type=int, default=0)
    parser.add_argument("--local-start", type=int, default=0)
    parser.add_argument("--local-stride", type=int, default=4)
    parser.add_argument("--local-frames", type=int, default=9)
    parser.add_argument("--hdr-frames", type=int, default=4)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("logs/visualizations/fastwamhdr_13_frames.mp4"))
    parser.add_argument("--label", action="store_true")
    args = parser.parse_args()

    agent_row = read_row(args.metadata, args.agent_camera_key, args.ordinal)
    hand_row = matching_camera_row(
        args.metadata,
        source_file=agent_row["source_file"],
        demo_id=agent_row["demo_id"],
        camera_key=args.hand_camera_key,
    )
    agent_video_path = args.base_path / agent_row["video"]
    hand_video_path = args.base_path / hand_row["video"]
    agent_frames = list(iio.imiter(agent_video_path))
    hand_frames = list(iio.imiter(hand_video_path))
    if not agent_frames:
        raise RuntimeError(f"No frames decoded from {agent_video_path}")
    if not hand_frames:
        raise RuntimeError(f"No frames decoded from {hand_video_path}")

    episode_len = min(len(agent_frames), len(hand_frames))
    local_indices = [min(args.local_start + i * args.local_stride, episode_len - 1) for i in range(args.local_frames)]
    local_end = min(args.local_start + (args.local_frames - 1) * args.local_stride, episode_len - 1)
    first_hdr = local_end + 1
    if first_hdr >= episode_len:
        raise RuntimeError(f"No frames after local_end={local_end} in episode_len={episode_len}")
    tail = episode_len - 1 - local_end
    hdr_indices = np.asarray(
        [np.ceil(local_end + tail * (i + 1) / args.hdr_frames) for i in range(args.hdr_frames)],
        dtype=np.int64,
    ).clip(first_hdr, episode_len - 1).tolist()
    indices = local_indices + [int(i) for i in hdr_indices]

    out_frames = []
    for pos, frame_idx in enumerate(indices):
        frame = make_fastwam_frame(agent_frames[frame_idx], hand_frames[frame_idx], args.height)
        if args.label:
            kind = "local" if pos < args.local_frames else "hdr"
            frame = label_frame(frame, f"{pos:02d} {kind} src={frame_idx}")
        out_frames.append(frame)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.output, out_frames, fps=args.fps, codec="libx264", quality=8)
    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "agent_video": str(agent_video_path),
                "hand_video": str(hand_video_path),
                "agent_camera_key": args.agent_camera_key,
                "hand_camera_key": args.hand_camera_key,
                "ordinal": args.ordinal,
                "episode_len": episode_len,
                "order": "local9_then_hdr4",
                "layout": "agentview_left_224x224__handview_right_224x224",
                "local_end": local_end,
                "local_indices": local_indices,
                "hdr_indices": hdr_indices,
                "indices": indices,
                "output": str(args.output),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {args.output}")
    print(f"saved {sidecar}")
    print(f"indices={indices}")


if __name__ == "__main__":
    main()
