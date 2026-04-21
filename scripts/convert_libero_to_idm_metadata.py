import argparse
import csv
import hashlib
import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm


DEFAULT_SUITES = (
    "libero_10",
    "libero_90",
    "libero_goal",
    "libero_object",
    "libero_spatial",
)


def deterministic_split(identifier: str, val_ratio: float, test_ratio: float) -> str:
    digest = hashlib.md5(identifier.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / float(16 ** 8)
    if value < test_ratio:
        return "test"
    if value < test_ratio + val_ratio:
        return "val"
    return "train"


def parse_args():
    parser = argparse.ArgumentParser(description="Convert LIBERO hdf5 to metadata for AnyPos IDM.")
    parser.add_argument("--libero-root", type=str, required=True, help="Raw LIBERO hdf5 root.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for media and metadata.")
    parser.add_argument("--suites", type=str, default=",".join(DEFAULT_SUITES), help="Comma-separated suite names.")
    parser.add_argument("--camera-key", type=str, default="agentview_rgb", help="Image key inside obs.")
    parser.add_argument("--target-key", type=str, default="ee_states", help="Absolute EE target key inside obs.")
    parser.add_argument("--gripper-key", type=str, default="gripper_states", help="Gripper state key inside obs.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Export every N-th frame.")
    parser.add_argument("--min-episode-len", type=int, default=1, help="Skip shorter episodes.")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.05, help="Test ratio.")
    parser.add_argument("--storage-mode", type=str, default="videos", choices=["videos", "images"], help="Export videos with action lists or per-frame images.")
    parser.add_argument("--image-format", type=str, default="jpg", choices=["jpg", "png"], help="Saved image format in image mode.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality when image-format=jpg.")
    parser.add_argument("--workers", type=int, default=8, help="Number of export threads.")
    parser.add_argument("--video-fps", type=int, default=10, help="Exported video fps in video mode.")
    return parser.parse_args()


def count_export_items(args, suites):
    total = 0
    for suite in suites:
        suite_dir = Path(args.libero_root) / suite
        if not suite_dir.is_dir():
            continue
        for hdf5_path in sorted(suite_dir.glob("*.hdf5")):
            with h5py.File(hdf5_path, "r") as h5_file:
                for demo_key in sorted(h5_file["data"].keys()):
                    episode = h5_file["data"][demo_key]
                    obs = episode["obs"]
                    if args.camera_key not in obs or args.target_key not in obs or args.gripper_key not in obs:
                        continue
                    images = obs[args.camera_key]
                    targets = obs[args.target_key]
                    gripper = obs[args.gripper_key]
                    length = min(len(images), len(targets), len(gripper))
                    if length < args.min_episode_len:
                        continue
                    if args.storage_mode == "videos":
                        total += 1
                    else:
                        total += len(range(0, length, max(1, int(args.frame_stride))))
    return total


def _flip_frame(frame):
    image = Image.fromarray(frame)
    image = ImageOps.flip(ImageOps.mirror(image))
    return image


def save_image(frame, image_path: str, image_format: str, jpeg_quality: int):
    image = _flip_frame(frame)
    if image_format == "jpg":
        image.save(image_path, quality=jpeg_quality)
    else:
        image.save(image_path)


def save_video(frames, video_path: str, fps: int):
    flipped = [np.asarray(_flip_frame(frame)) for frame in frames]
    with imageio.get_writer(
        video_path,
        fps=fps,
        codec="mpeg4",
        ffmpeg_log_level="error",
        output_params=["-threads", "1"],
    ) as writer:
        for frame in flipped:
            writer.append_data(frame)


def build_action_sequence(targets, gripper):
    action_sequence = []
    for target, grip in zip(targets, gripper):
        row = [float(v) for v in target]
        row.append(float(grip[0]))
        action_sequence.append(row)
    return action_sequence


def main():
    args = parse_args()
    suites = [suite.strip() for suite in args.suites.split(",") if suite.strip()]
    output_dir = Path(args.output_dir)
    media_root_name = "videos" if args.storage_mode == "videos" else "images"
    media_root = output_dir / media_root_name
    media_root.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / ("metadata_abs_action.jsonl" if args.storage_mode == "videos" else "metadata_abs_action.csv")
    total_items = count_export_items(args, suites)

    max_pending = max(8, int(args.workers) * 4)
    pending_futures = set()
    rows_written = 0

    if args.storage_mode == "videos":
        metadata_file = open(metadata_path, "w", encoding="utf-8")
    else:
        metadata_file = open(metadata_path, "w", encoding="utf-8", newline="")
    try:
        writer = None
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            with tqdm(total=total_items, desc="Export LIBERO IDM metadata") as pbar:
                for suite in suites:
                    suite_dir = Path(args.libero_root) / suite
                    if not suite_dir.is_dir():
                        continue
                    for hdf5_path in sorted(suite_dir.glob("*.hdf5")):
                        task_name = hdf5_path.stem
                        with h5py.File(hdf5_path, "r") as h5_file:
                            for demo_key in sorted(h5_file["data"].keys()):
                                episode = h5_file["data"][demo_key]
                                obs = episode["obs"]
                                if args.camera_key not in obs or args.target_key not in obs or args.gripper_key not in obs:
                                    continue
                                images = obs[args.camera_key]
                                targets = obs[args.target_key]
                                gripper = obs[args.gripper_key]
                                length = min(len(images), len(targets), len(gripper))
                                if length < args.min_episode_len:
                                    continue
                                split = deterministic_split(f"{suite}/{task_name}/{demo_key}", args.val_ratio, args.test_ratio)

                                if args.storage_mode == "videos":
                                    frame_indices = list(range(0, length, max(1, int(args.frame_stride))))
                                    video_rel = Path("videos") / suite / task_name / f"{demo_key}.mp4"
                                    video_abs = output_dir / video_rel
                                    video_abs.parent.mkdir(parents=True, exist_ok=True)
                                    selected_frames = [images[i].copy() for i in frame_indices]
                                    selected_targets = [targets[i] for i in frame_indices]
                                    selected_gripper = [gripper[i] for i in frame_indices]
                                    row = {
                                        "sample_id": f"{suite}/{task_name}/{demo_key}",
                                        "video": video_rel.as_posix(),
                                        "split": split,
                                        "suite": suite,
                                        "task_name": task_name,
                                        "demo_key": demo_key,
                                        "source_hdf5": str(hdf5_path),
                                        "frame_indices": frame_indices,
                                        "num_frames": len(frame_indices),
                                        "abs_action": build_action_sequence(selected_targets, selected_gripper),
                                    }
                                    metadata_file.write(json.dumps(row, ensure_ascii=True) + "\n")
                                    rows_written += 1

                                    future = executor.submit(
                                        save_video,
                                        selected_frames,
                                        str(video_abs),
                                        int(args.video_fps),
                                    )
                                    pending_futures.add(future)
                                    if len(pending_futures) >= max_pending:
                                        done, pending_futures = wait(pending_futures, return_when=FIRST_COMPLETED)
                                        for completed in done:
                                            completed.result()
                                    pbar.update(1)
                                    continue

                                for frame_index in range(0, length, max(1, int(args.frame_stride))):
                                    image_rel = Path("images") / suite / task_name / demo_key / f"{frame_index:06d}.{args.image_format}"
                                    image_abs = output_dir / image_rel
                                    image_abs.parent.mkdir(parents=True, exist_ok=True)

                                    row = {
                                        "sample_id": f"{suite}/{task_name}/{demo_key}/{frame_index}",
                                        "image": image_rel.as_posix(),
                                        "split": split,
                                        "suite": suite,
                                        "task_name": task_name,
                                        "demo_key": demo_key,
                                        "frame_index": frame_index,
                                        "source_hdf5": str(hdf5_path),
                                    }
                                    target = [float(v) for v in targets[frame_index]]
                                    target.append(float(gripper[frame_index][0]))
                                    for dim, value in enumerate(target):
                                        row[f"abs_action_{dim}"] = value

                                    if writer is None:
                                        writer = csv.DictWriter(metadata_file, fieldnames=list(row.keys()))
                                        writer.writeheader()
                                    writer.writerow(row)
                                    rows_written += 1

                                    future = executor.submit(
                                        save_image,
                                        images[frame_index].copy(),
                                        str(image_abs),
                                        args.image_format,
                                        args.jpeg_quality,
                                    )
                                    pending_futures.add(future)
                                    if len(pending_futures) >= max_pending:
                                        done, pending_futures = wait(pending_futures, return_when=FIRST_COMPLETED)
                                        for completed in done:
                                            completed.result()
                                    pbar.update(1)

            if pending_futures:
                done, _ = wait(pending_futures)
                for completed in done:
                    completed.result()
    finally:
        metadata_file.close()

    if rows_written == 0:
        raise RuntimeError("No LIBERO samples were exported. Check libero root / suite / keys.")

    print(
        {
            "output_dir": str(output_dir),
            "metadata_path": str(metadata_path),
            "storage_mode": args.storage_mode,
            "num_rows": rows_written,
            "camera_key": args.camera_key,
            "target_key": args.target_key,
            "gripper_key": args.gripper_key,
            "target_dim": 7,
            "workers": int(args.workers),
            "image_transform": "flip_left_right_then_flip_top_bottom",
            "note": "This exports 7D absolute targets: ee_states(6) + gripper_states[...,0](1). Do not use raw LIBERO relative actions for this IDM path.",
        }
    )


if __name__ == "__main__":
    main()
