import argparse
import csv
import hashlib
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import h5py
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
    parser = argparse.ArgumentParser(description="Convert LIBERO hdf5 to image + abs-action metadata for AnyPos IDM.")
    parser.add_argument("--libero-root", type=str, required=True, help="Raw LIBERO hdf5 root.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for images and metadata.")
    parser.add_argument("--suites", type=str, default=",".join(DEFAULT_SUITES), help="Comma-separated suite names.")
    parser.add_argument("--camera-key", type=str, default="agentview_rgb", help="Image key inside obs.")
    parser.add_argument("--target-key", type=str, default="ee_states", help="Absolute EE target key inside obs.")
    parser.add_argument("--gripper-key", type=str, default="gripper_states", help="Gripper state key inside obs.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Export every N-th frame.")
    parser.add_argument("--min-episode-len", type=int, default=1, help="Skip shorter episodes.")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.05, help="Test ratio.")
    parser.add_argument("--image-format", type=str, default="jpg", choices=["jpg", "png"], help="Saved image format.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality when image-format=jpg.")
    parser.add_argument("--workers", type=int, default=8, help="Number of image export threads.")
    return parser.parse_args()


def count_export_rows(args, suites):
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
                    total += len(range(0, length, max(1, int(args.frame_stride))))
    return total


def save_image(image_array, image_path: str, image_format: str, jpeg_quality: int):
    image = Image.fromarray(image_array)
    image = ImageOps.flip(ImageOps.mirror(image))
    if image_format == "jpg":
        image.save(image_path, quality=jpeg_quality)
    else:
        image.save(image_path)


def main():
    args = parse_args()
    suites = [suite.strip() for suite in args.suites.split(",") if suite.strip()]
    output_dir = Path(args.output_dir)
    image_root = output_dir / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata_abs_action.csv"
    total_rows = count_export_rows(args, suites)

    writer = None
    rows_written = 0
    max_pending = max(8, int(args.workers) * 4)
    pending_futures = set()

    with open(metadata_path, "w", encoding="utf-8", newline="") as metadata_file:
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            with tqdm(total=total_rows, desc="Export LIBERO IDM metadata") as pbar:
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
                                for frame_index in range(0, length, max(1, int(args.frame_stride))):
                                    image_rel = Path("images") / suite / task_name / demo_key / f"{frame_index:06d}.{args.image_format}"
                                    image_abs = output_dir / image_rel
                                    image_abs.parent.mkdir(parents=True, exist_ok=True)

                                    target = list(targets[frame_index])
                                    gripper_scalar = float(gripper[frame_index][0])
                                    target.append(gripper_scalar)
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
                                    for dim, value in enumerate(target):
                                        row[f"abs_action_{dim}"] = float(value)

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

    if rows_written == 0:
        raise RuntimeError("No LIBERO samples were exported. Check libero root / suite / keys.")

    print(
        {
            "output_dir": str(output_dir),
            "metadata_path": str(metadata_path),
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
