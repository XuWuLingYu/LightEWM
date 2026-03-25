#!/usr/bin/env python3
"""
Convert LIBERO hdf5 demos to LightEWM training metadata CSV.

Output format:
  <output_dir>/
    metadata.csv
    videos/
      <suite>__<task_file_stem>__<demo_id>__<camera_key>.mp4

CSV columns:
  video,prompt,source_file,demo_id,camera_key,num_frames
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

try:
    from tqdm.auto import tqdm
except Exception:
    class _NoOpTqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            if self.iterable is None:
                return iter(())
            return iter(self.iterable)
        def update(self, n=1):
            return None
        def set_postfix(self, *args, **kwargs):
            return None
        def close(self):
            return None
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
    def tqdm(iterable=None, **kwargs):
        return _NoOpTqdm(iterable=iterable, **kwargs)


PROMPT_ATTR_CANDIDATES = (
    "lang",
    "language",
    "language_instruction",
    "instruction",
    "task_description",
    "natural_language_instruction",
)


SCENE_PREFIX_PATTERN = re.compile(r"^\s*.*?\bSCENE\s*\d+\b[\s:._-]*", re.IGNORECASE)
DEFAULT_CAMERA_KEYS = (
    "agentview_rgb",
    "agentview_image",
    "front_rgb",
    "rgb",
    "robot0_eye_in_hand_rgb",
    "eye_in_hand_rgb",
)

CAMERA_KEY_ALIAS_GROUPS = (
    ("eye_in_hand_rgb", "robot0_eye_in_hand_rgb", "robot0_eye_in_hand_image"),
    ("agentview_rgb", "agentview_image"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LIBERO hdf5 demos to metadata.csv + videos.")
    parser.add_argument(
        "--libero-root",
        type=str,
        required=True,
        help="Root folder containing LIBERO suites (e.g. data/LIBERO-datasets).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output folder for metadata.csv and videos/.",
    )
    parser.add_argument(
        "--suites",
        type=str,
        default="libero_10,libero_90,libero_goal,libero_object,libero_spatial",
        help="Comma-separated suite folders to scan.",
    )
    parser.add_argument(
        "--glob-pattern",
        type=str,
        default="*.hdf5",
        help="Filename pattern inside each suite.",
    )
    parser.add_argument(
        "--camera-key",
        type=str,
        default="all",
        help="Comma-separated camera keys to export. Use `all` (default) to export all available RGB views.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="FPS used when writing output videos.",
    )
    parser.add_argument(
        "--max-demos-per-file",
        type=int,
        default=None,
        help="Optional limit per hdf5 file.",
    )
    parser.add_argument(
        "--prompt-source",
        type=str,
        default="attr_or_filename",
        choices=("attr_or_filename", "filename_only"),
        help="How to build text prompts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output videos.",
    )
    parser.add_argument(
        "--rotate-180",
        dest="rotate_180",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rotate exported frames by 180 degrees (default: enabled).",
    )
    return parser.parse_args()


def ensure_dependencies():
    try:
        import h5py  # noqa: F401
        import imageio.v3  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Missing dependencies. Install with: "
            "`pip install h5py imageio imageio-ffmpeg`"
        ) from e


def decode_if_bytes(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return value


def normalize_prompt(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = SCENE_PREFIX_PATTERN.sub("", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def prompt_from_filename(file_stem: str) -> str:
    text = file_stem
    text = text.replace("_demo", "")
    text = text.replace("_", " ")
    return normalize_prompt(text)


def prompt_from_attrs_or_filename(file_stem: str, file_attrs, demo_attrs) -> str:
    for key in PROMPT_ATTR_CANDIDATES:
        if key in demo_attrs:
            value = decode_if_bytes(demo_attrs[key])
            if isinstance(value, str) and value.strip() != "":
                return normalize_prompt(value)
    for key in PROMPT_ATTR_CANDIDATES:
        if key in file_attrs:
            value = decode_if_bytes(file_attrs[key])
            if isinstance(value, str) and value.strip() != "":
                return normalize_prompt(value)
    return prompt_from_filename(file_stem)


def is_rgb_tensor(dataset) -> bool:
    shape = getattr(dataset, "shape", None)
    if shape is None or len(shape) != 4:
        return False
    channels = shape[-1]
    return channels in (3, 4)


def parse_camera_key_arg(camera_key_arg: str | None) -> list[str]:
    if camera_key_arg is None:
        return ["all"]
    keys = [k.strip() for k in camera_key_arg.split(",") if k.strip()]
    if len(keys) == 0:
        return ["all"]
    if any(k.lower() == "all" for k in keys):
        return ["all"]
    return keys


def pick_camera_keys(obs_group, requested_keys: list[str]) -> list[str]:
    keys = list(obs_group.keys())
    rgb_keys = [k for k in keys if is_rgb_tensor(obs_group[k])]
    if len(rgb_keys) == 0:
        raise RuntimeError(f"No RGB-like camera tensor found in obs keys: {keys}")

    # Deterministic ordering by priority, then remaining keys.
    ordered_rgb_keys = []
    used = set()
    for k in DEFAULT_CAMERA_KEYS:
        if k in rgb_keys and k not in used:
            ordered_rgb_keys.append(k)
            used.add(k)
    for k in rgb_keys:
        if k not in used:
            ordered_rgb_keys.append(k)
            used.add(k)

    if len(requested_keys) == 1 and requested_keys[0].lower() == "all":
        return ordered_rgb_keys

    def alias_candidates(req: str) -> list[str]:
        cands = [req]
        for group in CAMERA_KEY_ALIAS_GROUPS:
            if req in group:
                for name in group:
                    if name not in cands:
                        cands.append(name)
        return cands

    selected = []
    missing = []
    for req in requested_keys:
        picked = None
        for cand in alias_candidates(req):
            if cand in obs_group and is_rgb_tensor(obs_group[cand]):
                picked = cand
                break
        if picked is None:
            missing.append(req)
            continue
        if picked not in selected:
            selected.append(picked)

    if len(missing) > 0:
        print(f"[WARN] Some requested camera keys are unavailable: {missing}.")

    if len(selected) > 0:
        return selected

    # Fallback to all RGB views if requested keys are not found.
    print(f"[WARN] Requested camera keys {requested_keys} not found in obs keys {keys}. Falling back to all RGB views.")
    return ordered_rgb_keys


def sanitize_view_name(name: str) -> str:
    name = name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    name = re.sub(r"[^0-9a-zA-Z_.-]", "_", name)
    return name


def to_uint8_frames(frames):
    import numpy as np

    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        if frames.max() <= 1.0:
            frames = (frames * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frames = frames.clip(0, 255).astype(np.uint8)
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    return frames


def apply_orientation_fix(frames, rotate_180: bool = True):
    if not rotate_180:
        return frames
    # LIBERO camera tensors are often upside-down and mirrored in our export path.
    # Flipping H and W equals a 180-degree rotation.
    return frames[:, ::-1, ::-1, :]


def write_video_mp4(path: Path, frames, fps: int):
    import imageio.v3 as iio

    iio.imwrite(path, frames, fps=fps)


def main():
    args = parse_args()
    ensure_dependencies()
    import h5py

    libero_root = Path(args.libero_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    suite_names = [x.strip() for x in args.suites.split(",") if x.strip()]
    requested_camera_keys = parse_camera_key_arg(args.camera_key)

    rows = []
    num_files = 0
    num_demos = 0
    num_errors = 0

    jobs = []
    for suite in suite_names:
        suite_dir = libero_root / suite
        if not suite_dir.exists():
            print(f"[WARN] Skip missing suite: {suite_dir}")
            continue
        for h5_path in sorted(suite_dir.glob(args.glob_pattern)):
            jobs.append((suite, h5_path))

    with tqdm(total=len(jobs), desc="LIBERO files", unit="file") as file_pbar:
        for suite, h5_path in jobs:
            num_files += 1
            stem = h5_path.stem
            try:
                with h5py.File(h5_path, "r") as f:
                    if "data" not in f:
                        print(f"[WARN] Skip file without /data group: {h5_path}")
                        continue
                    demo_keys = sorted(list(f["data"].keys()))
                    if args.max_demos_per_file is not None:
                        demo_keys = demo_keys[: args.max_demos_per_file]
                    demo_desc = f"{suite}:{stem[:36]}"
                    for demo_id in tqdm(demo_keys, desc=demo_desc, unit="demo", leave=False):
                        demo = f["data"][demo_id]
                        if "obs" not in demo:
                            print(f"[WARN] Skip demo without /obs: {h5_path}:{demo_id}")
                            continue
                        obs = demo["obs"]
                        try:
                            camera_keys = pick_camera_keys(obs, requested_camera_keys)
                        except Exception as e:
                            print(f"[WARN] Skip demo without usable camera: {h5_path}:{demo_id} ({e})")
                            continue

                        if args.prompt_source == "filename_only":
                            prompt = prompt_from_filename(stem)
                        else:
                            prompt = prompt_from_attrs_or_filename(stem, f.attrs, demo.attrs)

                        for camera_key in camera_keys:
                            frames = to_uint8_frames(obs[camera_key][...])
                            frames = apply_orientation_fix(frames, rotate_180=args.rotate_180)
                            if len(frames) == 0:
                                print(f"[WARN] Empty frames: {h5_path}:{demo_id}:{camera_key}")
                                continue

                            view_name = sanitize_view_name(camera_key)
                            video_name = f"{suite}__{stem}__{demo_id}__{view_name}.mp4"
                            rel_video_path = Path("videos") / video_name
                            out_video = output_dir / rel_video_path

                            if out_video.exists() and not args.overwrite:
                                pass
                            else:
                                out_video.parent.mkdir(parents=True, exist_ok=True)
                                write_video_mp4(out_video, frames, fps=args.fps)

                            rows.append({
                                "video": rel_video_path.as_posix(),
                                "prompt": prompt,
                                "source_file": h5_path.as_posix(),
                                "demo_id": demo_id,
                                "camera_key": camera_key,
                                "num_frames": len(frames),
                            })
                            num_demos += 1
            except Exception as e:
                num_errors += 1
                print(f"[ERROR] Failed file: {h5_path} ({e})")
            finally:
                file_pbar.update(1)
                file_pbar.set_postfix(exported=num_demos, errors=num_errors)

    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video", "prompt", "source_file", "demo_id", "camera_key", "num_frames"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] files_scanned={num_files} demos_exported={num_demos} file_errors={num_errors}")
    print(f"[DONE] metadata={metadata_path}")
    print(f"[DONE] videos_dir={videos_dir}")

    if num_demos == 0:
        print("[WARN] No demos exported. Check --suites / --camera-key / dataset path.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
