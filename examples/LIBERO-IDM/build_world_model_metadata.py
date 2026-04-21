#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

from lightewm.dataset.operators import ImageCropAndResize
from lightewm.runner.runner_util.wan_runtime import build_wan_i2v_pipeline_from_params


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_METADATA = "data/libero_idm_abs_action/metadata_abs_action.jsonl"
DEFAULT_SOURCE_BASE = "data/libero_idm_abs_action"
DEFAULT_OUTPUT_DIR = "data/libero_idm_abs_action_wm"
DEFAULT_PROMPT_METADATA = "data/libero_i2v_train/metadata_dense_prompt.csv"
DEFAULT_VIDEO_CKPT = "checkpoints/Wan2.2-5B-Libero/checkpoint.safetensors"
DEFAULT_INFER_CONFIG = "examples/LIBERO/infer_ti2v_5b.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate world-model videos for IDM augmentation metadata (video + abs_action list)."
    )
    parser.add_argument("--source-metadata-path", type=str, default=DEFAULT_SOURCE_METADATA)
    parser.add_argument("--source-base-path", type=str, default=DEFAULT_SOURCE_BASE)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt-metadata-path", type=str, default=DEFAULT_PROMPT_METADATA)
    parser.add_argument("--video-ckpt", type=str, default=DEFAULT_VIDEO_CKPT)
    parser.add_argument("--infer-config", type=str, default=DEFAULT_INFER_CONFIG)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--target-mix-ratio",
        type=float,
        default=0.2,
        help="Target generated ratio in final train samples. 0.2 means 20%% generated.",
    )
    parser.add_argument(
        "--num-generated-rows",
        type=int,
        default=0,
        help="If >0, override auto row count and generate this many rows.",
    )
    parser.add_argument("--video-fps", type=int, default=10)
    return parser.parse_args()


def _resolve_path(path_str: str):
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def _get_dist_info():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(global_rank)))
    if world_size < 1:
        raise ValueError(f"Invalid WORLD_SIZE={world_size}")
    if not (0 <= global_rank < world_size):
        raise ValueError(f"Invalid RANK={global_rank} for WORLD_SIZE={world_size}")
    return world_size, global_rank, local_rank


def _normalize_task_key(name: str):
    text = Path(str(name)).stem
    if text.endswith("_demo"):
        text = text[: -len("_demo")]
    return text


def _load_jsonl_rows(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if len(rows) == 0:
        raise RuntimeError(f"No rows in {path}")
    return rows


def _load_dense_prompt_lookup(metadata_path: str):
    prompt_counts = defaultdict(Counter)
    file_path = Path(metadata_path)
    if not file_path.exists():
        return {}
    with file_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("camera_key", "")).strip() not in ("", "agentview_rgb"):
                continue
            prompt = str(row.get("prompt", "")).strip()
            if not prompt:
                continue
            source_file = str(row.get("source_file", "")).strip()
            if source_file:
                prompt_counts[_normalize_task_key(source_file)][prompt] += 1
            video_field = str(row.get("video", "")).strip()
            if video_field:
                prompt_counts[_normalize_task_key(video_field)][prompt] += 1
    return {k: v.most_common(1)[0][0] for k, v in prompt_counts.items()}


def _load_wan_defaults(infer_config_path: str):
    with open(infer_config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model = dict(config["model"]["params"])
    model["pipeline_class_path"] = config["model"]["class_path"]
    runner_params = dict(config["runner"]["params"])
    infer_kwargs = dict(runner_params.get("infer_kwargs", {}))
    return model, runner_params, infer_kwargs


def _resolve_video_path(video_value: str, source_base_path: str, source_metadata_path: str):
    if os.path.isabs(video_value):
        return os.path.abspath(video_value)
    candidate = os.path.join(source_base_path, video_value)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    metadata_dir = os.path.dirname(os.path.abspath(source_metadata_path))
    return os.path.abspath(os.path.join(metadata_dir, video_value))


def _decode_frame(video_path: str, frame_index: int):
    reader = imageio.get_reader(video_path)
    try:
        frame = reader.get_data(int(frame_index))
    finally:
        reader.close()
    return np.asarray(frame, dtype=np.uint8)


def _save_video(frames, save_path: str, fps: int):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with imageio.get_writer(
        save_path,
        fps=int(fps),
        codec="mpeg4",
        ffmpeg_log_level="error",
        output_params=["-threads", "1"],
    ) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))


def _train_sample_count(row):
    actions = row.get("abs_action")
    if isinstance(actions, list) and len(actions) > 0 and isinstance(actions[0], list):
        return int(len(actions))
    return 1


def main():
    args = parse_args()
    if not (0.0 < float(args.target_mix_ratio) < 1.0):
        raise ValueError("--target-mix-ratio must be in (0,1)")
    if int(args.num_frames) <= 0:
        raise ValueError("--num-frames must be positive")

    source_metadata = _resolve_path(args.source_metadata_path)
    source_base = _resolve_path(args.source_base_path)
    output_dir = Path(_resolve_path(args.output_dir))
    prompt_metadata = _resolve_path(args.prompt_metadata_path)
    video_ckpt = _resolve_path(args.video_ckpt)
    infer_config = _resolve_path(args.infer_config)

    rows = _load_jsonl_rows(source_metadata)
    prompt_lookup = _load_dense_prompt_lookup(prompt_metadata)

    train_rows = []
    real_train_samples = 0
    for row in rows:
        if str(row.get("split", "")).strip() != "train":
            continue
        actions = np.asarray(row.get("abs_action", []), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] < int(args.num_frames) + 1:
            continue
        frame_indices = row.get("frame_indices", list(range(actions.shape[0])))
        if len(frame_indices) < int(args.num_frames) + 1:
            continue
        train_rows.append(row)
        real_train_samples += _train_sample_count(row)

    if len(train_rows) == 0:
        raise RuntimeError("No eligible train rows found for WM generation.")

    if int(args.num_generated_rows) > 0:
        target_rows = int(args.num_generated_rows)
        target_generated_samples = target_rows * int(args.num_frames)
    else:
        target_generated_samples = int(
            math.ceil(real_train_samples * float(args.target_mix_ratio) / (1.0 - float(args.target_mix_ratio)))
        )
        target_rows = int(math.ceil(target_generated_samples / float(args.num_frames)))

    world_size, global_rank, local_rank = _get_dist_info()
    assigned_row_indices = list(range(global_rank, target_rows, world_size))

    runtime_device = args.device
    if str(args.device).startswith("cuda") and ":" not in str(args.device):
        runtime_device = f"cuda:{local_rank}" if world_size > 1 else args.device

    model_params, infer_runner_params, infer_kwargs = _load_wan_defaults(infer_config)
    model_paths = list(model_params["model_paths"])
    model_paths[0] = video_ckpt
    model_params["model_paths"] = model_paths
    model_params["device"] = runtime_device
    model_params["torch_dtype"] = "bfloat16"
    infer_kwargs["num_frames"] = int(args.num_frames)
    infer_kwargs["num_inference_steps"] = int(args.num_inference_steps)
    infer_kwargs["cfg_scale"] = float(args.cfg_scale)

    pipe = build_wan_i2v_pipeline_from_params(model_params)
    input_image_resizer = ImageCropAndResize(
        height=int(infer_kwargs["height"]),
        width=int(infer_kwargs["width"]),
        max_pixels=None,
        height_division_factor=16,
        width_division_factor=16,
        resize_mode=str(infer_runner_params.get("input_image_resize_mode", "letterbox")),
    )

    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    if world_size == 1:
        metadata_path = output_dir / "metadata_abs_action_wm.jsonl"
    else:
        metadata_path = output_dir / f"metadata_abs_action_wm.rank{global_rank:05d}.jsonl"
    rng = random.Random(int(args.seed))
    total_generated_samples = 0
    generated_rows = 0

    with metadata_path.open("w", encoding="utf-8") as handle:
        for generated_idx in tqdm(
            assigned_row_indices,
            desc=f"Generate WM metadata rows [rank {global_rank}/{world_size}]",
        ):
            src_row = train_rows[rng.randrange(len(train_rows))]
            src_actions = np.asarray(src_row["abs_action"], dtype=np.float32)
            src_frame_indices = np.asarray(
                src_row.get("frame_indices", list(range(src_actions.shape[0]))),
                dtype=np.int64,
            )
            max_clip_start = int(src_actions.shape[0] - int(args.num_frames) - 1)
            clip_start = rng.randint(0, max_clip_start) if max_clip_start > 0 else 0
            condition_frame_index = int(src_frame_indices[clip_start])
            target_actions = src_actions[clip_start + 1 : clip_start + 1 + int(args.num_frames)]

            src_video_path = _resolve_video_path(
                str(src_row["video"]),
                source_base_path=source_base,
                source_metadata_path=source_metadata,
            )
            condition_frame = _decode_frame(src_video_path, condition_frame_index)
            condition_image = Image.fromarray(condition_frame).convert("RGB")
            condition_image = input_image_resizer(condition_image)

            task_name = str(src_row.get("task_name", ""))
            prompt = prompt_lookup.get(_normalize_task_key(task_name), task_name)
            if not prompt:
                prompt = str(src_row.get("sample_id", "libero task"))

            sample_seed = int(args.seed + generated_idx)
            generated_video = pipe(
                prompt=prompt,
                input_image=condition_image,
                seed=sample_seed,
                **infer_kwargs,
            )
            generated_frames = [np.asarray(frame, dtype=np.uint8) for frame in generated_video[: int(args.num_frames)]]
            if len(generated_frames) != int(args.num_frames):
                raise RuntimeError(
                    f"Generated frame count mismatch: expected {args.num_frames}, got {len(generated_frames)}"
                )

            suite = str(src_row.get("suite", "unknown_suite"))
            task = str(src_row.get("task_name", "unknown_task"))
            demo = str(src_row.get("demo_key", f"demo_{generated_idx}"))
            video_rel = Path("videos") / suite / task / f"{demo}_wm_{generated_idx:06d}.mp4"
            video_abs = output_dir / video_rel
            _save_video(generated_frames, str(video_abs), fps=int(args.video_fps))

            out_row = {
                "sample_id": f"wm/{src_row.get('sample_id', f'sample_{generated_idx}')}/{generated_idx:06d}",
                "video": video_rel.as_posix(),
                "split": "train",
                "suite": suite,
                "task_name": task,
                "demo_key": f"{demo}_wm_{generated_idx:06d}",
                "source_sample_id": src_row.get("sample_id"),
                "source_video": src_row.get("video"),
                "source_clip_start": int(clip_start),
                "source_condition_frame_index": int(condition_frame_index),
                "frame_indices": list(range(int(args.num_frames))),
                "num_frames": int(args.num_frames),
                "abs_action": target_actions.astype(np.float32).tolist(),
                "wm_seed": int(sample_seed),
                "wm_rank": int(global_rank),
                "prompt": prompt,
            }
            handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            generated_rows += 1
            total_generated_samples += int(args.num_frames)

    print(
        {
            "source_metadata_path": source_metadata,
            "output_dir": str(output_dir),
            "output_metadata_path": str(metadata_path),
            "world_size": int(world_size),
            "global_rank": int(global_rank),
            "local_rank": int(local_rank),
            "device": str(runtime_device),
            "target_rows_global": int(target_rows),
            "assigned_rows_local": int(len(assigned_row_indices)),
            "real_train_samples": int(real_train_samples),
            "target_mix_ratio": float(args.target_mix_ratio),
            "target_generated_samples": int(target_generated_samples),
            "generated_rows": int(generated_rows),
            "generated_samples": int(total_generated_samples),
            "num_frames_per_generated_row": int(args.num_frames),
            "video_ckpt": video_ckpt,
            "note": "Each generated row uses GT future 49-step abs_action from source row (clip_start+1 .. clip_start+49).",
        }
    )


if __name__ == "__main__":
    main()
