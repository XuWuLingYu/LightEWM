#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
ANYPOS_ROOT = REPO_ROOT / "third_parties" / "AnyPos"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ANYPOS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANYPOS_ROOT))

from idm.idm import IDM
from idm.preprocessor import DinoPreprocessor
from lightewm.dataset.operators import ImageCropAndResize
from lightewm.runner.runner_util.wan_runtime import build_wan_i2v_pipeline_from_params
from lightewm.utils.data import save_video

DEFAULT_VIDEO_CKPT = "checkpoints/Wan2.2-5B-Libero/checkpoint.safetensors"
DEFAULT_IDM_CKPT = "checkpoints/LIBERO-IDM/100000.pt"
DEFAULT_INFER_CONFIG = "examples/LIBERO/infer_ti2v_5b.yaml"
DEFAULT_METADATA = "data/libero_idm_abs_action/metadata_abs_action.jsonl"
DEFAULT_PROMPT_METADATA = "data/libero_i2v_train/metadata_dense_prompt.csv"
DIM_NAMES = ["x", "y", "z", "rx", "ry", "rz", "gripper"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare action curves from GT clip vs generated clip using IDM."
    )
    parser.add_argument("--video-ckpt", type=str, default=DEFAULT_VIDEO_CKPT)
    parser.add_argument("--idm-ckpt", type=str, default=DEFAULT_IDM_CKPT)
    parser.add_argument("--infer-config", type=str, default=DEFAULT_INFER_CONFIG)
    parser.add_argument("--metadata-path", type=str, default=DEFAULT_METADATA)
    parser.add_argument("--prompt-metadata-path", type=str, default=DEFAULT_PROMPT_METADATA)
    parser.add_argument("--output-dir", type=str, default="outputs/libero_idm_action_curve_compare")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--idm-model-name", type=str, default="direction_aware")
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--clip-start", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    return parser.parse_args()


def _resolve_path(path_str: str):
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def _normalize_task_key(name: str):
    text = Path(str(name)).stem
    if text.endswith("_demo"):
        text = text[: -len("_demo")]
    return text


def _load_wan_defaults(infer_config_path: str):
    with open(infer_config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model = dict(config["model"]["params"])
    model["pipeline_class_path"] = config["model"]["class_path"]
    runner_params = dict(config["runner"]["params"])
    infer_kwargs = dict(runner_params.get("infer_kwargs", {}))
    return model, runner_params, infer_kwargs


def _load_dense_prompt_lookup(metadata_path: str):
    prompt_counts = defaultdict(Counter)
    metadata_file = Path(metadata_path)
    if not metadata_file.exists():
        return {}
    with metadata_file.open("r", encoding="utf-8", newline="") as handle:
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


def _load_idm(idm_ckpt_path: str, model_name: str, device: str):
    try:
        checkpoint = torch.load(idm_ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(idm_ckpt_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    output_dim = int(state_dict["train_mean"].numel())
    net = IDM(
        model_name=model_name,
        dinov2_name="facebook/dinov2-with-registers-base",
        output_dim=output_dim,
        train_mean=state_dict.get("train_mean"),
        train_std=state_dict.get("train_std"),
    )
    net.load_state_dict(state_dict, strict=True)
    net.eval()
    net.to(device)
    return net


def _load_jsonl_rows(metadata_path: str):
    rows = []
    with open(metadata_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if len(rows) == 0:
        raise RuntimeError(f"No rows found in {metadata_path}")
    return rows


def _decode_video_frames(video_path: str, frame_indices):
    reader = imageio.get_reader(video_path)
    try:
        frames = [np.asarray(reader.get_data(int(idx)), dtype=np.uint8) for idx in frame_indices]
    finally:
        reader.close()
    return frames


def _predict_actions(idm, preprocessor, frames, device: str, chunk: int = 16):
    outputs = []
    with torch.no_grad():
        for start in range(0, len(frames), chunk):
            batch_frames = frames[start : start + chunk]
            batch = preprocessor.process_batch(batch_frames).to(device)
            pred = idm(batch).detach().cpu().numpy().astype(np.float32)
            outputs.append(pred)
    return np.concatenate(outputs, axis=0)


def _to_pil_rgb(frame):
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    return Image.fromarray(np.asarray(frame).astype(np.uint8)).convert("RGB")


def _center_square_crop(frame):
    arr = np.asarray(frame, dtype=np.uint8)
    h, w = arr.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    cropped = arr[top : top + side, left : left + side]
    return cropped


def _plot_curves(gt_actions, idm_on_gt, idm_on_gen, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = np.arange(gt_actions.shape[0])

    for dim in range(gt_actions.shape[1]):
        name = DIM_NAMES[dim] if dim < len(DIM_NAMES) else f"dim_{dim}"
        fig = plt.figure(figsize=(10, 4))
        plt.plot(steps, gt_actions[:, dim], label="gt_action", linewidth=2.0)
        plt.plot(steps, idm_on_gt[:, dim], label="idm(gt_frames)", linewidth=1.5)
        plt.plot(steps, idm_on_gen[:, dim], label="idm(gen_frames)", linewidth=1.5)
        plt.xlabel("step")
        plt.ylabel(name)
        plt.title(f"Action Curve - {name}")
        plt.grid(alpha=0.3)
        plt.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"curve_{dim:02d}_{name}.png", dpi=160)
        plt.close(fig)

    fig = plt.figure(figsize=(16, 18))
    for dim in range(gt_actions.shape[1]):
        name = DIM_NAMES[dim] if dim < len(DIM_NAMES) else f"dim_{dim}"
        ax = fig.add_subplot(gt_actions.shape[1], 1, dim + 1)
        ax.plot(steps, gt_actions[:, dim], label="gt_action", linewidth=1.8)
        ax.plot(steps, idm_on_gt[:, dim], label="idm(gt_frames)", linewidth=1.2)
        ax.plot(steps, idm_on_gen[:, dim], label="idm(gen_frames)", linewidth=1.2)
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
        if dim == 0:
            ax.legend(loc="upper right")
    ax.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(output_dir / "curves_all_dims.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()

    output_dir = Path(_resolve_path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    video_ckpt = _resolve_path(args.video_ckpt)
    idm_ckpt = _resolve_path(args.idm_ckpt)
    infer_config = _resolve_path(args.infer_config)
    metadata_path = _resolve_path(args.metadata_path)
    prompt_metadata_path = _resolve_path(args.prompt_metadata_path)

    rows = _load_jsonl_rows(metadata_path)
    if args.row_index < 0 or args.row_index >= len(rows):
        raise IndexError(f"row-index {args.row_index} out of range [0, {len(rows)-1}]")
    row = rows[args.row_index]

    abs_action = np.asarray(row["abs_action"], dtype=np.float32)
    frame_indices = np.asarray(row.get("frame_indices", list(range(abs_action.shape[0]))), dtype=np.int64)
    if abs_action.ndim != 2:
        raise ValueError(f"abs_action must be 2D, got shape {tuple(abs_action.shape)}")
    if frame_indices.shape[0] != abs_action.shape[0]:
        raise ValueError("frame_indices and abs_action lengths are inconsistent")
    if args.clip_start < 0 or args.clip_start + args.num_frames > abs_action.shape[0]:
        raise ValueError(
            f"clip range [{args.clip_start}, {args.clip_start + args.num_frames}) exceeds sequence length {abs_action.shape[0]}"
        )

    local_slice = slice(args.clip_start, args.clip_start + args.num_frames)
    gt_actions = abs_action[local_slice]
    gt_frame_indices = frame_indices[local_slice]

    media_base = Path(metadata_path).parent
    video_rel = row["video"]
    video_path = str((media_base / video_rel).resolve()) if not os.path.isabs(video_rel) else video_rel
    gt_frames = _decode_video_frames(video_path, gt_frame_indices.tolist())

    prompt_lookup = _load_dense_prompt_lookup(prompt_metadata_path)
    task_name = row.get("task_name", "")
    prompt = prompt_lookup.get(_normalize_task_key(task_name), task_name)

    model_params, infer_runner_params, infer_kwargs = _load_wan_defaults(infer_config)
    model_paths = list(model_params["model_paths"])
    model_paths[0] = video_ckpt
    model_params["model_paths"] = model_paths
    model_params["device"] = args.device
    model_params["torch_dtype"] = "bfloat16"
    infer_kwargs["num_frames"] = int(args.num_frames)
    infer_kwargs["num_inference_steps"] = int(args.num_inference_steps)
    infer_kwargs["cfg_scale"] = float(args.cfg_scale)

    pipe = build_wan_i2v_pipeline_from_params(model_params)
    idm = _load_idm(idm_ckpt, model_name=args.idm_model_name, device=args.device)
    preprocessor = DinoPreprocessor(SimpleNamespace(use_transform=False))
    input_image_resizer = ImageCropAndResize(
        height=int(infer_kwargs["height"]),
        width=int(infer_kwargs["width"]),
        max_pixels=None,
        height_division_factor=16,
        width_division_factor=16,
        resize_mode=str(infer_runner_params.get("input_image_resize_mode", "letterbox")),
    )

    condition_image = _to_pil_rgb(gt_frames[0])
    condition_image_resized = input_image_resizer(condition_image)
    gen_video = pipe(
        prompt=prompt,
        input_image=condition_image_resized,
        seed=int(args.seed),
        **infer_kwargs,
    )
    gen_frames = [np.asarray(_to_pil_rgb(frame), dtype=np.uint8) for frame in gen_video[: int(args.num_frames)]]
    if len(gen_frames) != int(args.num_frames):
        raise RuntimeError(f"Generated frame count mismatch: expected {args.num_frames}, got {len(gen_frames)}")
    gen_frames_center_square = [_center_square_crop(frame) for frame in gen_frames]

    idm_on_gt = _predict_actions(idm, preprocessor, gt_frames, device=args.device)
    idm_on_gen = _predict_actions(idm, preprocessor, gen_frames_center_square, device=args.device)

    plot_dir = output_dir / "plots"
    _plot_curves(gt_actions, idm_on_gt, idm_on_gen, plot_dir)

    save_video(gt_frames, str(output_dir / "gt_clip.mp4"), fps=int(infer_runner_params.get("fps", 10)), quality=5)
    save_video(gen_frames, str(output_dir / "generated_clip.mp4"), fps=int(infer_runner_params.get("fps", 10)), quality=5)
    save_video(
        gen_frames_center_square,
        str(output_dir / "generated_clip_center_square.mp4"),
        fps=int(infer_runner_params.get("fps", 10)),
        quality=5,
    )

    mae_idm_gt = np.mean(np.abs(idm_on_gt - gt_actions), axis=0)
    mae_idm_gen = np.mean(np.abs(idm_on_gen - gt_actions), axis=0)
    summary = {
        "row_index": int(args.row_index),
        "sample_id": row.get("sample_id"),
        "video_path": video_path,
        "task_name": task_name,
        "prompt": prompt,
        "clip_start": int(args.clip_start),
        "num_frames": int(args.num_frames),
        "generated_frame_shape": list(gen_frames[0].shape) if len(gen_frames) > 0 else None,
        "generated_center_square_shape": (
            list(gen_frames_center_square[0].shape) if len(gen_frames_center_square) > 0 else None
        ),
        "idm_ckpt": idm_ckpt,
        "video_ckpt": video_ckpt,
        "mae_idm_on_gt_per_dim": mae_idm_gt.tolist(),
        "mae_idm_on_gen_per_dim": mae_idm_gen.tolist(),
        "mae_idm_on_gt_mean": float(np.mean(mae_idm_gt)),
        "mae_idm_on_gen_mean": float(np.mean(mae_idm_gen)),
        "dim_names": DIM_NAMES[: gt_actions.shape[1]],
        "plot_dir": str(plot_dir),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[Saved] curves -> {plot_dir}")
    print(
        "[Saved] videos -> "
        f"{output_dir / 'gt_clip.mp4'}, "
        f"{output_dir / 'generated_clip.mp4'}, "
        f"{output_dir / 'generated_clip_center_square.mp4'}"
    )


if __name__ == "__main__":
    main()
