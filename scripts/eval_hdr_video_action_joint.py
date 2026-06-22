#!/usr/bin/env python3
"""Evaluate the HDR video-action-joint checkpoint on one LIBERO sample."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import h5py
import av
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf


ACTION_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


class CastingTextEncoder(torch.nn.Module):
    def __init__(self, text_encoder: torch.nn.Module, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.text_encoder = text_encoder
        self.device = device
        self.dtype = dtype

    def forward(self, text_prompts: list[str]) -> dict[str, torch.Tensor]:
        return {
            key: value.to(device=self.device, dtype=self.dtype)
            for key, value in self.text_encoder(text_prompts=text_prompts).items()
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-root", default="../HiDiT/Causal-Forcing")
    parser.add_argument(
        "--config-path",
        default="logs/LIBERO-HDR_train_video_action_joint/20260616_165344/causal_forcing_config.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="logs/LIBERO-HDR_train_video_action_joint/20260616_165344/checkpoint_model_020000/model.pt",
    )
    parser.add_argument("--output-dir", default="logs/eval/libero_hdr_video_action_joint_20k")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--camera-key", default="agentview_rgb")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--joint-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--tree-fixed-denoise-steps", type=int, default=5)
    parser.add_argument("--tree-sampling-steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--skip-local-video", action="store_true")
    parser.add_argument("--skip-open-loop-action", action="store_true")
    parser.add_argument("--skip-close-loop-action", action="store_true")
    parser.add_argument("--skip-closed-loop-video", action="store_true")
    parser.add_argument("--skip-tree-video", action="store_true")
    parser.add_argument(
        "--action-only",
        action="store_true",
        help="Do not append local future-video tokens during joint action denoising.",
    )
    parser.add_argument(
        "--use-generated-tree-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use HDR tree inference from the episode first frame as eval context instead of GT tree latents.",
    )
    parser.add_argument(
        "--tree-context-mode",
        choices=("generated", "gt-clean", "gt-noisy"),
        default=None,
        help="Override --use-generated-tree-context with generated, GT clean, or GT noisy hierarchical context.",
    )
    parser.add_argument(
        "--tree-x0-level-index",
        type=int,
        default=2,
        help="Hierarchical level index whose final pred-x0 decode is saved when using generated tree context.",
    )
    parser.add_argument(
        "--closed-loop-simulator",
        action="store_true",
        help="Run a true LIBERO simulator loop: encode the current observation as local_start before each action chunk.",
    )
    parser.add_argument(
        "--disable-proprio",
        action="store_true",
        help="Disable FastWAM-style proprio conditioning during eval. Useful for checkpoints trained before proprio_encoder was added.",
    )
    parser.add_argument("--closed-loop-chunks", type=int, default=1)
    parser.add_argument("--closed-loop-execute-steps", type=int, default=52)
    parser.add_argument("--closed-loop-timeout-steps", type=int, default=600)
    parser.add_argument("--physical-agent-root", default="/mnt/zezhong/physical_agent")
    parser.add_argument("--libero-root", default="/mnt/zezhong/LightEWM/third_parties/LIBERO")
    parser.add_argument("--libero-data-root", default="data/LIBERO-datasets")
    return parser.parse_args()


def resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def resolve_backend_path(value: str | Path, backend_root: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((backend_root / path).resolve())


def load_generator_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "generator" in state and state["generator"] is not None:
        state = state["generator"]
    elif "generator_ema" in state:
        state = state["generator_ema"]
    elif "model" in state:
        state = state["model"]
    fixed = {}
    for key, value in state.items():
        if key.startswith("model._fsdp_wrapped_module."):
            key = key.replace("model._fsdp_wrapped_module.", "model.", 1)
        fixed[key] = value
    return fixed


def load_action_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "action_dit" in state and state["action_dit"] is not None:
        return state["action_dit"]
    return state


def find_filtered_row(jsonl_path: Path, camera_key: str, ordinal: int) -> tuple[int, dict[str, Any]]:
    seen = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            row_camera = row.get("camera_key")
            row_video = str(row.get("video_path", row.get("video", "")))
            if row_camera == camera_key or camera_key in row_video:
                if seen == ordinal:
                    return line_idx, row
                seen += 1
    raise RuntimeError(f"No {camera_key} sample ordinal={ordinal} in {jsonl_path}")


def resolve_metadata_path(value: str | Path, base_dir: Path, repo_root: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidates = [(base_dir / path).resolve(), (repo_root / path).resolve()]
    stripped_parts = [part for part in path.parts if part not in ("..", ".")]
    if stripped_parts:
        candidates.append((repo_root / Path(*stripped_parts)).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def absolutize_row(row: dict[str, Any], base_dir: Path, repo_root: Path) -> dict[str, Any]:
    row = dict(row)
    for key in ("video_path", "video", "action_stats_path", "proprio_stats_path", "source_file", "video_latent_cache_path"):
        value = row.get(key)
        if not value:
            continue
        row[key] = str(resolve_metadata_path(value, base_dir, repo_root))
    return row


def tensor_video_to_uint8(video: torch.Tensor) -> np.ndarray:
    video = video.detach().float().cpu()
    if video.ndim == 5:
        video = video[0]
    if video.ndim != 4:
        raise ValueError(f"Expected [F,C,H,W] or [C,F,H,W], got {tuple(video.shape)}")
    if video.shape[1] == 3:
        frames = video
    elif video.shape[0] == 3:
        frames = video.permute(1, 0, 2, 3)
    else:
        raise ValueError(f"Cannot infer channel dim from {tuple(video.shape)}")
    if frames.min() < -0.05:
        frames = frames * 0.5 + 0.5
    frames = frames.clamp(0, 1)
    return (frames * 255.0).round().byte().permute(0, 2, 3, 1).numpy()


def resize_uint8(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    from PIL import Image

    image = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
    return np.asarray(image.resize((width, height), resample=Image.BICUBIC))


def save_video_mp4(path: Path, frames: list[np.ndarray] | np.ndarray, fps: int) -> None:
    frames = list(frames)
    if not frames:
        raise ValueError(f"No frames to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    first = np.asarray(frames[0])
    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError(f"Expected RGB frames, got shape={first.shape}")
    height, width = first.shape[:2]
    encode_height = height + (height % 2)
    encode_width = width + (width % 2)

    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=int(fps))
    stream.width = encode_width
    stream.height = encode_height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "medium"}
    try:
        for frame in frames:
            frame = np.asarray(frame)
            if frame.shape[:2] != (height, width):
                frame = resize_uint8(frame, height, width)
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"Expected RGB frame, got shape={frame.shape}")
            frame = np.ascontiguousarray(np.clip(frame, 0, 255).astype(np.uint8))
            if encode_height != height or encode_width != width:
                padded = np.zeros((encode_height, encode_width, 3), dtype=np.uint8)
                padded[:height, :width] = frame
                if encode_height != height:
                    padded[height:, :width] = frame[-1:, :]
                if encode_width != width:
                    padded[:, width:] = padded[:, width - 1: width]
                frame = padded
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def save_side_by_side(path: Path, left: np.ndarray, right: np.ndarray, fps: int) -> None:
    count = min(len(left), len(right))
    left = left[:count]
    right = right[:count]
    if left.shape[1:3] != right.shape[1:3]:
        right = np.stack([resize_uint8(frame, left.shape[1], left.shape[2]) for frame in right], axis=0)
    frames = [np.concatenate([l, r], axis=1) for l, r in zip(left, right)]
    save_video_mp4(path, frames, fps=fps)


def plot_action_curves(path: Path, gt: np.ndarray, pred: np.ndarray, title: str) -> None:
    xs = np.arange(gt.shape[0])
    fig, axes = plt.subplots(7, 1, figsize=(14, 18), sharex=True)
    for dim, ax in enumerate(axes):
        ax.plot(xs, gt[:, dim], label="GT", linewidth=1.6)
        ax.plot(xs, pred[:, dim], label="Gen", linewidth=1.3, linestyle="--")
        ax.set_ylabel(ACTION_NAMES[dim])
        ax.grid(True, alpha=0.25)
        if dim == 0:
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("step")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_action_csv(path: Path, gt: np.ndarray, pred: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["step"]
        for name in ACTION_NAMES:
            header.extend([f"gt_{name}", f"gen_{name}"])
        writer.writerow(header)
        for i in range(gt.shape[0]):
            row: list[Any] = [i]
            for dim in range(7):
                row.extend([float(gt[i, dim]), float(pred[i, dim])])
            writer.writerow(row)


def denormalize_actions(actions: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    min_v = np.asarray(stats["min"], dtype=np.float32)
    max_v = np.asarray(stats["max"], dtype=np.float32)
    eps = float(stats.get("eps", 1e-6))
    return (actions + 1.0) * 0.5 * (max_v - min_v + eps) + min_v


def action_norm_clip_from_config(cfg: Any) -> float:
    return float(getattr(cfg, "joint_norm_clip", 1.0))


def clip_normalized_actions(actions: np.ndarray, cfg: Any) -> np.ndarray:
    clip = action_norm_clip_from_config(cfg)
    return np.clip(actions, -clip, clip)


def normalize_vector(value: np.ndarray, stats: dict[str, Any] | None) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if stats is None:
        return value
    min_v = np.asarray(stats["min"], dtype=np.float32)
    max_v = np.asarray(stats["max"], dtype=np.float32)
    eps = float(stats.get("eps", 1e-6))
    return np.clip(2.0 * (value - min_v) / (max_v - min_v + eps) - 1.0, -1.0, 1.0)


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(0.0, 1.0 - float(quat[3] * quat[3])))
    if den < 1e-6:
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * np.arccos(quat[3]) / den)).astype(np.float32)


def proprio_from_obs(obs: dict[str, Any], stats: dict[str, Any] | None) -> np.ndarray:
    if "ee_pos" in obs and "ee_ori" in obs and "gripper_states" in obs:
        eef_pos = np.asarray(obs["ee_pos"], dtype=np.float32).reshape(-1)
        eef_axis_angle = np.asarray(obs["ee_ori"], dtype=np.float32).reshape(-1)
        gripper_qpos = np.asarray(obs["gripper_states"], dtype=np.float32).reshape(-1)
    else:
        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)
        eef_axis_angle = quat_to_axis_angle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(-1))
        gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    return normalize_vector(np.concatenate([eef_pos, eef_axis_angle, gripper_qpos], axis=0), stats)


def resolve_h5_path(source_file: str, libero_data_root: Path) -> Path:
    source = Path(source_file)
    if source.exists():
        return source
    matches = list(libero_data_root.rglob(source.name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Cannot resolve hdf5 source: {source_file}")


def problem_name_from_h5(h5_path: Path) -> str:
    stem = h5_path.stem
    if stem.endswith("_demo"):
        return stem[: -len("_demo")]
    return stem


def frame_from_obs(obs: dict[str, Any]) -> np.ndarray:
    for key in ("agentview_image", "agentview_rgb", "robot0_agentview_image"):
        if key in obs:
            frame = np.asarray(obs[key])
            break
    else:
        candidates = [v for v in obs.values() if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] == 3]
        if not candidates:
            raise KeyError(f"No RGB observation in keys={sorted(obs.keys())}")
        frame = np.asarray(candidates[0])
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame[::-1, ::-1].copy()


def fastwam_frame_from_obs(obs: dict[str, Any], camera_keys: list[str] | tuple[str, ...] | None = None) -> np.ndarray:
    if not camera_keys:
        return frame_from_obs(obs)
    key_aliases = {
        "agentview_rgb": ("agentview_image", "agentview_rgb", "robot0_agentview_image"),
        "eye_in_hand_rgb": ("robot0_eye_in_hand_image", "eye_in_hand_image", "eye_in_hand_rgb"),
    }
    frames = []
    for camera_key in camera_keys:
        aliases = key_aliases.get(str(camera_key), (str(camera_key),))
        for alias in aliases:
            if alias in obs:
                frame = np.asarray(obs[alias])
                break
        else:
            raise KeyError(f"No observation image for camera `{camera_key}`. Available={sorted(obs.keys())}")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frames.append(frame[::-1, ::-1].copy())
    return np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0]


def parse_camera_keys(value: Any) -> list[str]:
    if value is None:
        return ["agentview_rgb"]
    if isinstance(value, (list, tuple)):
        return [str(part) for part in value]
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        return [part.strip().strip("'\"") for part in text.strip("[]").split(",") if part.strip()]
    return [part.strip() for part in text.split(",") if part.strip()]


def uint8_frames_to_tensor(frames: list[np.ndarray], height: int, width: int) -> torch.Tensor:
    from PIL import Image

    tensors = []
    for frame in frames:
        image = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
        image = image.resize((width, height), resample=Image.BICUBIC)
        tensors.append(torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 127.5 - 1.0)
    return torch.stack(tensors, dim=1)


def uint8_fastwam_frames_to_tensor(frames: list[np.ndarray], height: int, width: int, num_cameras: int) -> torch.Tensor:
    from PIL import Image

    if num_cameras <= 1:
        if int(width) != int(height):
            raise ValueError(
                f"Single-camera FastWAM input must be square, got height={height}, width={width}."
            )
        return uint8_frames_to_tensor(frames, height=height, width=width)
    if int(width) % int(num_cameras) != 0:
        raise ValueError(f"width={width} must be divisible by num_cameras={num_cameras}.")
    per_camera_width = int(width) // int(num_cameras)
    if per_camera_width != int(height):
        raise ValueError(
            f"FastWAM two-camera input expects square views, got height={height}, "
            f"per_camera_width={per_camera_width}, num_cameras={num_cameras}."
        )
    tensors = []
    for frame in frames:
        frame = np.asarray(frame).astype(np.uint8)
        chunks = np.split(frame, int(num_cameras), axis=1)
        resized_chunks = []
        for chunk in chunks:
            image = Image.fromarray(chunk).convert("RGB")
            image = image.resize((per_camera_width, int(height)), resample=Image.BICUBIC)
            resized_chunks.append(np.asarray(image))
        merged = np.concatenate(resized_chunks, axis=1)
        tensors.append(torch.from_numpy(merged).permute(2, 0, 1).float() / 127.5 - 1.0)
    return torch.stack(tensors, dim=1)


def replay_actions(
    *,
    row: dict[str, Any],
    actions_raw: np.ndarray,
    output_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    sys.path.insert(0, str(Path(args.physical_agent_root)))
    from libero_agent import LiberoAgentInterface, LiberoActionError

    h5_path = resolve_h5_path(str(row["source_file"]), Path(args.libero_data_root))
    suite = h5_path.parent.name
    problem_name = problem_name_from_h5(h5_path)
    bddl_file = Path(args.libero_root) / "libero" / "libero" / "bddl_files" / suite / f"{problem_name}.bddl"
    if not bddl_file.exists():
        raise FileNotFoundError(f"Missing BDDL file: {bddl_file}")

    with h5py.File(h5_path, "r") as f:
        init_state = np.asarray(f["data"][str(row["demo_id"])].attrs["init_state"], dtype=np.float64)

    env = LiberoAgentInterface(
        bddl_file=bddl_file,
        camera_heights=int(args.camera_height),
        camera_widths=int(args.camera_width),
        action_repeat=1,
        libero_root=str(args.libero_root),
    )
    env.reset(init_state=init_state)
    frames = [resize_uint8(frame_from_obs(env.last_obs), args.camera_height, args.camera_width)]
    eef_positions = [np.asarray(env.state()["eef_pos"], dtype=np.float32)]
    executed = 0
    try:
        for step_id, action in enumerate(actions_raw):
            try:
                result = env.step(
                    action[:6],
                    mode="ee_delta",
                    gripper=float(np.clip(action[6], -1.0, 1.0)),
                )
            except LiberoActionError as exc:
                print(f"[CloseLoop] stop step={step_id}: {exc.to_dict()}")
                break
            frames.append(resize_uint8(frame_from_obs(result.observation), args.camera_height, args.camera_width))
            eef_positions.append(np.asarray(result.state_after["eef_pos"], dtype=np.float32))
            executed += 1
    finally:
        env.close()

    save_video_mp4(output_path, frames, fps=int(args.fps))
    eef = np.stack(eef_positions, axis=0)
    return {
        "executed_steps": executed,
        "eef_motion_total": float(np.linalg.norm(eef[-1] - eef[0])),
        "bddl_file": str(bddl_file),
        "video": str(output_path),
    }


def build_libero_env(row: dict[str, Any], args: argparse.Namespace):
    sys.path.insert(0, str(Path(args.physical_agent_root)))
    from libero_agent import LiberoAgentInterface

    h5_path = resolve_h5_path(str(row["source_file"]), Path(args.libero_data_root))
    suite = h5_path.parent.name
    problem_name = problem_name_from_h5(h5_path)
    bddl_file = Path(args.libero_root) / "libero" / "libero" / "bddl_files" / suite / f"{problem_name}.bddl"
    if not bddl_file.exists():
        raise FileNotFoundError(f"Missing BDDL file: {bddl_file}")
    with h5py.File(h5_path, "r") as f:
        init_state = np.asarray(f["data"][str(row["demo_id"])].attrs["init_state"], dtype=np.float64)
    env = LiberoAgentInterface(
        bddl_file=bddl_file,
        camera_heights=int(args.camera_height),
        camera_widths=int(args.camera_width),
        action_repeat=1,
        libero_root=str(args.libero_root),
    )
    env.reset(init_state=init_state)
    return env, str(bddl_file)


def make_autocast_ctx(device: torch.device, dtype: torch.dtype):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and dtype == torch.bfloat16
        else nullcontext()
    )


def joint_tree_count(cfg, vertical_info: dict[str, Any]) -> int:
    tree_num_levels = int(getattr(cfg, "joint_tree_num_levels", 0) or 0)
    if tree_num_levels <= 0:
        return int(vertical_info["num_tokens"])
    if tree_num_levels > len(vertical_info["level_sizes"]):
        raise ValueError(
            f"joint_tree_num_levels={tree_num_levels} exceeds hierarchy levels={vertical_info['level_sizes']}."
        )
    return int(sum(vertical_info["level_sizes"][:tree_num_levels]))


def truncate_joint_tree(
    cfg,
    vertical_info: dict[str, Any],
    tree_latents: torch.Tensor,
    tree_timestep: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    tree_count = joint_tree_count(cfg, vertical_info)
    return tree_latents[:, :tree_count], tree_timestep[:, :tree_count], list(range(tree_count))


def infer_tree_context(
    *,
    cfg,
    model,
    device: torch.device,
    dtype: torch.dtype,
    initial_latent: torch.Tensor,
    prompt: list[str],
    output_dir: Path,
    args: argparse.Namespace,
    save_video: bool,
    gt_frames: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    from model.diffusion import build_fixed_vertical_retained_timesteps
    from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline

    pipe = CausalDiffusionInferencePipeline(
        cfg,
        device=device,
        generator=model.generator,
        text_encoder=model.text_encoder,
        vae=model.vae,
        need_vae=True,
    ).to(device=device)
    pipe.eval()
    noise = torch.randn(
        (
            1,
            pipe.vertical_info["num_tokens"],
            initial_latent.shape[2],
            initial_latent.shape[3],
            initial_latent.shape[4],
        ),
        device=device,
        dtype=dtype,
    )
    with make_autocast_ctx(device, dtype):
        leaf_latents, vertical_latents, vertical_payload = pipe.inference(
            noise=noise,
            text_prompts=prompt,
            initial_latent=initial_latent,
            return_latents=True,
            return_vertical_layer_videos=True,
            return_video=False,
        )
        tree_video = model.vae.decode_to_pixel(leaf_latents.float())

    tree_timestep = build_fixed_vertical_retained_timesteps(
        token_step_budgets=list(pipe.vertical_token_step_budgets),
        num_train_timesteps=int(cfg.num_train_timestep),
        timestep_shift=float(cfg.timestep_shift),
        fixed_denoise_steps=int(args.tree_fixed_denoise_steps),
        preserve_budget_ratio=True,
        reference_total_steps=int(args.tree_sampling_steps),
        default_sampling_steps=int(args.tree_sampling_steps),
        device=device,
        dtype=dtype,
    ).unsqueeze(0)
    leaf_start = int(pipe.vertical_info["leaf_start_index"])
    tree_timestep[:, leaf_start:] = 0

    outputs: dict[str, Any] = {
        "tree_context": "generated_fixed_denoise",
        "tree_fixed_denoise_steps": int(args.tree_fixed_denoise_steps),
        "tree_sampling_steps": int(args.tree_sampling_steps),
        "tree_timestep_min": float(tree_timestep.float().min().detach().cpu()),
        "tree_timestep_max": float(tree_timestep.float().max().detach().cpu()),
        "leaf_timestep": 0.0,
    }
    if save_video:
        tree_np = tensor_video_to_uint8(tree_video)
        tree_path = output_dir / "closed_loop_hierarchical_tree_video_gen.mp4"
        save_video_mp4(tree_path, list(tree_np), fps=int(args.fps))
        outputs["hierarchical_tree_video"] = str(tree_path)
        leaf_np = tensor_video_to_uint8(model.vae.decode_to_pixel(leaf_latents.float()))
        leaf_path = output_dir / "closed_loop_hierarchical_tree_leaf_tokens.mp4"
        save_video_mp4(leaf_path, list(leaf_np), fps=int(args.fps))
        outputs["hierarchical_tree_leaf_tokens"] = str(leaf_path)
        if gt_frames is not None:
            gt_np = tensor_video_to_uint8(gt_frames[0])
            gt_tree_path = output_dir / "closed_loop_hierarchical_tree_video_gt_vs_gen.mp4"
            save_side_by_side(gt_tree_path, gt_np, tree_np, fps=int(args.fps))
            outputs["hierarchical_tree_gt_vs_gen"] = str(gt_tree_path)
        layer_videos = None if vertical_payload is None else vertical_payload.get("layer_videos")
        level_index = int(args.tree_x0_level_index)
        if layer_videos is not None and 0 <= level_index < len(layer_videos):
            level_np = tensor_video_to_uint8(layer_videos[level_index])
            level_path = output_dir / f"hierarchical_tree_level_{level_index}_pred_x0.mp4"
            save_video_mp4(level_path, list(level_np), fps=int(args.fps))
            outputs[f"hierarchical_tree_level_{level_index}_pred_x0"] = str(level_path)
            if gt_frames is not None:
                gt_np = tensor_video_to_uint8(gt_frames[0])
                gt_sampled = gt_np[np.linspace(0, len(gt_np) - 1, len(level_np)).round().astype(np.int64)]
                level_pair_path = output_dir / f"hierarchical_tree_level_{level_index}_pred_x0_gt_vs_pred.mp4"
                save_side_by_side(level_pair_path, gt_sampled, level_np, fps=int(args.fps))
                outputs[f"hierarchical_tree_level_{level_index}_pred_x0_gt_vs_pred"] = str(level_pair_path)

    vertical_latents, tree_timestep, _ = truncate_joint_tree(
        cfg,
        pipe.vertical_info,
        vertical_latents.to(dtype=dtype),
        tree_timestep,
    )
    outputs["tree_context_token_count"] = int(vertical_latents.shape[1])
    return vertical_latents, tree_timestep, outputs


def denoise_joint_action(
    *,
    model,
    cfg,
    condition: dict[str, torch.Tensor],
    tree_latents: torch.Tensor,
    tree_timestep: torch.Tensor,
    tree_token_ids: list[int],
    prefix_x: torch.Tensor,
    prefix_t: torch.Tensor,
    prefix_token_ids: list[int],
    local_start: torch.Tensor,
    local_video_clean: torch.Tensor | None,
    joint_proprio: torch.Tensor | None,
    action_shape: tuple[int, int, int],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    from utils.scheduler import WanContinuousFlowMatchScheduler

    dtype = model.dtype
    local_video_count = 0 if args.action_only else int(cfg.joint_local_video_tokens)
    if local_video_count > 0:
        if local_video_clean is None:
            local_video_clean = torch.zeros(
                local_start.shape[0],
                local_video_count,
                local_start.shape[2],
                local_start.shape[3],
                local_start.shape[4],
                device=device,
                dtype=dtype,
            )
        if local_video_clean.shape[1] < local_video_count:
            pad = local_video_clean[:, -1:].expand(-1, local_video_count - local_video_clean.shape[1], -1, -1, -1)
            local_video_clean = torch.cat([local_video_clean, pad], dim=1)
        elif local_video_clean.shape[1] > local_video_count:
            local_video_clean = local_video_clean[:, :local_video_count]
        local_video = torch.randn_like(local_video_clean)
    else:
        local_video = local_start.new_zeros(
            local_start.shape[0],
            0,
            local_start.shape[2],
            local_start.shape[3],
            local_start.shape[4],
        )

    action_scheduler = WanContinuousFlowMatchScheduler(
        num_train_timesteps=int(cfg.num_train_timestep),
        shift=float(getattr(cfg, "action_infer_shift", getattr(cfg, "action_train_shift", 5.0))),
    )
    video_scheduler = WanContinuousFlowMatchScheduler(
        num_train_timesteps=int(cfg.num_train_timestep),
        shift=float(getattr(cfg, "action_infer_shift", getattr(cfg, "action_train_shift", 5.0))),
    )
    action_timesteps, action_deltas = action_scheduler.build_inference_schedule(
        int(args.joint_steps), device=device, dtype=dtype
    )
    if local_video_count > 0:
        video_timesteps, video_deltas = video_scheduler.build_inference_schedule(
            int(args.joint_steps), device=device, dtype=dtype
        )
    else:
        video_timesteps = torch.zeros_like(action_timesteps)
        video_deltas = torch.zeros_like(action_deltas)

    actions = torch.randn(action_shape, device=device, dtype=dtype)
    frame_seq_len = (local_start.shape[-2] // model.generator_patch_size_hw[0]) * (
        local_start.shape[-1] // model.generator_patch_size_hw[1]
    )
    seq_len_override = (len(tree_token_ids) + 1 + local_video_count) * frame_seq_len
    local_start_timestep = torch.zeros((local_start.shape[0], 1), device=device, dtype=dtype)
    autocast_ctx = make_autocast_ctx(device, dtype)

    for step, (vt, vd, at, ad) in enumerate(zip(video_timesteps, video_deltas, action_timesteps, action_deltas)):
        video_latents = torch.cat([tree_latents, local_start, local_video], dim=1)
        video_timestep = torch.cat(
            [
                tree_timestep.to(device=device, dtype=dtype),
                local_start_timestep,
                torch.full((local_start.shape[0], local_video_count), float(vt), device=device, dtype=dtype),
            ],
            dim=1,
        )
        with autocast_ctx:
            flow_video, flow_action = model.action_dit.forward_video_action_joint(
                noisy_actions=actions,
                action_timestep=torch.full((local_start.shape[0],), float(at), device=device, dtype=dtype),
                video_latents=video_latents,
                video_timestep=video_timestep,
                conditional_dict=condition,
                prefix_x=prefix_x,
                prefix_t=prefix_t,
                prefix_token_ids=prefix_token_ids,
                tree_token_ids=tree_token_ids,
                vertical_info=model.vertical_info,
                vertical_use_representative_rope=bool(cfg.vertical_use_representative_rope),
                local_start_count=1,
                local_video_count=local_video_count,
                detach_action_video_kv=False,
                action_attend_video=str(getattr(cfg, "joint_action_attend_video", "local_start")),
                action_video_kv_scale=float(getattr(cfg, "joint_action_video_kv_scale", 1.0)),
                joint_proprio=joint_proprio,
                seq_len_override=seq_len_override,
            )
        if local_video_count > 0:
            local_flow = flow_video[:, len(tree_token_ids) + 1: len(tree_token_ids) + 1 + local_video_count]
            local_video = video_scheduler.step(local_flow, vd, local_video)
        actions = action_scheduler.step(flow_action, ad, actions)
        print(
            f"[JointEval] step={step + 1}/{args.joint_steps} "
            f"vt={float(vt):.2f} at={float(at):.2f} "
            f"local_tokens={local_video_count} action_abs={float(actions.float().abs().mean()):.4f}",
            flush=True,
        )
    return actions, local_video if local_video_count > 0 else None


def main() -> None:
    args = parse_args()
    if args.tree_context_mode is None:
        args.tree_context_mode = "generated" if args.use_generated_tree_context else "gt-clean"
    repo_root = Path.cwd()
    backend_root = resolve_path(args.backend_root, repo_root)
    sys.path.insert(0, str(backend_root))

    from model.diffusion import CausalDiffusion
    from utils.dataset import TextVideoDataset
    from utils.vertical_hierarchy import gather_vertical_latents

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    output_dir = resolve_path(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = resolve_path(args.config_path, repo_root)
    checkpoint = resolve_path(args.checkpoint, repo_root)
    cfg = OmegaConf.load(config_path)
    cfg.batch_size = 1
    cfg.image_or_video_shape[0] = 1
    cfg.disable_wandb = True
    cfg.gradient_checkpointing = False
    cfg.data_path = resolve_backend_path(str(cfg.data_path), backend_root)
    cfg.generator_ckpt = str(checkpoint)
    cfg.model_kwargs.model_root = resolve_backend_path(str(cfg.model_kwargs.model_root), backend_root)
    if not hasattr(cfg, "independent_first_frame"):
        cfg.independent_first_frame = False
    cfg.sampling_steps = int(args.tree_sampling_steps)
    cfg.vertical_infer_fixed_denoise_steps = int(args.tree_fixed_denoise_steps)
    cfg.vertical_infer_preserve_budget_ratio = True
    cfg.vertical_infer_reference_total_steps = int(args.tree_sampling_steps)
    if args.guidance_scale is not None:
        cfg.guidance_scale = float(args.guidance_scale)

    data_path = Path(cfg.data_path)
    row_index, raw_row = find_filtered_row(data_path, args.camera_key, args.sample_index)
    row = absolutize_row(raw_row, data_path.parent, repo_root)
    # Closed-loop eval needs RGB local windows and simulator observations, so do
    # not short-circuit through training-time latent caches for the one-row item.
    row.pop("preencoded_cache_path", None)
    row.pop("video_latent_cache_path", None)
    sample_jsonl = output_dir / "sample.jsonl"
    sample_jsonl.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    dataset = TextVideoDataset(
        metadata_path=str(sample_jsonl),
        height=int(cfg.height),
        width=int(cfg.width),
        num_frames=int(cfg.num_frames),
        variable_num_frames=bool(getattr(cfg, "variable_num_frames_train", False)),
        max_num_frames=getattr(cfg, "max_training_video_frames", None),
        video_action_joint=True,
        joint_window_frames=int(cfg.joint_window_frames),
        joint_source_fps=float(cfg.joint_source_fps),
        joint_target_fps=float(cfg.joint_target_fps),
        joint_video_frame_stride=int(cfg.joint_video_frame_stride),
        joint_camera_key=OmegaConf.to_container(cfg.joint_camera_key, resolve=True) if hasattr(cfg, "joint_camera_key") else "agentview_rgb",
        joint_drop_tree_tokens=bool(getattr(cfg, "joint_drop_tree_tokens", False)),
        joint_proprio_stats_path=getattr(cfg, "joint_proprio_stats_path", None),
    )
    item = dataset[0]
    stats_path = Path(row["action_stats_path"])
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    proprio_stats = None
    proprio_stats_path = row.get("proprio_stats_path", getattr(cfg, "joint_proprio_stats_path", None))
    if proprio_stats_path:
        proprio_stats = json.loads(Path(proprio_stats_path).read_text(encoding="utf-8"))

    print(f"[JointEval] selected row={row_index} demo={row.get('demo_id')} prompt={item['prompts']}", flush=True)
    print("[JointEval] building CausalDiffusion model", flush=True)
    model = CausalDiffusion(cfg, device=device).to(device=device)
    model.eval()
    print(f"[JointEval] loading checkpoint {checkpoint}", flush=True)
    model.generator.load_state_dict(load_generator_state(checkpoint), strict=True)
    action_load = model.action_dit.load_state_dict(load_action_state(checkpoint), strict=False)
    if action_load.missing_keys or action_load.unexpected_keys:
        print(f"[Load] action missing={len(action_load.missing_keys)} unexpected={len(action_load.unexpected_keys)}")
    missing_proprio = any(str(key).startswith("proprio_encoder.") for key in action_load.missing_keys)
    use_proprio = (
        not bool(args.disable_proprio)
        and getattr(model.action_dit, "proprio_encoder", None) is not None
        and not missing_proprio
    )
    if not use_proprio and getattr(model.action_dit, "proprio_encoder", None) is not None:
        model.action_dit.proprio_encoder = None
        model.action_dit.proprio_dim = None
    print(
        "[JointEval] proprio_conditioning="
        f"{'on' if use_proprio else 'off'}"
        + (" (checkpoint has no proprio_encoder weights)" if missing_proprio else ""),
        flush=True,
    )
    model.action_dit.bind_video_model(model.generator.model)
    model.text_encoder = CastingTextEncoder(model.text_encoder, device, model.dtype)
    for module in (model.generator, model.action_dit, model.text_encoder, model.vae):
        module.eval()
        module.requires_grad_(False)
    print("[JointEval] checkpoint loaded", flush=True)

    drop_tree_tokens = bool(getattr(cfg, "joint_drop_tree_tokens", False))
    frames = (
        item["frames"].unsqueeze(0).to(device=device, dtype=model.dtype)
        if "frames" in item
        else None
    )
    local_frames = item["joint_local_frames"].unsqueeze(0).to(device=device, dtype=model.dtype)
    gt_actions_norm = item["joint_actions"].unsqueeze(0).to(device=device, dtype=model.dtype)
    joint_proprio = item.get("joint_proprio")
    joint_proprio = (
        joint_proprio.unsqueeze(0).to(device=device, dtype=model.dtype)
        if use_proprio and joint_proprio is not None
        else None
    )
    prompt = [item["prompts"]]
    condition = model.text_encoder(prompt)

    with torch.no_grad():
        print("[JointEval] encoding videos", flush=True)
        # Match training: action KV conditioning uses the local start frame encoded
        # by itself, not the first latent from a temporally encoded local clip.
        local_start = model.vae.encode_to_latent(local_frames[:, :, :1].float()).to(dtype=model.dtype)
        local_video_clean = model.vae.encode_to_latent(local_frames.float()).to(dtype=model.dtype)[:, 1:]
        if local_video_clean.shape[1] < int(cfg.joint_local_video_tokens):
            pad = local_video_clean[:, -1:].expand(-1, int(cfg.joint_local_video_tokens) - local_video_clean.shape[1], -1, -1, -1)
            local_video_clean = torch.cat([local_video_clean, pad], dim=1)
        else:
            local_video_clean = local_video_clean[:, : int(cfg.joint_local_video_tokens)]

        if drop_tree_tokens:
            clean_latent = local_start
            tree_clean = local_start.new_zeros(
                1,
                0,
                local_start.shape[2],
                local_start.shape[3],
                local_start.shape[4],
            )
            tree_token_ids = []
            prefix_x = local_start[:, :0]
            prefix_t = torch.zeros((1, 0), device=device, dtype=model.dtype)
            prefix_token_ids = []
            tree_timestep = torch.zeros((1, 0), device=device, dtype=model.dtype)
            tree_context_outputs = {"tree_context": "dropped"}
        else:
            if frames is None:
                raise KeyError("Non-drop-tree eval requires dataset item['frames'].")
            clean_latent = model.vae.encode_to_latent(frames.float()).to(dtype=model.dtype)
            runtime_vertical_info, runtime_vertical_token_step_budgets = model._get_runtime_vertical(clean_latent.shape[1])
            prefix_x = clean_latent[:, :1]
            prefix_t = torch.zeros((1, 1), device=device, dtype=model.dtype)
            prefix_token_ids = [-1]
            if args.tree_context_mode == "generated":
                tree_clean, tree_timestep, tree_context_outputs = infer_tree_context(
                    cfg=cfg,
                    model=model,
                    device=device,
                    dtype=model.dtype,
                    initial_latent=clean_latent[:, :1],
                    prompt=prompt,
                    output_dir=output_dir,
                    args=args,
                    save_video=not args.skip_tree_video,
                    gt_frames=frames,
                )
                tree_token_ids = list(range(tree_clean.shape[1]))
            else:
                full_tree_clean = gather_vertical_latents(clean_latent, runtime_vertical_info)
                if args.tree_context_mode == "gt-noisy":
                    tree_count = joint_tree_count(cfg, runtime_vertical_info)
                    tree_budgets = list(runtime_vertical_token_step_budgets)[:tree_count]
                    tree_clean, _, tree_token_ids = truncate_joint_tree(
                        cfg,
                        runtime_vertical_info,
                        full_tree_clean,
                        torch.zeros((1, full_tree_clean.shape[1]), device=device, dtype=model.dtype),
                    )
                    tree_noise = torch.randn_like(tree_clean)
                    tree_timestep = model._sample_vertical_timesteps(1, tree_budgets).to(
                        device=device,
                        dtype=model.dtype,
                    )
                    tree_clean = model.scheduler.add_noise(
                        tree_clean.flatten(0, 1),
                        tree_noise.flatten(0, 1),
                        tree_timestep.flatten(0, 1),
                    ).unflatten(0, (1, tree_clean.shape[1]))
                    tree_context_outputs = {
                        "tree_context": "gt_noisy",
                        "tree_context_token_count": int(tree_clean.shape[1]),
                        "tree_timestep_min": float(tree_timestep.float().min().detach().cpu()),
                        "tree_timestep_max": float(tree_timestep.float().max().detach().cpu()),
                    }
                    if not args.skip_tree_video:
                        gt_noisy_video = tensor_video_to_uint8(model.vae.decode_to_pixel(tree_clean.float()))
                        gt_noisy_path = output_dir / "hierarchical_tree_gt_noisy_tokens.mp4"
                        save_video_mp4(gt_noisy_path, list(gt_noisy_video), fps=int(args.fps))
                        tree_context_outputs["hierarchical_tree_gt_noisy_tokens"] = str(gt_noisy_path)
                elif args.tree_context_mode == "gt-clean":
                    tree_timestep_full = torch.zeros((1, full_tree_clean.shape[1]), device=device, dtype=model.dtype)
                    tree_clean, tree_timestep, tree_token_ids = truncate_joint_tree(
                        cfg,
                        runtime_vertical_info,
                        full_tree_clean,
                        tree_timestep_full,
                    )
                    tree_context_outputs = {
                        "tree_context": "gt_clean",
                        "tree_context_token_count": int(tree_clean.shape[1]),
                    }
                else:
                    raise ValueError(f"Unsupported tree_context_mode={args.tree_context_mode}")

        actions, local_video = denoise_joint_action(
            model=model,
            cfg=cfg,
            condition=condition,
            tree_latents=tree_clean,
            tree_timestep=tree_timestep,
            tree_token_ids=tree_token_ids,
            prefix_x=prefix_x,
            prefix_t=prefix_t,
            prefix_token_ids=prefix_token_ids,
            local_start=local_start,
            local_video_clean=local_video_clean,
            joint_proprio=joint_proprio,
            action_shape=tuple(gt_actions_norm.shape),
            args=args,
            device=device,
        )
        torch.cuda.empty_cache()

        outputs: dict[str, Any] = {
            "checkpoint": str(checkpoint),
            "config_path": str(config_path),
            "row_index": row_index,
            "prompt": item["prompts"],
            "source_file": row["source_file"],
            "demo_id": row["demo_id"],
            "window_start": int(item["joint_window_start"].item()),
            "joint_window_indices": item["joint_window_indices"].cpu().tolist(),
            "joint_video_indices": item["joint_video_indices"].cpu().tolist(),
            "action_only": bool(args.action_only),
        }
        outputs.update(tree_context_outputs)

        gt_actions_norm_np = gt_actions_norm[0].float().cpu().numpy()
        pred_actions_norm_np = actions[0].float().cpu().numpy()
        gt_actions_raw = denormalize_actions(gt_actions_norm_np, stats)
        pred_actions_raw = denormalize_actions(clip_normalized_actions(pred_actions_norm_np, cfg), stats)
        outputs["open_loop_mse_norm"] = float(np.mean((pred_actions_norm_np - gt_actions_norm_np) ** 2))
        outputs["open_loop_mse_raw"] = float(np.mean((pred_actions_raw - gt_actions_raw) ** 2))

        if not args.skip_local_video:
            if local_video is None:
                raise ValueError("--skip-local-video must be used when --action-only is enabled.")
            local_video_dec = tensor_video_to_uint8(model.vae.decode_to_pixel(torch.cat([local_start, local_video], dim=1).float()))
            local_gt_dec = tensor_video_to_uint8(model.vae.decode_to_pixel(torch.cat([local_start, local_video_clean], dim=1).float()))
            local_path = output_dir / "local_video_gen_vs_gt.mp4"
            save_side_by_side(local_path, local_gt_dec, local_video_dec, fps=max(1, args.fps // max(1, int(cfg.joint_video_frame_stride))))
            outputs["local_video_side_by_side"] = str(local_path)

        if not args.skip_open_loop_action:
            plot_path = output_dir / "open_loop_action_curves.png"
            csv_path = output_dir / "open_loop_actions.csv"
            plot_action_curves(plot_path, gt_actions_raw, pred_actions_raw, "GT vs generated controller actions")
            write_action_csv(csv_path, gt_actions_raw, pred_actions_raw)
            np.savez(
                output_dir / "open_loop_actions.npz",
                gt_norm=gt_actions_norm_np,
                pred_norm=pred_actions_norm_np,
                gt_raw=gt_actions_raw,
                pred_raw=pred_actions_raw,
            )
            outputs["open_loop_action_plot"] = str(plot_path)
            outputs["open_loop_action_csv"] = str(csv_path)

        if not args.skip_close_loop_action:
            close_gt = replay_actions(
                row=row,
                actions_raw=gt_actions_raw,
                output_path=output_dir / "close_loop_gt_action.mp4",
                args=args,
            )
            close_gen = replay_actions(
                row=row,
                actions_raw=pred_actions_raw,
                output_path=output_dir / "close_loop_gen_action.mp4",
                args=args,
            )
            gt_video = imageio.mimread(close_gt["video"])
            gen_video = imageio.mimread(close_gen["video"])
            count = min(len(gt_video), len(gen_video))
            pair = [np.concatenate([gt_video[i], gen_video[i]], axis=1) for i in range(count)]
            pair_path = output_dir / "close_loop_gt_vs_gen_action.mp4"
            save_video_mp4(pair_path, pair, fps=int(args.fps))
            outputs["close_loop_gt"] = close_gt
            outputs["close_loop_gen"] = close_gen
            outputs["close_loop_side_by_side"] = str(pair_path)

        if args.closed_loop_simulator:
            sys.path.insert(0, str(Path(args.physical_agent_root)))
            from libero_agent import LiberoActionError

            env, bddl_file = build_libero_env(row, args)
            rollout_camera_keys = parse_camera_keys(getattr(cfg, "joint_camera_key", args.camera_key))
            sim_frames = [resize_uint8(frame_from_obs(env.last_obs), args.camera_height, args.camera_width)]
            sim_actions_norm: list[np.ndarray] = []
            sim_actions_raw: list[np.ndarray] = []
            sim_infer_seconds: list[float] = []
            executed = 0
            sim_success = bool(env.state().get("success", False))
            success_step = 0 if sim_success else None
            try:
                timeout_steps = max(1, int(args.closed_loop_timeout_steps))
                for chunk_id in range(max(1, int(args.closed_loop_chunks))):
                    if executed >= timeout_steps or sim_success:
                        break
                    obs_frame = fastwam_frame_from_obs(env.last_obs, rollout_camera_keys)
                    obs_tensor = uint8_fastwam_frames_to_tensor(
                        [obs_frame],
                        height=int(cfg.height),
                        width=int(cfg.width),
                        num_cameras=len(rollout_camera_keys),
                    ).unsqueeze(0).to(device=device)
                    current_local_start = model.vae.encode_to_latent(obs_tensor.float()).to(dtype=model.dtype)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    infer_start = time.perf_counter()
                    current_actions, _ = denoise_joint_action(
                        model=model,
                        cfg=cfg,
                        condition=condition,
                        tree_latents=tree_clean,
                        tree_timestep=tree_timestep,
                        tree_token_ids=tree_token_ids,
                        prefix_x=prefix_x,
                        prefix_t=prefix_t,
                        prefix_token_ids=prefix_token_ids,
                        local_start=current_local_start,
                        local_video_clean=None,
                        joint_proprio=(
                            torch.from_numpy(proprio_from_obs(env.last_obs, proprio_stats))
                            .unsqueeze(0)
                            .to(device=device, dtype=model.dtype)
                            if use_proprio
                            else None
                        ),
                        action_shape=tuple(gt_actions_norm.shape),
                        args=argparse.Namespace(**{**vars(args), "action_only": True}),
                        device=device,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    sim_infer_seconds.append(time.perf_counter() - infer_start)
                    action_norm_np = current_actions[0].float().cpu().numpy()
                    action_raw_np = denormalize_actions(clip_normalized_actions(action_norm_np, cfg), stats)
                    steps_this_chunk = min(
                        int(args.closed_loop_execute_steps),
                        action_raw_np.shape[0],
                        timeout_steps - executed,
                    )
                    print(
                        f"[ClosedLoopSimulator] chunk={chunk_id} executing {steps_this_chunk} actions "
                        f"(executed={executed}/{timeout_steps})",
                        flush=True,
                    )
                    for step_id, action in enumerate(action_raw_np[:steps_this_chunk]):
                        try:
                            result = env.step(
                                action[:6],
                                mode="ee_delta",
                                gripper=float(np.clip(action[6], -1.0, 1.0)),
                            )
                        except LiberoActionError as exc:
                            print(f"[ClosedLoopSimulator] stop chunk={chunk_id} step={step_id}: {exc.to_dict()}")
                            raise
                        if not args.skip_closed_loop_video:
                            sim_frames.append(
                                resize_uint8(frame_from_obs(result.observation), args.camera_height, args.camera_width)
                            )
                        sim_actions_norm.append(action_norm_np[step_id])
                        sim_actions_raw.append(action.astype(np.float32))
                        executed += 1
                        sim_success = bool(result.state_after.get("success", False))
                        if sim_success:
                            success_step = executed
                            print(
                                f"[ClosedLoopSimulator] success after executed_steps={executed}",
                                flush=True,
                            )
                            break
                    if sim_success:
                        break
            except LiberoActionError:
                pass
            finally:
                env.close()
            sim_path = output_dir / "closed_loop_simulator_action.mp4"
            if not args.skip_closed_loop_video:
                save_video_mp4(sim_path, sim_frames, fps=int(args.fps))
            sim_npz = output_dir / "closed_loop_simulator_actions.npz"
            np.savez(
                sim_npz,
                action_norm=np.asarray(sim_actions_norm, dtype=np.float32),
                action_raw=np.asarray(sim_actions_raw, dtype=np.float32),
            )
            outputs["closed_loop_simulator"] = {
                "video": None if args.skip_closed_loop_video else str(sim_path),
                "actions": str(sim_npz),
                "executed_steps": int(executed),
                "success": bool(sim_success),
                "success_step": success_step,
                "chunks": int(args.closed_loop_chunks),
                "timeout_steps": int(args.closed_loop_timeout_steps),
                "bddl_file": bddl_file,
                "action_only": True,
                "guidance_scale": float(args.guidance_scale),
                "joint_steps": int(args.joint_steps),
                "inference_calls": len(sim_infer_seconds),
                "inference_seconds_total": float(np.sum(sim_infer_seconds)) if sim_infer_seconds else 0.0,
                "inference_seconds_mean": float(np.mean(sim_infer_seconds)) if sim_infer_seconds else None,
                "inference_seconds_min": float(np.min(sim_infer_seconds)) if sim_infer_seconds else None,
                "inference_seconds_max": float(np.max(sim_infer_seconds)) if sim_infer_seconds else None,
                "inference_ms_per_generated_action": (
                    float(1000.0 * np.sum(sim_infer_seconds) / (len(sim_infer_seconds) * int(gt_actions_norm.shape[1])))
                    if sim_infer_seconds
                    else None
                ),
                "inference_ms_per_executed_action": (
                    float(1000.0 * np.sum(sim_infer_seconds) / max(1, executed))
                    if sim_infer_seconds
                    else None
                ),
            }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    print(f"[JointEval] wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
