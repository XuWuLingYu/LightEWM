#!/usr/bin/env python3
"""Run HDR action open-loop evaluation from a LIBERO first frame."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

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
        default="logs/LIBERO-HDR_train_action/20260613_103637/causal_forcing_config.yaml",
    )
    parser.add_argument(
        "--action-checkpoint",
        default="logs/LIBERO-HDR_train_action/20260613_103637/checkpoint_model_005000/model.pt",
    )
    parser.add_argument("--output-dir", default="logs/debug/libero_hdr_action_open_loop_5k")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--camera-key", default="agentview_rgb")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--action-steps", type=int, default=50)
    parser.add_argument("--action-shift", type=float, default=None)
    parser.add_argument("--video-sampling-steps", type=int, default=None)
    parser.add_argument("--vertical-fixed-denoise-steps", type=int, default=5)
    parser.add_argument("--no-vertical-preserve-budget-ratio", action="store_true")
    parser.add_argument("--skip-video-decode", action="store_true")
    parser.add_argument(
        "--use-video-latent-cache",
        action="store_true",
        help="Use the sample's cached train-set video tree latents instead of running video inference.",
    )
    return parser.parse_args()


def resolve_repo_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def resolve_backend_path(value: str | Path, backend_root: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((backend_root / path).resolve())


def load_generator_state(path: str | Path) -> dict[str, torch.Tensor]:
    state_dict = torch.load(path, map_location="cpu")
    if "generator" in state_dict and state_dict["generator"] is not None:
        state_dict = state_dict["generator"]
    elif "generator_ema" in state_dict:
        state_dict = state_dict["generator_ema"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]
    fixed = {}
    for key, value in state_dict.items():
        if key.startswith("model._fsdp_wrapped_module."):
            key = key.replace("model._fsdp_wrapped_module.", "model.", 1)
        fixed[key] = value
    return fixed


def load_action_state(path: str | Path) -> dict[str, torch.Tensor]:
    state_dict = torch.load(path, map_location="cpu")
    if "action_dit" in state_dict:
        return state_dict["action_dit"]
    return state_dict


def tensor_video_to_uint8(video: torch.Tensor) -> np.ndarray:
    video = video.detach().float().cpu()
    if video.ndim != 4:
        raise ValueError(f"Expected video [F,C,H,W] or [C,F,H,W], got {tuple(video.shape)}")
    if video.shape[1] == 3:
        frames = video
    elif video.shape[0] == 3:
        frames = video.permute(1, 0, 2, 3)
    else:
        raise ValueError(f"Cannot infer channel dimension from {tuple(video.shape)}")
    if frames.min() < -0.05:
        frames = frames * 0.5 + 0.5
    frames = frames.clamp(0, 1)
    return (frames * 255.0).round().byte().permute(0, 2, 3, 1).numpy()


def sample_video_frames(frames: np.ndarray, target_count: int) -> np.ndarray:
    if len(frames) == target_count:
        return frames
    if len(frames) == 1:
        return np.repeat(frames, target_count, axis=0)
    indices = np.linspace(0, len(frames) - 1, target_count).round().astype(np.int64)
    return frames[indices]


def find_filtered_sample_index(jsonl_path: Path, camera_key: str, ordinal: int) -> tuple[int, dict[str, Any]]:
    matches = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_index, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            row_camera = row.get("camera_key")
            row_video = str(row.get("video_path", row.get("video", "")))
            if row_camera == camera_key or camera_key in row_video:
                matches.append((line_index, row))
                if len(matches) > ordinal:
                    return matches[ordinal]
    raise RuntimeError(f"No sample ordinal={ordinal} found for camera_key={camera_key} in {jsonl_path}")


def absolutize_sample_paths(row: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    row = dict(row)
    for key in (
        "video_path",
        "video",
        "video_latent_cache_path",
        "action_path",
        "action_stats_path",
        "source_file",
    ):
        value = row.get(key)
        if not value:
            continue
        path = Path(str(value))
        row[key] = str(path if path.is_absolute() else (base_dir / path).resolve())
    return row


def denormalize_actions(actions_norm: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    min_v = np.asarray(stats["min"], dtype=np.float32)
    max_v = np.asarray(stats["max"], dtype=np.float32)
    eps = float(stats.get("eps", 1e-6))
    return (actions_norm + 1.0) * 0.5 * (max_v - min_v + eps) + min_v


def save_action_csv(path: Path, gt: np.ndarray, pred: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["step"]
        for name in ACTION_NAMES:
            header.extend([f"gt_{name}", f"gen_{name}"])
        writer.writerow(header)
        for step in range(gt.shape[0]):
            row: list[Any] = [step]
            for dim in range(gt.shape[1]):
                row.extend([float(gt[step, dim]), float(pred[step, dim])])
            writer.writerow(row)


def plot_action_curves(path: Path, gt: np.ndarray, pred: np.ndarray, title: str) -> None:
    fig, axes = plt.subplots(7, 1, figsize=(14, 18), sharex=True)
    xs = np.arange(gt.shape[0])
    for dim, ax in enumerate(axes):
        ax.plot(xs, gt[:, dim], label="GT", linewidth=1.8)
        ax.plot(xs, pred[:, dim], label="Generated", linewidth=1.4, linestyle="--")
        ax.set_ylabel(ACTION_NAMES[dim])
        ax.grid(True, alpha=0.25)
        if dim == 0:
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("Action step")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    backend_root = resolve_repo_path(args.backend_root, repo_root)
    sys.path.insert(0, str(backend_root))

    from model.diffusion import CausalDiffusion
    from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
    from utils.dataset import TextVideoDataset
    from utils.vertical_hierarchy import CONDITION_TOKEN_ID
    from utils.scheduler import WanContinuousFlowMatchScheduler

    output_dir = resolve_repo_path(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    action_checkpoint = resolve_repo_path(args.action_checkpoint, repo_root)
    config_path = resolve_repo_path(args.config_path, repo_root)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = OmegaConf.load(config_path)
    cfg.batch_size = 1
    cfg.image_or_video_shape[0] = 1
    cfg.disable_wandb = True
    if not hasattr(cfg, "independent_first_frame"):
        cfg.independent_first_frame = False
    cfg.data_path = resolve_backend_path(str(cfg.data_path), backend_root)
    cfg.generator_ckpt = resolve_backend_path(str(cfg.generator_ckpt), backend_root)
    cfg.model_kwargs.model_root = resolve_backend_path(str(cfg.model_kwargs.model_root), backend_root)
    if args.video_sampling_steps is not None:
        cfg.sampling_steps = int(args.video_sampling_steps)
    elif not hasattr(cfg, "sampling_steps"):
        cfg.sampling_steps = int(max(cfg.vertical_step_budgets))
    cfg.vertical_infer_fixed_denoise_steps = int(args.vertical_fixed_denoise_steps)
    cfg.vertical_infer_preserve_budget_ratio = not bool(args.no_vertical_preserve_budget_ratio)
    cfg.vertical_infer_reference_total_steps = int(getattr(cfg, "sampling_steps", 50))

    data_path = Path(cfg.data_path)
    filtered_index, raw_row = find_filtered_sample_index(
        data_path,
        camera_key=args.camera_key,
        ordinal=int(args.sample_index),
    )
    raw_row = absolutize_sample_paths(raw_row, data_path.parent)
    sample_jsonl = output_dir / "agentview_sample.jsonl"
    sample_jsonl.write_text(json.dumps(raw_row, ensure_ascii=False) + "\n", encoding="utf-8")

    dataset = TextVideoDataset(
        metadata_path=str(sample_jsonl),
        height=int(cfg.height),
        width=int(cfg.width),
        num_frames=int(cfg.num_frames),
        variable_num_frames=bool(getattr(cfg, "variable_num_frames_train", False)),
        max_num_frames=getattr(cfg, "max_training_video_frames", None),
    )
    item = dataset[0]
    if "actions" not in item:
        raise KeyError("Selected sample does not contain action_path/actions.")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if bool(cfg.mixed_precision) else torch.float32

    model = CausalDiffusion(cfg, device=device)
    model.generator = model.generator.to(device=device, dtype=dtype).eval().requires_grad_(False)
    model.text_encoder = model.text_encoder.to(device=device).eval().requires_grad_(False)
    model.vae = model.vae.to(device=device, dtype=torch.bfloat16 if bool(cfg.mixed_precision) else torch.float32)
    model.vae.eval().requires_grad_(False)
    model.action_dit = model.action_dit.to(device=device, dtype=dtype).eval().requires_grad_(False)

    model.generator.load_state_dict(load_generator_state(cfg.generator_ckpt), strict=True)
    model.action_dit.bind_video_model(model.generator.model)
    load_info = model.action_dit.load_state_dict(load_action_state(action_checkpoint), strict=False)
    if load_info.missing_keys or load_info.unexpected_keys:
        print(
            f"[OpenLoop] action ckpt load missing={list(load_info.missing_keys)} "
            f"unexpected={list(load_info.unexpected_keys)}",
            flush=True,
        )

    frames = item["frames"].unsqueeze(0).to(device=device, dtype=dtype)
    prompts = [item["prompts"]]
    gt_actions_norm = item["actions"].numpy().astype(np.float32)
    stats_path = Path(item["action_path"]).parents[1] / "hdr_actions_leaf8_stats.json"
    if raw_row.get("action_stats_path"):
        raw_stats_path = Path(raw_row["action_stats_path"])
        stats_path = raw_stats_path if raw_stats_path.is_absolute() else (sample_jsonl.parent / raw_stats_path).resolve()
    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    with torch.no_grad():
        clean_latent = model.vae.encode_to_latent(frames).to(device=device, dtype=dtype)
        first_frame_latent = clean_latent[:, :1]
        if args.use_video_latent_cache:
            if "video_leaf_latents" in item:
                leaf_latents = item["video_leaf_latents"].unsqueeze(0).to(device=device, dtype=dtype)
                vertical_latents = leaf_latents
            elif "video_latents" in item:
                vertical_latents = item["video_latents"].unsqueeze(0).to(device=device, dtype=dtype)
                leaf_latents = vertical_latents[:, -int(model.vertical_leaf_frames) :]
            else:
                raise KeyError("Selected sample does not contain video_latent_cache_path/video leaf latents.")
            generated_video = None
        else:
            noise = torch.randn(
                [
                    1,
                    int(model.vertical_info["num_tokens"]),
                    clean_latent.shape[2],
                    clean_latent.shape[3],
                    clean_latent.shape[4],
                ],
                device=device,
                dtype=dtype,
            )

            video_pipe = CausalDiffusionInferencePipeline(
                cfg,
                device=device,
                generator=model.generator,
                text_encoder=CastingTextEncoder(model.text_encoder, device=device, dtype=dtype),
                vae=model.vae,
            )
            video_pipe = video_pipe.to(device=device).eval()
            leaf_latents, vertical_latents = video_pipe.inference(
                noise=noise,
                text_prompts=prompts,
                initial_latent=first_frame_latent,
                return_latents=True,
                return_video=False,
            )
            if args.skip_video_decode:
                generated_video = None
            else:
                generated_video = model.vae.decode_to_pixel(leaf_latents)
                generated_video = (generated_video * 0.5 + 0.5).clamp(0, 1)

        conditional_dict = {
            key: value.to(device=device, dtype=dtype)
            for key, value in model.text_encoder(text_prompts=prompts).items()
        }
        prefix_t = torch.zeros([1, 1], device=device, dtype=dtype)
        video_timestep = model._build_action_cache_video_timestep(vertical_latents)
        frame_seq_len = (
            clean_latent.shape[-2] // model.generator_patch_size_hw[0]
        ) * (
            clean_latent.shape[-1] // model.generator_patch_size_hw[1]
        )
        seq_len_override = int(model.vertical_info["num_tokens"]) * frame_seq_len

        actions = torch.randn(
            [1, int(model.vertical_leaf_frames) * int(model.actions_per_leaf), int(cfg.action_dit_config.action_dim)],
            device=device,
            dtype=dtype,
        )
        action_shift = (
            float(args.action_shift)
            if args.action_shift is not None
            else float(getattr(cfg, "action_infer_shift", getattr(cfg, "action_train_shift", 5.0)))
        )
        action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(cfg.num_train_timestep),
            shift=action_shift,
        )
        action_timesteps, action_deltas = action_scheduler.build_inference_schedule(
            num_inference_steps=int(args.action_steps),
            device=device,
            dtype=actions.dtype,
        )
        for timestep_value, delta_value in zip(action_timesteps, action_deltas):
            action_timestep = timestep_value * torch.ones([1], device=device, dtype=torch.float32)
            flow_pred = model.action_dit(
                noisy_actions=actions,
                action_timestep=action_timestep,
                video_latents=vertical_latents,
                video_timestep=video_timestep,
                conditional_dict=conditional_dict,
                prefix_x=first_frame_latent,
                prefix_t=prefix_t,
                prefix_token_ids=[CONDITION_TOKEN_ID],
                noisy_token_ids=list(range(int(model.vertical_info["num_tokens"]))),
                vertical_info=model.vertical_info,
                vertical_use_representative_rope=model.vertical_use_representative_rope,
                seq_len_override=seq_len_override,
            )
            actions = action_scheduler.step(flow_pred, delta_value, actions)

    gen_actions_norm = actions.detach().float().cpu().numpy()[0]
    gt_actions_raw = denormalize_actions(gt_actions_norm, stats)
    gen_actions_raw = denormalize_actions(np.clip(gen_actions_norm, -1.5, 1.5), stats)

    np.savez(
        output_dir / "actions_gt_vs_gen.npz",
        gt_norm=gt_actions_norm,
        gen_norm=gen_actions_norm,
        gt_raw=gt_actions_raw,
        gen_raw=gen_actions_raw,
        source_jsonl_index=np.asarray([filtered_index], dtype=np.int64),
    )
    save_action_csv(output_dir / "actions_raw.csv", gt_actions_raw, gen_actions_raw)
    save_action_csv(output_dir / "actions_normalized.csv", gt_actions_norm, gen_actions_norm)
    plot_action_curves(
        output_dir / "action_raw_curves.png",
        gt_actions_raw,
        gen_actions_raw,
        "HDR Action Open-Loop: Raw GT vs Generated",
    )
    plot_action_curves(
        output_dir / "action_normalized_curves.png",
        gt_actions_norm,
        gen_actions_norm,
        "HDR Action Open-Loop: Normalized GT vs Generated",
    )

    if generated_video is not None:
        gen_frames = tensor_video_to_uint8(generated_video[0])
        gt_frames = tensor_video_to_uint8(frames[0])
        gt_frames = sample_video_frames(gt_frames, len(gen_frames))
        paired = [np.concatenate([gt, gen], axis=1) for gt, gen in zip(gt_frames, gen_frames)]
        imageio.mimsave(output_dir / "gt_video.mp4", gt_frames, fps=args.video_fps)
        imageio.mimsave(output_dir / "generated_video.mp4", gen_frames, fps=args.video_fps)
        imageio.mimsave(output_dir / "gt_left_generated_right.mp4", paired, fps=args.video_fps)

    summary = {
        "config_path": str(config_path),
        "action_checkpoint": str(action_checkpoint),
        "source_jsonl_index": filtered_index,
        "camera_key": raw_row.get("camera_key"),
        "video_path": item.get("video_path"),
        "action_path": item.get("action_path"),
        "prompt": prompts[0],
        "video_sampling_steps": int(cfg.sampling_steps),
        "vertical_infer_fixed_denoise_steps": int(cfg.vertical_infer_fixed_denoise_steps),
        "vertical_infer_preserve_budget_ratio": bool(cfg.vertical_infer_preserve_budget_ratio),
        "action_sampling_steps": int(args.action_steps),
        "action_shift": float(action_shift),
        "use_video_latent_cache": bool(args.use_video_latent_cache),
        "gt_norm_abs_mean": float(np.mean(np.abs(gt_actions_norm))),
        "gen_norm_abs_mean": float(np.mean(np.abs(gen_actions_norm))),
        "gt_raw_min": gt_actions_raw.min(axis=0).tolist(),
        "gt_raw_max": gt_actions_raw.max(axis=0).tolist(),
        "gen_raw_min": gen_actions_raw.min(axis=0).tolist(),
        "gen_raw_max": gen_actions_raw.max(axis=0).tolist(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[OpenLoop] wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
