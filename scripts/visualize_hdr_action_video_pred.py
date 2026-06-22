#!/usr/bin/env python3
"""Decode frozen HDR video DiT leaf x0 predictions for an action-training batch."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-root", default="../HiDiT/Causal-Forcing")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output", default="logs/debug/hdr_action_video_leaf_pred_x0.mp4")
    parser.add_argument("--side-by-side-output", default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=4)
    return parser.parse_args()


def resolve_backend_path(value: str, backend_root: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((backend_root / path).resolve())


def load_generator_state(path: str) -> dict:
    state_dict = torch.load(path, map_location="cpu")
    if "generator" in state_dict:
        state_dict = state_dict["generator"]
        fixed = {}
        for key, value in state_dict.items():
            if key.startswith("model._fsdp_wrapped_module."):
                key = key.replace("model._fsdp_wrapped_module.", "model.", 1)
            fixed[key] = value
        return fixed
    if "model" in state_dict:
        return state_dict["model"]
    if "generator_ema" in state_dict:
        fixed = {}
        for key, value in state_dict["generator_ema"].items():
            if key.startswith("model._fsdp_wrapped_module."):
                key = key.replace("model._fsdp_wrapped_module.", "model.", 1)
            fixed[key] = value
        return fixed
    return state_dict


def tensor_video_to_uint8(video: torch.Tensor) -> np.ndarray:
    video = video.detach().float().cpu().clamp(-1, 1)
    if video.ndim != 4:
        raise ValueError(f"Expected [F,C,H,W] or [C,F,H,W], got {tuple(video.shape)}")
    if video.shape[1] == 3:
        frames = video
    elif video.shape[0] == 3:
        frames = video.permute(1, 0, 2, 3)
    else:
        raise ValueError(f"Cannot infer channel dimension from {tuple(video.shape)}")
    frames = ((frames + 1.0) * 127.5).byte().permute(0, 2, 3, 1).numpy()
    return frames


def resize_frame(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    import cv2

    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def sample_frames(frames: np.ndarray, target_count: int) -> np.ndarray:
    if target_count <= 0:
        raise ValueError(f"target_count must be positive, got {target_count}")
    if len(frames) == target_count:
        return frames
    if len(frames) == 1:
        return np.repeat(frames, target_count, axis=0)
    indices = np.linspace(0, len(frames) - 1, target_count).round().astype(np.int64)
    return frames[indices]


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    backend_root = (repo_root / args.backend_root).resolve()
    sys.path.insert(0, str(backend_root))

    from model.diffusion import CausalDiffusion
    from utils.dataset import TextVideoDataset
    from utils.vertical_hierarchy import CONDITION_TOKEN_ID, gather_vertical_latents

    cfg = OmegaConf.load(args.config_path)
    cfg.action_training = False
    cfg.batch_size = 1
    cfg.image_or_video_shape[0] = 1
    cfg.data_path = resolve_backend_path(str(cfg.data_path), backend_root)
    cfg.generator_ckpt = resolve_backend_path(str(cfg.generator_ckpt), backend_root)
    cfg.model_kwargs.model_root = resolve_backend_path(str(cfg.model_kwargs.model_root), backend_root)
    cfg.disable_wandb = True

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if cfg.mixed_precision else torch.float32

    dataset = TextVideoDataset(
        metadata_path=cfg.data_path,
        height=int(cfg.height),
        width=int(cfg.width),
        num_frames=int(cfg.num_frames),
        variable_num_frames=bool(getattr(cfg, "variable_num_frames_train", False)),
        max_num_frames=getattr(cfg, "max_training_video_frames", None),
    )
    item = dataset[int(args.sample_index)]
    frames = item["frames"].unsqueeze(0).to(device=device, dtype=dtype)
    prompts = [item["prompts"]]

    model = CausalDiffusion(cfg, device=device)
    model.generator = model.generator.to(device=device, dtype=dtype).eval().requires_grad_(False)
    model.text_encoder = model.text_encoder.to(device=device).eval().requires_grad_(False)
    model.vae = model.vae.to(device=device, dtype=torch.bfloat16 if cfg.mixed_precision else torch.float32)
    model.vae.eval().requires_grad_(False)
    model.generator.load_state_dict(load_generator_state(cfg.generator_ckpt), strict=True)

    with torch.no_grad():
        clean_latent = model.vae.encode_to_latent(frames).to(device=device, dtype=dtype)
        first_frame_latent = clean_latent[:, :1]
        runtime_vertical_info, runtime_vertical_token_step_budgets = model._get_runtime_vertical(clean_latent.shape[1])
        vertical_clean_latent = gather_vertical_latents(clean_latent, runtime_vertical_info)
        vertical_noise = torch.randn_like(vertical_clean_latent)
        timestep = model._sample_vertical_timesteps(clean_latent.shape[0], runtime_vertical_token_step_budgets)
        noisy_latents = model.scheduler.add_noise(
            vertical_clean_latent.flatten(0, 1),
            vertical_noise.flatten(0, 1),
            timestep.flatten(0, 1),
        ).unflatten(0, (clean_latent.shape[0], vertical_clean_latent.shape[1]))
        conditional_dict = model.text_encoder(text_prompts=prompts)
        conditional_dict = {
            key: value.to(device=device, dtype=dtype)
            for key, value in conditional_dict.items()
        }
        prefix_t = torch.zeros([clean_latent.shape[0], 1], device=device, dtype=dtype)
        frame_seq_len = (
            clean_latent.shape[-2] // model.generator_patch_size_hw[0]
        ) * (
            clean_latent.shape[-1] // model.generator_patch_size_hw[1]
        )
        seq_len_override = runtime_vertical_info["num_tokens"] * frame_seq_len
        _, x0_pred = model.generator(
            noisy_image_or_video=noisy_latents,
            conditional_dict=conditional_dict,
            timestep=timestep,
            prefix_x=first_frame_latent,
            prefix_t=prefix_t,
            prefix_token_ids=[CONDITION_TOKEN_ID],
            noisy_token_ids=list(range(runtime_vertical_info["num_tokens"])),
            vertical_info=runtime_vertical_info,
            vertical_use_representative_rope=model.vertical_use_representative_rope,
            seq_len_override=seq_len_override,
        )
        leaf_start = int(runtime_vertical_info["leaf_start_index"])
        pred_leaf_latent = x0_pred[:, leaf_start:]
        pred_leaf_video = model.vae.decode_to_pixel(pred_leaf_latent)[0]

    pred_frames = tensor_video_to_uint8(pred_leaf_video)
    source_frames_full = tensor_video_to_uint8(frames[0])
    source_frames = sample_frames(source_frames_full, len(pred_frames))
    height, width = pred_frames[0].shape[:2]
    source_frames = [resize_frame(frame, height, width) for frame in source_frames]
    paired = [np.concatenate([src, pred], axis=1) for src, pred in zip(source_frames, pred_frames)]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, pred_frames, fps=args.fps)
    side_by_side_output = (
        Path(args.side_by_side_output)
        if args.side_by_side_output
        else output.with_name(f"{output.stem}_side_by_side{output.suffix}")
    )
    imageio.mimsave(side_by_side_output, paired, fps=args.fps)
    print(f"[VideoPred] prompt={item['prompts']}")
    print(f"[VideoPred] video_path={item.get('video_path')}")
    print(f"[VideoPred] timestep_mean={float(timestep.float().mean().item()):.4f}")
    print(f"[VideoPred] x0_pred_leaf_shape={tuple(pred_leaf_latent.shape)}")
    print(f"[VideoPred] source_frame_count={len(source_frames_full)}")
    print(f"[VideoPred] decoded_pred_frame_count={len(pred_frames)}")
    print(f"[VideoPred] leaf_representative_indices={[runtime_vertical_info['representative_indices'][token_id] for token_id in runtime_vertical_info['leaf_token_ids']]}")
    print(f"[VideoPred] wrote_pred_leaf_video={output}")
    print(f"[VideoPred] wrote_side_by_side={side_by_side_output}")


if __name__ == "__main__":
    main()
