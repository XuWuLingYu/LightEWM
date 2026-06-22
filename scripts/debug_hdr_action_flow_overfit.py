#!/usr/bin/env python3
"""Overfit one HDR action batch to verify the action flow-matching objective."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-root", default="../HiDiT/Causal-Forcing")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--action-attend-video", default=None, choices=[None, "none", "parents", "all"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
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
    elif "model" in state_dict:
        state_dict = state_dict["model"]
    elif "generator_ema" in state_dict:
        state_dict = state_dict["generator_ema"]
    fixed = {}
    for key, value in state_dict.items():
        if key.startswith("model._fsdp_wrapped_module."):
            key = key.replace("model._fsdp_wrapped_module.", "model.", 1)
        fixed[key] = value
    return fixed


def load_action_state(path: str) -> dict:
    state_dict = torch.load(path, map_location="cpu")
    if "action_dit" in state_dict:
        return state_dict["action_dit"]
    if "backbone_state_dict" in state_dict:
        return state_dict["backbone_state_dict"]
    return state_dict


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    backend_root = (repo_root / args.backend_root).resolve()
    sys.path.insert(0, str(backend_root))

    from model.diffusion import CausalDiffusion
    from utils.dataset import TextVideoDataset
    from utils.vertical_hierarchy import CONDITION_TOKEN_ID, gather_vertical_latents

    torch.manual_seed(args.seed)
    cfg = OmegaConf.load(args.config_path)
    cfg.batch_size = 1
    cfg.image_or_video_shape[0] = 1
    cfg.data_path = resolve_backend_path(str(cfg.data_path), backend_root)
    cfg.generator_ckpt = resolve_backend_path(str(cfg.generator_ckpt), backend_root)
    cfg.model_kwargs.model_root = resolve_backend_path(str(cfg.model_kwargs.model_root), backend_root)
    if getattr(cfg, "action_dit_ckpt", None):
        cfg.action_dit_ckpt = resolve_backend_path(str(cfg.action_dit_ckpt), backend_root)
    if args.action_attend_video is not None:
        cfg.action_dit_config.action_attend_video = args.action_attend_video
    cfg.disable_wandb = True

    device = torch.device(args.device)
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
    actions = item["actions"].unsqueeze(0).to(device=device, dtype=dtype)
    prompts = [item["prompts"]]

    model = CausalDiffusion(cfg, device=device)
    model.generator = model.generator.to(device=device, dtype=dtype).eval().requires_grad_(False)
    model.text_encoder = model.text_encoder.to(device=device).eval().requires_grad_(False)
    model.vae = model.vae.to(device=device, dtype=torch.bfloat16 if cfg.mixed_precision else torch.float32)
    model.vae.eval().requires_grad_(False)
    model.action_dit = model.action_dit.to(device=device, dtype=dtype).train()
    model.generator.load_state_dict(load_generator_state(cfg.generator_ckpt), strict=True)
    model.action_dit.bind_video_model(model.generator.model)
    if getattr(cfg, "action_dit_ckpt", None):
        info = model.action_dit.load_state_dict(load_action_state(cfg.action_dit_ckpt), strict=False)
        print(f"[Overfit] loaded action ckpt missing={list(info.missing_keys)} unexpected={list(info.unexpected_keys)}")

    optimizer = torch.optim.AdamW(
        [p for p in model.action_dit.parameters() if p.requires_grad],
        lr=float(args.lr),
        betas=(float(getattr(cfg, "beta1", 0.9)), float(getattr(cfg, "beta2", 0.999))),
        weight_decay=float(getattr(cfg, "weight_decay", 0.0)),
    )

    with torch.no_grad():
        clean_latent = model.vae.encode_to_latent(frames).to(device=device, dtype=dtype)
        first_frame_latent = clean_latent[:, :1]
        runtime_vertical_info, runtime_vertical_token_step_budgets = model._get_runtime_vertical(clean_latent.shape[1])
        vertical_clean_latent = gather_vertical_latents(clean_latent, runtime_vertical_info)
        vertical_noise = torch.randn_like(vertical_clean_latent)
        video_timestep = model._sample_vertical_timesteps(clean_latent.shape[0], runtime_vertical_token_step_budgets)
        noisy_latents = model.scheduler.add_noise(
            vertical_clean_latent.flatten(0, 1),
            vertical_noise.flatten(0, 1),
            video_timestep.flatten(0, 1),
        ).unflatten(0, (clean_latent.shape[0], vertical_clean_latent.shape[1]))
        conditional_dict = {
            key: value.to(device=device, dtype=dtype)
            for key, value in model.text_encoder(text_prompts=prompts).items()
        }
        prefix_t = torch.zeros([clean_latent.shape[0], 1], device=device, dtype=dtype)
        frame_seq_len = (
            clean_latent.shape[-2] // model.generator_patch_size_hw[0]
        ) * (
            clean_latent.shape[-1] // model.generator_patch_size_hw[1]
        )
        seq_len_override = runtime_vertical_info["num_tokens"] * frame_seq_len
        action_noise = torch.randn_like(actions)
        action_timestep_sample = torch.full(
            [actions.shape[0]],
            float(model.scheduler.timesteps[len(model.scheduler.timesteps) // 2].item()),
            device=device,
            dtype=dtype,
        )
        action_timestep = action_timestep_sample[:, None].expand(-1, actions.shape[1])
        noisy_actions = model.scheduler.add_noise(
            actions.flatten(0, 1).unsqueeze(-1).unsqueeze(-1),
            action_noise.flatten(0, 1).unsqueeze(-1).unsqueeze(-1),
            action_timestep.flatten(0, 1),
        ).squeeze(-1).squeeze(-1).unflatten(0, actions.shape[:2])
        action_target = model.scheduler.training_target(actions, action_noise, action_timestep)

    print(f"[Overfit] prompt={prompts[0]}")
    print(f"[Overfit] action_attend_video={cfg.action_dit_config.action_attend_video}")
    print(f"[Overfit] actions={tuple(actions.shape)} target_abs_mean={float(action_target.float().abs().mean()):.6f}")
    for step in range(1, int(args.steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        pred = model.action_dit(
            noisy_actions=noisy_actions,
            action_timestep=action_timestep_sample,
            video_latents=noisy_latents,
            video_timestep=video_timestep,
            conditional_dict=conditional_dict,
            prefix_x=first_frame_latent,
            prefix_t=prefix_t,
            prefix_token_ids=[CONDITION_TOKEN_ID],
            noisy_token_ids=list(range(runtime_vertical_info["num_tokens"])),
            vertical_info=runtime_vertical_info,
            vertical_use_representative_rope=model.vertical_use_representative_rope,
            seq_len_override=seq_len_override,
        )
        loss = torch.nn.functional.mse_loss(pred.float(), action_target.float())
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.action_dit.parameters(), 10.0)
        optimizer.step()
        if step <= 5 or step % 20 == 0:
            print(
                f"[Overfit] step={step} loss={float(loss.item()):.6f} "
                f"pred_abs_mean={float(pred.detach().float().abs().mean()):.6f} "
                f"grad_norm={float(grad_norm):.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
