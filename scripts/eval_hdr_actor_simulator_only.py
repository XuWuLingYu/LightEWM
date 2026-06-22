#!/usr/bin/env python3
"""Run HDR actor closed-loop rollout from a LIBERO simulator reset only."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from eval_hdr_video_action_joint import (
    CastingTextEncoder,
    clip_normalized_actions,
    denoise_joint_action,
    denormalize_actions,
    fastwam_frame_from_obs,
    frame_from_obs,
    infer_tree_context,
    load_action_state,
    load_generator_state,
    make_autocast_ctx,
    parse_camera_keys,
    proprio_from_obs,
    resolve_backend_path,
    resolve_path,
    save_video_mp4,
    tensor_video_to_uint8,
    uint8_fastwam_frames_to_tensor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-root", default="lightewm/vendor/causal_forcing")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--benchmark", default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--joint-steps", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--tree-fixed-denoise-steps", type=int, default=5)
    parser.add_argument("--tree-sampling-steps", type=int, default=50)
    parser.add_argument("--tree-x0-level-index", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--execute-steps-per-chunk", type=int, default=32)
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--physical-agent-root", default="/mnt/zezhong/physical_agent")
    parser.add_argument("--libero-root", default="/mnt/zezhong/physical_agent/LIBERO")
    parser.add_argument("--disable-proprio", action="store_true")
    parser.add_argument("--skip-tree-video", action="store_true")
    parser.add_argument(
        "--save-every-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save every simulator step frame. Disable to save only chunk boundary frames.",
    )
    return parser.parse_args()


def resolve_config_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [(repo_root / path).resolve()]
    stripped = [part for part in path.parts if part not in ("..", ".")]
    if stripped:
        candidates.append((repo_root / Path(*stripped)).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def truncate_tree_context(cfg: Any, vertical_info: dict[str, Any], tree_latents: torch.Tensor, tree_timestep: torch.Tensor):
    tree_num_levels = int(getattr(cfg, "joint_tree_num_levels", 0) or 0)
    if tree_num_levels <= 0:
        tree_count = int(vertical_info["num_tokens"])
    else:
        tree_count = int(sum(list(vertical_info["level_sizes"])[:tree_num_levels]))
    return tree_latents[:, :tree_count], tree_timestep[:, :tree_count], list(range(tree_count))


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    backend_root = resolve_path(args.backend_root, repo_root)
    sys.path.insert(0, str(backend_root))
    sys.path.insert(0, str(Path(args.physical_agent_root)))

    from libero_agent import LiberoActionError, LiberoAgentInterface
    from model.diffusion import CausalDiffusion

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
    repo_model_root = repo_root / "checkpoints"
    if not (Path(str(cfg.model_kwargs.model_root)) / str(cfg.model_kwargs.model_name)).exists():
        if (repo_model_root / str(cfg.model_kwargs.model_name)).exists():
            cfg.model_kwargs.model_root = str(repo_model_root)
    if not hasattr(cfg, "independent_first_frame"):
        cfg.independent_first_frame = False
    cfg.sampling_steps = int(args.tree_sampling_steps)
    cfg.vertical_infer_fixed_denoise_steps = int(args.tree_fixed_denoise_steps)
    cfg.vertical_infer_preserve_budget_ratio = True
    cfg.vertical_infer_reference_total_steps = int(args.tree_sampling_steps)
    cfg.guidance_scale = float(args.guidance_scale)

    action_stats_path = resolve_config_path(getattr(cfg, "action_stats_path"), repo_root)
    action_stats = json.loads(action_stats_path.read_text(encoding="utf-8"))
    proprio_stats = None
    proprio_stats_value = getattr(cfg, "joint_proprio_stats_path", None)
    if proprio_stats_value:
        proprio_stats_path = resolve_config_path(proprio_stats_value, repo_root)
        proprio_stats = json.loads(proprio_stats_path.read_text(encoding="utf-8"))

    print("[SimOnlyEval] building model", flush=True)
    model = CausalDiffusion(cfg, device=device).to(device=device)
    model.eval()
    print(f"[SimOnlyEval] loading checkpoint {checkpoint}", flush=True)
    model.generator.load_state_dict(load_generator_state(checkpoint), strict=True)
    action_load = model.action_dit.load_state_dict(load_action_state(checkpoint), strict=False)
    missing_proprio = any(str(key).startswith("proprio_encoder.") for key in action_load.missing_keys)
    use_proprio = (
        not bool(args.disable_proprio)
        and getattr(model.action_dit, "proprio_encoder", None) is not None
        and not missing_proprio
        and proprio_stats is not None
    )
    if not use_proprio and getattr(model.action_dit, "proprio_encoder", None) is not None:
        model.action_dit.proprio_encoder = None
        model.action_dit.proprio_dim = None
    model.action_dit.bind_video_model(model.generator.model)
    model.text_encoder = CastingTextEncoder(model.text_encoder, device, model.dtype)
    for module in (model.generator, model.action_dit, model.text_encoder, model.vae):
        module.eval()
        module.requires_grad_(False)
    print(f"[SimOnlyEval] proprio_conditioning={'on' if use_proprio else 'off'}", flush=True)

    torch_load = torch.load

    def torch_load_libero_compat(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return torch_load(*load_args, **load_kwargs)

    torch.load = torch_load_libero_compat
    try:
        env = LiberoAgentInterface(
            benchmark_name=args.benchmark,
            task_id=int(args.task_id),
            init_state_id=int(args.init_state_id),
            camera_heights=int(args.camera_height),
            camera_widths=int(args.camera_width),
            action_repeat=1,
            libero_root=str(args.libero_root),
        )
    finally:
        torch.load = torch_load
    prompt_text = env.state().get("task", {}).get("language") or env.state().get("task", {}).get("name") or args.benchmark
    prompt = [str(prompt_text)]
    print(f"[SimOnlyEval] prompt={prompt[0]}", flush=True)
    condition = model.text_encoder(prompt)
    rollout_camera_keys = parse_camera_keys(getattr(cfg, "joint_camera_key", "agentview_rgb"))

    with torch.no_grad(), make_autocast_ctx(device, model.dtype):
        initial_frame = fastwam_frame_from_obs(env.last_obs, rollout_camera_keys)
        initial_tensor = uint8_fastwam_frames_to_tensor(
            [initial_frame],
            height=int(cfg.height),
            width=int(cfg.width),
            num_cameras=len(rollout_camera_keys),
        ).unsqueeze(0).to(device=device, dtype=model.dtype)
        initial_latent = model.vae.encode_to_latent(initial_tensor.float()).to(dtype=model.dtype)
        tree_start = time.perf_counter()
        tree_latents, tree_timestep, tree_outputs = infer_tree_context(
            cfg=cfg,
            model=model,
            device=device,
            dtype=model.dtype,
            initial_latent=initial_latent,
            prompt=prompt,
            output_dir=output_dir,
            args=args,
            save_video=not bool(args.skip_tree_video),
            gt_frames=None,
        )
        tree_latents, tree_timestep, tree_token_ids = truncate_tree_context(
            cfg, model.vertical_info, tree_latents, tree_timestep
        )
        tree_context_video = output_dir / "hierarchical_tree_context_tokens.mp4"
        save_video_mp4(
            tree_context_video,
            list(tensor_video_to_uint8(model.vae.decode_to_pixel(tree_latents.float()))),
            fps=int(args.fps),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tree_seconds = time.perf_counter() - tree_start

    frames = [frame_from_obs(env.last_obs)]
    actions_norm_records: list[np.ndarray] = []
    actions_raw_records: list[np.ndarray] = []
    inference_seconds: list[float] = []
    executed = 0
    success = bool(env.state().get("success", False))
    success_step = 0 if success else None
    error = None
    action_shape = (1, int(cfg.joint_window_frames), 7)
    prefix_x = initial_latent[:, :1]
    prefix_t = torch.zeros((1, 1), device=device, dtype=model.dtype)
    prefix_token_ids = [-1]

    try:
        while executed < int(args.max_steps) and not success:
            obs_frame = fastwam_frame_from_obs(env.last_obs, rollout_camera_keys)
            obs_tensor = uint8_fastwam_frames_to_tensor(
                [obs_frame],
                height=int(cfg.height),
                width=int(cfg.width),
                num_cameras=len(rollout_camera_keys),
            ).unsqueeze(0).to(device=device, dtype=model.dtype)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            infer_start = time.perf_counter()
            with torch.no_grad(), make_autocast_ctx(device, model.dtype):
                local_start = model.vae.encode_to_latent(obs_tensor.float()).to(dtype=model.dtype)
                actions_norm, _ = denoise_joint_action(
                    model=model,
                    cfg=cfg,
                    condition=condition,
                    tree_latents=tree_latents,
                    tree_timestep=tree_timestep,
                    tree_token_ids=tree_token_ids,
                    prefix_x=prefix_x,
                    prefix_t=prefix_t,
                    prefix_token_ids=prefix_token_ids,
                    local_start=local_start,
                    local_video_clean=None,
                    joint_proprio=(
                        torch.from_numpy(proprio_from_obs(env.last_obs, proprio_stats))
                        .unsqueeze(0)
                        .to(device=device, dtype=model.dtype)
                        if use_proprio
                        else None
                    ),
                    action_shape=action_shape,
                    args=argparse.Namespace(**vars(args), action_only=True),
                    device=device,
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds.append(time.perf_counter() - infer_start)

            actions_norm_np = actions_norm[0].float().cpu().numpy()
            actions_raw = denormalize_actions(clip_normalized_actions(actions_norm_np, cfg), action_stats)
            steps_this_chunk = min(
                int(args.execute_steps_per_chunk),
                actions_raw.shape[0],
                int(args.max_steps) - executed,
            )
            print(
                f"[SimOnlyEval] chunk={len(inference_seconds)-1} executing={steps_this_chunk} "
                f"executed={executed}/{int(args.max_steps)} success={success}",
                flush=True,
            )
            for step_id, action in enumerate(actions_raw[:steps_this_chunk]):
                try:
                    result = env.step(action[:6], mode="ee_delta", gripper=float(np.clip(action[6], -1.0, 1.0)))
                except LiberoActionError as exc:
                    error = exc.to_dict()
                    print(f"[SimOnlyEval] action error at global_step={executed}: {error}", flush=True)
                    raise
                if args.save_every_frame:
                    frames.append(frame_from_obs(result.observation))
                actions_norm_records.append(actions_norm_np[step_id].astype(np.float32))
                actions_raw_records.append(action.astype(np.float32))
                executed += 1
                success = bool(result.state_after.get("success", False))
                if success:
                    success_step = executed
                    print(f"[SimOnlyEval] success at step={executed}", flush=True)
                    break
            if not args.save_every_frame:
                frames.append(frame_from_obs(env.last_obs))
    except LiberoActionError:
        pass
    finally:
        env.close()

    rollout_path = output_dir / "closed_loop_simulator_max500.mp4"
    save_video_mp4(rollout_path, frames, fps=int(args.fps))
    actions_path = output_dir / "closed_loop_actions.npz"
    np.savez(
        actions_path,
        action_norm=np.asarray(actions_norm_records, dtype=np.float32),
        action_raw=np.asarray(actions_raw_records, dtype=np.float32),
    )
    summary = {
        "checkpoint": str(checkpoint),
        "config_path": str(config_path),
        "benchmark": str(args.benchmark),
        "task_id": int(args.task_id),
        "init_state_id": int(args.init_state_id),
        "prompt": prompt[0],
        "max_steps": int(args.max_steps),
        "executed_steps": int(executed),
        "success": bool(success),
        "success_step": success_step,
        "rollout_video": str(rollout_path),
        "actions": str(actions_path),
        "tree_context_video": str(tree_context_video),
        "tree_seconds": float(tree_seconds),
        "tree_token_count": int(len(tree_token_ids)),
        "tree_timestep_min": float(tree_timestep.detach().float().min().cpu()) if tree_timestep.numel() else 0.0,
        "tree_timestep_max": float(tree_timestep.detach().float().max().cpu()) if tree_timestep.numel() else 0.0,
        "tree_outputs": tree_outputs,
        "inference_calls": len(inference_seconds),
        "inference_seconds_total": float(np.sum(inference_seconds)) if inference_seconds else 0.0,
        "inference_seconds_mean": float(np.mean(inference_seconds)) if inference_seconds else None,
        "inference_ms_per_generated_action": (
            float(1000.0 * np.sum(inference_seconds) / max(1, len(inference_seconds) * int(cfg.joint_window_frames)))
            if inference_seconds
            else None
        ),
        "error": error,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[SimOnlyEval] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
