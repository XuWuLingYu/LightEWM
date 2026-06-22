#!/usr/bin/env python3
"""Batch online LIBERO eval for action-only HDR/FastWAM-local checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import imageio.v2 as imageio
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
    load_action_state,
    load_generator_state,
    parse_camera_keys,
    problem_name_from_h5,
    proprio_from_obs,
    resolve_backend_path,
    resolve_path,
    uint8_frames_to_tensor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-root", default="../HiDiT/Causal-Forcing")
    parser.add_argument(
        "--config-path",
        default="logs/LIBERO-HDR_train_video_action_joint_fastwam_local/20260617_090745/causal_forcing_config.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="logs/LIBERO-HDR_train_video_action_joint_fastwam_local/20260617_090745/checkpoint_model_030000/model.pt",
    )
    parser.add_argument("--metadata", default="data/libero_i2v_train/metadata_dense_prompt_hdr_video_action_joint.csv")
    parser.add_argument("--output-dir", default="logs/eval/libero_10_action_only_online_30k_step10_cfg1_40")
    parser.add_argument("--suites", nargs="+", default=["libero_10"])
    parser.add_argument("--tasks-per-suite", type=int, default=10)
    parser.add_argument("--episodes-per-task", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--joint-steps", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--closed-loop-execute-steps", type=int, default=8)
    parser.add_argument("--closed-loop-timeout-steps", type=int, default=200)
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--physical-agent-root", default="/mnt/zezhong/physical_agent")
    parser.add_argument("--libero-root", default="/mnt/zezhong/LightEWM/third_parties/LIBERO")
    parser.add_argument("--disable-proprio", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def suite_task_name(source_file: str) -> tuple[str, str]:
    path = Path(source_file)
    return path.parent.name, path.stem.removesuffix("_demo")


def read_eval_rows(
    metadata_path: Path,
    suites: list[str],
    tasks_per_suite: int,
    episodes_per_task: int,
) -> list[dict[str, Any]]:
    by_suite_task: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("camera_key") != "agentview_rgb":
                continue
            suite, task = suite_task_name(row["source_file"])
            if suite in suites:
                by_suite_task[suite][task].append(row)

    selected: list[dict[str, Any]] = []
    for suite in suites:
        tasks = sorted(by_suite_task[suite].keys())[:tasks_per_suite]
        for task in tasks:
            rows = sorted(by_suite_task[suite][task], key=lambda r: r.get("demo_id", ""))[:episodes_per_task]
            selected.extend(rows)
    return selected


def resolve_row(row: dict[str, Any], metadata_dir: Path, repo_root: Path) -> dict[str, Any]:
    row = dict(row)
    for key in ("video", "source_file", "action_stats_path", "proprio_stats_path"):
        value = row.get(key)
        if not value:
            continue
        path = Path(str(value))
        if path.is_absolute():
            row[key] = str(path)
        elif path.exists():
            row[key] = str(path.resolve())
        elif (repo_root / path).exists():
            row[key] = str((repo_root / path).resolve())
        else:
            row[key] = str((metadata_dir / path).resolve())
    return row


def bddl_for_row(row: dict[str, Any], libero_root: Path) -> Path:
    h5_path = Path(row["source_file"])
    suite = h5_path.parent.name
    problem_name = problem_name_from_h5(h5_path)
    bddl_file = libero_root / "libero" / "libero" / "bddl_files" / suite / f"{problem_name}.bddl"
    if not bddl_file.exists():
        raise FileNotFoundError(f"Missing BDDL file: {bddl_file}")
    return bddl_file


def load_init_state(row: dict[str, Any]) -> np.ndarray:
    with h5py.File(row["source_file"], "r") as f:
        return np.asarray(f["data"][row["demo_id"]].attrs["init_state"], dtype=np.float64)


def safe_name(value: str, max_len: int = 96) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:max_len] if len(cleaned) > max_len else cleaned


def save_rgb_video(path: Path, frames: list[np.ndarray] | np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(frames)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    imageio.mimsave(path, list(array), fps=int(fps))


def summarize(results: list[dict[str, Any]], checkpoint: Path, summary_path: Path) -> dict[str, Any]:
    by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in results:
        by_suite[record["suite"]].append(record)
        by_task[f"{record['suite']}/{record['task']}"].append(record)

    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "num_rollouts": len(results),
        "successes": int(sum(bool(r["success"]) for r in results)),
        "overall_success_rate": float(np.mean([r["success"] for r in results])) if results else 0.0,
        "mean_executed_steps": float(np.mean([r["executed_steps"] for r in results])) if results else 0.0,
        "mean_inference_seconds": float(np.mean([r["inference_seconds_mean"] for r in results])) if results else 0.0,
        "mean_ms_per_generated_action": (
            float(np.mean([r["inference_ms_per_generated_action"] for r in results])) if results else 0.0
        ),
        "by_suite": {},
        "by_task": {},
    }
    for suite, rows in sorted(by_suite.items()):
        summary["by_suite"][suite] = {
            "n": len(rows),
            "successes": int(sum(bool(r["success"]) for r in rows)),
            "success_rate": float(np.mean([r["success"] for r in rows])),
        }
    for task, rows in sorted(by_task.items()):
        summary["by_task"][task] = {
            "n": len(rows),
            "successes": int(sum(bool(r["success"]) for r in rows)),
            "success_rate": float(np.mean([r["success"] for r in rows])),
        }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards).")

    repo_root = Path.cwd()
    backend_root = resolve_path(args.backend_root, repo_root)
    sys.path.insert(0, str(backend_root))
    sys.path.insert(0, str(Path(args.physical_agent_root)))

    from libero_agent import LiberoActionError, LiberoAgentInterface
    from model.diffusion import CausalDiffusion

    torch.manual_seed(int(args.seed) + int(args.shard_index))
    np.random.seed(int(args.seed) + int(args.shard_index))
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
    cfg.guidance_scale = float(args.guidance_scale)
    if not hasattr(cfg, "independent_first_frame"):
        cfg.independent_first_frame = False
    rollout_camera_keys = parse_camera_keys(getattr(cfg, "joint_camera_key", "agentview_rgb"))

    metadata_path = resolve_path(args.metadata, repo_root)
    all_rows = [
        resolve_row(row, metadata_path.parent, repo_root)
        for row in read_eval_rows(
            metadata_path,
            suites=list(args.suites),
            tasks_per_suite=int(args.tasks_per_suite),
            episodes_per_task=int(args.episodes_per_task),
        )
    ]
    shard_rows = [row for i, row in enumerate(all_rows) if i % int(args.num_shards) == int(args.shard_index)]
    print(
        f"[BatchEval] shard={args.shard_index}/{args.num_shards} "
        f"selected={len(shard_rows)} total={len(all_rows)}",
        flush=True,
    )

    model = CausalDiffusion(cfg, device=device).to(device=device)
    model.eval()
    model.generator.load_state_dict(load_generator_state(checkpoint), strict=True)
    action_load = model.action_dit.load_state_dict(load_action_state(checkpoint), strict=False)
    missing_proprio = any(str(key).startswith("proprio_encoder.") for key in action_load.missing_keys)
    use_proprio = (
        not bool(args.disable_proprio)
        and getattr(model.action_dit, "proprio_encoder", None) is not None
        and not missing_proprio
    )
    if not use_proprio and getattr(model.action_dit, "proprio_encoder", None) is not None:
        model.action_dit.proprio_encoder = None
        model.action_dit.proprio_dim = None
    model.action_dit.bind_video_model(model.generator.model)
    model.text_encoder = CastingTextEncoder(model.text_encoder, device, model.dtype)
    for module in (model.generator, model.action_dit, model.text_encoder, model.vae):
        module.eval()
        module.requires_grad_(False)
    print(f"[BatchEval] proprio_conditioning={'on' if use_proprio else 'off'}", flush=True)

    stats_cache: dict[str, dict[str, Any]] = {}
    proprio_stats_cache: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    results_path = output_dir / f"results_shard_{args.shard_index:02d}.jsonl"
    results_path.write_text("", encoding="utf-8")

    for local_idx, row in enumerate(shard_rows):
        global_idx = int(args.shard_index) + local_idx * int(args.num_shards)
        suite, task = suite_task_name(row["source_file"])
        rollout_dir = output_dir / f"{global_idx:04d}_{suite}_{safe_name(task)}_{row['demo_id']}"
        rollout_dir.mkdir(parents=True, exist_ok=True)

        action_stats_path = row["action_stats_path"]
        if action_stats_path not in stats_cache:
            stats_cache[action_stats_path] = json.loads(Path(action_stats_path).read_text(encoding="utf-8"))
        action_stats = stats_cache[action_stats_path]

        proprio_stats = None
        proprio_stats_path = row.get("proprio_stats_path", getattr(cfg, "joint_proprio_stats_path", None))
        if proprio_stats_path:
            proprio_stats_path = str(resolve_path(proprio_stats_path, repo_root))
            if proprio_stats_path not in proprio_stats_cache:
                proprio_stats_cache[proprio_stats_path] = json.loads(Path(proprio_stats_path).read_text(encoding="utf-8"))
            proprio_stats = proprio_stats_cache[proprio_stats_path]

        prompt = [row.get("dense_prompt") or row.get("prompt") or row.get("sparse_prompt") or task]
        condition = model.text_encoder(prompt)
        latent_channels = int(cfg.image_or_video_shape[2])
        latent_h = int(cfg.image_or_video_shape[3])
        latent_w = int(cfg.image_or_video_shape[4])
        tree_latents = torch.zeros((1, 0, latent_channels, latent_h, latent_w), device=device, dtype=model.dtype)
        tree_timestep = torch.zeros((1, 0), device=device, dtype=model.dtype)
        tree_token_ids: list[int] = []
        prefix_x = tree_latents[:, :0]
        prefix_t = torch.zeros((1, 0), device=device, dtype=model.dtype)
        prefix_token_ids: list[int] = []
        action_shape = (1, int(cfg.joint_window_frames), 7)

        env = None
        sim_frames: list[np.ndarray] = []
        sim_actions_norm: list[np.ndarray] = []
        sim_actions_raw: list[np.ndarray] = []
        infer_seconds: list[float] = []
        executed = 0
        success = False
        success_step = None
        error = None
        rollout_start = time.perf_counter()
        try:
            env = LiberoAgentInterface(
                bddl_file=bddl_for_row(row, Path(args.libero_root)),
                camera_heights=int(args.camera_height),
                camera_widths=int(args.camera_width),
                action_repeat=1,
                libero_root=str(args.libero_root),
            )
            env.reset(init_state=load_init_state(row))
            sim_frames.append(frame_from_obs(env.last_obs))
            success = bool(env.state().get("success", False))
            success_step = 0 if success else None
            while executed < int(args.closed_loop_timeout_steps) and not success:
                obs_frame = fastwam_frame_from_obs(env.last_obs, rollout_camera_keys)
                obs_tensor = uint8_frames_to_tensor(
                    [obs_frame],
                    height=int(cfg.height),
                    width=int(cfg.width),
                ).unsqueeze(0).to(device=device)
                with torch.no_grad():
                    local_start = model.vae.encode_to_latent(obs_tensor.float()).to(dtype=model.dtype)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    infer_start = time.perf_counter()
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
                    infer_seconds.append(time.perf_counter() - infer_start)
                action_norm_np = actions_norm[0].float().cpu().numpy()
                action_raw_np = denormalize_actions(clip_normalized_actions(action_norm_np, cfg), action_stats)
                steps_this_chunk = min(
                    int(args.closed_loop_execute_steps),
                    action_raw_np.shape[0],
                    int(args.closed_loop_timeout_steps) - executed,
                )
                for step_id, action in enumerate(action_raw_np[:steps_this_chunk]):
                    result = env.step(action[:6], mode="ee_delta", gripper=float(np.clip(action[6], -1.0, 1.0)))
                    sim_frames.append(frame_from_obs(result.observation))
                    sim_actions_norm.append(action_norm_np[step_id])
                    sim_actions_raw.append(action.astype(np.float32))
                    executed += 1
                    success = bool(result.state_after.get("success", False))
                    if success:
                        success_step = executed
                        break
        except LiberoActionError as exc:
            error = exc.to_dict()
        except Exception as exc:  # keep the batch alive and record the failure.
            error = {"error": type(exc).__name__, "message": str(exc)}
        finally:
            if env is not None:
                env.close()

        sim_video_path = rollout_dir / "closed_loop_sim.mp4"
        actions_path = rollout_dir / "closed_loop_actions.npz"
        if sim_frames:
            save_rgb_video(sim_video_path, sim_frames, fps=int(args.fps))
        np.savez(
            actions_path,
            action_norm=np.asarray(sim_actions_norm, dtype=np.float32),
            action_raw=np.asarray(sim_actions_raw, dtype=np.float32),
        )
        record = {
            "global_index": global_idx,
            "suite": suite,
            "task": task,
            "demo_id": row["demo_id"],
            "prompt": prompt[0],
            "success": bool(success),
            "success_step": success_step,
            "executed_steps": int(executed),
            "rollout_seconds": float(time.perf_counter() - rollout_start),
            "inference_calls": len(infer_seconds),
            "inference_seconds_total": float(np.sum(infer_seconds)) if infer_seconds else 0.0,
            "inference_seconds_mean": float(np.mean(infer_seconds)) if infer_seconds else 0.0,
            "inference_ms_per_generated_action": (
                float(1000.0 * np.sum(infer_seconds) / (len(infer_seconds) * int(cfg.joint_window_frames)))
                if infer_seconds
                else 0.0
            ),
            "sim_video": str(sim_video_path),
            "actions": str(actions_path),
            "error": error,
        }
        results.append(record)
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"[BatchEval] shard={args.shard_index} {local_idx + 1}/{len(shard_rows)} "
            f"global={global_idx} {suite}/{task}/{row['demo_id']} "
            f"success={success} steps={executed} time={record['rollout_seconds']:.1f}s",
            flush=True,
        )

    summary = summarize(results, checkpoint, output_dir / f"summary_shard_{args.shard_index:02d}.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
