#!/usr/bin/env python3
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
import numpy as np
import torch
from omegaconf import OmegaConf

from eval_hdr_video_action_joint import (
    CastingTextEncoder,
    denoise_joint_action,
    denormalize_actions,
    fastwam_frame_from_obs,
    frame_from_obs,
    infer_tree_context,
    load_action_state,
    load_generator_state,
    make_autocast_ctx,
    parse_camera_keys,
    problem_name_from_h5,
    proprio_from_obs,
    resolve_backend_path,
    resolve_path,
    save_video_mp4,
    tensor_video_to_uint8,
    uint8_fastwam_frames_to_tensor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-root", default="../HiDiT/Causal-Forcing")
    parser.add_argument("--config-path", default="logs/LIBERO-HDR_train_video_action_joint/20260617_082026/causal_forcing_config.yaml")
    parser.add_argument("--checkpoint", default="logs/LIBERO-HDR_train_video_action_joint/20260617_082026/checkpoint_model_030000/model.pt")
    parser.add_argument("--metadata", default="data/libero_i2v_train/metadata_dense_prompt_hdr_video_action_joint.csv")
    parser.add_argument("--output-dir", default="logs/eval/libero_hdr_video_action_joint_30k_suites")
    parser.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    parser.add_argument("--tasks-per-suite", type=int, default=10)
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--joint-steps", type=int, default=20)
    parser.add_argument("--tree-fixed-denoise-steps", type=int, default=5)
    parser.add_argument("--tree-sampling-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--closed-loop-execute-steps", type=int, default=52)
    parser.add_argument("--closed-loop-timeout-steps", type=int, default=600)
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--physical-agent-root", default="/mnt/zezhong/physical_agent")
    parser.add_argument("--libero-root", default="/mnt/zezhong/LightEWM/third_parties/LIBERO")
    parser.add_argument("--libero-data-root", default="data/LIBERO-datasets")
    parser.add_argument("--disable-proprio", action="store_true")
    parser.add_argument("--save-tree-video", action="store_true")
    return parser.parse_args()


def suite_task_name(source_file: str) -> tuple[str, str]:
    path = Path(source_file)
    return path.parent.name, path.stem.removesuffix("_demo")


def read_eval_rows(metadata_path: Path, suites: list[str], tasks_per_suite: int, episodes_per_task: int) -> list[dict[str, Any]]:
    by_suite_task: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("camera_key") != "agentview_rgb":
                continue
            suite, task = suite_task_name(row["source_file"])
            if suite in suites:
                by_suite_task[suite][task].append(row)

    selected = []
    for suite in suites:
        tasks = sorted(by_suite_task[suite].keys())[:tasks_per_suite]
        for task in tasks:
            rows = sorted(by_suite_task[suite][task], key=lambda r: r.get("demo_id", ""))[:episodes_per_task]
            selected.extend(rows)
    return selected


def resolve_row(row: dict[str, Any], metadata_dir: Path) -> dict[str, Any]:
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


def _hdf5_obs_frame(obs_group, camera_keys: list[str]) -> np.ndarray:
    key_aliases = {
        "agentview_rgb": ("agentview_rgb", "agentview_image"),
        "eye_in_hand_rgb": ("eye_in_hand_rgb", "robot0_eye_in_hand_rgb", "robot0_eye_in_hand_image"),
    }
    frames = []
    for camera_key in camera_keys:
        aliases = key_aliases.get(str(camera_key), (str(camera_key),))
        for alias in aliases:
            if alias in obs_group:
                frame = np.asarray(obs_group[alias][0])
                break
        else:
            raise KeyError(f"No camera `{camera_key}` in HDF5 obs keys={list(obs_group.keys())}")
        frames.append(frame[::-1, ::-1, :].copy())
    return np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0]


def first_frame_tensor(row: dict[str, Any], cfg, device: torch.device, camera_keys: list[str]) -> torch.Tensor:
    with h5py.File(row["source_file"], "r") as f:
        demo = f["data"][row["demo_id"]]
        frame = _hdf5_obs_frame(demo["obs"], camera_keys)
    return uint8_fastwam_frames_to_tensor(
        [frame],
        height=int(cfg.height),
        width=int(cfg.width),
        num_cameras=len(camera_keys),
    ).unsqueeze(0).to(device=device)


def load_init_state(row: dict[str, Any]) -> np.ndarray:
    with h5py.File(row["source_file"], "r") as f:
        return np.asarray(f["data"][row["demo_id"]].attrs["init_state"], dtype=np.float64)


def safe_name(value: str, max_len: int = 96) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:max_len] if len(cleaned) > max_len else cleaned


def truncate_tree_context(cfg, model, tree_latents: torch.Tensor, tree_timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    tree_num_levels = int(getattr(cfg, "joint_tree_num_levels", 0) or 0)
    total_tokens = int(model.vertical_info["num_tokens"])
    if tree_num_levels <= 0:
        tree_count = total_tokens
    else:
        level_sizes = list(model.vertical_info["level_sizes"])
        if tree_num_levels > len(level_sizes):
            raise ValueError(
                f"joint_tree_num_levels={tree_num_levels} exceeds hierarchy levels={level_sizes}."
            )
        tree_count = int(sum(level_sizes[:tree_num_levels]))
    tree_latents = tree_latents[:, :tree_count]
    tree_timestep = tree_timestep[:, :tree_count]
    return tree_latents, tree_timestep, list(range(tree_count))


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
    model_root_path = Path(str(cfg.model_kwargs.model_root))
    if not (model_root_path / str(cfg.model_kwargs.model_name)).exists():
        repo_model_root = repo_root / "checkpoints"
        if (repo_model_root / str(cfg.model_kwargs.model_name)).exists():
            cfg.model_kwargs.model_root = str(repo_model_root)
    if not hasattr(cfg, "independent_first_frame"):
        cfg.independent_first_frame = False
    cfg.sampling_steps = int(args.tree_sampling_steps)
    cfg.vertical_infer_fixed_denoise_steps = int(args.tree_fixed_denoise_steps)
    cfg.vertical_infer_preserve_budget_ratio = True
    cfg.vertical_infer_reference_total_steps = int(args.tree_sampling_steps)
    if args.guidance_scale is not None:
        cfg.guidance_scale = float(args.guidance_scale)
    rollout_camera_keys = parse_camera_keys(getattr(cfg, "joint_camera_key", "agentview_rgb"))

    metadata_path = resolve_path(args.metadata, repo_root)
    rows = [resolve_row(row, metadata_path.parent) for row in read_eval_rows(
        metadata_path,
        suites=list(args.suites),
        tasks_per_suite=int(args.tasks_per_suite),
        episodes_per_task=int(args.episodes_per_task),
    )]
    if not rows:
        raise RuntimeError("No rows selected for eval.")
    print(f"[SuiteEval] selected {len(rows)} rollouts", flush=True)

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
    print(f"[SuiteEval] proprio_conditioning={'on' if use_proprio else 'off'}", flush=True)
    model.action_dit.bind_video_model(model.generator.model)
    model.text_encoder = CastingTextEncoder(model.text_encoder, device, model.dtype)
    for module in (model.generator, model.action_dit, model.text_encoder, model.vae):
        module.eval()
        module.requires_grad_(False)

    results = []
    stats_cache: dict[str, dict[str, Any]] = {}
    proprio_stats_cache: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        suite, task = suite_task_name(row["source_file"])
        rollout_dir = output_dir / f"{idx:04d}_{suite}_{safe_name(task)}_{row['demo_id']}"
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
        with torch.no_grad():
            initial_frame = first_frame_tensor(row, cfg, device, rollout_camera_keys).to(dtype=model.dtype)
            initial_latent = model.vae.encode_to_latent(initial_frame.float()).to(dtype=model.dtype)
            t0 = time.perf_counter()
            tree_latents, tree_timestep, _ = infer_tree_context(
                cfg=cfg,
                model=model,
                device=device,
                dtype=model.dtype,
                initial_latent=initial_latent,
                prompt=prompt,
                output_dir=output_dir,
                args=argparse.Namespace(
                    tree_fixed_denoise_steps=args.tree_fixed_denoise_steps,
                    tree_sampling_steps=args.tree_sampling_steps,
                    fps=16,
                ),
                save_video=bool(args.save_tree_video),
                gt_frames=None,
            )
            tree_latents, tree_timestep, tree_token_ids = truncate_tree_context(
                cfg,
                model,
                tree_latents,
                tree_timestep,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            pre_hier_time = time.perf_counter() - t0
            tree_video = model.vae.decode_to_pixel(tree_latents.float())
            tree_video_path = rollout_dir / "hierarchical_tree_context_decode.mp4"
            save_video_mp4(tree_video_path, list(tensor_video_to_uint8(tree_video)), fps=int(args.fps))

        bddl_file = bddl_for_row(row, Path(args.libero_root))
        env = LiberoAgentInterface(
            bddl_file=bddl_file,
            camera_heights=int(args.camera_height),
            camera_widths=int(args.camera_width),
            action_repeat=1,
            libero_root=str(args.libero_root),
        )
        env.reset(init_state=load_init_state(row))
        sim_frames = [frame_from_obs(env.last_obs)]
        executed = 0
        success = bool(env.state().get("success", False))
        success_step = 0 if success else None
        timeout_steps = int(args.closed_loop_timeout_steps)
        action_shape = (1, int(cfg.joint_window_frames), 7)
        prefix_x = initial_latent[:, :1]
        prefix_t = torch.zeros((1, 1), device=device, dtype=model.dtype)
        prefix_token_ids = [-1]
        closed_t0 = time.perf_counter()
        try:
            while executed < timeout_steps and not success:
                obs_frame = fastwam_frame_from_obs(env.last_obs, rollout_camera_keys)
                obs_tensor = uint8_fastwam_frames_to_tensor(
                    [obs_frame],
                    height=int(cfg.height),
                    width=int(cfg.width),
                    num_cameras=len(rollout_camera_keys),
                ).unsqueeze(0).to(device=device)
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
                        args=argparse.Namespace(**vars(args), action_only=False),
                        device=device,
                    )
                actions_raw = denormalize_actions(actions_norm[0].float().cpu().numpy().clip(
                    -float(getattr(cfg, "joint_norm_clip", 1.0)),
                    float(getattr(cfg, "joint_norm_clip", 1.0)),
                ), action_stats)
                max_steps = min(int(args.closed_loop_execute_steps), actions_raw.shape[0], timeout_steps - executed)
                for step_id, action in enumerate(actions_raw[:max_steps]):
                    result = env.step(action[:6], mode="ee_delta", gripper=float(np.clip(action[6], -1.0, 1.0)))
                    sim_frames.append(frame_from_obs(result.observation))
                    executed += 1
                    success = bool(result.state_after.get("success", False))
                    if success:
                        success_step = executed
                        break
        except LiberoActionError as exc:
            error = exc.to_dict()
        else:
            error = None
        finally:
            env.close()
        closed_loop_time = time.perf_counter() - closed_t0
        sim_video_path = rollout_dir / "closed_loop_sim.mp4"
        save_video_mp4(sim_video_path, sim_frames, fps=int(args.fps))
        record = {
            "index": idx,
            "suite": suite,
            "task": task,
            "demo_id": row["demo_id"],
            "success": bool(success),
            "success_step": success_step,
            "executed_steps": int(executed),
            "pre_hierarchical_time_sec": float(pre_hier_time),
            "closed_loop_time_sec": float(closed_loop_time),
            "tree_context_video": str(tree_video_path),
            "tree_token_count": int(len(tree_token_ids)),
            "tree_timestep_min": float(tree_timestep.detach().float().min().cpu()) if tree_timestep.numel() else 0.0,
            "tree_timestep_max": float(tree_timestep.detach().float().max().cpu()) if tree_timestep.numel() else 0.0,
            "sim_video": str(sim_video_path),
            "error": error,
        }
        results.append(record)
        print(
            f"[SuiteEval] {idx + 1}/{len(rows)} {suite}/{task}/{row['demo_id']} "
            f"success={success} pre={pre_hier_time:.2f}s closed={closed_loop_time:.2f}s steps={executed}",
            flush=True,
        )
        with (output_dir / "results.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    by_suite_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in results:
        by_suite_task[(record["suite"], record["task"])].append(record)
        by_suite[record["suite"]].append(record)

    summary = {
        "checkpoint": str(checkpoint),
        "num_rollouts": len(results),
        "overall_success_rate": float(np.mean([r["success"] for r in results])) if results else 0.0,
        "overall_pre_hierarchical_time_mean_sec": float(np.mean([r["pre_hierarchical_time_sec"] for r in results])) if results else 0.0,
        "overall_closed_loop_time_mean_sec": float(np.mean([r["closed_loop_time_sec"] for r in results])) if results else 0.0,
        "by_suite": {},
        "by_task": {},
    }
    for suite, suite_records in sorted(by_suite.items()):
        summary["by_suite"][suite] = {
            "n": len(suite_records),
            "success_rate": float(np.mean([r["success"] for r in suite_records])),
            "pre_hierarchical_time_mean_sec": float(np.mean([r["pre_hierarchical_time_sec"] for r in suite_records])),
            "closed_loop_time_mean_sec": float(np.mean([r["closed_loop_time_sec"] for r in suite_records])),
        }
    for (suite, task), task_records in sorted(by_suite_task.items()):
        summary["by_task"][f"{suite}/{task}"] = {
            "n": len(task_records),
            "success_rate": float(np.mean([r["success"] for r in task_records])),
            "pre_hierarchical_time_mean_sec": float(np.mean([r["pre_hierarchical_time_sec"] for r in task_records])),
            "closed_loop_time_mean_sec": float(np.mean([r["closed_loop_time_sec"] for r in task_records])),
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
