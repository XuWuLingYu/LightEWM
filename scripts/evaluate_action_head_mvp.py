#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import imageio.v3 as iio
import h5py
import numpy as np
import torch

from lightewm.action_head import ActionHeadMVP, load_action_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Action Head MVP in LIBERO simulator.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suite", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--num-cases", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--save-videos", type=int, default=3)
    parser.add_argument("--init-source", choices=("official", "hdf5"), default="official")
    parser.add_argument("--data-root", default="/pfs-verdent/zhangyu/robot-trial/data/LIBERO-datasets")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def env_proprio(obs: dict) -> np.ndarray:
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
    return np.concatenate([gripper, eef_pos, eef_quat], axis=0)


def model_action(model: ActionHeadMVP, obs: dict, task_id: int, device: str) -> np.ndarray:
    image = np.asarray(obs["agentview_image"], dtype=np.uint8)
    image_t = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    proprio_t = torch.from_numpy(env_proprio(obs)).float().unsqueeze(0).to(device)
    task_t = torch.tensor([task_id], dtype=torch.long, device=device)
    with torch.no_grad():
        action = model(image_t, proprio_t, task_t).squeeze(0).cpu().numpy()
    return np.clip(action, -1.0, 1.0)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    specs = load_action_manifest(ckpt["manifest"])
    suite = args.suite or specs[0].suite
    task_name = args.task or specs[0].task
    task_to_id = ckpt["task_to_id"]
    if task_name not in task_to_id:
        raise KeyError(f"Task {task_name!r} not in checkpoint task vocab: {sorted(task_to_id)}")

    model = ActionHeadMVP(
        num_tasks=len(task_to_id),
        proprio_dim=int(ckpt["proprio_dim"]),
        action_dim=int(ckpt["action_dim"]),
    ).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    bench_cls = benchmark.get_benchmark(suite)
    bench = bench_cls()
    task_names = bench.get_task_names()
    if task_name not in task_names:
        raise KeyError(f"Task {task_name!r} not found in {suite}; available={task_names}")
    task_idx = task_names.index(task_name)
    task = bench.get_task(task_idx)
    init_states = bench.get_task_init_states(task_idx)
    hdf5_states = None
    if args.init_source == "hdf5":
        hdf5_path = Path(args.data_root) / suite / f"{task_name}_demo.hdf5"
        with h5py.File(hdf5_path, "r") as f:
            demo_ids = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
            hdf5_states = [np.asarray(f[f"data/{demo_id}/states"][0]) for demo_id in demo_ids]
    env = OffScreenRenderEnv(
        bddl_file_name=bench.get_task_bddl_file_path(task_idx),
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
    )
    env.seed(args.seed)

    output_dir = Path(args.output_dir)
    video_dir = output_dir / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    task_id = int(task_to_id[task_name])
    try:
        for case_id in range(args.num_cases):
            env.reset()
            if hdf5_states is not None:
                init_state = hdf5_states[case_id % len(hdf5_states)]
            else:
                init_state = init_states[case_id % len(init_states)]
            obs = env.set_init_state(init_state)
            for _ in range(args.warmup_steps):
                obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
            done = False
            total_reward = 0.0
            frames = []
            for step in range(args.max_steps):
                if case_id < args.save_videos:
                    frames.append(np.asarray(obs["agentview_image"], dtype=np.uint8))
                action = model_action(model, obs, task_id, args.device)
                obs, reward, step_done, info = env.step(action)
                total_reward += float(reward)
                done = bool(done or step_done)
                if done:
                    break
            if case_id < args.save_videos:
                frames.append(np.asarray(obs["agentview_image"], dtype=np.uint8))
                iio.imwrite(video_dir / f"case_{case_id:03d}.mp4", np.stack(frames), fps=20)
            rows.append(
                {
                    "case_id": case_id,
                    "suite": suite,
                    "task": task_name,
                    "language": task.language,
                    "success": int(done),
                    "steps": step + 1,
                    "total_reward": total_reward,
                    "init_source": args.init_source,
                }
            )
            print(f"[Eval] case={case_id} success={int(done)} steps={step + 1}", flush=True)
    finally:
        env.close()

    results_csv = output_dir / "rollout_results.csv"
    with results_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    success_rate = sum(row["success"] for row in rows) / max(1, len(rows))
    summary = {
        "suite": suite,
        "task": task_name,
        "num_cases": len(rows),
        "success_rate": success_rate,
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "init_source": args.init_source,
        "results_csv": str(results_csv),
        "video_dir": str(video_dir),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Eval] success_rate={success_rate:.4f} num_cases={len(rows)}")
    print(f"[Eval] summary={output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
