#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay LIBERO expert hdf5 actions in simulator.")
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task", required=True)
    parser.add_argument("--data-root", default="/pfs-verdent/zhangyu/robot-trial/data/LIBERO-datasets")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-demos", type=int, default=50)
    parser.add_argument("--init-source", choices=("official", "hdf5"), default="official")
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    bench_cls = benchmark.get_benchmark(args.suite)
    bench = bench_cls()
    task_names = bench.get_task_names()
    if args.task not in task_names:
        raise KeyError(f"Task {args.task!r} not found in {args.suite}")
    task_idx = task_names.index(args.task)
    official_init_states = bench.get_task_init_states(task_idx)
    hdf5_path = Path(args.data_root) / args.suite / f"{args.task}_demo.hdf5"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = OffScreenRenderEnv(
        bddl_file_name=bench.get_task_bddl_file_path(task_idx),
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
    )
    rows = []
    try:
        with h5py.File(hdf5_path, "r") as f:
            demo_ids = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))[: args.max_demos]
            for demo_id in demo_ids:
                demo_idx = int(demo_id.split("_")[-1])
                demo = f[f"data/{demo_id}"]
                actions = np.asarray(demo["actions"], dtype=np.float32)
                hdf5_states = np.asarray(demo["states"])
                if args.init_source == "official":
                    init_state = official_init_states[demo_idx % len(official_init_states)]
                else:
                    init_state = hdf5_states[0]
                env.reset()
                env.set_init_state(init_state)
                success = False
                step_count = 0
                for step_count, action in enumerate(actions, start=1):
                    _, _, done, _ = env.step(action)
                    if done:
                        success = True
                        break
                final_success = bool(env.check_success())
                rows.append(
                    {
                        "demo_id": demo_id,
                        "success": int(success or final_success),
                        "done_success": int(success),
                        "final_success": int(final_success),
                        "steps": step_count,
                        "num_actions": len(actions),
                    }
                )
                print(f"[ExpertReplay] {demo_id} success={rows[-1]['success']} steps={step_count}", flush=True)
    finally:
        env.close()

    csv_path = output_dir / "expert_replay_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    success_rate = sum(row["success"] for row in rows) / max(1, len(rows))
    summary = {
        "suite": args.suite,
        "task": args.task,
        "init_source": args.init_source,
        "num_demos": len(rows),
        "success_rate": success_rate,
        "results_csv": str(csv_path),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[ExpertReplay] success_rate={success_rate:.4f} num_demos={len(rows)}")
    print(f"[ExpertReplay] summary={output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
