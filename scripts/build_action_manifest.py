#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a LIBERO action-head demo manifest.")
    parser.add_argument("--data-root", default="/pfs-verdent/zhangyu/robot-trial/data/LIBERO-datasets")
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task", default=None, help="Optional exact task name without _demo.hdf5.")
    parser.add_argument("--max-demos", type=int, default=10)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_dir = Path(args.data_root) / args.suite
    if args.task:
        hdf5_files = [suite_dir / f"{args.task}_demo.hdf5"]
    else:
        hdf5_files = sorted(suite_dir.glob("*_demo.hdf5"))
    rows = []
    for hdf5_path in hdf5_files:
        if not hdf5_path.exists():
            raise FileNotFoundError(hdf5_path)
        task = hdf5_path.name.removesuffix("_demo.hdf5")
        with h5py.File(hdf5_path, "r") as f:
            demo_ids = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
            for demo_id in demo_ids[: args.max_demos]:
                demo = f[f"data/{demo_id}"]
                rows.append(
                    {
                        "suite": args.suite,
                        "task": task,
                        "hdf5_path": str(hdf5_path),
                        "demo_id": demo_id,
                        "num_steps": int(demo["actions"].shape[0]),
                    }
                )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[Manifest] wrote {len(rows)} demos to {output}")


if __name__ == "__main__":
    main()
