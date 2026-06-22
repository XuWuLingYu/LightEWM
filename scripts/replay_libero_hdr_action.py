#!/usr/bin/env python3
"""Replay one prepared HDR action sequence in LIBERO and save agentview video."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-path", default="data/libero_i2v_train/metadata_dense_prompt_hdr_actions_leaf8.csv")
    parser.add_argument("--stats-path", default="data/libero_i2v_train/hdr_actions_leaf8_stats.json")
    parser.add_argument("--output", default="logs/debug/libero_hdr_action_replay.mp4")
    parser.add_argument("--side-by-side-output", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--row-index", type=int, default=None)
    parser.add_argument("--replay-seed", type=int, default=None)
    parser.add_argument("--replay-row-index", type=int, default=None)
    parser.add_argument(
        "--replay-different-scene",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Execute the source action sequence in a different LIBERO scene/init state when possible.",
    )
    parser.add_argument("--action-mode", choices=["ee_delta", "ee_abs"], default="ee_delta")
    parser.add_argument(
        "--delta-to-controller",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Legacy option for metric EE deltas. Prepared HDR actions are already LIBERO controller actions.",
    )
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=104)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--libero-root", default="/mnt/zezhong/LightEWM/third_parties/LIBERO")
    parser.add_argument("--libero-data-root", default="data/LIBERO-datasets")
    parser.add_argument("--physical-agent-root", default="/mnt/zezhong/physical_agent")
    return parser.parse_args()


def load_rows(metadata_path: Path) -> list[dict[str, str]]:
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in metadata: {metadata_path}")
    return rows


def load_row(metadata_path: Path, seed: int, row_index: int | None) -> dict[str, str]:
    rows = load_rows(metadata_path)
    if row_index is not None:
        return rows[int(row_index)]
    return random.Random(seed).choice(rows)


def select_replay_row(
    metadata_path: Path,
    action_row: dict[str, str],
    *,
    seed: int,
    row_index: int | None,
    different_scene: bool,
) -> dict[str, str]:
    rows = load_rows(metadata_path)
    if row_index is not None:
        return rows[int(row_index)]
    if not different_scene:
        return action_row

    source_key = (action_row.get("source_file"), action_row.get("demo_id"))
    candidates = [
        row for row in rows
        if (row.get("source_file"), row.get("demo_id")) != source_key
    ]
    if not candidates:
        return action_row
    return random.Random(seed).choice(candidates)


def find_agentview_video(row: dict[str, str], metadata_path: Path) -> str:
    if row.get("camera_key") == "agentview_rgb" or "agentview" in row.get("video", ""):
        return row["video"]
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        for candidate in csv.DictReader(f):
            if candidate.get("source_file") != row.get("source_file"):
                continue
            if candidate.get("demo_id") != row.get("demo_id"):
                continue
            if candidate.get("camera_key") == "agentview_rgb" or "agentview" in candidate.get("video", ""):
                return candidate["video"]
    return row["video"]


def denormalize(action_norm: np.ndarray, stats: dict) -> np.ndarray:
    min_v = np.asarray(stats["min"], dtype=np.float32)
    max_v = np.asarray(stats["max"], dtype=np.float32)
    eps = float(stats.get("eps", 1e-6))
    return (action_norm + 1.0) * 0.5 * (max_v - min_v + eps) + min_v


def normalize(action: np.ndarray, stats: dict) -> np.ndarray:
    min_v = np.asarray(stats["min"], dtype=np.float32)
    max_v = np.asarray(stats["max"], dtype=np.float32)
    eps = float(stats.get("eps", 1e-6))
    return 2.0 * (action - min_v) / (max_v - min_v + eps) - 1.0


def resolve_h5_path(source_file: str, libero_data_root: Path) -> Path:
    source = Path(source_file)
    matches = list(libero_data_root.rglob(source.name))
    if matches:
        return matches[0]
    if source.exists():
        return source
    raise FileNotFoundError(f"Cannot resolve hdf5 source: {source_file}")


def problem_name_from_h5(h5_path: Path) -> str:
    stem = h5_path.stem
    if stem.endswith("_demo"):
        return stem[: -len("_demo")]
    return stem


def gripper_to_command(value: float) -> float:
    return float(np.clip(float(value), -1.0, 1.0))


def frame_from_obs(obs: dict) -> np.ndarray:
    for key in ("agentview_image", "agentview_rgb", "robot0_agentview_image"):
        if key in obs:
            frame = obs[key]
            break
    else:
        image_keys = [k for k, v in obs.items() if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] == 3]
        if not image_keys:
            raise KeyError(f"No RGB observation key found. Available keys: {sorted(obs.keys())}")
        frame = obs[image_keys[0]]
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame[::-1, ::-1].copy()


def resize_frame(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    import cv2

    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def read_video_frames(path: Path, count: int, height: int, width: int) -> list[np.ndarray]:
    reader = imageio.get_reader(path)
    try:
        frames = [np.asarray(frame) for frame in reader]
    finally:
        reader.close()
    if not frames:
        raise RuntimeError(f"No frames found in video: {path}")
    source_idx = np.linspace(0, len(frames) - 1, count)
    selected = []
    for idx in source_idx:
        frame = frames[int(round(idx))]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        selected.append(resize_frame(frame, height, width))
    return selected


def write_side_by_side(
    *,
    dataset_video_path: Path,
    replay_frames: list[np.ndarray],
    output_path: Path,
    fps: int,
) -> None:
    height, width = replay_frames[0].shape[:2]
    dataset_frames = read_video_frames(dataset_video_path, len(replay_frames), height, width)
    paired = [
        np.concatenate([dataset_frame, replay_frame], axis=1)
        for dataset_frame, replay_frame in zip(dataset_frames, replay_frames)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, paired, fps=fps)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.physical_agent_root)))
    import torch

    torch_load = torch.load

    def torch_load_compat(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return torch_load(*load_args, **load_kwargs)

    torch.load = torch_load_compat
    from libero_agent import LiberoAgentInterface, LiberoActionError

    metadata_path = Path(args.metadata_path)
    row = load_row(metadata_path, args.seed, args.row_index)
    replay_row = select_replay_row(
        metadata_path,
        row,
        seed=args.replay_seed if args.replay_seed is not None else args.seed + 1,
        row_index=args.replay_row_index,
        different_scene=args.replay_different_scene,
    )
    stats = json.loads(Path(args.stats_path).read_text(encoding="utf-8"))
    action_npz = np.load(row["action_path"])
    action = action_npz["action"].reshape(-1, 7).astype(np.float32)

    # Verify the min/max round trip that training/inference will use.
    action_norm = normalize(action, stats).astype(np.float32)
    action = denormalize(action_norm, stats).astype(np.float32)

    h5_path = resolve_h5_path(replay_row["source_file"], Path(args.libero_data_root))
    problem_name = problem_name_from_h5(h5_path)
    suite = h5_path.parent.name
    bddl_file = Path(args.libero_root) / "libero" / "libero" / "bddl_files" / suite / f"{problem_name}.bddl"
    if not bddl_file.exists():
        raise FileNotFoundError(f"Cannot find BDDL for replay: {bddl_file}")

    import h5py

    with h5py.File(h5_path, "r") as f:
        init_state = np.asarray(f["data"][replay_row["demo_id"]].attrs["init_state"], dtype=np.float64)

    env = LiberoAgentInterface(
        bddl_file=bddl_file,
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
        action_repeat=args.action_repeat,
        libero_root=args.libero_root,
    )
    env.reset(init_state=init_state)
    frames = [frame_from_obs(env.last_obs)]
    eef_positions = [np.asarray(env.state()["eef_pos"], dtype=np.float32)]
    executed = 0
    try:
        for step_id, act in enumerate(action[: args.max_steps]):
            arm_action = act[:6]
            if args.action_mode == "ee_delta" and args.delta_to_controller:
                output_max = np.asarray([0.05, 0.05, 0.05, 0.5, 0.5, 0.5], dtype=np.float32)
                arm_action = np.clip(arm_action.astype(np.float32) / output_max, -1.0, 1.0)
            try:
                result = env.step(
                    arm_action,
                    mode=args.action_mode,
                    gripper=gripper_to_command(float(act[6])),
                )
            except LiberoActionError as exc:
                print(f"[Replay] stop at step={step_id}: {exc.to_dict()}")
                break
            frames.append(frame_from_obs(result.observation))
            eef_positions.append(np.asarray(result.state_after["eef_pos"], dtype=np.float32))
            executed += 1
    finally:
        env.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, fps=args.fps)
    side_by_side_output = (
        Path(args.side_by_side_output)
        if args.side_by_side_output
        else output.with_name(f"{output.stem}_side_by_side{output.suffix}")
    )
    agentview_video = find_agentview_video(row, metadata_path)
    dataset_video_path = metadata_path.parent / agentview_video
    if not dataset_video_path.exists():
        dataset_video_path = Path(agentview_video)
    if dataset_video_path.exists():
        write_side_by_side(
            dataset_video_path=dataset_video_path,
            replay_frames=frames,
            output_path=side_by_side_output,
            fps=args.fps,
        )
    else:
        side_by_side_output = None

    print(f"[Replay] metadata_row_video={row.get('video')}")
    print(f"[Replay] side_by_side_dataset_video={agentview_video}")
    print(f"[Replay] action_source_h5={resolve_h5_path(row['source_file'], Path(args.libero_data_root))} demo_id={row['demo_id']}")
    print(f"[Replay] replay_h5={h5_path} demo_id={replay_row['demo_id']} bddl={bddl_file}")
    print(
        f"[Replay] action_mode={args.action_mode} "
        f"delta_to_controller={args.delta_to_controller} "
        f"action_repeat={args.action_repeat} "
        "gripper=raw_libero_dim6_clipped"
    )
    print(f"[Replay] action_path={row['action_path']}")
    print(f"[Replay] action_shape={action_npz['action'].shape} flattened={action.shape}")
    print(f"[Replay] action_norm_range=({float(action_norm.min()):.4f}, {float(action_norm.max()):.4f})")
    eef_positions_arr = np.stack(eef_positions, axis=0)
    eef_motion = np.linalg.norm(eef_positions_arr[-1] - eef_positions_arr[0])
    eef_step_motion = np.linalg.norm(np.diff(eef_positions_arr, axis=0), axis=1)
    print(
        f"[Replay] eef_motion_total={float(eef_motion):.4f} "
        f"eef_step_motion_max={float(eef_step_motion.max(initial=0.0)):.4f}"
    )
    print(f"[Replay] executed_steps={executed}")
    print(f"[Replay] wrote={output}")
    if side_by_side_output is not None:
        print(f"[Replay] wrote_side_by_side={side_by_side_output}")
    else:
        print("[Replay] side_by_side_skipped=dataset_video_missing")


if __name__ == "__main__":
    main()
