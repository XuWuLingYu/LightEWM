#!/usr/bin/env python3
"""Fine-grained LIBERO HDR action sanity check.

The script compares a 104-step rollout reconstructed from prepared action
deltas against the 13 decoded leaf latents used by HDR action training.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import av
import h5py
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CAUSAL_FORCING_ROOT = (REPO_ROOT / "../HiDiT/Causal-Forcing").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-path", default="data/libero_i2v_train/metadata_dense_prompt_hdr_actions_leaf8_agentview_video_cache.csv")
    parser.add_argument("--stats-path", default="data/libero_i2v_train/hdr_actions_leaf8_stats.json")
    parser.add_argument("--row-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replay-row-index", type=int, default=None)
    parser.add_argument("--replay-seed", type=int, default=17)
    parser.add_argument(
        "--replay-different-scene",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replay the source action in another scene. Disabled by default because absolute EEF trajectories are scene/init dependent.",
    )
    parser.add_argument(
        "--delta-to-controller",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Convert metric EEF deltas to robosuite OSC controller commands for rollout visualization.",
    )
    parser.add_argument(
        "--rollout-source",
        choices=["eef_delta", "raw_controller"],
        default="eef_delta",
        help=(
            "eef_delta replays the current model input after denormalization; raw_controller replays "
            "the original LIBERO controller actions interpolated to the same 104 timestamps."
        ),
    )
    parser.add_argument("--output-dir", default="logs/debug/libero_hdr_action_finegrained")
    parser.add_argument("--libero-root", default="third_parties/LIBERO")
    parser.add_argument("--libero-data-root", default="data/LIBERO-datasets")
    parser.add_argument("--physical-agent-root", default="/mnt/zezhong/physical_agent")
    parser.add_argument("--model-root", default="checkpoints")
    parser.add_argument("--model-name", default="Wan2.2-TI2V-5B")
    parser.add_argument("--camera-height", type=int, default=224)
    parser.add_argument("--camera-width", type=int, default=224)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=104)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def choose_row(rows: list[dict[str, str]], *, row_index: int | None, seed: int) -> tuple[int, dict[str, str]]:
    if row_index is not None:
        idx = int(row_index)
    else:
        idx = random.Random(seed).randrange(len(rows))
    return idx, rows[idx]


def choose_different_replay_row(
    rows: list[dict[str, str]],
    source_row: dict[str, str],
    *,
    row_index: int | None,
    seed: int,
) -> tuple[int, dict[str, str]]:
    if row_index is not None:
        idx = int(row_index)
        return idx, rows[idx]
    if not source_row.get("source_file") or not source_row.get("demo_id"):
        return choose_row(rows, row_index=None, seed=seed)
    source_key = (source_row.get("source_file"), source_row.get("demo_id"))
    candidates = [
        (idx, row) for idx, row in enumerate(rows)
        if (row.get("source_file"), row.get("demo_id")) != source_key
    ]
    if not candidates:
        return choose_row(rows, row_index=None, seed=seed)
    return random.Random(seed).choice(candidates)


def resolve_rel(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def resolve_h5_path(source_file: str, libero_data_root: Path) -> Path:
    source = Path(source_file)
    if source.exists():
        return source
    matches = list(libero_data_root.rglob(source.name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Cannot resolve hdf5 source: {source_file}")


def problem_name_from_h5(h5_path: Path) -> str:
    stem = h5_path.stem
    return stem[: -len("_demo")] if stem.endswith("_demo") else stem


def normalize_action(action: np.ndarray, stats: dict) -> np.ndarray:
    min_v = np.asarray(stats["min"], dtype=np.float32)
    max_v = np.asarray(stats["max"], dtype=np.float32)
    eps = float(stats.get("eps", 1e-6))
    return np.clip(2.0 * (action - min_v) / (max_v - min_v + eps) - 1.0, -1.0, 1.0)


def denormalize_action(action_norm: np.ndarray, stats: dict) -> np.ndarray:
    min_v = np.asarray(stats["min"], dtype=np.float32)
    max_v = np.asarray(stats["max"], dtype=np.float32)
    eps = float(stats.get("eps", 1e-6))
    return (action_norm + 1.0) * 0.5 * (max_v - min_v + eps) + min_v


def interpolate_sequence(values: np.ndarray, sample_times: np.ndarray) -> np.ndarray:
    raw_t = np.arange(values.shape[0], dtype=np.float64)
    flat_times = sample_times.reshape(-1)
    interp = np.stack(
        [np.interp(flat_times, raw_t, values[:, dim]) for dim in range(values.shape[1])],
        axis=-1,
    )
    return interp.reshape(*sample_times.shape, values.shape[1]).astype(np.float32)


def reconstruct_abs_eef(init_eef: np.ndarray, delta_actions: np.ndarray) -> np.ndarray:
    abs_eef = np.zeros((delta_actions.shape[0], 6), dtype=np.float32)
    current = init_eef.astype(np.float32).copy()
    for idx, delta in enumerate(delta_actions[:, :6]):
        current = current + delta.astype(np.float32)
        abs_eef[idx] = current
    return abs_eef


def gripper_to_command(value: float) -> float:
    return float(np.clip(value, -1.0, 1.0))


def delta_to_controller_command(delta: np.ndarray) -> np.ndarray:
    output_max = np.asarray([0.05, 0.05, 0.05, 0.5, 0.5, 0.5], dtype=np.float32)
    return np.clip(delta.astype(np.float32) / output_max, -1.0, 1.0)


def frame_from_obs(obs: dict) -> np.ndarray:
    for key in ("agentview_image", "agentview_rgb", "robot0_agentview_image"):
        if key in obs:
            frame = np.asarray(obs[key])
            break
    else:
        image_keys = [k for k, v in obs.items() if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] == 3]
        if not image_keys:
            raise KeyError(f"No RGB observation key found. Available keys: {sorted(obs.keys())}")
        frame = np.asarray(obs[image_keys[0]])
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame[::-1, ::-1].copy()


def resize_frame(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    import cv2

    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def tensor_video_to_uint8_frames(video: torch.Tensor) -> list[np.ndarray]:
    if video.ndim != 5:
        raise ValueError(f"Expected decoded video [B,T,C,H,W], got {tuple(video.shape)}")
    frames = []
    for frame in video[0].detach().cpu().permute(0, 2, 3, 1):
        arr = ((frame.float().clamp(-1, 1) + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8).numpy()
        frames.append(arr)
    return frames


def resample_frames(frames: list[np.ndarray], count: int, height: int, width: int) -> list[np.ndarray]:
    if not frames:
        raise ValueError("Cannot resample empty frames.")
    ids = np.linspace(0, len(frames) - 1, count)
    out = []
    for idx in ids:
        out.append(resize_frame(frames[int(round(idx))], height, width))
    return out


def write_mp4_pyav(frames: list[np.ndarray], path: Path, fps: int) -> None:
    if not frames:
        raise ValueError("No frames to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "medium"}
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = resize_frame(frame, height, width)
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def write_csv(path: Path, header: list[str], array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(array.tolist())


def decode_leaf_latents(cache_path: Path, *, model_root: str, model_name: str, device: str) -> list[np.ndarray]:
    if str(CAUSAL_FORCING_ROOT) not in sys.path:
        sys.path.insert(0, str(CAUSAL_FORCING_ROOT))
    from utils.wan_wrapper import WanVAEWrapper

    loaded = np.load(cache_path)
    vertical_latents = loaded["vertical_latents"].astype(np.float32)
    leaf_latents = torch.from_numpy(vertical_latents[-13:]).unsqueeze(0).to(device=device, dtype=torch.float32)
    vae = WanVAEWrapper(model_name=model_name, model_root=model_root).to(device)
    with torch.no_grad():
        decoded = vae.decode_to_pixel(leaf_latents)
    return tensor_video_to_uint8_frames(decoded)


def rollout_delta_actions(
    *,
    replay_row: dict[str, str],
    delta_actions: np.ndarray,
    gripper: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, Path, Path]:
    sys.path.insert(0, str(Path(args.physical_agent_root)))
    from libero_agent import LiberoActionError, LiberoAgentInterface

    libero_root = Path(args.libero_root).resolve()
    h5_path = resolve_h5_path(replay_row["source_file"], Path(args.libero_data_root))
    suite = h5_path.parent.name
    bddl_file = libero_root / "libero" / "libero" / "bddl_files" / suite / f"{problem_name_from_h5(h5_path)}.bddl"
    if not bddl_file.exists():
        raise FileNotFoundError(f"Cannot find BDDL: {bddl_file}")

    with h5py.File(h5_path, "r") as f:
        init_state = np.asarray(f["data"][replay_row["demo_id"]].attrs["init_state"], dtype=np.float64)

    env = LiberoAgentInterface(
        bddl_file=bddl_file,
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
        action_repeat=1,
        libero_root=libero_root,
    )
    env.reset(init_state=init_state)
    frames = [frame_from_obs(env.last_obs)]
    actual_eef = [np.asarray(env.state()["eef_pos"], dtype=np.float32)]
    initial_eef_pos = actual_eef[0].copy()
    try:
        for step_id in range(min(args.max_steps, delta_actions.shape[0])):
            arm_action = delta_actions[step_id, :6].astype(np.float32)
            if args.delta_to_controller:
                arm_action = delta_to_controller_command(arm_action)
            try:
                result = env.step(
                    arm_action,
                    mode="ee_delta",
                    gripper=gripper_to_command(float(gripper[step_id])),
                )
            except LiberoActionError as exc:
                print(f"[FineAction] rollout stopped step={step_id}: {exc.to_dict()}", flush=True)
                break
            frames.append(frame_from_obs(result.observation))
            actual_eef.append(np.asarray(result.state_after["eef_pos"], dtype=np.float32))
    finally:
        env.close()
    return frames[: args.max_steps], np.stack(actual_eef, axis=0), initial_eef_pos, h5_path, bddl_file


def rollout_controller_actions(
    *,
    replay_row: dict[str, str],
    controller_actions: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, Path, Path]:
    sys.path.insert(0, str(Path(args.physical_agent_root)))
    from libero_agent import LiberoActionError, LiberoAgentInterface

    libero_root = Path(args.libero_root).resolve()
    h5_path = resolve_h5_path(replay_row["source_file"], Path(args.libero_data_root))
    suite = h5_path.parent.name
    bddl_file = libero_root / "libero" / "libero" / "bddl_files" / suite / f"{problem_name_from_h5(h5_path)}.bddl"
    if not bddl_file.exists():
        raise FileNotFoundError(f"Cannot find BDDL: {bddl_file}")

    with h5py.File(h5_path, "r") as f:
        init_state = np.asarray(f["data"][replay_row["demo_id"]].attrs["init_state"], dtype=np.float64)

    env = LiberoAgentInterface(
        bddl_file=bddl_file,
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
        action_repeat=1,
        libero_root=libero_root,
    )
    env.reset(init_state=init_state)
    frames = [frame_from_obs(env.last_obs)]
    actual_eef = [np.asarray(env.state()["eef_pos"], dtype=np.float32)]
    initial_eef_pos = actual_eef[0].copy()
    try:
        for step_id in range(min(args.max_steps, controller_actions.shape[0])):
            try:
                result = env.step(
                    np.clip(controller_actions[step_id, :6], -1.0, 1.0),
                    mode="ee_delta",
                    gripper=gripper_to_command(float(controller_actions[step_id, 6])),
                )
            except LiberoActionError as exc:
                print(f"[FineAction] controller rollout stopped step={step_id}: {exc.to_dict()}", flush=True)
                break
            frames.append(frame_from_obs(result.observation))
            actual_eef.append(np.asarray(result.state_after["eef_pos"], dtype=np.float32))
    finally:
        env.close()
    return frames[: args.max_steps], np.stack(actual_eef, axis=0), initial_eef_pos, h5_path, bddl_file


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata_path)
    metadata_base = metadata_path.parent
    rows = load_rows(metadata_path)
    row_idx, row = choose_row(rows, row_index=args.row_index, seed=args.seed)
    if args.replay_different_scene:
        replay_idx, replay_row = choose_different_replay_row(
            rows,
            row,
            row_index=args.replay_row_index,
            seed=args.replay_seed,
        )
    else:
        replay_idx, replay_row = row_idx, row
    stats = json.loads(Path(args.stats_path).read_text(encoding="utf-8"))

    action_path = resolve_rel(metadata_base, row["action_path"])
    action_npz = np.load(action_path)
    action_raw = action_npz["action"].astype(np.float32).reshape(-1, 7)
    prepared_abs_eef = action_npz["abs_eef"].astype(np.float32).reshape(-1, 6) if "abs_eef" in action_npz else None
    leaf_action_times = action_npz["leaf_action_times"].astype(np.float32) if "leaf_action_times" in action_npz else None
    action_norm = normalize_action(action_raw, stats).astype(np.float32)
    action_delta = denormalize_action(action_norm, stats).astype(np.float32)

    source_h5 = resolve_h5_path(row["source_file"], Path(args.libero_data_root))
    with h5py.File(source_h5, "r") as f:
        source_demo = f["data"][row["demo_id"]]
        init_eef = np.asarray(source_demo["obs"]["ee_states"][0], dtype=np.float32)
        source_controller_actions = np.asarray(source_demo["actions"][...], dtype=np.float32)
    if leaf_action_times is not None:
        controller_actions_104 = interpolate_sequence(source_controller_actions, leaf_action_times).reshape(-1, 7)
    else:
        ids = np.linspace(0, source_controller_actions.shape[0] - 1, action_delta.shape[0])
        controller_actions_104 = interpolate_sequence(source_controller_actions, ids).reshape(-1, 7)
    abs_eef = reconstruct_abs_eef(init_eef, action_delta)
    abs_action = np.concatenate([abs_eef, action_delta[:, 6:7]], axis=-1)

    if args.rollout_source == "raw_controller":
        replay_frames, actual_eef, replay_init_eef_pos, replay_h5, bddl_file = rollout_controller_actions(
            replay_row=replay_row,
            controller_actions=controller_actions_104,
            args=args,
        )
    else:
        replay_frames, actual_eef, replay_init_eef_pos, replay_h5, bddl_file = rollout_delta_actions(
            replay_row=replay_row,
            delta_actions=action_delta,
            gripper=action_delta[:, 6],
            args=args,
        )
    leaf_cache_path = resolve_rel(metadata_base, row["video_latent_cache_path"])
    leaf_frames = decode_leaf_latents(
        leaf_cache_path,
        model_root=args.model_root,
        model_name=args.model_name,
        device=args.device,
    )
    leaf_frames_104 = resample_frames(leaf_frames, len(replay_frames), args.camera_height, args.camera_width)
    paired = [
        np.concatenate([rollout, leaf], axis=1)
        for rollout, leaf in zip(replay_frames, leaf_frames_104)
    ]

    output_dir = Path(args.output_dir)
    prefix = f"row{row_idx:05d}_replay{replay_idx:05d}"
    comparison_path = output_dir / f"{prefix}_rollout_vs_leaf_decode.mp4"
    rollout_path = output_dir / f"{prefix}_rollout.mp4"
    leaf_path = output_dir / f"{prefix}_leaf_decode_resampled.mp4"
    write_mp4_pyav(paired, comparison_path, fps=args.fps)
    write_mp4_pyav(replay_frames, rollout_path, fps=args.fps)
    write_mp4_pyav(leaf_frames_104, leaf_path, fps=args.fps)
    write_csv(
        output_dir / f"{prefix}_normalized_delta_actions.csv",
        ["dx", "dy", "dz", "drot_x", "drot_y", "drot_z", "gripper"],
        action_norm,
    )
    write_csv(
        output_dir / f"{prefix}_denormalized_delta_actions.csv",
        ["dx", "dy", "dz", "drot_x", "drot_y", "drot_z", "gripper"],
        action_delta,
    )
    write_csv(
        output_dir / f"{prefix}_raw_controller_actions_104.csv",
        ["x", "y", "z", "rot_x", "rot_y", "rot_z", "gripper"],
        controller_actions_104,
    )
    write_csv(
        output_dir / f"{prefix}_absolute_actions.csv",
        ["x", "y", "z", "rot_x", "rot_y", "rot_z", "gripper"],
        abs_action,
    )
    if prepared_abs_eef is not None:
        write_csv(
            output_dir / f"{prefix}_prepared_abs_eef_from_npz.csv",
            ["x", "y", "z", "rot_x", "rot_y", "rot_z"],
            prepared_abs_eef,
        )
    write_csv(
        output_dir / f"{prefix}_actual_rollout_eef.csv",
        ["x", "y", "z"],
        actual_eef,
    )

    summary = {
        "row_index": row_idx,
        "replay_row_index": replay_idx,
        "source_prompt": row.get("sparse_prompt"),
        "source_h5": str(source_h5),
        "source_demo_id": row.get("demo_id"),
        "action_path": str(action_path),
        "video_latent_cache_path": str(leaf_cache_path),
        "replay_h5": str(replay_h5),
        "replay_demo_id": replay_row.get("demo_id"),
        "replay_different_scene": bool(args.replay_different_scene),
        "rollout_source": args.rollout_source,
        "delta_to_controller": bool(args.delta_to_controller),
        "bddl_file": str(bddl_file),
        "init_eef_from_source_demo": init_eef.tolist(),
        "init_eef_pos_from_replay_env": replay_init_eef_pos.tolist(),
        "action_norm_shape": list(action_norm.shape),
        "action_norm_min": action_norm.min(axis=0).tolist(),
        "action_norm_max": action_norm.max(axis=0).tolist(),
        "action_delta_min": action_delta.min(axis=0).tolist(),
        "action_delta_max": action_delta.max(axis=0).tolist(),
        "raw_controller_action_min": controller_actions_104.min(axis=0).tolist(),
        "raw_controller_action_max": controller_actions_104.max(axis=0).tolist(),
        "abs_action_first": abs_action[0].tolist(),
        "abs_action_last": abs_action[-1].tolist(),
        "prepared_abs_eef_first": prepared_abs_eef[0].tolist() if prepared_abs_eef is not None else None,
        "prepared_abs_eef_last": prepared_abs_eef[-1].tolist() if prepared_abs_eef is not None else None,
        "rollout_frames": len(replay_frames),
        "leaf_decoded_frames": len(leaf_frames),
        "comparison_mp4": str(comparison_path),
        "rollout_mp4": str(rollout_path),
        "leaf_decode_mp4": str(leaf_path),
    }
    summary_path = output_dir / f"{prefix}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
