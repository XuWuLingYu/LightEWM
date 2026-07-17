#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import json
import socket
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


_THIS_FILE = Path(__file__).resolve()
ROOT = _THIS_FILE.parents[1] if _THIS_FILE.parent.name == "scripts" else _THIS_FILE.parent
FASTWAM_ROOT = ROOT / "lightewm" / "vendor" / "fastwam"


def _insert_paths() -> None:
    for path in (
        FASTWAM_ROOT,
        ROOT / "data" / "python-packages" / "fastwam_pydeps",
    ):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _to_numpy(obj: Any) -> Any:
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, bytes):
        return {"__bytes__": True, "data": base64.b64encode(obj).decode("ascii")}
    if isinstance(obj, Mapping):
        return {k: _to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_numpy(v) for v in obj]
    return obj


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return {
                "__numpy_array__": True,
                "data": base64.b64encode(obj.tobytes()).decode("ascii"),
                "dtype": str(obj.dtype),
                "shape": obj.shape,
            }
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def numpy_to_json(data: Any) -> str:
    return json.dumps(_to_numpy(data), cls=NumpyEncoder, ensure_ascii=False)


def json_to_numpy(json_str: str) -> Any:
    def object_hook(dct: dict[str, Any]) -> Any:
        if "__numpy_array__" in dct:
            raw = base64.b64decode(dct["data"])
            return np.frombuffer(raw, dtype=np.dtype(dct["dtype"])).reshape(dct["shape"])
        if "__bytes__" in dct:
            return base64.b64decode(dct["data"])
        return dct

    return json.loads(json_str, object_hook=object_hook)


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConnectionError("connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class Client:
    def __init__(self, host: str, port: int, timeout: float) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.connect((host, int(port)))

    def call(self, cmd: str, obs: Any = None) -> Any:
        payload = numpy_to_json({"cmd": cmd, "obs": obs}).encode("utf-8")
        self.sock.sendall(len(payload).to_bytes(4, "big"))
        self.sock.sendall(payload)
        header = recv_exact(self.sock, 4)
        response = json_to_numpy(recv_exact(self.sock, int.from_bytes(header, "big")).decode("utf-8"))
        if "error" in response:
            raise RuntimeError(response["error"] + "\n" + response.get("traceback", ""))
        return response

    def close(self) -> None:
        self.sock.close()


def _repo_abs(path: str | Path) -> str:
    return str((ROOT / path).resolve())


def build_dataset(dataset_dir: str, text_cache: str, stats_path: str):
    _insert_paths()
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    from fastwam.utils import misc
    from fastwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    out_dir = ROOT / "logs" / "remote_policy" / "dataset_client_workdir"
    out_dir.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(str(out_dir))
    overrides = [
        "task=robotwin_uncond_3cam_384_1e-4",
        "model=fastwam_joint",
        "data=robotwin",
        "model.redirect_common_files=false",
        "model.mot_checkpoint_mixed_attn=false",
        "model.action_attend_video=local_clean_first",
        f"data.train.dataset_dirs=[{Path(dataset_dir).expanduser().resolve()}]",
        f"data.train.text_embedding_cache_dir={Path(text_cache).expanduser().resolve()}",
        f"data.train.pretrained_norm_stats={Path(stats_path).expanduser().resolve()}",
        "data.train.is_training_set=false",
        "data.train.video_size=[384,320]",
        "data.train.concat_multi_camera=robotwin",
        "data.train.processor.action_state_transforms=null",
        "data.train.processor.norm_default_mode=z-score",
        "+data.train.hdr_enabled=true",
        "+data.train.hdr_local_rgb_frames=9",
        "+data.train.hdr_tree_rgb_frames=4",
        "+data.train.hdr_total_rgb_frames=13",
        "+data.train.hdr_tree_sampling=uniform_local_start_to_end",
    ]
    with initialize_config_dir(version_base="1.3", config_dir=str((FASTWAM_ROOT / "configs").resolve())):
        cfg = compose(config_name="train", overrides=overrides)
    dataset = instantiate(cfg.data.train)
    dataset.lerobot_dataset.processor.set_normalizer_from_stats(load_dataset_stats_from_json(stats_path))
    return dataset


def denormalize_field(processor: Any, category: str, value: Any) -> np.ndarray:
    import torch

    meta = processor.shape_meta[category]
    if len(meta) != 1:
        raise ValueError(f"expected one {category} key, got {meta}")
    key = meta[0]["key"]
    tensor = value.detach().cpu().float() if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
    normalizer = processor.normalizer.normalizers[category][key]
    return normalizer.backward(tensor).numpy().astype(np.float32)


def tensor_image_to_uint8(image: Any) -> np.ndarray:
    arr = image.detach().cpu().float().numpy() if hasattr(image, "detach") else np.asarray(image, dtype=np.float32)
    if arr.shape[0] == 3:
        arr = np.moveaxis(arr, 0, -1)
    if arr.min() < -0.05:
        arr = (arr + 1.0) * 127.5
    elif arr.max() <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def split_robotwin_frame(frame_chw: Any) -> dict[str, np.ndarray]:
    frame = tensor_image_to_uint8(frame_chw)
    if frame.shape[:2] != (384, 320):
        raise ValueError(f"expected composed robotwin frame 384x320, got {frame.shape}")
    head = frame[:256, :, :]
    bottom = frame[256:, :, :]
    left = bottom[:, :160, :]
    right = bottom[:, 160:, :]
    return {"head": head, "left": left, "right": right}


def state_to_obs_dict(state: np.ndarray) -> dict[str, np.ndarray]:
    if state.shape[-1] != 14:
        raise ValueError(f"expected 14-D state, got {state.shape}")
    return {
        "left_arm_joint_state": state[0:6].astype(np.float32),
        "left_ee_joint_state": state[6:7].astype(np.float32),
        "right_arm_joint_state": state[7:13].astype(np.float32),
        "right_ee_joint_state": state[13:14].astype(np.float32),
    }


def action_dicts_to_array(actions: list[dict[str, Any]]) -> np.ndarray:
    rows = []
    for action in actions:
        rows.append(
            np.concatenate(
                [
                    np.asarray(action["left_arm_joint_state"], dtype=np.float32),
                    np.asarray(action["left_ee_joint_state"], dtype=np.float32),
                    np.asarray(action["right_arm_joint_state"], dtype=np.float32),
                    np.asarray(action["right_ee_joint_state"], dtype=np.float32),
                ],
                axis=-1,
            )
        )
    return np.stack(rows, axis=0).astype(np.float32)


def sample_to_obs(dataset: Any, index: int, env_idx: int) -> tuple[dict[str, Any], np.ndarray, str]:
    sample = dataset[int(index)]
    frame = split_robotwin_frame(sample["video"][:, 0])
    state = denormalize_field(dataset.lerobot_dataset.processor, "state", sample["proprio"][0])
    gt_action = denormalize_field(dataset.lerobot_dataset.processor, "action", sample["action"])
    prompt = str(sample["prompt"])
    prefix = "A video recorded from a robot's point of view executing the following instruction:"
    instruction = prompt[len(prefix) :].strip() if prompt.startswith(prefix) else prompt
    obs = {
        "env_idx": int(env_idx),
        "task_instruction": instruction,
        "vision": {
            "cam_head": {"color": frame["head"]},
            "cam_left_wrist": {"color": frame["left"]},
            "cam_right_wrist": {"color": frame["right"]},
        },
        "state": state_to_obs_dict(state),
    }
    return obs, gt_action, instruction


def plot_action(out_path: Path, gt: np.ndarray, pred: np.ndarray, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dims = gt.shape[-1]
    rows = 7 if dims == 14 else int(np.ceil(dims / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(16, 2.3 * rows), sharex=True)
    axes_flat = np.asarray(axes).reshape(-1)
    timesteps = np.arange(gt.shape[0])
    for i in range(dims):
        ax = axes_flat[i]
        ax.plot(timesteps, gt[:, i], label="gt", linewidth=1.8)
        ax.plot(timesteps, pred[:, i], label="pred", linewidth=1.3)
        ax.set_title(f"dim_{i} MAE={np.mean(np.abs(gt[:, i] - pred[:, i])):.4f}")
        ax.grid(alpha=0.3)
    for ax in axes_flat[dims:]:
        ax.axis("off")
    axes_flat[0].legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--dataset-dir", default=_repo_abs("data/robodojo_fastwam/robodojo-v21-video/lerobot"))
    parser.add_argument("--text-cache", default=_repo_abs("data/text_embeds_cache/robodojo_fastwam/full"))
    parser.add_argument("--dataset-stats", default=_repo_abs("data/robodojo_fastwam/robodojo-v21-video/dataset_stats.json"))
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--indices", default=None, help="comma separated dataset indices; overrides num-samples")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output-dir", default="logs/remote_policy/fastwamhdr_dataset_open_loop")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    dataset = build_dataset(args.dataset_dir, args.text_cache, args.dataset_stats)
    if args.indices:
        indices = [int(x) for x in args.indices.split(",") if x.strip()]
    else:
        if args.num_samples <= 1:
            indices = [0]
        else:
            indices = [round(i * (len(dataset) - 1) / (args.num_samples - 1)) for i in range(args.num_samples)]

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    client = Client(args.host, args.port, args.timeout)
    rows = []
    try:
        client.call("reset")
        for offset in range(0, len(indices), args.batch_size):
            batch_indices = indices[offset : offset + args.batch_size]
            obs_list = []
            gt_actions = []
            instructions = []
            env_ids = list(range(len(batch_indices)))
            for env_idx, index in zip(env_ids, batch_indices):
                obs, gt, instruction = sample_to_obs(dataset, index, env_idx)
                obs_list.append(obs)
                gt_actions.append(gt)
                instructions.append(instruction)

            client.call("update_obs_batch", obs_list)
            response = client.call("get_action_batch", env_ids)
            pred_chunks = response["res"]
            for index, gt, pred_dicts, instruction in zip(batch_indices, gt_actions, pred_chunks, instructions):
                pred = action_dicts_to_array(pred_dicts)
                steps = min(len(pred), len(gt))
                diff = pred[:steps] - gt[:steps]
                row = {
                    "index": int(index),
                    "steps": int(steps),
                    "action_l1": float(np.mean(np.abs(diff))),
                    "action_l2": float(np.mean(diff**2)),
                    "action_rmse": float(np.sqrt(np.mean(diff**2))),
                    "max_abs": float(np.max(np.abs(diff))),
                    "instruction": instruction,
                }
                rows.append(row)
                np.savez_compressed(out_dir / f"idx_{int(index):06d}.npz", pred=pred, gt=gt, diff=diff)
                (out_dir / f"idx_{int(index):06d}.json").write_text(json.dumps(row, indent=2, ensure_ascii=True) + "\n")
                if args.plot:
                    plot_action(out_dir / f"idx_{int(index):06d}.png", gt[:steps], pred[:steps], f"RoboDojo remote policy open-loop idx={index}")
                print(
                    f"idx={index} steps={steps} l1={row['action_l1']:.6f} rmse={row['action_rmse']:.6f} max={row['max_abs']:.6f}",
                    flush=True,
                )
    finally:
        client.close()

    l1 = np.asarray([row["action_l1"] for row in rows], dtype=np.float64)
    l2 = np.asarray([row["action_l2"] for row in rows], dtype=np.float64)
    summary = {
        "server": f"{args.host}:{args.port}",
        "dataset_dir": args.dataset_dir,
        "dataset_stats": args.dataset_stats,
        "indices": indices,
        "num_samples": len(rows),
        "batch_size": args.batch_size,
        "mean_action_l1": float(l1.mean()) if len(l1) else None,
        "median_action_l1": float(np.median(l1)) if len(l1) else None,
        "mean_action_l2": float(l2.mean()) if len(l2) else None,
        "mean_action_rmse": float(np.sqrt(l2.mean())) if len(l2) else None,
        "max_sample_l1": float(l1.max()) if len(l1) else None,
        "min_sample_l1": float(l1.min()) if len(l1) else None,
        "per_index": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "steps", "action_l1", "action_l2", "action_rmse", "max_abs", "instruction"])
        writer.writeheader()
        writer.writerows(rows)
    print("SUMMARY", json.dumps({k: summary[k] for k in ("num_samples", "mean_action_l1", "mean_action_rmse", "median_action_l1", "max_sample_l1")}, indent=2))


if __name__ == "__main__":
    main()
