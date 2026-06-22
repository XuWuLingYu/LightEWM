from utils.lmdb_ import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import json
from array import array
from pathlib import Path
from PIL import Image
import os
import imageio
import h5py
import ast

try:
    import lmdb
except ImportError:
    lmdb = None

try:
    import av
except ImportError:
    av = None


def _require_lmdb():
    if lmdb is None:
        raise ImportError("lmdb is required for LMDB-backed datasets but is not installed.")


def _resize_to_target(image: Image.Image, target_height: int, target_width: int) -> Image.Image:
    return image.resize((target_width, target_height), resample=Image.BICUBIC)


def _sample_video_frame_ids(total_frames: int, target_frames: int) -> list[int]:
    if total_frames <= 0:
        raise ValueError("Video contains no frames.")
    if total_frames == 1:
        return [0] * target_frames
    return np.linspace(0, total_frames - 1, target_frames).round().astype(np.int64).tolist()


def _load_video_as_tensor(
    video_path: Path,
    num_frames: int | None,
    height: int,
    width: int,
    max_frames: int | None = None,
) -> torch.Tensor:
    def _to_tensor(rgb_frame: np.ndarray) -> torch.Tensor:
        frame = Image.fromarray(rgb_frame).convert("RGB")
        frame = _resize_to_target(frame, height, width)
        return torch.from_numpy(np.array(frame)).permute(2, 0, 1).float() / 127.5 - 1.0

    def _stack_frames_by_ids(frame_ids: list[int], frame_cache: dict[int, torch.Tensor]) -> torch.Tensor:
        if not frame_cache:
            raise ValueError(f"Video contains no decodable frames: {video_path}")
        existing_ids = sorted(frame_cache.keys())
        frames = []
        for frame_id in frame_ids:
            if frame_id in frame_cache:
                frames.append(frame_cache[frame_id])
                continue
            nearest_id = min(existing_ids, key=lambda x: abs(x - frame_id))
            frames.append(frame_cache[nearest_id])
        return torch.stack(frames, dim=1)

    def _target_count(total_frames: int) -> int:
        if max_frames is not None and total_frames > max_frames:
            return max_frames
        if num_frames is not None:
            return num_frames
        return total_frames

    av_error = None
    if av is not None:
        try:
            with av.open(str(video_path), mode="r") as container:
                stream = container.streams.video[0] if container.streams.video else None
                if stream is None:
                    raise ValueError(f"No video stream in {video_path}")

                stream_frame_count = int(stream.frames) if stream.frames and stream.frames > 0 else 0
                if stream_frame_count > 0:
                    count = _target_count(stream_frame_count)
                    frame_ids = _sample_video_frame_ids(stream_frame_count, count)
                    required_ids = set(frame_ids)
                    selected_frames: dict[int, torch.Tensor] = {}
                    for decoded_id, frame in enumerate(container.decode(stream)):
                        if decoded_id in required_ids and decoded_id not in selected_frames:
                            selected_frames[decoded_id] = _to_tensor(frame.to_ndarray(format="rgb24"))
                            if len(selected_frames) == len(required_ids):
                                break
                    return _stack_frames_by_ids(frame_ids, selected_frames)

                decoded_frames: list[torch.Tensor] = []
                for frame in container.decode(stream):
                    decoded_frames.append(_to_tensor(frame.to_ndarray(format="rgb24")))
                if not decoded_frames:
                    raise ValueError(f"Video contains no decodable frames: {video_path}")
                total = len(decoded_frames)
                count = _target_count(total)
                frame_ids = _sample_video_frame_ids(total, count)
                return torch.stack([decoded_frames[i] for i in frame_ids], dim=1)
        except Exception as e:
            av_error = e

    # Fallback for environments without PyAV or on decode edge cases.
    reader = imageio.get_reader(str(video_path))
    try:
        total_frames = reader.count_frames()
        count = _target_count(total_frames)
        frame_ids = _sample_video_frame_ids(total_frames, count)
        frames = []
        for frame_id in frame_ids:
            frame = reader.get_data(int(frame_id))
            frames.append(_to_tensor(frame))
        return torch.stack(frames, dim=1)
    except Exception as imageio_error:
        if av_error is not None:
            raise RuntimeError(
                f"Failed to decode video {video_path} with both PyAV and imageio. "
                f"PyAV error: {repr(av_error)}; imageio error: {repr(imageio_error)}"
            ) from imageio_error
        raise
    finally:
        reader.close()


def _rgb_frames_to_tensor(frames: np.ndarray, height: int, width: int) -> torch.Tensor:
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    tensors = []
    for frame in frames:
        image = Image.fromarray(frame).convert("RGB")
        image = _resize_to_target(image, height, width)
        tensors.append(torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1.0)
    return torch.stack(tensors, dim=1)


def _parse_camera_keys(camera_keys: str | list[str]) -> list[str]:
    if isinstance(camera_keys, str):
        stripped = camera_keys.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, (list, tuple)):
                    camera_keys = [str(part).strip() for part in parsed if str(part).strip()]
                else:
                    camera_keys = [str(parsed).strip()]
            except (SyntaxError, ValueError):
                camera_keys = [part.strip().strip("'\"") for part in stripped.strip("[]").split(",") if part.strip()]
        else:
            camera_keys = [part.strip() for part in camera_keys.split(",") if part.strip()]
    else:
        camera_keys = list(camera_keys)
    if not camera_keys:
        raise ValueError("At least one LIBERO camera key is required.")
    return [str(camera_key) for camera_key in camera_keys]


def _resize_rgb_frames(frames: np.ndarray, height: int, width: int) -> np.ndarray:
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    resized = []
    for frame in frames:
        image = Image.fromarray(frame).convert("RGB")
        image = _resize_to_target(image, height, width)
        resized.append(np.asarray(image))
    return np.stack(resized, axis=0)


def _load_libero_camera_frames(obs, camera_keys: str | list[str], video_indices: np.ndarray) -> np.ndarray:
    camera_keys = _parse_camera_keys(camera_keys)

    camera_frames = []
    for camera_key in camera_keys:
        if camera_key not in obs:
            raise KeyError(f"Camera key `{camera_key}` not found. Available={list(obs.keys())}")
        frames = np.asarray(obs[camera_key][video_indices])
        # Match exported LIBERO videos and FastWAM eval: camera tensors need a 180-degree rotation.
        frames = frames[:, ::-1, ::-1, :]
        camera_frames.append(frames)
    if len(camera_frames) == 1:
        return camera_frames[0]
    return np.concatenate(camera_frames, axis=2)


def _load_libero_camera_tensor(
    obs,
    camera_keys: str | list[str],
    video_indices: np.ndarray,
    height: int,
    width: int,
) -> torch.Tensor:
    """Load LIBERO camera frames as FastWAM-style per-camera squares.

    For two-camera LIBERO, FastWAM resizes each camera to 224x224 and then
    concatenates horizontally into a 224x448 frame. Resizing a single
    agentview image into 224x448 is a data bug, so this path rejects that
    shape when only one camera is requested.
    """
    camera_keys = _parse_camera_keys(camera_keys)
    num_cameras = len(camera_keys)
    if num_cameras == 1:
        if int(width) != int(height):
            raise ValueError(
                f"Single-camera LIBERO input must stay square, got height={height}, width={width}. "
                "Use two camera keys for FastWAM-style 224x448 inputs."
            )
        frames = _load_libero_camera_frames(obs, camera_keys, video_indices)
        return _rgb_frames_to_tensor(frames, height, width)

    if int(width) % num_cameras != 0:
        raise ValueError(f"width={width} must be divisible by num_cameras={num_cameras}.")
    per_camera_width = int(width) // num_cameras
    if per_camera_width != int(height):
        raise ValueError(
            f"FastWAM-style LIBERO multi-camera input expects square views, got "
            f"height={height}, per_camera_width={per_camera_width}, cameras={camera_keys}."
        )

    camera_frames = []
    for camera_key in camera_keys:
        if camera_key not in obs:
            raise KeyError(f"Camera key `{camera_key}` not found. Available={list(obs.keys())}")
        frames = np.asarray(obs[camera_key][video_indices])
        frames = frames[:, ::-1, ::-1, :]
        camera_frames.append(_resize_rgb_frames(frames, int(height), per_camera_width))
    merged = np.concatenate(camera_frames, axis=2)
    return _rgb_frames_to_tensor(merged, int(height), int(width))


_ACTION_STATS_CACHE: dict[str, dict] = {}


def _load_action_stats(stats_path: Path) -> dict:
    key = str(stats_path)
    cached = _ACTION_STATS_CACHE.get(key)
    if cached is not None:
        return cached
    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)
    _ACTION_STATS_CACHE[key] = stats
    return stats


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _load_action_as_tensor(action_path: Path, stats_path: Path | None = None, norm_clip: float = 1.0) -> torch.Tensor:
    loaded = np.load(action_path)
    if "action" not in loaded:
        raise KeyError(f"Action npz must contain `action`: {action_path}")
    action = loaded["action"].astype(np.float32)
    if action.ndim == 3:
        action = action.reshape(-1, action.shape[-1])
    if action.ndim != 2:
        raise ValueError(f"Expected action shape [T, D] or [leaf, per_leaf, D], got {action.shape} in {action_path}")

    if stats_path is not None:
        stats = _load_action_stats(stats_path)
        min_v = np.asarray(stats["min"], dtype=np.float32)
        max_v = np.asarray(stats["max"], dtype=np.float32)
        eps = float(stats.get("eps", 1e-6))
        action = 2.0 * (action - min_v) / (max_v - min_v + eps) - 1.0
        action = np.clip(action, -float(norm_clip), float(norm_clip))
    return torch.from_numpy(action).float()


def _normalize_actions(action: np.ndarray, stats_path: Path | None, norm_clip: float = 1.0) -> np.ndarray:
    action = action.astype(np.float32)
    if stats_path is not None:
        stats = _load_action_stats(stats_path)
        min_v = np.asarray(stats["min"], dtype=np.float32)
        max_v = np.asarray(stats["max"], dtype=np.float32)
        eps = float(stats.get("eps", 1e-6))
        action = 2.0 * (action - min_v) / (max_v - min_v + eps) - 1.0
        action = np.clip(action, -float(norm_clip), float(norm_clip))
    return action


def _normalize_vector(value: np.ndarray, stats_path: Path | None, norm_clip: float = 1.0) -> np.ndarray:
    value = value.astype(np.float32)
    if stats_path is not None:
        stats = _load_action_stats(stats_path)
        min_v = np.asarray(stats["min"], dtype=np.float32)
        max_v = np.asarray(stats["max"], dtype=np.float32)
        eps = float(stats.get("eps", 1e-6))
        value = 2.0 * (value - min_v) / (max_v - min_v + eps) - 1.0
        value = np.clip(value, -float(norm_clip), float(norm_clip))
    return value


def _quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(0.0, 1.0 - float(quat[3] * quat[3])))
    if den < 1e-6:
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * np.arccos(quat[3]) / den)).astype(np.float32)


def _extract_libero_proprio(obs, frame_index: int) -> np.ndarray:
    if "ee_pos" in obs and "ee_ori" in obs and "gripper_states" in obs:
        eef_pos = np.asarray(obs["ee_pos"][frame_index], dtype=np.float32).reshape(-1)
        eef_axis_angle = np.asarray(obs["ee_ori"][frame_index], dtype=np.float32).reshape(-1)
        gripper_qpos = np.asarray(obs["gripper_states"][frame_index], dtype=np.float32).reshape(-1)
    else:
        required_keys = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
        missing = [key for key in required_keys if key not in obs]
        if missing:
            raise KeyError(f"LIBERO proprio requires obs keys {required_keys} or exported ee_* keys, missing={missing}.")
        eef_pos = np.asarray(obs["robot0_eef_pos"][frame_index], dtype=np.float32).reshape(-1)
        eef_axis_angle = _quat_to_axis_angle(np.asarray(obs["robot0_eef_quat"][frame_index], dtype=np.float32).reshape(-1))
        gripper_qpos = np.asarray(obs["robot0_gripper_qpos"][frame_index], dtype=np.float32).reshape(-1)
    return np.concatenate([eef_pos, eef_axis_angle, gripper_qpos], axis=0).astype(np.float32)


def _fps10_indices(num_frames: int, source_fps: float, target_fps: float) -> np.ndarray:
    if num_frames <= 0:
        raise ValueError("LIBERO demo contains no frames.")
    if target_fps >= source_fps:
        return np.arange(num_frames, dtype=np.int64)
    step = float(source_fps) / float(target_fps)
    indices = np.floor(np.arange(0, num_frames, step)).astype(np.int64)
    indices = np.clip(indices, 0, num_frames - 1)
    return np.unique(indices)


def _load_libero_joint_window(
    *,
    source_file: Path,
    demo_id: str,
    stats_path: Path | None,
    proprio_stats_path: Path | None,
    height: int,
    width: int,
    window_frames: int,
    source_fps: float,
    target_fps: float,
    video_frame_stride: int,
    camera_key: str | list[str],
    include_terminal_video_frame: bool = False,
    norm_clip: float = 1.0,
) -> dict[str, torch.Tensor]:
    with h5py.File(source_file, "r") as f:
        demo = f["data"][demo_id]
        obs = demo["obs"]
        if isinstance(camera_key, str):
            stripped = camera_key.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = ast.literal_eval(stripped)
                    camera_keys = [str(part).strip() for part in parsed if str(part).strip()]
                except (SyntaxError, ValueError):
                    camera_keys = [part.strip().strip("'\"") for part in stripped.strip("[]").split(",") if part.strip()]
            else:
                camera_keys = [part.strip() for part in camera_key.split(",") if part.strip()]
        else:
            camera_keys = list(camera_key)
        if not camera_keys:
            raise ValueError("video_action_joint requires at least one `joint_camera_key`.")
        if camera_keys[0] not in obs:
            raise KeyError(f"Camera key `{camera_keys[0]}` not found in {source_file}:{demo_id}. Available={list(obs.keys())}")
        raw_frames = obs[camera_keys[0]]
        raw_actions = demo["actions"]
        frame_count = int(min(raw_frames.shape[0], raw_actions.shape[0]))
        indices = _fps10_indices(frame_count, source_fps=source_fps, target_fps=target_fps)
        required_index_count = window_frames + (1 if include_terminal_video_frame else 0)
        if indices.shape[0] < required_index_count:
            pad = np.full(required_index_count - indices.shape[0], indices[-1], dtype=np.int64)
            indices = np.concatenate([indices, pad], axis=0)
        max_start = max(0, indices.shape[0] - required_index_count)
        if max_start > 0:
            start = int(np.random.randint(0, max_start + 1))
        else:
            start = 0
        window_indices = indices[start:start + window_frames]
        video_source_indices = indices[start:start + required_index_count]
        video_stride = max(1, int(video_frame_stride))
        video_indices = video_source_indices[::video_stride]
        if include_terminal_video_frame:
            expected_video_frames = int((window_frames // video_stride) + 1)
        else:
            expected_video_frames = int(np.ceil(float(window_frames) / float(video_stride)))
        if video_indices.shape[0] < expected_video_frames:
            pad = np.full(expected_video_frames - video_indices.shape[0], video_indices[-1], dtype=np.int64)
            video_indices = np.concatenate([video_indices, pad], axis=0)
        joint_local_frames = _load_libero_camera_tensor(obs, camera_keys, video_indices, height, width)
        actions = np.asarray(raw_actions[window_indices], dtype=np.float32)
        proprio = _extract_libero_proprio(obs, int(window_indices[0]))
    return {
        "joint_local_frames": joint_local_frames,
        "joint_actions": torch.from_numpy(_normalize_actions(actions, stats_path, norm_clip=norm_clip)).float(),
        "joint_proprio": torch.from_numpy(_normalize_vector(proprio, proprio_stats_path, norm_clip=norm_clip)).float(),
        "joint_window_start": torch.tensor(start, dtype=torch.long),
        "joint_window_indices": torch.from_numpy(window_indices.astype(np.int64)),
        "joint_video_indices": torch.from_numpy(video_indices.astype(np.int64)),
    }


def _load_libero_episode_frames(
    *,
    source_file: Path,
    demo_id: str,
    height: int,
    width: int,
    num_frames: int,
    camera_key: str | list[str],
) -> torch.Tensor:
    with h5py.File(source_file, "r") as f:
        demo = f["data"][demo_id]
        obs = demo["obs"]
        camera_keys = _parse_camera_keys(camera_key)
        if camera_keys[0] not in obs:
            raise KeyError(f"Camera key `{camera_keys[0]}` not found in {source_file}:{demo_id}. Available={list(obs.keys())}")
        frame_count = int(obs[camera_keys[0]].shape[0])
        frame_ids = np.asarray(_sample_video_frame_ids(frame_count, int(num_frames)), dtype=np.int64)
        return _load_libero_camera_tensor(obs, camera_keys, frame_ids, height, width)


def _load_video_latent_cache(cache_path: Path) -> dict[str, torch.Tensor]:
    loaded = np.load(cache_path)
    payload = {}
    if "leaf_latents" in loaded:
        payload["video_leaf_latents"] = torch.from_numpy(loaded["leaf_latents"]).float()
    elif "vertical_latents" in loaded:
        payload["video_latents"] = torch.from_numpy(loaded["vertical_latents"]).float()
    else:
        raise KeyError(f"Video latent cache must contain `leaf_latents` or `vertical_latents`: {cache_path}")
    if "leaf_k" in loaded and "leaf_v" in loaded:
        payload["video_leaf_k"] = torch.from_numpy(loaded["leaf_k"]).float()
        payload["video_leaf_v"] = torch.from_numpy(loaded["leaf_v"]).float()
    if "first_frame_latent" in loaded:
        payload["first_frame_latent"] = torch.from_numpy(loaded["first_frame_latent"]).float()
    if "video_timestep" in loaded:
        payload["video_timestep"] = torch.from_numpy(loaded["video_timestep"]).float()
    return payload


def _load_preencoded_joint_cache(cache_path: Path) -> dict:
    loaded = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise TypeError(f"Preencoded cache must be a dict, got {type(loaded)} in {cache_path}")

    def _tensor(name: str, required: bool = True):
        value = loaded.get(name)
        if value is None:
            if required:
                raise KeyError(f"Preencoded cache missing `{name}`: {cache_path}")
            return None
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        return value.float()

    batch = {
        "clean_latent": _tensor("clean_latent"),
        "joint_local_start_latent": _tensor("joint_local_start_latent"),
        "joint_local_video_latents": _tensor("joint_local_video_latents"),
        "joint_actions": _tensor("joint_actions"),
        "joint_proprio": _tensor("joint_proprio"),
        "prompt_embeds": _tensor("prompt_embeds"),
    }
    for key in ("joint_window_start", "joint_window_indices", "joint_video_indices"):
        value = loaded.get(key)
        if value is not None:
            batch[key] = value if torch.is_tensor(value) else torch.as_tensor(value)
    return batch


class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class TextVideoDataset(Dataset):
    def __init__(
        self,
        metadata_path: str,
        height: int,
        width: int,
        num_frames: int | None = 81,
        variable_num_frames: bool = False,
        max_num_frames: int | None = None,
        video_action_joint: bool = False,
        joint_window_frames: int = 13,
        joint_source_fps: float = 16.0,
        joint_target_fps: float = 10.0,
        joint_video_frame_stride: int = 1,
        joint_camera_key: str = "agentview_rgb",
        joint_include_terminal_video_frame: bool = False,
        joint_norm_clip: float = 1.0,
        joint_drop_tree_tokens: bool = False,
        joint_tree_from_hdf5: bool = False,
        joint_tree_camera_key: str | None = None,
        joint_proprio_stats_path: str | None = None,
    ):
        self.metadata_path = Path(metadata_path)
        self.base_dir = self.metadata_path.parent
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.variable_num_frames = variable_num_frames
        self.max_num_frames = max_num_frames
        self.video_action_joint = bool(video_action_joint)
        self.joint_window_frames = int(joint_window_frames)
        self.joint_source_fps = float(joint_source_fps)
        self.joint_target_fps = float(joint_target_fps)
        self.joint_video_frame_stride = int(joint_video_frame_stride)
        self.joint_camera_key = joint_camera_key
        self.joint_include_terminal_video_frame = bool(joint_include_terminal_video_frame)
        self.joint_norm_clip = float(joint_norm_clip)
        self.joint_drop_tree_tokens = bool(joint_drop_tree_tokens)
        self.joint_tree_from_hdf5 = bool(joint_tree_from_hdf5)
        self.joint_tree_camera_key = joint_tree_camera_key if joint_tree_camera_key is not None else self.joint_camera_key
        self.joint_proprio_stats_path = joint_proprio_stats_path
        self._jsonl_file = None

        if self.metadata_path.suffix == ".jsonl":
            # Keep memory stable for very large jsonl files (e.g. 1M+ samples)
            # by indexing line offsets instead of loading all rows into memory.
            self.metadata = None
            self._jsonl_offsets = array("Q")
            with open(self.metadata_path, "rb") as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    if line.strip():
                        self._jsonl_offsets.append(offset)
        elif self.metadata_path.suffix == ".json":
            self._jsonl_offsets = None
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            raise ValueError(f"Unsupported metadata format: {self.metadata_path}")

    def __len__(self):
        if self._jsonl_offsets is not None:
            return len(self._jsonl_offsets)
        return len(self.metadata)

    def __getstate__(self):
        # Avoid serializing open file handles into dataloader workers.
        state = self.__dict__.copy()
        state["_jsonl_file"] = None
        return state

    def _get_jsonl_file(self):
        if self._jsonl_file is None or self._jsonl_file.closed:
            self._jsonl_file = open(self.metadata_path, "r", encoding="utf-8")
        return self._jsonl_file

    def _resolve_video_path(self, video_path: str) -> Path:
        return _resolve_path(self.base_dir, video_path)

    def __getitem__(self, idx):
        if self._jsonl_offsets is not None:
            if idx < 0 or idx >= len(self._jsonl_offsets):
                raise IndexError(f"Index {idx} out of range for dataset of size {len(self._jsonl_offsets)}.")
            jsonl_file = self._get_jsonl_file()
            jsonl_file.seek(int(self._jsonl_offsets[idx]))
            line = jsonl_file.readline()
            if not line:
                raise RuntimeError(f"Failed to read jsonl line at index {idx}.")
            item = json.loads(line)
        else:
            item = self.metadata[idx]
        prompt = item.get("prompt", item.get("caption"))
        if prompt is None:
            raise KeyError("Each metadata item must contain `prompt`.")
        preencoded_cache_path = item.get("preencoded_cache_path")
        video_path = item.get("video_path", item.get("video"))
        skip_tree_video = self.video_action_joint and self.joint_drop_tree_tokens
        tree_from_hdf5 = self.video_action_joint and self.joint_tree_from_hdf5 and not skip_tree_video
        if video_path is None and not skip_tree_video and not tree_from_hdf5:
            raise KeyError("Each metadata item must contain `video_path`.")
        video_path = self._resolve_video_path(video_path) if video_path is not None else None
        video_latent_cache_path = item.get("video_latent_cache_path")
        batch = {
            "prompts": prompt,
            "idx": idx,
        }
        if preencoded_cache_path is not None:
            resolved_cache_path = _resolve_path(self.base_dir, str(preencoded_cache_path))
            batch.update(_load_preencoded_joint_cache(resolved_cache_path))
            batch["preencoded_cache_path"] = str(resolved_cache_path)
            batch["num_frames"] = int(batch["clean_latent"].shape[0])
            return batch
        source_file = item.get("source_file")
        demo_id = item.get("demo_id")
        if skip_tree_video:
            if video_path is not None:
                batch["video_path"] = str(video_path)
            batch["num_frames"] = 0
        elif tree_from_hdf5:
            if source_file is None or demo_id is None:
                raise KeyError("joint_tree_from_hdf5 requires `source_file` and `demo_id` in each item.")
            frames = _load_libero_episode_frames(
                source_file=_resolve_path(self.base_dir, str(source_file)),
                demo_id=str(demo_id),
                height=self.height,
                width=self.width,
                num_frames=int(self.num_frames),
                camera_key=self.joint_tree_camera_key,
            )
            batch.update({
                "frames": frames,
                "video_path": str(video_path) if video_path is not None else "",
                "num_frames": frames.shape[1],
            })
        else:
            frames = _load_video_as_tensor(
                video_path=video_path,
                num_frames=1 if video_latent_cache_path is not None else (None if self.variable_num_frames else self.num_frames),
                height=self.height,
                width=self.width,
                max_frames=self.max_num_frames if self.variable_num_frames else None,
            )
            batch.update({
                "frames": frames,
                "video_path": str(video_path),
                "num_frames": frames.shape[1],
            })
        action_path = item.get("action_path")
        if action_path is not None and not self.video_action_joint:
            stats_path = item.get("action_stats_path")
            resolved_action_path = _resolve_path(self.base_dir, str(action_path))
            resolved_stats_path = _resolve_path(self.base_dir, str(stats_path)) if stats_path else None
            actions = _load_action_as_tensor(resolved_action_path, resolved_stats_path, norm_clip=self.joint_norm_clip)
            batch.update({
                "actions": actions,
                "action_is_pad": torch.zeros(actions.shape[0], dtype=torch.bool),
                "action_path": str(resolved_action_path),
            })
        if video_latent_cache_path is not None:
            resolved_cache_path = _resolve_path(self.base_dir, str(video_latent_cache_path))
            batch.update(_load_video_latent_cache(resolved_cache_path))
            batch["video_latent_cache_path"] = str(resolved_cache_path)
        if self.video_action_joint:
            if source_file is None or demo_id is None:
                raise KeyError("video_action_joint dataset requires `source_file` and `demo_id` in each item.")
            stats_path = item.get("action_stats_path")
            proprio_stats_path = item.get("proprio_stats_path", self.joint_proprio_stats_path)
            resolved_stats_path = _resolve_path(self.base_dir, str(stats_path)) if stats_path else None
            resolved_proprio_stats_path = (
                _resolve_path(self.base_dir, str(proprio_stats_path)) if proprio_stats_path else None
            )
            batch.update(_load_libero_joint_window(
                source_file=_resolve_path(self.base_dir, str(source_file)),
                demo_id=str(demo_id),
                stats_path=resolved_stats_path,
                proprio_stats_path=resolved_proprio_stats_path,
                height=self.height,
                width=self.width,
                window_frames=self.joint_window_frames,
                source_fps=self.joint_source_fps,
                target_fps=self.joint_target_fps,
                video_frame_stride=self.joint_video_frame_stride,
                camera_key=self.joint_camera_key,
                include_terminal_video_frame=self.joint_include_terminal_video_frame,
                norm_clip=self.joint_norm_clip,
            ))
        return batch


class ODERegressionLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        _require_lmdb()
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }





class LatentLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        _require_lmdb()
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "clean_latent": torch.tensor(latents, dtype=torch.float32)[-1]
        }


class ShardingLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        _require_lmdb()
        self.envs = []
        self.index = []

        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            env = lmdb.open(path,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            self.envs.append(env)

        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            self.latents_shape[shard_id] = get_array_shape_from_lmdb(env, 'latents')
            for local_i in range(self.latents_shape[shard_id][0]):
                self.index.append((shard_id, local_i))

            # print("shard_id ", shard_id, " local_i ", local_i)

        self.max_pair = max_pair

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
            Outputs:
                - prompts: List of Strings
                - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        shard_id, local_idx = self.index[idx]

        latents = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "latents", np.float16, local_idx,
            shape=self.latents_shape[shard_id][1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "prompts", str, local_idx
        )

        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }



class TextImagePairDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        eval_first_n=-1,
        pad_to_multiple_of=None
    ):
        """
        Args:
            data_dir (str): Path to the directory containing:
                - target_crop_info_*.json (metadata file)
                - */ (subdirectory containing images with matching aspect ratio)
            transform (callable, optional): Optional transform to be applied on the image
        """
        self.transform = transform
        data_dir = Path(data_dir)

        # Find the metadata JSON file
        metadata_files = list(data_dir.glob('target_crop_info_*.json'))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {data_dir}")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files found in {data_dir}")

        metadata_path = metadata_files[0]
        # Extract aspect ratio from metadata filename (e.g. target_crop_info_26-15.json -> 26-15)
        aspect_ratio = metadata_path.stem.split('_')[-1]

        # Use aspect ratio subfolder for images
        self.image_dir = data_dir / aspect_ratio
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        eval_first_n = eval_first_n if eval_first_n != -1 else len(self.metadata)
        self.metadata = self.metadata[:eval_first_n]

        # Verify all images exist
        for item in self.metadata:
            image_path = self.image_dir / item['file_name']
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        self.dummy_prompt = "DUMMY PROMPT"
        self.pre_pad_len = len(self.metadata)
        if pad_to_multiple_of is not None and len(self.metadata) % pad_to_multiple_of != 0:
            # Duplicate the last entry
            self.metadata += [self.metadata[-1]] * (
                pad_to_multiple_of - len(self.metadata) % pad_to_multiple_of
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary containing:
                - image: PIL Image
                - caption: str
                - target_bbox: list of int [x1, y1, x2, y2]
                - target_ratio: str
                - type: str
                - origin_size: tuple of int (width, height)
        """
        item = self.metadata[idx]

        # Load image
        image_path = self.image_dir / item['file_name']
        image = Image.open(image_path).convert('RGB')

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'prompts': item['caption'],
            'target_bbox': item['target_crop']['target_bbox'],
            'target_ratio': item['target_crop']['target_ratio'],
            'type': item['type'],
            'origin_size': (item['origin_width'], item['origin_height']),
            'idx': idx
        }



def cycle(dl):
    while True:
        for data in dl:
            yield data
