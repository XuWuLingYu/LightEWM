#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import sys
import threading
import time
import traceback
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
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


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


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConnectionError("connection closed while receiving payload")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class ModelServer:
    def __init__(self, model: Any, host: str, port: int) -> None:
        self.model = model
        self.host = host
        self.port = int(port)
        self.running = False
        self.server_socket: socket.socket | None = None
        self.threads: list[threading.Thread] = []

    def start(self) -> None:
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(64)
        self.server_socket.settimeout(2.0)
        self.running = True
        print(f"[server] listening on {self.host}:{self.port}", flush=True)
        while self.running:
            try:
                client, addr = self.server_socket.accept()
            except socket.timeout:
                continue
            print(f"[server] client connected: {addr}", flush=True)
            thread = threading.Thread(target=self._handle_client, args=(client,), daemon=True)
            thread.start()
            self.threads.append(thread)

    def stop(self) -> None:
        self.running = False
        if self.server_socket is not None:
            self.server_socket.close()

    def _send(self, sock: socket.socket, data: dict[str, Any]) -> None:
        payload = numpy_to_json(data).encode("utf-8")
        sock.sendall(len(payload).to_bytes(4, "big"))
        sock.sendall(payload)

    def _handle_client(self, sock: socket.socket) -> None:
        with sock:
            while self.running:
                try:
                    header = sock.recv(4)
                    if not header:
                        return
                    payload = _recv_exact(sock, int.from_bytes(header, "big"))
                    request = json_to_numpy(payload.decode("utf-8"))
                    cmd = request.get("cmd")
                    obs = request.get("obs")
                    if not isinstance(cmd, str):
                        raise ValueError(f"request cmd must be a string, got {cmd!r}")
                    method = getattr(self.model, cmd, None)
                    if not callable(method):
                        raise AttributeError(f"no model method named {cmd!r}")
                    t0 = time.perf_counter()
                    result = method(obs) if obs is not None else method()
                    dt = time.perf_counter() - t0
                    self._send(sock, {"res": result, "server_timing": {"cmd": cmd, "seconds": dt}})
                except (BrokenPipeError, ConnectionResetError):
                    return
                except Exception as exc:
                    self._send(
                        sock,
                        {
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        },
                    )
                    return


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _standardize_rgb(image: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    import cv2

    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB image with 3 channels, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    target_h, target_w = size_hw
    if arr.shape[:2] != (target_h, target_w):
        arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return arr


def _get_nested(data: Mapping[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            raise KeyError(".".join(keys))
        cur = cur[key]
    return cur


def _first_existing(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    raise KeyError(f"none of {keys!r} exists")


def _extract_color(obs: Mapping[str, Any], keys: tuple[str, ...]) -> np.ndarray:
    vision = _get_nested(obs, "vision")
    cam = _first_existing(vision, keys)
    if isinstance(cam, Mapping):
        if "color" in cam:
            return np.asarray(cam["color"])
        if "rgb" in cam:
            return np.asarray(cam["rgb"])
    return np.asarray(cam)


def pack_joint_state(obs: Mapping[str, Any]) -> np.ndarray:
    state = _get_nested(obs, "state")
    parts = [
        np.asarray(state["left_arm_joint_state"], dtype=np.float32),
        np.asarray(state["left_ee_joint_state"], dtype=np.float32),
        np.asarray(state["right_arm_joint_state"], dtype=np.float32),
        np.asarray(state["right_ee_joint_state"], dtype=np.float32),
    ]
    expected = [6, 1, 6, 1]
    for value, dim in zip(parts, expected):
        if value.shape[-1] != dim:
            raise ValueError(f"state dim mismatch: expected {dim}, got {value.shape}")
    return np.concatenate(parts, axis=-1).astype(np.float32)


def unpack_joint_action(action: np.ndarray) -> dict[str, np.ndarray] | list[dict[str, np.ndarray]]:
    arr = np.asarray(action, dtype=np.float32)
    if arr.shape[-1] != 14:
        raise ValueError(f"action last dim must be 14, got {arr.shape}")

    def one(row: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "left_arm_joint_state": row[0:6].astype(np.float32),
            "left_ee_joint_state": row[6:7].astype(np.float32),
            "right_arm_joint_state": row[7:13].astype(np.float32),
            "right_ee_joint_state": row[13:14].astype(np.float32),
        }

    if arr.ndim == 1:
        return one(arr)
    if arr.ndim == 2:
        return [one(row) for row in arr]
    raise ValueError(f"action must be 1D or 2D, got {arr.shape}")


def get_instruction(obs: Mapping[str, Any], fallback: str) -> str:
    for key in ("task_instruction", "instruction", "instructions"):
        if key not in obs:
            continue
        value = obs[key]
        if isinstance(value, (list, tuple)):
            value = value[0] if value else fallback
        if hasattr(value, "item"):
            value = value.item()
        text = str(value).strip()
        if text:
            return text
    return fallback


class LightEWMRoboDojoFastWAMHDRPolicy:
    def __init__(self, args: argparse.Namespace) -> None:
        print(f"[policy] init start port={args.port} pid={os.getpid()}", flush=True)
        self.args = args
        self.lock = threading.Lock()
        self.default_instruction = args.default_instruction
        self.replan_steps = int(args.replan_steps)
        self.action_horizon = int(args.action_horizon)
        self.num_video_frames = int(args.num_video_frames)
        self.num_inference_steps = int(args.num_inference_steps)
        self.seed = None if args.seed < 0 else int(args.seed)
        self.allow_dummy_policy = bool(args.allow_dummy_policy)
        self._last_obs: dict[str, Any] | None = None
        self._last_instruction = self.default_instruction
        self._batch_obs: dict[int, dict[str, Any]] = {}
        self._batch_instruction: dict[int, str] = {}
        self._episode_first_caches: dict[int, dict[str, Any]] = {}

        if self.allow_dummy_policy:
            print("[policy] allow-dummy-policy enabled; model loading skipped", flush=True)
            self.model = None
            self.processor = None
            return

        _insert_paths()
        self._load_model()

    def _load_model(self) -> None:
        t_total = time.perf_counter()
        print("[policy] importing torch/hydra/model deps", flush=True)
        import torch
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
        from fastwam.utils.config_resolvers import register_default_resolvers
        from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

        print(f"[policy] imports done in {time.perf_counter() - t_total:.2f}s", flush=True)
        register_default_resolvers()
        self.default_prompt_template = DEFAULT_PROMPT
        checkpoint = Path(self.args.checkpoint).expanduser().resolve()
        stats = Path(self.args.dataset_stats).expanduser().resolve()
        print(f"[policy] checkpoint={checkpoint}", flush=True)
        print(f"[policy] dataset_stats={stats}", flush=True)
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        if not stats.exists():
            raise FileNotFoundError(f"dataset stats not found: {stats}")

        overrides = [
            "task=robotwin_uncond_3cam_384_1e-4",
            "model=fastwam_joint",
            "data=robotwin",
            "model.redirect_common_files=false",
            "model.mot_checkpoint_mixed_attn=false",
            "model.action_attend_video=local_clean_first",
            "model.hdr_mode=episode_back_mixed",
            "model.load_text_encoder=true",
            f"model.video_dit_pretrained_path={ROOT / 'checkpoints/Wan2.2-5B-Robot/checkpoint.safetensors'}",
            f"model.action_dit_pretrained_path={ROOT / 'checkpoints/ActionDiT_linear_interp_Wan22Robot_alphascale_1024hdim.pt'}",
            "model.model_id=Wan-AI/Wan2.2-TI2V-5B",
            "model.tokenizer_model_id=Wan-AI/Wan2.1-T2V-1.3B",
            f"data.train.pretrained_norm_stats={stats}",
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
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        t0 = time.perf_counter()
        print("[policy] composing hydra config", flush=True)
        with initialize_config_dir(version_base="1.3", config_dir=str((FASTWAM_ROOT / "configs").resolve())):
            cfg = compose(config_name="train", overrides=overrides)
        cfg_model = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        print(f"[policy] hydra config ready in {time.perf_counter() - t0:.2f}s", flush=True)
        dtype = torch.bfloat16 if self.args.mixed_precision == "bf16" else torch.float16 if self.args.mixed_precision == "fp16" else torch.float32
        device = self.args.device
        print("[policy] checking CUDA availability", flush=True)
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[policy] CUDA unavailable; falling back to cpu", flush=True)
            device = "cpu"
        print(f"[policy] loading model on {device}: {checkpoint}", flush=True)
        t0 = time.perf_counter()
        self.model = instantiate(cfg_model, model_dtype=dtype, device=device)
        print(f"[policy] instantiate done in {time.perf_counter() - t0:.2f}s", flush=True)
        t0 = time.perf_counter()
        self.model.load_checkpoint(str(checkpoint))
        print(f"[policy] checkpoint loaded in {time.perf_counter() - t0:.2f}s", flush=True)
        t0 = time.perf_counter()
        self.model = self.model.to(device).eval()
        print(f"[policy] model.to/eval done in {time.perf_counter() - t0:.2f}s", flush=True)
        t0 = time.perf_counter()
        self.processor = instantiate(cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats)))
        print(f"[policy] processor ready in {time.perf_counter() - t0:.2f}s", flush=True)
        print(f"[policy] model ready total={time.perf_counter() - t_total:.2f}s", flush=True)

    def _encode_obs(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        head = _standardize_rgb(_extract_color(obs, ("cam_head", "cam_high", "head_camera")), (240, 320))
        left = _standardize_rgb(_extract_color(obs, ("cam_left_wrist", "left_camera")), (240, 320))
        right = _standardize_rgb(_extract_color(obs, ("cam_right_wrist", "right_camera")), (240, 320))
        return {
            "head": head,
            "left": left,
            "right": right,
            "state": pack_joint_state(obs),
        }

    def update_obs(self, obs: Mapping[str, Any]) -> None:
        with self.lock:
            self._last_obs = self._encode_obs(obs)
            self._last_instruction = get_instruction(obs, self.default_instruction)

    def update_obs_batch(self, obs_list: list[Mapping[str, Any]]) -> None:
        if not obs_list:
            raise ValueError("update_obs_batch received empty obs_list")
        with self.lock:
            self._batch_obs = {}
            self._batch_instruction = {}
            for i, obs in enumerate(obs_list):
                env_idx = int(obs.get("env_idx", i))
                self._batch_obs[env_idx] = self._encode_obs(obs)
                self._batch_instruction[env_idx] = get_instruction(obs, self.default_instruction)

    def get_action(self) -> list[dict[str, np.ndarray]]:
        with self.lock:
            if self._last_obs is None:
                raise ValueError("call update_obs before get_action")
            chunk = self._infer_action_chunk_batch([self._last_obs], [self._last_instruction], [0])[0]
            return unpack_joint_action(chunk[: self.replan_steps])  # type: ignore[return-value]

    def get_action_batch(self, env_idx_list: list[int]) -> list[list[dict[str, np.ndarray]]]:
        if env_idx_list is None:
            env_idx_list = sorted(self._batch_obs)
        with self.lock:
            encoded = [self._batch_obs[int(env_idx)] for env_idx in env_idx_list]
            instructions = [self._batch_instruction[int(env_idx)] for env_idx in env_idx_list]
            chunks = self._infer_action_chunk_batch(encoded, instructions, [int(env_idx) for env_idx in env_idx_list])
            return [
                unpack_joint_action(chunk[: self.replan_steps])  # type: ignore[list-item]
                for chunk in chunks
            ]

    def _reset_episode_state_locked(self, env_indices: list[int] | None = None) -> None:
        if env_indices is None:
            self._last_obs = None
            self._last_instruction = self.default_instruction
            self._batch_obs = {}
            self._batch_instruction = {}
            self._episode_first_caches = {}
            return
        for env_idx in env_indices:
            self._batch_obs.pop(int(env_idx), None)
            self._batch_instruction.pop(int(env_idx), None)
            self._episode_first_caches.pop(int(env_idx), None)
        if 0 in set(int(x) for x in env_indices):
            self._last_obs = None
            self._last_instruction = self.default_instruction

    def _parse_reset_env_indices(self, request: Any | None) -> list[int] | None:
        if request is None:
            return None
        if isinstance(request, Mapping):
            if _is_true(request.get("all", False)):
                return None
            for key in ("env_indices", "env_idx_list", "env_ids"):
                if key in request:
                    return [int(x) for x in request[key]]
            if "env_idx" in request:
                return [int(request["env_idx"])]
        if isinstance(request, (list, tuple)):
            return [int(x) for x in request]
        return [int(request)]

    def reset(self, request: Any | None = None) -> None:
        env_indices = self._parse_reset_env_indices(request)
        with self.lock:
            self._reset_episode_state_locked(env_indices)
        target = "all" if env_indices is None else ",".join(str(x) for x in env_indices)
        print(f"[policy] reset: cleared episode-first caches env_idx={target}", flush=True)

    def _zero_chunk(self, batch_size: int) -> np.ndarray:
        return np.zeros((batch_size, self.action_horizon, 14), dtype=np.float32)

    def _build_image_batch(self, obs_batch: list[dict[str, Any]]) -> Any:
        import torch

        frames = []
        for obs in obs_batch:
            head = _standardize_rgb(obs["head"], (256, 320))
            left = _standardize_rgb(obs["left"], (128, 160))
            right = _standardize_rgb(obs["right"], (128, 160))
            image = np.concatenate([head, np.concatenate([left, right], axis=1)], axis=0)
            frames.append(torch.from_numpy(image).permute(2, 0, 1))
        image_tensor = torch.stack(frames, dim=0).to(device=self.model.device, dtype=self.model.torch_dtype)
        return image_tensor * (2.0 / 255.0) - 1.0

    def _normalize_state_batch(self, obs_batch: list[dict[str, Any]]) -> Any:
        import torch

        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("expected exactly one state key")
        key = state_meta[0]["key"]
        states = np.stack([np.asarray(obs["state"], dtype=np.float32) for obs in obs_batch], axis=0)
        batch = {"state": {key: torch.as_tensor(states, dtype=torch.float32)}}
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        return batch["state"][key]

    def _denormalize_action_batch(self, action: Any) -> np.ndarray:
        import torch

        if action.ndim != 3:
            raise ValueError(f"expected action [B,T,D], got {tuple(action.shape)}")
        meta = self.processor.shape_meta["action"]
        if len(meta) != 1:
            raise ValueError("expected exactly one action key")
        key = meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy().astype(np.float32)

    def _encode_first_frame_latents_batch(self, image_batch: Any) -> Any:
        latents = []
        for image in image_batch:
            latents.append(self.model._encode_input_image_latents_tensor(image.unsqueeze(0), tiled=self.args.tiled))
        return __import__("torch").cat(latents, dim=0)

    def _episode_context(self, instruction: str, proprio: Any) -> tuple[Any, Any]:
        prompt = self.default_prompt_template.format(task=instruction)
        context, context_mask = self.model.encode_prompt([prompt])
        return self.model._append_proprio_to_context(context, context_mask, proprio)

    def _get_episode_first_latents(self, env_idx: int, obs: dict[str, Any]) -> Any:
        if env_idx not in self._episode_first_caches:
            episode_first_latents = self._encode_first_frame_latents_batch(self._build_image_batch([obs])).detach()
            self._episode_first_caches[env_idx] = {
                "episode_first_latents": episode_first_latents,
            }
            print(f"[policy] episode-first latent cached env_idx={env_idx}", flush=True)
        return self._episode_first_caches[env_idx]["episode_first_latents"]

    def _prefill_episode_video_cache(
        self, env_idx: int, obs: dict[str, Any], instruction: str, proprio: Any
    ) -> dict[str, Any]:
        import torch

        episode_context, episode_context_mask = self._episode_context(instruction, proprio)
        episode_first_latents = self._get_episode_first_latents(env_idx, obs)
        local_first_latents = self._encode_first_frame_latents_batch(self._build_image_batch([obs]))
        latents_video = torch.cat([episode_first_latents, local_first_latents], dim=2)
        timestep_video = torch.zeros((1,), dtype=latents_video.dtype, device=self.model.device)
        fuse_flag = bool(getattr(self.model.video_expert, "fuse_vae_embedding_in_latents", False))
        video_pre = self.model.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=episode_context,
            context_mask=episode_context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
            clean_latent_indices=torch.arange(2, dtype=torch.long, device=self.model.device),
        )
        if hasattr(self.model, "_episode_first_hdr_action_only_video_freqs"):
            video_pre["freqs"] = self.model._episode_first_hdr_action_only_video_freqs(video_pre)
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self.model._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=self.action_horizon,
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        return {
            "video_kv_cache": self.model.mot.prefill_video_cache(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={"context": video_pre["context"], "mask": video_pre["context_mask"]},
                video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
            ),
            "attention_mask": attention_mask,
            "video_seq_len": video_seq_len,
        }

    def _infer_action_chunk_one(
        self, obs: dict[str, Any], instruction: str, env_idx: int
    ) -> np.ndarray:
        if self.allow_dummy_policy:
            return self._zero_chunk(1)[0]
        import torch

        proprio = self._normalize_state_batch([obs]).to(device=self.model.device, dtype=self.model.torch_dtype)
        cache = self._prefill_episode_video_cache(env_idx, obs, instruction, proprio)
        context, context_mask = self._episode_context(instruction, proprio)
        generator = None if self.seed is None else torch.Generator(device=self.args.rand_device).manual_seed(self.seed)
        latents_action = torch.randn(
            (1, self.action_horizon, self.model.action_expert.action_dim),
            generator=generator,
            device=self.args.rand_device,
            dtype=torch.float32,
        ).to(device=self.model.device, dtype=self.model.torch_dtype)
        infer_timesteps_action, infer_deltas_action = self.model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=self.num_inference_steps,
            device=self.model.device,
            dtype=latents_action.dtype,
            shift_override=self.args.sigma_shift,
        )
        with torch.no_grad():
            for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
                timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.model.device)
                pred_action = self.model._predict_action_noise_with_cache(
                    latents_action=latents_action,
                    timestep_action=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    video_kv_cache=cache["video_kv_cache"],
                    attention_mask=cache["attention_mask"],
                    video_seq_len=cache["video_seq_len"],
                )
                latents_action = self.model.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
        return self._denormalize_action_batch(latents_action.detach().cpu())[0]

    def _infer_action_chunk_batch(
        self,
        obs_batch: list[dict[str, Any]],
        instructions: list[str],
        env_idx_list: list[int],
    ) -> np.ndarray:
        if not (len(obs_batch) == len(instructions) == len(env_idx_list)):
            raise ValueError("obs, instruction, and env_idx batch sizes must match")
        chunks = [
            self._infer_action_chunk_one(obs, instruction, env_idx)
            for obs, instruction, env_idx in zip(obs_batch, instructions, env_idx_list, strict=True)
        ]
        return np.stack(chunks, axis=0)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-stats", default=str(ROOT / "data/robodojo_fastwam/robodojo-v21-video/dataset_stats.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--replan-steps", type=int, default=24)
    parser.add_argument("--num-video-frames", type=int, default=13)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--sigma-shift", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42, help="negative disables deterministic sampling")
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--default-instruction", default="follow the instruction")
    parser.add_argument("--allow-dummy-policy", action="store_true")
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = latest_robodojo_checkpoint()
    return args


def main() -> None:
    args = parse_args()
    model = LightEWMRoboDojoFastWAMHDRPolicy(args)
    server = ModelServer(model, host=args.host, port=args.port)
    try:
        server.start()
    except KeyboardInterrupt:
        print("[server] interrupted", flush=True)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
