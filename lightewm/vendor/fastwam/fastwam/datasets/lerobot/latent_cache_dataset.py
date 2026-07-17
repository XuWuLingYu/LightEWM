import hashlib
import os
import time
from pathlib import Path
from typing import Optional

import torch

from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


class FastWAMLatentCacheDataset(torch.utils.data.Dataset):
    """Read pre-encoded FastWAM cache payloads directly.

    This is used for mixed-source training where raw datasets may have different
    fps or schema. The precache stage is responsible for normalizing video layout,
    action/proprio tensors, adapter metadata, and held-out indices.
    """

    def __init__(
        self,
        latent_cache_dir: str,
        index_file: Optional[str] = None,
        text_embedding_cache_dir: Optional[str] = None,
        context_len: int = 128,
        max_samples: Optional[int] = None,
        require_video: bool = False,
        **_ignored,
    ):
        self.latent_cache_dir = Path(latent_cache_dir)
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = int(context_len)
        self.require_video = bool(require_video)
        if index_file is not None:
            path = Path(index_file)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(payload, dict):
                indices = payload.get("indices") or payload.get("train_indices") or payload.get("val_indices")
            else:
                indices = payload
            if indices is None:
                raise ValueError(f"No indices found in {path}")
            self.indices = [int(x) for x in indices]
        else:
            meta_path = self.latent_cache_dir / "metadata.pt"
            if not meta_path.exists():
                raise FileNotFoundError(f"Missing latent cache metadata: {meta_path}")
            meta = torch.load(meta_path, map_location="cpu", weights_only=False)
            self.indices = list(range(int(meta["num_samples"])))
        if max_samples is not None:
            self.indices = self.indices[: max(int(max_samples), 0)]

    def __len__(self):
        return len(self.indices)

    def _cache_path(self, idx: int) -> Path:
        shard = int(idx) // 10000
        return self.latent_cache_dir / f"shard_{shard:05d}" / f"{int(idx):08d}.pt"

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is required when cache payload lacks context/context_mask.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Missing text embedding cache: {cache_path}")
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        context = payload["context"]
        context_mask = payload["mask"].bool()
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        return context, context_mask

    def _load_payload(self, logical_idx: int):
        cache_path = self._cache_path(logical_idx)
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing FastWAM latent cache file: {cache_path}")
        for attempt in range(5):
            try:
                return torch.load(cache_path, map_location="cpu", weights_only=False)
            except OSError as exc:
                if attempt == 4:
                    raise
                wait_s = min(2.0 * (attempt + 1), 8.0)
                logger.warning("Failed to load %s (%s); retrying in %.1fs", cache_path, exc, wait_s)
                time.sleep(wait_s)

    def _payload_to_sample(self, payload: dict):
        prompt = payload["prompt"]
        if "context" in payload and "context_mask" in payload:
            context = payload["context"]
            context_mask = payload["context_mask"]
        else:
            context, context_mask = self._get_cached_text_context(prompt)

        hdr_mode = str(payload.get("hdr_mode", "back_hdr"))
        if hdr_mode in {"episode_first_hdr", "episode_back_mixed", "episode_first_special_latent"}:
            episode_latents = payload["episode_latents"]
            local_latents = payload["local_latents"]
            input_latents = torch.cat([episode_latents, local_latents], dim=1).contiguous()
            clean_latent_indices = torch.tensor([0, int(episode_latents.shape[1])], dtype=torch.long)
            first_frame_latents = None
        else:
            input_latents = payload["input_latents"]
            clean_latent_indices = None
            first_frame_latents = payload.get("first_frame_latents")

        action = payload["action"]
        sample = {
            "input_latents": input_latents,
            "action": action,
            "action_is_pad": payload["action_is_pad"],
            "action_dim_is_pad": payload.get("action_dim_is_pad", torch.zeros(action.shape[-1], dtype=torch.bool)),
            "proprio": payload["proprio"],
            "proprio_is_pad": payload["proprio_is_pad"],
            "image_is_pad": payload["image_is_pad"],
            "prompt": prompt,
            "context": context,
            "context_mask": context_mask,
            "num_video_frames": int(payload["num_video_frames"]),
            "hdr_mode": hdr_mode,
        }
        if first_frame_latents is not None:
            sample["first_frame_latents"] = first_frame_latents
        if clean_latent_indices is not None:
            sample["clean_latent_indices"] = clean_latent_indices
            sample["episode_video_latent_indices"] = torch.arange(1, int(payload["episode_latents"].shape[1]), dtype=torch.long)
            local_start = int(payload["episode_latents"].shape[1])
            sample["local_video_latent_indices"] = torch.arange(local_start + 1, local_start + int(payload["local_latents"].shape[1]), dtype=torch.long)
        for key in (
            "local_video_frames",
            "action_video_transition_count",
            "hdr_tree_frame_indices",
            "hdr_local_frame_indices",
            "hdr_episode_frame_indices",
            "hdr_episode_latent_indices",
            "hdr_episode_latent_source_frame_indices",
            "hdr_episode_latent_padded_frames",
            "action_adapter",
            "source_name",
            "source_episode_index",
        ):
            if key in payload:
                sample[key] = payload[key]
        for key in ("video", "episode_video", "tree_video", "tree_image_is_pad"):
            if key in payload:
                sample[key] = payload[key]
        if self.require_video and "video" not in sample:
            raise KeyError("Cache payload lacks `video` but require_video=True")
        return sample

    def _get(self, idx: int):
        return self._payload_to_sample(self._load_payload(self.indices[int(idx)]))

    def __getitem__(self, idx: int):
        return self._get(idx)
