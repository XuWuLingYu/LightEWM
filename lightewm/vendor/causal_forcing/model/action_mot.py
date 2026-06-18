import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from wan.modules.attention import attention
from wan.modules.causal_model import causal_rope_apply, flex_attention
from wan.modules.model import sinusoidal_embedding_1d
from torch.nn.attention.flex_attention import create_block_mask
from utils.vertical_hierarchy import CONDITION_TOKEN_ID, get_vertical_allowed_token_ids


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def _rope_1d(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    x_complex = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(x_complex * freqs.to(x.device)).flatten(3)
    return out.to(x.dtype)


def _precompute_1d_freqs(head_dim: int, end: int = 2048, theta: float = 10000.0) -> torch.Tensor:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}.")
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    freqs = torch.outer(torch.arange(end, dtype=torch.float64), freqs)
    return torch.polar(torch.ones_like(freqs), freqs).view(end, 1, -1)


def _interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def _resize_tensor_to_shape(src: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(src.shape) == tuple(target_shape):
        return src

    out = src.to(torch.float32)
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(
                f"Cannot reduce tensor rank for resize: src shape={tuple(src.shape)}, target={target_shape}."
            )
        out = out.squeeze(0)

    for dim, new_size in enumerate(target_shape):
        if out.shape[dim] == new_size:
            continue
        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        inv_perm = [0] * out.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        prefix_shape = out.permute(*perm).contiguous().shape[:-1]
        out_perm = _interpolate_last_dim(out.permute(*perm).contiguous(), new_size)
        out = out_perm.reshape(*prefix_shape, new_size).permute(*inv_perm).contiguous()

    if tuple(out.shape) != tuple(target_shape):
        raise ValueError(
            f"Resize produced wrong shape: src={tuple(src.shape)}, target={target_shape}, got={tuple(out.shape)}."
        )
    return out.to(dtype=src.dtype)


class ActionSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim
        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)


class ActionCrossAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim
        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, s = x.shape[:2]
        q = self.norm_q(self.q(x)).view(b, s, self.num_heads, self.attn_head_dim)
        k = self.norm_k(self.k(context)).view(b, context.shape[1], self.num_heads, self.attn_head_dim)
        v = self.v(context).view(b, context.shape[1], self.num_heads, self.attn_head_dim)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        ).transpose(1, 2)
        return self.o(out.flatten(2))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class GateModule(nn.Module):
    def forward(self, x: torch.Tensor, gate: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return x + gate * residual


class ActionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        attn_head_dim: int,
        num_heads: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_head_dim = attn_head_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.self_attn = ActionSelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.cross_attn = ActionCrossAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)
        self.gate = GateModule()


class HDRActionMoT(nn.Module):
    """FastWAM-style action expert that attends same-layer frozen video K/V."""

    ACTION_BACKBONE_SKIP_PREFIXES = ("action_encoder.", "head.", "proprio_encoder.")

    def __init__(
        self,
        *,
        video_model,
        action_dim: int = 7,
        hidden_dim: int = 1024,
        ffn_dim: int = 4096,
        freq_dim: int = 256,
        eps: float = 1e-6,
        actions_per_leaf: int = 8,
        action_attend_video: str = "parents",
        use_gradient_checkpointing: bool = False,
        proprio_dim: int | None = None,
    ):
        super().__init__()
        object.__setattr__(self, "_video_model", video_model)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.ffn_dim = int(ffn_dim)
        self.freq_dim = int(freq_dim)
        self.actions_per_leaf = int(actions_per_leaf)
        self.action_attend_video = str(action_attend_video)
        self.proprio_dim = None if proprio_dim is None or int(proprio_dim) <= 0 else int(proprio_dim)
        self.disable_action_text_cross_attn = False
        if self.action_attend_video not in {"none", "parents", "all"}:
            raise ValueError("action_attend_video must be one of: none, parents, all.")
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

        self.video_dim = int(video_model.dim)
        self.num_heads = int(video_model.num_heads)
        self.attn_dim = self.video_dim
        if self.attn_dim % self.num_heads != 0:
            raise ValueError(f"video dim={self.attn_dim} must be divisible by num_heads={self.num_heads}.")
        self.attn_head_dim = self.attn_dim // self.num_heads
        if self.attn_head_dim % 2 != 0:
            raise ValueError(f"attn_head_dim={self.attn_head_dim} must be even for RoPE.")

        self.action_encoder = nn.Linear(self.action_dim, self.hidden_dim)
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim * 6),
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(int(video_model.text_dim), self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.proprio_encoder = (
            nn.Linear(self.proprio_dim, int(video_model.text_dim))
            if self.proprio_dim is not None
            else None
        )
        self.blocks = nn.ModuleList([
            ActionBlock(
                hidden_dim=self.hidden_dim,
                ffn_dim=self.ffn_dim,
                attn_head_dim=self.attn_head_dim,
                num_heads=self.num_heads,
                eps=eps,
            )
            for _ in range(len(video_model.blocks))
        ])
        self.head = nn.Linear(self.hidden_dim, self.action_dim)
        self._action_freqs_cpu = _precompute_1d_freqs(self.attn_head_dim)
        self._action_freqs_cache: dict[torch.device, torch.Tensor] = {}

        if len(self.blocks) != len(video_model.blocks):
            raise ValueError("Action and video experts must have the same number of layers for MoT.")

    def _append_proprio_to_prompt_embeds(
        self,
        conditional_dict: dict,
        proprio: torch.Tensor | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict:
        if self.proprio_encoder is None or proprio is None:
            return conditional_dict
        if proprio.ndim != 2:
            raise ValueError(f"`joint_proprio` must be [B, D], got {tuple(proprio.shape)}.")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(f"`joint_proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}.")
        prompt_embeds = conditional_dict["prompt_embeds"].to(device=device, dtype=dtype)
        proprio_token = self.proprio_encoder(
            proprio.to(device=device, dtype=dtype)
        ).to(dtype=prompt_embeds.dtype).unsqueeze(1)
        with_proprio = dict(conditional_dict)
        with_proprio["prompt_embeds"] = torch.cat([prompt_embeds, proprio_token], dim=1)
        return with_proprio

    @classmethod
    def backbone_key_set(cls, keys) -> set[str]:
        return {
            key
            for key in keys
            if not any(key.startswith(prefix) for prefix in cls.ACTION_BACKBONE_SKIP_PREFIXES)
        }

    def build_interpolated_video_backbone_state_dict(
        self,
        *,
        apply_alpha_scaling: bool = True,
    ) -> tuple[dict[str, torch.Tensor], dict[str, int | bool | str]]:
        """Build a FastWAM-style ActionDiT backbone from the bound video expert.

        The action input encoder and output head stay randomly initialized,
        matching FastWAM's preprocessing policy.
        """
        action_state = self.state_dict()
        video_state = self.video_model.state_dict()
        backbone_keys = self.backbone_key_set(action_state.keys())

        backbone_state_dict: dict[str, torch.Tensor] = {}
        copied = 0
        interpolated = 0
        for key in sorted(backbone_keys):
            if key not in video_state:
                raise ValueError(f"Cannot initialize action backbone: video expert has no key `{key}`.")
            src = video_state[key]
            target = action_state[key]
            if tuple(src.shape) == tuple(target.shape):
                value = src
                copied += 1
            else:
                value = _resize_tensor_to_shape(src, tuple(target.shape))
                if apply_alpha_scaling and src.ndim >= 2 and src.shape[-1] != target.shape[-1]:
                    value = value.to(torch.float32) * (float(src.shape[-1]) / float(target.shape[-1])) ** 0.5
                interpolated += 1
            backbone_state_dict[key] = value.detach().to(dtype=target.dtype, device="cpu").contiguous()

        summary = {
            "copied": copied,
            "interpolated": interpolated,
            "skipped": len(action_state) - len(backbone_keys),
            "alpha_scaling": bool(apply_alpha_scaling),
            "interpolation": "sequential_1d_linear_align_corners_true",
        }
        return backbone_state_dict, summary

    @property
    def video_model(self):
        return self._video_model

    def _action_freqs(self, device: torch.device, length: int) -> torch.Tensor:
        freqs = self._action_freqs_cache.get(device)
        if freqs is None:
            freqs = self._action_freqs_cpu.to(device=device)
            self._action_freqs_cache[device] = freqs
        if length > freqs.shape[0]:
            freqs = _precompute_1d_freqs(self.attn_head_dim, end=length * 2).to(device=device)
            self._action_freqs_cache[device] = freqs
        return freqs[:length]

    def _action_leaf_local_freqs(self, device: torch.device, length: int) -> torch.Tensor:
        if length <= 0:
            raise ValueError(f"Action sequence length must be positive, got {length}.")
        base_freqs = self._action_freqs(device, max(self.actions_per_leaf, 1))
        local_positions = torch.arange(length, device=device) % self.actions_per_leaf
        return base_freqs.index_select(0, local_positions)

    def bind_video_model(self, video_model) -> None:
        object.__setattr__(self, "_video_model", video_model)

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _prepare_video_state(
        self,
        video_latents: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict,
        *,
        prefix_x: torch.Tensor,
        prefix_t: torch.Tensor,
        prefix_token_ids: list[int],
        noisy_token_ids: list[int],
        vertical_info: dict,
        vertical_use_representative_rope: bool,
        seq_len_override: int,
    ) -> dict[str, Any]:
        video = self.video_model
        device = video.patch_embedding.weight.device
        if video.freqs.device != device:
            video.freqs = video.freqs.to(device)

        x = video_latents.permute(0, 2, 1, 3, 4)
        has_prefix = len(prefix_token_ids) > 0
        if has_prefix:
            prefix = prefix_x.permute(0, 2, 1, 3, 4)
        frame_seqlen = x.shape[-2] * x.shape[-1] // (video.patch_size[1] * video.patch_size[2])
        block_mask_key = (
            "vertical_action_mot",
            frame_seqlen,
            tuple(prefix_token_ids),
            tuple(noisy_token_ids),
            bool(vertical_info.get("allow_condition_for_all_frames", False)),
        )
        if video.block_mask is None or video.block_mask_key != block_mask_key:
            video.block_mask = video._prepare_vertical_attn_mask(
                device=device,
                frame_seqlen=frame_seqlen,
                sequence_token_ids=list(prefix_token_ids) + list(noisy_token_ids),
                token_to_parent=vertical_info["token_to_parent"],
                token_to_level=vertical_info["token_to_level"],
                token_to_level_pos=vertical_info["token_to_level_pos"],
                level_offsets=vertical_info["level_offsets"],
                level_sizes=vertical_info["level_sizes"],
                allow_condition_for_all_frames=bool(vertical_info.get("allow_condition_for_all_frames", False)),
            )
            video.block_mask_key = block_mask_key

        x = [video.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long, device=device) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long, device=device)
        assert seq_lens.max() <= seq_len_override
        max_seq_len = seq_lens.max().item()
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, max_seq_len - u.size(1), u.size(2))], dim=1)
            for u in x
        ])

        e = video.time_embedding(sinusoidal_embedding_1d(video.freq_dim, timestep.flatten()).type_as(x))
        head_e = e.unflatten(dim=0, sizes=timestep.shape)
        e0 = video.time_projection(e).unflatten(1, (6, video.dim)).unflatten(dim=0, sizes=timestep.shape)

        context = conditional_dict["prompt_embeds"].to(device=device, dtype=x.dtype)
        context = video.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(video.text_len - u.size(0), u.size(1))])
                for u in context
            ])
        )

        if has_prefix:
            prefix = [video.patch_embedding(u.unsqueeze(0)) for u in prefix]
            prefix = [u.flatten(2).transpose(1, 2) for u in prefix]
            seq_lens_prefix = torch.tensor([u.size(1) for u in prefix], dtype=torch.long, device=device)
            assert seq_lens_prefix.max() <= seq_len_override
            max_seq_len_prefix = seq_lens_prefix.max().item()
            prefix = torch.cat([
                torch.cat([u, u.new_zeros(1, max_seq_len_prefix - u.size(1), u.size(2))], dim=1)
                for u in prefix
            ])
            x = torch.cat([prefix, x], dim=1)

            e_prefix = video.time_embedding(sinusoidal_embedding_1d(video.freq_dim, prefix_t.flatten()).type_as(x))
            head_e_prefix = e_prefix.unflatten(dim=0, sizes=prefix_t.shape)
            e0_prefix = video.time_projection(e_prefix).unflatten(1, (6, video.dim)).unflatten(dim=0, sizes=prefix_t.shape)
            e0 = torch.cat([e0_prefix, e0], dim=1)
            head_e = torch.cat([head_e_prefix, head_e], dim=1)

        attn_grid_sizes = grid_sizes.clone()
        attn_grid_sizes[:, 0] += len(prefix_token_ids)

        temporal_positions = None
        if vertical_use_representative_rope:
            sequence_temporal_positions = []
            for token_id in list(prefix_token_ids) + list(noisy_token_ids):
                if token_id < 0:
                    sequence_temporal_positions.append(0)
                else:
                    sequence_temporal_positions.append(vertical_info["representative_indices"][token_id])
            temporal_positions = torch.tensor(
                sequence_temporal_positions,
                device=device,
                dtype=torch.long,
            ).unsqueeze(0).expand(timestep.shape[0], -1)

        return {
            "tokens": x,
            "e": e0,
            "head_e": head_e,
            "seq_lens": seq_lens,
            "grid_sizes": attn_grid_sizes,
            "freqs": video.freqs,
            "context": context,
            "context_lens": None,
            "block_mask": video.block_mask,
            "temporal_positions": temporal_positions,
            "prefix_seq_len": len(prefix_token_ids) * frame_seqlen,
        }

    def _prepare_joint_video_state(
        self,
        video_latents: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict,
        *,
        prefix_x: torch.Tensor,
        prefix_t: torch.Tensor,
        prefix_token_ids: list[int],
        tree_token_ids: list[int],
        vertical_info: dict,
        vertical_use_representative_rope: bool,
        local_start_count: int,
        local_video_count: int,
        seq_len_override: int,
        local_conditional_dict: dict | None = None,
    ) -> dict[str, Any]:
        video = self.video_model
        device = video.patch_embedding.weight.device
        if video.freqs.device != device:
            video.freqs = video.freqs.to(device)

        x = video_latents.permute(0, 2, 1, 3, 4)
        has_prefix = len(prefix_token_ids) > 0
        if has_prefix:
            prefix = prefix_x.permute(0, 2, 1, 3, 4)
        frame_seqlen = x.shape[-2] * x.shape[-1] // (video.patch_size[1] * video.patch_size[2])
        block_mask_key = (
            "video_action_joint",
            frame_seqlen,
            tuple(prefix_token_ids),
            tuple(tree_token_ids),
            int(local_start_count),
            int(local_video_count),
            bool(vertical_info.get("allow_condition_for_all_frames", False)),
        )
        if video.block_mask is None or video.block_mask_key != block_mask_key:
            video.block_mask = self._joint_video_mask(
                frame_seqlen=frame_seqlen,
                prefix_token_ids=prefix_token_ids,
                tree_token_ids=tree_token_ids,
                local_start_count=local_start_count,
                local_video_count=local_video_count,
                vertical_info=vertical_info,
                device=device,
            )
            video.block_mask_key = block_mask_key

        x = [video.patch_embedding(u.unsqueeze(0)) for u in x]
        latent_grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long, device=device) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long, device=device)
        assert seq_lens.max() <= seq_len_override
        max_seq_len = seq_lens.max().item()
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, max_seq_len - u.size(1), u.size(2))], dim=1)
            for u in x
        ])

        e = video.time_embedding(sinusoidal_embedding_1d(video.freq_dim, timestep.flatten()).type_as(x))
        head_e = e.unflatten(dim=0, sizes=timestep.shape)
        e0 = video.time_projection(e).unflatten(1, (6, video.dim)).unflatten(dim=0, sizes=timestep.shape)

        context = conditional_dict["prompt_embeds"].to(device=device, dtype=x.dtype)
        context = video.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(video.text_len - u.size(0), u.size(1))])
                for u in context
            ])
        )
        local_context = None
        if local_conditional_dict is not None:
            raw_local_context = local_conditional_dict["prompt_embeds"].to(device=device, dtype=x.dtype)
            local_text_len = max(int(video.text_len), int(raw_local_context.shape[1]))
            local_context = video.text_embedding(
                torch.stack([
                    u[:local_text_len]
                    if u.size(0) >= local_text_len
                    else torch.cat([u, u.new_zeros(local_text_len - u.size(0), u.size(1))])
                    for u in raw_local_context
                ])
            )

        if has_prefix:
            prefix = [video.patch_embedding(u.unsqueeze(0)) for u in prefix]
            prefix = [u.flatten(2).transpose(1, 2) for u in prefix]
            seq_lens_prefix = torch.tensor([u.size(1) for u in prefix], dtype=torch.long, device=device)
            assert seq_lens_prefix.max() <= seq_len_override
            max_seq_len_prefix = seq_lens_prefix.max().item()
            prefix = torch.cat([
                torch.cat([u, u.new_zeros(1, max_seq_len_prefix - u.size(1), u.size(2))], dim=1)
                for u in prefix
            ])
            x = torch.cat([prefix, x], dim=1)

            e_prefix = video.time_embedding(sinusoidal_embedding_1d(video.freq_dim, prefix_t.flatten()).type_as(x))
            head_e_prefix = e_prefix.unflatten(dim=0, sizes=prefix_t.shape)
            e0_prefix = video.time_projection(e_prefix).unflatten(1, (6, video.dim)).unflatten(dim=0, sizes=prefix_t.shape)
            e0 = torch.cat([e0_prefix, e0], dim=1)
            head_e = torch.cat([head_e_prefix, head_e], dim=1)

        attn_grid_sizes = latent_grid_sizes.clone()
        attn_grid_sizes[:, 0] += len(prefix_token_ids)

        temporal_positions = None
        if vertical_use_representative_rope:
            sequence_temporal_positions = []
            for token_id in list(prefix_token_ids) + list(tree_token_ids):
                if token_id < 0:
                    sequence_temporal_positions.append(0)
                else:
                    sequence_temporal_positions.append(vertical_info["representative_indices"][token_id])
            # When the HDR tree prefix is dropped, the local FastWAM-style clip
            # is the whole video sequence, so its RoPE time index must start at
            # zero instead of being offset after the HDR leaf positions.
            leaf_count = int(vertical_info["num_leaf_frames"]) if (prefix_token_ids or tree_token_ids) else 0
            sequence_temporal_positions.extend(
                range(leaf_count, leaf_count + int(local_start_count) + int(local_video_count))
            )
            temporal_positions = torch.tensor(
                sequence_temporal_positions,
                device=device,
                dtype=torch.long,
            ).unsqueeze(0).expand(timestep.shape[0], -1)

        return {
            "tokens": x,
            "e": e0,
            "head_e": head_e,
            "seq_lens": seq_lens,
            "grid_sizes": attn_grid_sizes,
            "latent_grid_sizes": latent_grid_sizes,
            "freqs": video.freqs,
            "context": context,
            "local_context": local_context,
            "context_lens": None,
            "block_mask": video.block_mask,
            "temporal_positions": temporal_positions,
            "prefix_seq_len": len(prefix_token_ids) * frame_seqlen,
            "frame_seqlen": frame_seqlen,
            "local_start_sequence_index": len(prefix_token_ids) + len(tree_token_ids),
        }

    def _prepare_action_state(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict,
    ) -> dict[str, Any]:
        if noisy_actions.ndim != 3 or noisy_actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected noisy_actions [B, T, {self.action_dim}], got {tuple(noisy_actions.shape)}."
            )
        batch_size, seq_len = noisy_actions.shape[:2]
        action_dtype = self.action_encoder.weight.dtype
        t_embed = sinusoidal_embedding_1d(self.freq_dim, timestep.reshape(-1)).to(
            device=noisy_actions.device,
            dtype=action_dtype,
        )
        t = self.time_embedding(t_embed)
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))
        context = self.text_embedding(
            conditional_dict["prompt_embeds"].to(device=noisy_actions.device, dtype=action_dtype)
        )
        tokens = self.action_encoder(noisy_actions.to(dtype=action_dtype))
        return {
            "tokens": tokens,
            "t": t,
            "t_mod": t_mod,
            "context": context,
            "batch_size": batch_size,
            "seq_len": seq_len,
        }

    def _video_qkv(self, block, x: torch.Tensor, e: torch.Tensor, grid_sizes, freqs, temporal_positions):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = (
            block.modulation.unsqueeze(1).to(dtype=e.dtype, device=e.device) + e
        ).chunk(6, dim=2)
        attn_input = (
            block.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + scale) + shift
        ).flatten(1, 2)
        q = block.self_attn.norm_q(block.self_attn.q(attn_input)).view(
            x.shape[0], x.shape[1], block.self_attn.num_heads, block.self_attn.head_dim
        )
        k = block.self_attn.norm_k(block.self_attn.k(attn_input)).view(
            x.shape[0], x.shape[1], block.self_attn.num_heads, block.self_attn.head_dim
        )
        v = block.self_attn.v(attn_input).view(
            x.shape[0], x.shape[1], block.self_attn.num_heads, block.self_attn.head_dim
        )
        if temporal_positions is not None:
            from wan.modules.causal_model import rope_apply_with_positions

            q = rope_apply_with_positions(q, grid_sizes, freqs, temporal_positions).type_as(v)
            k = rope_apply_with_positions(k, grid_sizes, freqs, temporal_positions).type_as(v)
        else:
            q = causal_rope_apply(q, grid_sizes, freqs).type_as(v)
            k = causal_rope_apply(k, grid_sizes, freqs).type_as(v)
        return q, k, v, (shift_mlp, scale_mlp, gate, gate_mlp)

    @staticmethod
    def _video_post(
        block,
        x,
        mixed,
        e,
        context,
        context_lens,
        *,
        local_context=None,
        local_context_start: int | None = None,
    ):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e_chunks = (block.modulation.unsqueeze(1).to(dtype=e.dtype, device=e.device) + e).chunk(6, dim=2)
        x = x + (
            block.self_attn.o(mixed.flatten(2)).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e_chunks[2]
        ).flatten(1, 2)
        if local_context is None or local_context_start is None:
            x = x + block.cross_attn(block.norm3(x), context, context_lens)
        else:
            if local_context_start <= 0:
                x = x + block.cross_attn(block.norm3(x), local_context, context_lens)
            elif local_context_start >= x.shape[1]:
                x = x + block.cross_attn(block.norm3(x), context, context_lens)
            else:
                x_tree = x[:, :local_context_start]
                x_local = x[:, local_context_start:]
                x = torch.cat([
                    x_tree + block.cross_attn(block.norm3(x_tree), context, context_lens),
                    x_local + block.cross_attn(block.norm3(x_local), local_context, context_lens),
                ], dim=1)
        y = block.ffn(
            (
                block.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e_chunks[4])
                + e_chunks[3]
            ).flatten(1, 2)
        )
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e_chunks[5]).flatten(1, 2)
        return x

    @staticmethod
    def _video_self_attention_from_qkv(q, k, v, block_mask):
        padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
        if padded_length:
            q = torch.cat(
                [q, torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]], device=q.device, dtype=v.dtype)],
                dim=1,
            )
            k = torch.cat(
                [k, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]], device=k.device, dtype=v.dtype)],
                dim=1,
            )
            v = torch.cat(
                [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]], device=v.device, dtype=v.dtype)],
                dim=1,
            )
        out = flex_attention(
            query=q.transpose(2, 1),
            key=k.transpose(2, 1),
            value=v.transpose(2, 1),
            block_mask=block_mask,
        ).transpose(2, 1)
        if padded_length:
            out = out[:, :-padded_length]
        return out

    @staticmethod
    def _full_block_mask(total_length: int, device):
        padded_length = math.ceil(total_length / 128) * 128 - total_length
        valid_positions = torch.zeros(total_length + padded_length, device=device, dtype=torch.bool)
        valid_positions[:total_length] = True

        def attention_mask(b, h, q_idx, kv_idx):
            return valid_positions[q_idx] & valid_positions[kv_idx]

        return create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False,
            device=device,
        )

    @torch.no_grad()
    def _compute_leaf_video_kv(
        self,
        leaf_latents: torch.Tensor,
        conditional_dict: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video = self.video_model
        device = video.patch_embedding.weight.device
        if video.freqs.device != device:
            video.freqs = video.freqs.to(device)
        if leaf_latents.ndim != 5:
            raise ValueError(f"Expected leaf latents [B, T, C, H, W], got {tuple(leaf_latents.shape)}.")
        batch_size, num_leaf = leaf_latents.shape[:2]
        timestep = torch.zeros([batch_size, num_leaf], device=device, dtype=leaf_latents.dtype)

        x = leaf_latents.permute(0, 2, 1, 3, 4)
        x = [video.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long, device=device) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        max_seq_len = max(u.size(1) for u in x)
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, max_seq_len - u.size(1), u.size(2))], dim=1)
            for u in x
        ])

        e = video.time_embedding(sinusoidal_embedding_1d(video.freq_dim, timestep.flatten()).type_as(x))
        e = video.time_projection(e).unflatten(1, (6, video.dim)).unflatten(dim=0, sizes=timestep.shape)
        context = conditional_dict["prompt_embeds"].to(device=device, dtype=x.dtype)
        context = video.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(video.text_len - u.size(0), u.size(1))])
                for u in context
            ])
        )
        block_mask = self._full_block_mask(int(x.shape[1]), device)

        layer_k = []
        layer_v = []
        for block in video.blocks:
            q, k, v, _ = self._video_qkv(block, x, e, grid_sizes, video.freqs, temporal_positions=None)
            layer_k.append(k)
            layer_v.append(v)
            mixed = self._video_self_attention_from_qkv(q, k, v, block_mask)
            x = self._video_post(block, x, mixed, e, context, context_lens=None)
        return torch.stack(layer_k, dim=1), torch.stack(layer_v, dim=1)

    def _action_qkv(self, block: ActionBlock, x: torch.Tensor, t_mod: torch.Tensor):
        shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(block, t_mod)
        attn_input = _modulate(block.norm1(x), shift, scale)
        q = block.self_attn.norm_q(block.self_attn.q(attn_input)).view(
            x.shape[0], x.shape[1], self.num_heads, block.self_attn.attn_head_dim
        )
        k = block.self_attn.norm_k(block.self_attn.k(attn_input)).view(
            x.shape[0], x.shape[1], self.num_heads, block.self_attn.attn_head_dim
        )
        v = block.self_attn.v(attn_input).view(
            x.shape[0], x.shape[1], self.num_heads, block.self_attn.attn_head_dim
        )
        freqs = self._action_freqs(x.device, x.shape[1])
        q = _rope_1d(q, freqs)
        k = _rope_1d(k, freqs)
        return q, k, v, (gate, shift_mlp, scale_mlp, gate_mlp)

    @staticmethod
    def _action_post(
        block: ActionBlock,
        x,
        mixed,
        context,
        gate,
        shift_mlp,
        scale_mlp,
        gate_mlp,
        *,
        disable_text_cross_attn: bool = False,
    ):
        x = block.gate(x, gate, block.self_attn.o(mixed.flatten(2)))
        if not disable_text_cross_attn:
            x = x + block.cross_attn(block.norm3(x), context)
        x = block.gate(x, gate_mlp, block.ffn(_modulate(block.norm2(x), shift_mlp, scale_mlp)))
        return x

    def _leaf_action_ranges(self, vertical_info: dict) -> list[tuple[int, int]]:
        leaf_token_ids = list(vertical_info["leaf_token_ids"])
        if len(leaf_token_ids) * self.actions_per_leaf <= 0:
            raise ValueError("Action hierarchy must contain at least one action token.")
        ranges = []
        for leaf_pos, _ in enumerate(leaf_token_ids):
            start = leaf_pos * self.actions_per_leaf
            ranges.append((start, start + self.actions_per_leaf))
        return ranges

    def _action_hierarchy_mask(self, action_len: int, vertical_info: dict, device) -> torch.Tensor:
        leaf_ranges = self._leaf_action_ranges(vertical_info)
        expected_len = len(leaf_ranges) * self.actions_per_leaf
        if action_len != expected_len:
            raise ValueError(
                f"Expected {expected_len} action tokens from {len(leaf_ranges)} leaves and "
                f"actions_per_leaf={self.actions_per_leaf}, got {action_len}."
            )

        token_to_parent = list(vertical_info["token_to_parent"])
        leaf_token_ids = list(vertical_info["leaf_token_ids"])
        leaf_token_to_range = dict(zip(leaf_token_ids, leaf_ranges))
        mask = torch.zeros((action_len, action_len), dtype=torch.bool, device=device)
        for leaf_token_id, (row_start, row_end) in leaf_token_to_range.items():
            parent_token_id = token_to_parent[leaf_token_id]
            sibling_leaf_ids = [
                other_leaf_id
                for other_leaf_id in leaf_token_ids
                if token_to_parent[other_leaf_id] == parent_token_id
            ]
            for sibling_leaf_id in sibling_leaf_ids:
                col_start, col_end = leaf_token_to_range[sibling_leaf_id]
                mask[row_start:row_end, col_start:col_end] = True
        return mask

    @staticmethod
    def _action_full_mask(action_len: int, video_len: int, device) -> torch.Tensor:
        return torch.ones((action_len, video_len + action_len), dtype=torch.bool, device=device)

    @staticmethod
    def _action_local_start_mask(action_len: int, video_len: int, device) -> torch.Tensor:
        mask = torch.zeros((action_len, video_len + action_len), dtype=torch.bool, device=device)
        mask[:, :video_len] = True
        mask[:, video_len:] = True
        return mask

    @staticmethod
    def _action_self_only_mask(action_len: int, video_len: int, device) -> torch.Tensor:
        mask = torch.zeros((action_len, video_len + action_len), dtype=torch.bool, device=device)
        mask[:, video_len:] = True
        return mask

    @staticmethod
    def _joint_video_mask(
        *,
        frame_seqlen: int,
        prefix_token_ids: list[int],
        tree_token_ids: list[int],
        local_start_count: int,
        local_video_count: int,
        vertical_info: dict,
        device,
    ):
        sequence_token_ids = list(prefix_token_ids) + list(tree_token_ids)
        num_prefix = len(prefix_token_ids)
        num_tree = len(tree_token_ids)
        local_start_sequence_index = num_prefix + num_tree
        local_video_start_sequence_index = local_start_sequence_index + local_start_count
        num_sequence_frames = num_prefix + num_tree + local_start_count + local_video_count
        total_length = num_sequence_frames * frame_seqlen
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        allowed_frames = torch.zeros(
            [num_sequence_frames, num_sequence_frames],
            device=device,
            dtype=torch.bool,
        )
        token_id_to_sequence_index = {
            token_id: sequence_index for sequence_index, token_id in enumerate(sequence_token_ids)
        }

        for sequence_index, token_id in enumerate(sequence_token_ids):
            allowed_token_ids = get_vertical_allowed_token_ids(
                token_id,
                vertical_info,
                include_self=True,
                condition_token_id=CONDITION_TOKEN_ID,
            )
            for allowed_token_id in allowed_token_ids:
                allowed_sequence_index = token_id_to_sequence_index.get(allowed_token_id)
                if allowed_sequence_index is not None:
                    allowed_frames[sequence_index, allowed_sequence_index] = True

        leaf_sequence_indices = [
            token_id_to_sequence_index[token_id]
            for token_id in vertical_info["leaf_token_ids"]
            if token_id in token_id_to_sequence_index
        ]
        for offset in range(local_start_count):
            row = local_start_sequence_index + offset
            allowed_frames[row, row] = True
            for leaf_sequence_index in leaf_sequence_indices:
                allowed_frames[row, leaf_sequence_index] = True

        local_video_indices = list(
            range(local_video_start_sequence_index, local_video_start_sequence_index + local_video_count)
        )
        for row in local_video_indices:
            allowed_frames[row, local_start_sequence_index:local_start_sequence_index + local_start_count] = True
            for col in local_video_indices:
                allowed_frames[row, col] = True
            for leaf_sequence_index in leaf_sequence_indices:
                allowed_frames[row, leaf_sequence_index] = True

        token_frame_ids = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        valid_positions = torch.zeros(total_length + padded_length, device=device, dtype=torch.bool)
        for frame_index in range(num_sequence_frames):
            start = frame_index * frame_seqlen
            end = start + frame_seqlen
            token_frame_ids[start:end] = frame_index
            valid_positions[start:end] = True

        def attention_mask(b, h, q_idx, kv_idx):
            q_frame = token_frame_ids[q_idx]
            kv_frame = token_frame_ids[kv_idx]
            return valid_positions[q_idx] & valid_positions[kv_idx] & allowed_frames[q_frame, kv_frame]

        return create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False,
            device=device,
        )

    @staticmethod
    def _select_leaf_video_tokens(tensor: torch.Tensor, video_state: dict, vertical_info: dict) -> torch.Tensor:
        frame_seq_len = int(video_state["tokens"].shape[1] // video_state["e"].shape[1])
        prefix_seq_len = int(video_state.get("prefix_seq_len", 0))
        leaf_chunks = []
        for leaf_token_id in vertical_info["leaf_token_ids"]:
            start = prefix_seq_len + int(leaf_token_id) * frame_seq_len
            leaf_chunks.append(tensor[:, start:start + frame_seq_len])
        return torch.cat(leaf_chunks, dim=1)

    def _action_video_mask(
        self,
        action_len: int,
        video_len: int,
        video_tokens_per_token: int,
        prefix_token_count: int,
        noisy_token_ids: list[int],
        vertical_info: dict,
        device,
    ) -> torch.Tensor:
        if self.action_attend_video == "all":
            return torch.ones((action_len, video_len + action_len), dtype=torch.bool, device=device)
        mask = torch.zeros((action_len, video_len + action_len), dtype=torch.bool, device=device)
        if self.action_attend_video == "none":
            return mask

        sequence_index_by_token_id = {
            token_id: prefix_token_count + position
            for position, token_id in enumerate(noisy_token_ids)
        }
        token_to_parent = list(vertical_info["token_to_parent"])
        token_to_level = list(vertical_info["token_to_level"])
        token_to_level_pos = list(vertical_info["token_to_level_pos"])
        level_offsets = list(vertical_info["level_offsets"])
        level_sizes = list(vertical_info["level_sizes"])
        for leaf_pos, leaf_token_id in enumerate(vertical_info["leaf_token_ids"]):
            row_start = leaf_pos * self.actions_per_leaf
            row_end = row_start + self.actions_per_leaf
            parent_token_id = token_to_parent[leaf_token_id]
            if parent_token_id < 0:
                continue
            parent_level = token_to_level[parent_token_id]
            parent_level_pos = token_to_level_pos[parent_token_id]
            if parent_level_pos == 0 and prefix_token_count > 0:
                mask[row_start:row_end, : video_tokens_per_token * prefix_token_count] = True
            parent_neighbor_ids = []
            for neighbor_level_pos in (parent_level_pos - 1, parent_level_pos, parent_level_pos + 1):
                if 0 <= neighbor_level_pos < level_sizes[parent_level]:
                    parent_neighbor_ids.append(level_offsets[parent_level] + neighbor_level_pos)
            for visible_token_id in parent_neighbor_ids:
                sequence_index = sequence_index_by_token_id.get(visible_token_id)
                if sequence_index is not None:
                    col_start = sequence_index * video_tokens_per_token
                    col_end = min(col_start + video_tokens_per_token, video_len)
                    mask[row_start:row_end, col_start:col_end] = True
        return mask

    def _action_layer(
        self,
        x_action: torch.Tensor,
        action_block: ActionBlock,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        action_state: dict,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        q_action, k_action, v_action, action_post = self._action_qkv(
            action_block,
            x_action,
            action_state["t_mod"],
        )
        k_full = torch.cat([k_cat, k_action], dim=1)
        v_full = torch.cat([v_cat, v_action], dim=1)
        mixed_action = F.scaled_dot_product_attention(
            q_action.transpose(1, 2),
            k_full.transpose(1, 2),
            v_full.transpose(1, 2),
            attn_mask=attn_mask.view(1, 1, q_action.shape[1], k_full.shape[1]),
        ).transpose(1, 2)

        gate, shift_mlp, scale_mlp, gate_mlp = action_post
        return self._action_post(
            action_block,
            x_action,
            mixed_action,
            action_state["context"],
            gate,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            disable_text_cross_attn=self.disable_action_text_cross_attn,
        )

    def _action_layer_checkpointed(
        self,
        x_action: torch.Tensor,
        action_block: ActionBlock,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        t_mod: torch.Tensor,
        context: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        def build_qkv(x_in: torch.Tensor, t_mod_in: torch.Tensor):
            q_action, k_action, v_action, action_post = self._action_qkv(
                action_block,
                x_in,
                t_mod_in,
            )
            k_full = torch.cat([k_cat, k_action], dim=1)
            v_full = torch.cat([v_cat, v_action], dim=1)
            mixed_action = F.scaled_dot_product_attention(
                q_action.transpose(1, 2),
                k_full.transpose(1, 2),
                v_full.transpose(1, 2),
                attn_mask=attn_mask.view(1, 1, q_action.shape[1], k_full.shape[1]),
            ).transpose(1, 2)
            gate, shift_mlp, scale_mlp, gate_mlp = action_post
            return mixed_action, gate, shift_mlp, scale_mlp, gate_mlp

        mixed_action, gate, shift_mlp, scale_mlp, gate_mlp = checkpoint(
            build_qkv,
            x_action,
            t_mod,
            use_reentrant=False,
        )

        def post_fn(
            x_in: torch.Tensor,
            mixed_in: torch.Tensor,
            gate_in: torch.Tensor,
            shift_mlp_in: torch.Tensor,
            scale_mlp_in: torch.Tensor,
            gate_mlp_in: torch.Tensor,
            context_in: torch.Tensor,
        ) -> torch.Tensor:
            return self._action_post(
                action_block,
                x_in,
                mixed_in,
                context_in,
                gate_in,
                shift_mlp_in,
                scale_mlp_in,
                gate_mlp_in,
                disable_text_cross_attn=self.disable_action_text_cross_attn,
            )

        return checkpoint(
            post_fn,
            x_action,
            mixed_action,
            gate,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            context,
            use_reentrant=False,
        )

    def forward(
        self,
        *,
        noisy_actions: torch.Tensor,
        action_timestep: torch.Tensor,
        video_latents: torch.Tensor | None = None,
        video_timestep: torch.Tensor | None = None,
        video_leaf_k: torch.Tensor | None = None,
        video_leaf_v: torch.Tensor | None = None,
        conditional_dict: dict,
        prefix_x: torch.Tensor,
        prefix_t: torch.Tensor,
        prefix_token_ids: list[int],
        noisy_token_ids: list[int],
        vertical_info: dict,
        vertical_use_representative_rope: bool,
        seq_len_override: int,
        video_action_joint: bool = False,
        tree_token_ids: list[int] | None = None,
        local_start_count: int = 1,
        local_video_count: int = 5,
        detach_action_video_kv: bool = False,
        action_attend_video: str = "local_start",
        action_video_kv_scale: float = 1.0,
        joint_proprio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if video_action_joint:
            return self.forward_video_action_joint(
                noisy_actions=noisy_actions,
                action_timestep=action_timestep,
                video_latents=video_latents,
                video_timestep=video_timestep,
                conditional_dict=conditional_dict,
                prefix_x=prefix_x,
                prefix_t=prefix_t,
                prefix_token_ids=prefix_token_ids,
                tree_token_ids=tree_token_ids if tree_token_ids is not None else noisy_token_ids,
                vertical_info=vertical_info,
                vertical_use_representative_rope=vertical_use_representative_rope,
                local_start_count=local_start_count,
                local_video_count=local_video_count,
                seq_len_override=seq_len_override,
                detach_action_video_kv=detach_action_video_kv,
                action_attend_video=action_attend_video,
                action_video_kv_scale=action_video_kv_scale,
                joint_proprio=joint_proprio,
            )
        action_state = self._prepare_action_state(noisy_actions, action_timestep, conditional_dict)
        x_action = action_state["tokens"]
        action_seq_len = int(x_action.shape[1])

        if (video_leaf_k is None and video_leaf_v is None and video_latents is not None
                and video_latents.shape[1] == int(vertical_info["num_leaf_frames"])):
            video_leaf_k, video_leaf_v = self._compute_leaf_video_kv(video_latents, conditional_dict)

        if video_leaf_k is not None or video_leaf_v is not None:
            if video_leaf_k is None or video_leaf_v is None:
                raise ValueError("`video_leaf_k` and `video_leaf_v` must be provided together.")
            if video_leaf_k.ndim != 5 or video_leaf_v.ndim != 5:
                raise ValueError(
                    f"Expected leaf KV cache [B, L, S, H, D], got k={tuple(video_leaf_k.shape)}, v={tuple(video_leaf_v.shape)}."
                )
            if video_leaf_k.shape != video_leaf_v.shape:
                raise ValueError(f"Leaf KV cache shape mismatch: k={tuple(video_leaf_k.shape)}, v={tuple(video_leaf_v.shape)}.")
            if video_leaf_k.shape[0] != x_action.shape[0] or video_leaf_k.shape[1] != len(self.blocks):
                raise ValueError(
                    f"Leaf KV cache must be [B, {len(self.blocks)}, S, H, D], got {tuple(video_leaf_k.shape)}."
                )
            action_video_mask = self._action_full_mask(action_seq_len, int(video_leaf_k.shape[2]), x_action.device)
            for layer_idx, action_block in enumerate(self.blocks):
                k_video = video_leaf_k[:, layer_idx].to(device=x_action.device, dtype=x_action.dtype)
                v_video = video_leaf_v[:, layer_idx].to(device=x_action.device, dtype=x_action.dtype)
                if self.training and self.use_gradient_checkpointing:
                    x_action = self._action_layer_checkpointed(
                        x_action,
                        action_block,
                        k_video.detach(),
                        v_video.detach(),
                        action_state["t_mod"],
                        action_state["context"],
                        action_video_mask,
                    )
                else:
                    x_action = self._action_layer(
                        x_action,
                        action_block,
                        k_video.detach(),
                        v_video.detach(),
                        action_state,
                        action_video_mask,
                    )
            return self.head(x_action)

        if video_latents is None or video_timestep is None:
            raise ValueError("Either leaf KV cache or `video_latents`/`video_timestep` must be provided.")

        video_state = self._prepare_video_state(
            video_latents,
            video_timestep,
            conditional_dict,
            prefix_x=prefix_x,
            prefix_t=prefix_t,
            prefix_token_ids=prefix_token_ids,
            noisy_token_ids=noisy_token_ids,
            vertical_info=vertical_info,
            vertical_use_representative_rope=vertical_use_representative_rope,
            seq_len_override=seq_len_override,
        )

        x_video = video_state["tokens"]
        leaf_video_seq_len = len(vertical_info["leaf_token_ids"]) * (
            int(video_state["tokens"].shape[1]) // int(video_state["e"].shape[1])
        )
        action_video_mask = self._action_full_mask(action_seq_len, leaf_video_seq_len, x_action.device)

        for layer_idx, (video_block, action_block) in enumerate(zip(self.video_model.blocks, self.blocks)):
            with torch.no_grad():
                q_video, k_video, v_video, _ = self._video_qkv(
                    video_block,
                    x_video,
                    video_state["e"],
                    video_state["grid_sizes"],
                    video_state["freqs"],
                    video_state["temporal_positions"],
                )
                k_action_video = self._select_leaf_video_tokens(k_video, video_state, vertical_info)
                v_action_video = self._select_leaf_video_tokens(v_video, video_state, vertical_info)
            if self.training and self.use_gradient_checkpointing:
                x_action = self._action_layer_checkpointed(
                    x_action,
                    action_block,
                    k_action_video.detach(),
                    v_action_video.detach(),
                    action_state["t_mod"],
                    action_state["context"],
                    action_video_mask,
                )
            else:
                x_action = self._action_layer(
                    x_action,
                    action_block,
                    k_action_video.detach(),
                    v_action_video.detach(),
                    action_state,
                    action_video_mask,
                )

            with torch.no_grad():
                mixed_video = self._video_self_attention_from_qkv(
                    q_video,
                    k_video,
                    v_video,
                    video_state["block_mask"],
                )
                x_video = self._video_post(
                    video_block,
                    x_video,
                    mixed_video,
                    video_state["e"],
                    video_state["context"],
                    video_state["context_lens"],
                )

        return self.head(x_action)

    def forward_video_action_joint(
        self,
        *,
        noisy_actions: torch.Tensor,
        action_timestep: torch.Tensor,
        video_latents: torch.Tensor,
        video_timestep: torch.Tensor,
        conditional_dict: dict,
        prefix_x: torch.Tensor,
        prefix_t: torch.Tensor,
        prefix_token_ids: list[int],
        tree_token_ids: list[int],
        vertical_info: dict,
        vertical_use_representative_rope: bool,
        local_start_count: int,
        local_video_count: int,
        seq_len_override: int,
        detach_action_video_kv: bool = False,
        action_attend_video: str = "local_start",
        action_video_kv_scale: float = 1.0,
        joint_proprio: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_conditional_dict = self._append_proprio_to_prompt_embeds(
            conditional_dict,
            joint_proprio,
            device=noisy_actions.device,
            dtype=self.action_encoder.weight.dtype,
        )
        action_state = self._prepare_action_state(noisy_actions, action_timestep, local_conditional_dict)
        x_action = action_state["tokens"]
        action_seq_len = int(x_action.shape[1])
        debug_video_kv = bool(getattr(self, "debug_action_video_kv", False))
        debug_kv_stats = []

        video_state = self._prepare_joint_video_state(
            video_latents,
            video_timestep,
            conditional_dict,
            prefix_x=prefix_x,
            prefix_t=prefix_t,
            prefix_token_ids=prefix_token_ids,
            tree_token_ids=tree_token_ids,
            vertical_info=vertical_info,
            vertical_use_representative_rope=vertical_use_representative_rope,
            local_start_count=local_start_count,
            local_video_count=local_video_count,
            seq_len_override=seq_len_override,
            local_conditional_dict=local_conditional_dict if local_conditional_dict is not conditional_dict else None,
        )

        x_video = video_state["tokens"]
        frame_seqlen = int(video_state["frame_seqlen"])
        local_start_sequence_index = int(video_state["local_start_sequence_index"])
        local_start_seq_start = local_start_sequence_index * frame_seqlen
        local_start_seq_end = local_start_seq_start + int(local_start_count) * frame_seqlen
        local_start_video_len = local_start_seq_end - local_start_seq_start
        if action_attend_video == "none":
            action_video_mask = self._action_self_only_mask(
                action_seq_len,
                local_start_video_len,
                x_action.device,
            )
        elif action_attend_video == "local_start":
            action_video_mask = self._action_local_start_mask(
                action_seq_len,
                local_start_video_len,
                x_action.device,
            )
        else:
            raise ValueError(f"Unsupported joint action_attend_video: {action_attend_video}")

        for video_block, action_block in zip(self.video_model.blocks, self.blocks):
            q_video, k_video, v_video, _ = self._video_qkv(
                video_block,
                x_video,
                video_state["e"],
                video_state["grid_sizes"],
                video_state["freqs"],
                video_state["temporal_positions"],
            )
            k_action_video = k_video[:, local_start_seq_start:local_start_seq_end]
            v_action_video = v_video[:, local_start_seq_start:local_start_seq_end]
            if debug_video_kv and len(debug_kv_stats) < 6:
                debug_kv_stats.append((
                    float(k_action_video.detach().float().norm(dim=-1).mean().cpu()),
                    float(k_action_video.detach().float().abs().max().cpu()),
                    float(v_action_video.detach().float().norm(dim=-1).mean().cpu()),
                    float(v_action_video.detach().float().abs().max().cpu()),
                ))
            if action_video_kv_scale != 1.0:
                k_action_video = k_action_video * action_video_kv_scale
                v_action_video = v_action_video * action_video_kv_scale
            if detach_action_video_kv:
                k_action_video = k_action_video.detach()
                v_action_video = v_action_video.detach()

            if self.training and self.use_gradient_checkpointing:
                x_action = self._action_layer_checkpointed(
                    x_action,
                    action_block,
                    k_action_video,
                    v_action_video,
                    action_state["t_mod"],
                    action_state["context"],
                    action_video_mask,
                )
            else:
                x_action = self._action_layer(
                    x_action,
                    action_block,
                    k_action_video,
                    v_action_video,
                    action_state,
                    action_video_mask,
                )

            mixed_video = self._video_self_attention_from_qkv(
                q_video,
                k_video,
                v_video,
                video_state["block_mask"],
            )
            x_video = self._video_post(
                video_block,
                x_video,
                mixed_video,
                video_state["e"],
                video_state["context"],
                video_state["context_lens"],
                local_context=video_state["local_context"],
                local_context_start=local_start_seq_start,
            )
        if debug_video_kv and debug_kv_stats:
            print(
                "[JointActionKV] "
                + " ".join(
                    f"L{i}:k_norm={k_norm:.3f},k_max={k_max:.3f},v_norm={v_norm:.3f},v_max={v_max:.3f}"
                    for i, (k_norm, k_max, v_norm, v_max) in enumerate(debug_kv_stats)
                ),
                flush=True,
            )

        prefix_seq_len = int(video_state["prefix_seq_len"])
        x_video_no_prefix = x_video[:, prefix_seq_len:]
        head_e_no_prefix = video_state["head_e"][:, len(prefix_token_ids):]
        video_flow_patch = self.video_model.head(x_video_no_prefix, head_e_no_prefix.unsqueeze(2))
        video_flow = torch.stack(
            self.video_model.unpatchify(video_flow_patch, video_state["latent_grid_sizes"])
        ).permute(0, 2, 1, 3, 4)
        return video_flow, self.head(x_action)


class HDRVideoActionJointMoT(nn.Module):
    """FSDP-owned joint video/action expert wrapper.

    The action expert reads the video expert's same-layer K/V inside this
    module's forward, so wrapping this module keeps the full MoT computation
    under a single FSDP-managed forward pass.
    """

    def __init__(self, *, generator: nn.Module, action_dit: HDRActionMoT):
        super().__init__()
        self.generator = generator
        self.action_dit = action_dit
        self.action_dit.bind_video_model(self.generator.model)

    def forward(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        self.action_dit.bind_video_model(self.generator.model)
        if "noisy_token_ids" not in kwargs:
            kwargs["noisy_token_ids"] = kwargs.get("tree_token_ids")
        return self.action_dit(video_action_joint=True, **kwargs)
