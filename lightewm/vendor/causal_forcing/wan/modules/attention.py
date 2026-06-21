# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import torch

try:
    import flash_attn_interface

    def is_hopper_gpu():
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return "h100" in device_name or "hopper" in device_name
    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = hasattr(flash_attn, "flash_attn_varlen_func")
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from flash_attn.cute import flash_attn_varlen_func as flash_attn_4_varlen_func
    FLASH_ATTN_4_AVAILABLE = os.environ.get("CAUSAL_ENABLE_FLASH_ATTN_4") == "1"
except (ImportError, ModuleNotFoundError):
    flash_attn_4_varlen_func = None
    FLASH_ATTN_4_AVAILABLE = False

if os.environ.get("CAUSAL_FORCE_ATTENTION_FALLBACK") == "1":
    FLASH_ATTN_2_AVAILABLE = False
    FLASH_ATTN_3_AVAILABLE = False
    FLASH_ATTN_4_AVAILABLE = False

# FLASH_ATTN_3_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
]


def _has_real_padding(lens, full_len):
    if lens is None:
        return False
    if torch.is_tensor(lens):
        return bool((lens != full_len).any().item())
    return any(int(length) != full_len for length in lens)


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    use_flash_attn_4 = FLASH_ATTN_4_AVAILABLE and not FLASH_ATTN_2_AVAILABLE and not FLASH_ATTN_3_AVAILABLE and dropout_p == 0
    if not FLASH_ATTN_2_AVAILABLE and not FLASH_ATTN_3_AVAILABLE and not use_flash_attn_4:
        if _has_real_padding(q_lens, lq) or _has_real_padding(k_lens, lk) or window_size != (-1, -1):
            raise RuntimeError(
                'scaled_dot_product_attention fallback does not preserve varlen padding '
                'or windowed attention semantics. Install a compatible flash attention '
                f'backend instead: q_lens={q_lens}, k_lens={k_lens}, '
                f'window_size={window_size}, FLASH_ATTN_2_AVAILABLE={FLASH_ATTN_2_AVAILABLE}, '
                f'FLASH_ATTN_3_AVAILABLE={FLASH_ATTN_3_AVAILABLE}, '
                f'FLASH_ATTN_4_AVAILABLE={FLASH_ATTN_4_AVAILABLE}.'
            )
        if q_scale is not None:
            q = q * q_scale
        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=causal, dropout_p=dropout_p, scale=softmax_scale)
        return x.transpose(1, 2).contiguous().type(out_dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available; using another available flash attention backend if present.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    elif FLASH_ATTN_2_AVAILABLE:
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))
    elif use_flash_attn_4:
        fa4_window_size = (None, None) if window_size == (-1, -1) else window_size
        x = flash_attn_4_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=fa4_window_size,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        raise RuntimeError(
            'No compatible flash attention backend is available for the requested '
            f'configuration: version={version}, dropout_p={dropout_p}, '
            f'FLASH_ATTN_2_AVAILABLE={FLASH_ATTN_2_AVAILABLE}, '
            f'FLASH_ATTN_3_AVAILABLE={FLASH_ATTN_3_AVAILABLE}, '
            f'FLASH_ATTN_4_AVAILABLE={FLASH_ATTN_4_AVAILABLE}.'
        )

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE or FLASH_ATTN_4_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if _has_real_padding(q_lens, q.size(1)) or _has_real_padding(k_lens, k.size(1)) or window_size != (-1, -1):
            raise RuntimeError(
                'scaled_dot_product_attention fallback does not preserve varlen padding '
                'or windowed attention semantics. Install a compatible flash attention '
                f'backend instead: q_lens={q_lens}, k_lens={k_lens}, '
                f'window_size={window_size}, FLASH_ATTN_2_AVAILABLE={FLASH_ATTN_2_AVAILABLE}, '
                f'FLASH_ATTN_3_AVAILABLE={FLASH_ATTN_3_AVAILABLE}, '
                f'FLASH_ATTN_4_AVAILABLE={FLASH_ATTN_4_AVAILABLE}.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
