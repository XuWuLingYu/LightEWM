#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FASTWAM_SRC = ROOT / "lightewm" / "vendor" / "fastwam"
HIDIT_ROOT = ROOT.parent / "HiDiT" / "Causal-Forcing"
for path in (str(FASTWAM_SRC), str(HIDIT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _bucket_norms(module, grad: bool = False):
    buckets = {}
    top = []
    for name, param in module.named_parameters():
        value = param.grad if grad else param
        if value is None:
            continue
        norm = float(value.detach().float().norm().item())
        top.append((norm, name, tuple(param.shape)))
        if "action_encoder" in name:
            key = "action_encoder"
        elif "head" in name:
            key = "head"
        elif "time_embedding" in name or "time_projection" in name:
            key = "time"
        elif "text_embedding" in name:
            key = "text"
        elif "blocks" in name:
            if ".self_attn." in name:
                key = "blocks.self_attn"
            elif ".cross_attn." in name:
                key = "blocks.cross_attn"
            elif ".ffn." in name:
                key = "blocks.ffn"
            elif ".modulation" in name:
                key = "blocks.modulation"
            else:
                key = "blocks.other"
        else:
            key = "other"
        buckets[key] = buckets.get(key, 0.0) + norm * norm
    return {k: math.sqrt(v) for k, v in sorted(buckets.items())}, sorted(top, reverse=True)[:20]


def _print_norms(title, module):
    buckets, top = _bucket_norms(module, grad=False)
    print(f"\n[{title}] param_norm_buckets")
    for key, value in buckets.items():
        print(f"  {key}: {value:.6f}")
    print(f"[{title}] top_param_norms")
    for norm, name, shape in top[:10]:
        print(f"  {norm:.6f} {name} {shape}")


def _grad_summary(title, module):
    buckets, top = _bucket_norms(module, grad=True)
    print(f"\n[{title}] grad_norm_buckets")
    for key, value in buckets.items():
        print(f"  {key}: {value:.6f}")
    print(f"[{title}] top_grad_norms")
    for norm, name, shape in top[:15]:
        print(f"  {norm:.6f} {name} {shape}")


def _run_action_only_compare(official_action, hdr_action, *, device, dtype):
    torch.manual_seed(1234)
    batch_size = 1
    action_len = 52
    context_len = 512
    action = torch.randn(batch_size, action_len, 7, device=device, dtype=dtype)
    target = torch.randn_like(action)
    timestep = torch.rand(batch_size, device=device, dtype=dtype) * 1000.0
    context = torch.randn(batch_size, context_len, 4096, device=device, dtype=dtype)
    context_mask = torch.ones(batch_size, context_len, device=device, dtype=torch.bool)

    official_action.zero_grad(set_to_none=True)
    hdr_action.zero_grad(set_to_none=True)

    official_pred = official_action(
        action_tokens=action,
        timestep=timestep,
        context=context,
        context_mask=context_mask,
    )
    official_loss = torch.nn.functional.mse_loss(official_pred.float(), target.float())
    official_loss.backward()
    _grad_summary("official_action_only", official_action)

    action_state = hdr_action._prepare_action_state(
        action,
        timestep,
        {"prompt_embeds": context},
    )
    x = action_state["tokens"]
    empty_k = torch.empty(
        batch_size,
        0,
        hdr_action.num_heads,
        hdr_action.attn_head_dim,
        device=device,
        dtype=dtype,
    )
    empty_v = torch.empty_like(empty_k)
    mask = hdr_action._action_self_only_mask(action_len, 0, device)
    for block in hdr_action.blocks:
        x = hdr_action._action_layer(x, block, empty_k, empty_v, action_state, mask)
    hdr_pred = hdr_action.head(x)
    hdr_loss = torch.nn.functional.mse_loss(hdr_pred.float(), target.float())
    hdr_loss.backward()
    _grad_summary("hdr_action_only", hdr_action)

    delta = (official_pred.float() - hdr_pred.float()).abs()
    print(
        "\n[action_only_output_compare] "
        f"official_loss={float(official_loss.detach().cpu()):.6f} "
        f"hdr_loss={float(hdr_loss.detach().cpu()):.6f} "
        f"max_abs_delta={float(delta.max().detach().cpu()):.6f} "
        f"mean_abs_delta={float(delta.mean().detach().cpu()):.6f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    device = torch.device(args.device)

    from fastwam.models.wan22.action_dit import ActionDiT
    from fastwam.models.wan22.fastwam import FastWAM
    from model.action_mot import HDRActionMoT

    video_dit_config = {
        "has_image_input": False,
        "patch_size": [1, 2, 2],
        "in_dim": 48,
        "hidden_dim": 3072,
        "ffn_dim": 14336,
        "freq_dim": 256,
        "text_dim": 4096,
        "out_dim": 48,
        "num_heads": 24,
        "attn_head_dim": 128,
        "num_layers": 30,
        "eps": 1.0e-6,
        "seperated_timestep": True,
        "require_clip_embedding": False,
        "require_vae_embedding": False,
        "fuse_vae_embedding_in_latents": True,
        "use_gradient_checkpointing": False,
        "video_attention_mask_mode": "first_frame_causal",
        "action_conditioned": False,
        "action_dim": 7,
        "action_group_causal_mask_mode": "group_diagonal",
    }
    action_dit_config = {
        "hidden_dim": 1024,
        "action_dim": 7,
        "ffn_dim": 4096,
        "text_dim": 4096,
        "freq_dim": 256,
        "eps": 1.0e-6,
        "num_heads": 24,
        "attn_head_dim": 128,
        "num_layers": 30,
        "use_gradient_checkpointing": False,
    }

    # Load FastWAM once to get a known-good official video/action pair.
    fastwam = FastWAM.from_wan22_pretrained(
        device=str(device),
        torch_dtype=dtype,
        model_id="Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
        load_text_encoder=False,
        redirect_common_files=False,
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=str(ROOT / "checkpoints" / "ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"),
        action_train_shift=5.0,
        action_infer_shift=5.0,
    )
    official_action = fastwam.action_expert
    video_model = fastwam.video_expert
    # HDRActionMoT was written against HiDiT/Wan field names. For this static
    # weight comparison, the official FastWAM video model is structurally
    # equivalent but exposes `hidden_dim` instead of `dim`.
    if not hasattr(video_model, "dim"):
        video_model.dim = int(video_dit_config["hidden_dim"])
    if not hasattr(video_model, "text_dim"):
        video_model.text_dim = int(video_dit_config["text_dim"])

    hdr_action = HDRActionMoT(
        video_model=video_model,
        action_dim=7,
        hidden_dim=1024,
        ffn_dim=4096,
        freq_dim=256,
        eps=1.0e-6,
        actions_per_leaf=52,
        action_attend_video="all",
        use_gradient_checkpointing=False,
    ).to(device=device, dtype=dtype)
    backbone, summary = hdr_action.build_interpolated_video_backbone_state_dict(apply_alpha_scaling=True)
    hdr_action.load_state_dict(backbone, strict=False)
    print(f"[HDRActionMoT] interpolation_summary={summary}")

    hdr_action_matched = HDRActionMoT(
        video_model=video_model,
        action_dim=7,
        hidden_dim=1024,
        ffn_dim=4096,
        freq_dim=256,
        eps=1.0e-6,
        actions_per_leaf=52,
        action_attend_video="all",
        use_gradient_checkpointing=False,
    ).to(device=device, dtype=dtype)
    matched_state = hdr_action_matched.state_dict()
    official_state_for_match = official_action.state_dict()
    copied_match_keys = 0
    for key, value in official_state_for_match.items():
        if key in matched_state and tuple(matched_state[key].shape) == tuple(value.shape):
            matched_state[key] = value.to(device=matched_state[key].device, dtype=matched_state[key].dtype)
            copied_match_keys += 1
    hdr_action_matched.load_state_dict(matched_state, strict=True)
    print(f"[HDRActionMoT] copied_full_official_matching_keys={copied_match_keys}")

    official_random = ActionDiT(
        hidden_dim=1024,
        action_dim=7,
        ffn_dim=4096,
        text_dim=4096,
        freq_dim=256,
        eps=1.0e-6,
        num_heads=24,
        attn_head_dim=128,
        num_layers=30,
    ).to(device=device, dtype=dtype)

    _print_norms("official_pretrained", official_action)
    _print_norms("official_random", official_random)
    _print_norms("hdr_interpolated", hdr_action)
    _run_action_only_compare(official_action, hdr_action, device=device, dtype=dtype)
    _run_action_only_compare(official_action, hdr_action_matched, device=device, dtype=dtype)

    # Direct key comparison where module structures are intended to match.
    official_state = official_action.state_dict()
    hdr_state = hdr_action.state_dict()
    diffs = []
    for key, value in official_state.items():
        if key not in hdr_state or tuple(value.shape) != tuple(hdr_state[key].shape):
            continue
        if key.startswith("action_encoder.") or key.startswith("head."):
            continue
        delta = (hdr_state[key].float().cpu() - value.float().cpu()).norm().item()
        base = value.float().cpu().norm().item()
        diffs.append((delta / (base + 1e-12), delta, base, key))
    diffs.sort(reverse=True)
    print("\n[official_vs_hdr] top_relative_backbone_diffs")
    for rel, delta, base, key in diffs[:20]:
        print(f"  rel={rel:.6f} delta={delta:.6f} base={base:.6f} {key}")


if __name__ == "__main__":
    main()
