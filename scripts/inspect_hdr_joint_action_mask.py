#!/usr/bin/env python3
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
HIDIT_ROOT = ROOT.parent / "HiDiT" / "Causal-Forcing"
sys.path.insert(0, str(HIDIT_ROOT))

from model.action_mot import HDRActionMoT


def summarize(name: str, mask: torch.Tensor, video_len: int):
    video_visible = mask[:, :video_len].sum(dim=1)
    action_visible = mask[:, video_len:].sum(dim=1)
    print(
        f"{name}: shape={tuple(mask.shape)} "
        f"video_visible_unique={sorted(set(video_visible.tolist()))} "
        f"action_visible_unique={sorted(set(action_visible.tolist()))} "
        f"all_true={bool(mask.all())}"
    )
    print(f"  row0 first_video={mask[0, :min(video_len, 16)].int().tolist()}")
    print(f"  row0 first_action={mask[0, video_len:video_len + min(mask.shape[0], 16)].int().tolist()}")


def main():
    action_len = 52
    local_start_video_len = 784
    summarize("none", HDRActionMoT._action_self_only_mask(action_len, local_start_video_len, "cpu"), local_start_video_len)
    summarize("local_start", HDRActionMoT._action_local_start_mask(action_len, local_start_video_len, "cpu"), local_start_video_len)
    summarize("all", HDRActionMoT._action_full_mask(action_len, local_start_video_len, "cpu"), local_start_video_len)


if __name__ == "__main__":
    main()
