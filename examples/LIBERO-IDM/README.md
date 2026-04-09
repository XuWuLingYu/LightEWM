# LIBERO IDM

This example trains AnyPos on LIBERO through exported `video + abs action list` metadata.

If you have not downloaded the LIBERO dataset yet, see [examples/LIBERO/README.md](/mnt/world_foundational_model/wfm_ckp-fileset/qianzezhong/LightEWM/examples/LIBERO/README.md) first.

Important:
- This path assumes absolute action targets.
- Do not use raw LIBERO `actions` here. Those are relative control commands.
- The default export uses `7D = obs/ee_states (6) + obs/gripper_states[...,0] (1)`.
- The default export writes one video per demo plus a per-frame `abs_action` list.
- The exported frames are flipped left-right and top-bottom before being written into the mp4.
- Training itself does not apply an extra fixed flip on top of the exported metadata.

## 1) Export metadata

```bash
bash examples/LIBERO-IDM/build_metadata.sh \
  --libero-root data/LIBERO-datasets \
  --output-dir data/libero_idm_abs_action
```

This creates:

```text
data/libero_idm_abs_action/
├── metadata_abs_action.jsonl
└── videos/
    └── ...
```

## 2) Train


```bash
accelerate launch \
  --num_processes $NPROC_PER_NODE \
  third_parties/AnyPos/train_metadata_abs_action.py \
  --metadata_path data/libero_idm_abs_action/metadata_abs_action.jsonl \
  --image_base_path data/libero_idm_abs_action \
  --video_key video \
  --action_key abs_action \
  --model_name direction_aware \
  --dinov2_name facebook/dinov2-with-registers-base \
  --learning_rate 1e-4 \
  --batch_size 32 \
  --eval_batch_size 64 \
  --num_workers 8 \
  --eval_interval 2000 \
  --save_interval 2000 \
  --save_dir logs/libero_idm \
  --wandb_project IDM_LIBERO_abs_action \
  --run_name libero_abs_ee
```

## Metadata Format

The default exported JSONL uses:
- `video`
- `split`
- `frame_indices`
- `num_frames`
- `abs_action`

Example JSONL row:

```json
{
  "sample_id": "libero_10/KITCHEN_SCENE10_turn_on_the_stove_and_put_the_moka_pot_on_it/demo_0",
  "video": "videos/libero_10/KITCHEN_SCENE10_turn_on_the_stove_and_put_the_moka_pot_on_it/demo_0.mp4",
  "split": "train",
  "frame_indices": [0, 1, 2],
  "num_frames": 3,
  "abs_action": [
    [-0.1, 0.0, 0.9, 3.0, -0.2, -0.1, 0.04],
    [-0.1, 0.0, 0.9, 3.0, -0.2, -0.1, 0.04],
    [-0.1, 0.0, 0.9, 3.0, -0.2, -0.1, 0.04]
  ]
}
```

The generic trainer also still supports:
- per-frame CSV/JSONL image metadata with `image`
- `abs_action` as a single JSON list
- `abs_action_0 ... abs_action_N` indexed columns
