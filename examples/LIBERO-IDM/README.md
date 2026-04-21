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

### 2.1) Train With 20% World-Model Generated Samples

This pipeline does three steps in sequence:
- generate world-model videos with GT future abs actions (49-step)
- mix real + WM metadata to target 20% generated train samples
- launch IDM training on the mixed metadata

World-model generation now auto-detects local GPU count and uses multi-GPU `torch.distributed.run` by default.
You can override with `--wm-num-processes N` in the one-click script or `--num-processes N` in `build_world_model_metadata.sh`.

```bash
bash examples/LIBERO-IDM/train_abs_ee_wm_mix.sh \
  --target-generated-ratio 0.2 \
  --run-name libero_abs_ee_wm20
```

If you want to run each step manually:

```bash
# 1) Build WM-generated metadata/videos
bash examples/LIBERO-IDM/build_world_model_metadata.sh \
  --target-mix-ratio 0.2

# 2) Mix metadata to 20% generated samples
bash examples/LIBERO-IDM/mix_metadata_with_world_model.sh \
  --target-generated-ratio 0.2

# 3) Train IDM on mixed metadata
accelerate launch \
  --num_processes $NPROC_PER_NODE \
  third_parties/AnyPos/train_metadata_abs_action.py \
  --metadata_path data/libero_idm_abs_action_mix/metadata_abs_action_mix.jsonl \
  --video_key video \
  --action_key abs_action \
  --model_name direction_aware \
  --save_dir logs/libero_idm \
  --wandb_project IDM_LIBERO_abs_action \
  --run_name libero_abs_ee_wm20
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

## 3) Evaluate With Wan + IDM

This rollout evaluator:
- uses the current `agentview` observation as the conditioning frame
- uses the task dense prompt from `data/libero_i2v_train/metadata_dense_prompt.csv`
- generates a future video with Wan
- feeds the first predicted future frame into the IDM
- converts the predicted absolute `7D = ee_states(6) + gripper(1)` target into an `OSC_POSE` delta action
- executes the action in LIBERO and reports success rates

Default checkpoints:
- Wan: `checkpoints/Wan2.2-5B-Libero/checkpoint.safetensors`
- IDM: `checkpoints/LIBERO-IDM/100000.pt`

Default evaluation coverage:
- `libero_object`: 10 tasks, 10 trials per task, horizon `240`
- `libero_goal`: 10 tasks, 10 trials per task, horizon `320`
- `libero_spatial`: 10 tasks, 10 trials per task, horizon `240`
- `libero_10`: 10 tasks, 10 trials per task, horizon `512`

Run:

```bash
bash examples/LIBERO-IDM/eval_video_idm.sh
```

Outputs:
- `outputs/libero_video_idm_eval/summary.json`

Notes:
- This evaluator assumes the LIBERO environment dependencies are installed in the active Python environment.
- It uses the same 180-degree orientation fix as the IDM export path before feeding observations into Wan.
- By default it replans every environment step. This is accurate to the requested setup but expensive. You can increase `--replan-every` if you want a cheaper rollout.
