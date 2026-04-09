# LIBERO IDM

This example trains AnyPos on LIBERO through exported `image + abs action` metadata.

Important:
- This path assumes absolute action targets.
- Do not use raw LIBERO `actions` here. Those are relative control commands.
- The default export uses `obs/ee_states` and writes it as `abs_action_*` columns.
- The exported images are flipped left-right and top-bottom by default.
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
├── metadata_abs_action.csv
└── images/
    └── ...
```

## 2) Train


```bash
accelerate launch \
  --num_processes $NPROC_PER_NODE \
  third_parties/AnyPos/train_metadata_abs_action.py \
  --metadata_path data/libero_idm_abs_action/metadata_abs_action.csv \
  --image_base_path data/libero_idm_abs_action \
  --image_key image \
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

The exported CSV uses:
- `image`
- `split`
- `abs_action_0 ... abs_action_5`

The train script also supports JSONL metadata and a single `abs_action` field when it is a JSON list string / list.
