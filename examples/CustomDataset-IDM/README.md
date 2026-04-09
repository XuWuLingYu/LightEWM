# CustomDataset IDM

This example trains AnyPos from custom `image + abs action` metadata.

Important:
- Only absolute actions are supported here.
- Do not feed relative / delta actions into this IDM path.

## Metadata Format

Supported metadata files:
- `.csv`
- `.jsonl`

Minimum fields:
- `image`
- absolute action, using either:
  - `abs_action` as a JSON list
  - `abs_action_0`, `abs_action_1`, ...

Recommended field:
- `split`

Example CSV:

```csv
image,split,abs_action_0,abs_action_1,abs_action_2,abs_action_3,abs_action_4,abs_action_5
images/000001.jpg,train,0.1,0.2,0.3,1.2,-0.1,0.0
images/000002.jpg,val,0.0,0.1,0.4,1.1,-0.2,0.1
```

Example JSONL:

```jsonl
{"image":"images/000001.jpg","split":"train","abs_action":[0.1,0.2,0.3,1.2,-0.1,0.0]}
{"image":"images/000002.jpg","split":"val","abs_action":[0.0,0.1,0.4,1.1,-0.2,0.1]}
```

If `split` is absent, the trainer will create deterministic `train/val/test` splits from `image` or `id_key`.

## Train

```bash
bash examples/CustomDataset-IDM/train_abs_action.sh \
  --metadata-path data/your_dataset_idm/metadata_abs_action.csv \
  --image-base-path data/your_dataset_idm \
  --save-dir logs/customdataset_idm
```

## Direct Python Command

```bash
accelerate launch \
  --num_processes 4 \
  third_parties/AnyPos/train_metadata_abs_action.py \
  --metadata_path data/your_dataset_idm/metadata_abs_action.csv \
  --image_base_path data/your_dataset_idm \
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
  --save_dir logs/customdataset_idm \
  --wandb_project IDM_CustomDataset_abs_action \
  --run_name custom_abs_action
```
