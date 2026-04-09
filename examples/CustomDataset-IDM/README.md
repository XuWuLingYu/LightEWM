# CustomDataset IDM

This example trains AnyPos from custom metadata. The metadata can be either:
- `image + abs action`
- `video + per-frame abs action list`

Important:
- Only absolute actions are supported here.
- Do not feed relative / delta actions into this IDM path.

## Metadata Format

Supported metadata files:
- `.csv`
- `.jsonl`

Minimum fields:
- one of:
  - `image`
  - `video`
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

Example video JSONL:

```jsonl
{"video":"videos/demo_0001.mp4","split":"train","frame_indices":[0,1,2],"abs_action":[[0.1,0.2,0.3,1.2,-0.1,0.0,0.04],[0.1,0.2,0.3,1.2,-0.1,0.0,0.04],[0.0,0.1,0.4,1.1,-0.2,0.1,0.02]]}
{"video":"videos/demo_0002.mp4","split":"val","abs_action":[[0.0,0.1,0.4,1.1,-0.2,0.1,0.02],[0.0,0.1,0.4,1.1,-0.2,0.1,0.02]]}
```

For video metadata:
- `abs_action` must be a list with one action vector per exported frame
- `frame_indices` is optional; if omitted, the trainer assumes `0..T-1`

If `split` is absent, the trainer will create deterministic `train/val/test` splits from `image`, `video`, or `id_key`.

## Train

```bash
bash examples/CustomDataset-IDM/train_abs_action.sh \
  --metadata-path data/your_dataset_idm/metadata_abs_action.jsonl \
  --image-base-path data/your_dataset_idm \
  --save-dir logs/customdataset_idm
```

## Direct Python Command

```bash
accelerate launch \
  --num_processes 4 \
  third_parties/AnyPos/train_metadata_abs_action.py \
  --metadata_path data/your_dataset_idm/metadata_abs_action.jsonl \
  --image_base_path data/your_dataset_idm \
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
  --save_dir logs/customdataset_idm \
  --wandb_project IDM_CustomDataset_abs_action \
  --run_name custom_abs_action
```
