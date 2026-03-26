# LIBERO Guide

## 1) Download LIBERO dataset

```bash
mkdir -p data
cd data
hf download \
  yifengzhu-hf/LIBERO-datasets \
  --repo-type dataset \
  --local-dir ./LIBERO-datasets
cd ..
```

## 2) Convert LIBERO hdf5 to trainable CSV + videos

```bash
python scripts/convert_libero_to_csv.py \
  --libero-root data/LIBERO-datasets \
  --output-dir data/libero_i2v_train \
  --suites libero_10,libero_90,libero_goal,libero_object,libero_spatial \
  --fps 16 \
  --camera-key agentview_rgb,eye_in_hand_rgb \
  --prompt-source attr_or_filename
```

Outputs:
- `data/libero_i2v_train/metadata.csv`
- `data/libero_i2v_train/videos/*.mp4`

## 3) Generate latent cache

```bash
bash scripts/process_libero_cache.sh
```

## 4) Full training

Optional W&B setup:

```bash
wandb login
# or disable
export WANDB_MODE=disabled
```

```bash
accelerate launch run.py --config configs/libero/train_full.yaml
```

Or:

```bash
bash scripts/train_libero_full.sh
```

## 5) Batch inference

```bash
python run.py --config configs/libero/infer.yaml
```

Or:

```bash
bash scripts/infer_libero.sh
```

Output naming (default):
- `<row_id>__<demo_id>__<camera_key>.mp4`

Output directory layout:
- `logs/libero_infer/<timestamp>/`
