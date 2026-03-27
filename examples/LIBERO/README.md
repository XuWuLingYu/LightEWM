# LIBERO Example

This example shows the full Wan2.1-1.3B I2V workflow on LIBERO:
- dataset download
- HDF5 to video/CSV conversion
- latent-cache preprocessing
- full training
- batch inference

## 1) Download LIBERO

```bash
cd /root/to/LightEWM/
mkdir -p data
cd data
hf download \
  yifengzhu-hf/LIBERO-datasets \
  --repo-type dataset \
  --local-dir ./LIBERO-datasets
cd ..
```

## 2) Convert LIBERO to training CSV + videos

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

## 3) Build latent cache

```bash
bash examples/LIBERO/process_cache.sh
```

## 4) Train

Optional W&B setup:

```bash
wandb login
# or disable logging
export WANDB_MODE=disabled
```

```bash
bash examples/LIBERO/train_full.sh
```

## 5) Infer

```bash
bash examples/LIBERO/infer.sh
```

Optional checkpoint override:

```bash
bash examples/LIBERO/infer.sh --ckpt /path/to/your/ckpt.safetensors
```

If `--ckpt` is not provided, inference uses the official Wan2.1-Fun-1.3B I2V checkpoint from the example config.
