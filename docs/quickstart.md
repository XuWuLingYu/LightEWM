# Training Quickstart (Wan2.1-Fun-1.3B I2V)

This page focuses on training commands.

For dataset conversion/preprocessing details, see [Data Guide](./data.md) and [LIBERO Guide](./libero.md).

## 0) W&B setup (optional but enabled by default in train configs)

```bash
# enable online logging
wandb login

# disable W&B logging globally
export WANDB_MODE=disabled
```

## 1) Run full training by config

```bash
# generic base config (override dataset paths as needed)
accelerate launch run.py --config configs/wan/i2v/base_train_full.yaml

# LIBERO training config
accelerate launch run.py --config configs/libero/train_full.yaml
```

## 2) Useful overrides

Run output layout is standardized automatically:

```text
logs/{config_name}/{timestamp}/
```

For train runs, this folder contains:
- `checkpoint_XXXXXXXX.safetensors`
- `merged_config.yaml`
- `config_sources.txt`
- `configs/` (current config + all inherited base configs)

```bash
# change epochs / lr
accelerate launch run.py \
  --config configs/libero/train_full.yaml \
  --overrides runner.params.num_epochs=5 runner.params.learning_rate=5e-6

# change per-process batch size
accelerate launch run.py \
  --config configs/libero/train_full.yaml \
  --overrides runner.params.batch_size=2

# disable wandb for a single run
accelerate launch run.py \
  --config configs/libero/train_full.yaml \
  --overrides runner.params.wandb_enabled=false
```

## 3) One-line helper script

```bash
bash scripts/train_libero_full.sh
```

## 4) Inference (optional custom checkpoint)

```bash
# default inference (official Wan2.1-Fun-1.3B I2V checkpoint from config)
python run.py --config configs/libero/infer.yaml

# override with your finetuned DiT checkpoint (.safetensors)
python run.py --config configs/libero/infer.yaml --ckpt /path/to/your/ckpt.safetensors
```
