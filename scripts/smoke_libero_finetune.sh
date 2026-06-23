#!/usr/bin/env bash
set -euo pipefail

ASSET_ROOT=${ASSET_ROOT:-/pfs-verdent/zhangyu/robot-trial}
CONFIG=${CONFIG:-examples/LIBERO/train_full_ti2v_5b.yaml}
CACHE_PATH=""
VALIDATION_BASE_PATH=""
VALIDATION_METADATA_PATH=""
BASE_MODEL_DIR=""
CKPT_PATH=""
MAX_TRAIN_STEPS=1
MAX_DATA_ITEMS=1
NUM_PROCESSES=${NUM_PROCESSES:-1}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
DYNAMO_BACKEND=${DYNAMO_BACKEND:-no}
RUN_ID=${LIGHTEWM_RUN_ID:-smoke_libero_finetune_$(date -u +%Y%m%d_%H%M%S)}

usage() {
  cat <<'EOF'
Usage: bash scripts/smoke_libero_finetune.sh [options]

Options:
  --asset-root PATH          Large-file root. Default: /pfs-verdent/zhangyu/robot-trial
  --config PATH              Training config. Default: examples/LIBERO/train_full_ti2v_5b.yaml
  --cache-path PATH          Latent cache path. Default: <asset-root>/data/libero_i2v_train/latent_cache_ti2v_5b
  --validation-base-path PATH     Default: <asset-root>/data/libero_i2v_train
  --validation-metadata-path PATH Default: <asset-root>/data/libero_i2v_train/metadata_dense_prompt.csv
  --base-model-dir PATH      Default: <asset-root>/checkpoints/Wan2.2-TI2V-5B
  --ckpt PATH                Default: <asset-root>/checkpoints/Wan2.2-5B-Robot/checkpoint.safetensors
  --max-train-steps N        Default: 1
  --max-data-items N         Default: 1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --asset-root) ASSET_ROOT="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --cache-path) CACHE_PATH="$2"; shift 2 ;;
    --validation-base-path) VALIDATION_BASE_PATH="$2"; shift 2 ;;
    --validation-metadata-path) VALIDATION_METADATA_PATH="$2"; shift 2 ;;
    --base-model-dir) BASE_MODEL_DIR="$2"; shift 2 ;;
    --ckpt) CKPT_PATH="$2"; shift 2 ;;
    --max-train-steps) MAX_TRAIN_STEPS="$2"; shift 2 ;;
    --max-data-items) MAX_DATA_ITEMS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

CACHE_PATH=${CACHE_PATH:-$ASSET_ROOT/data/libero_i2v_train/latent_cache_ti2v_5b}
VALIDATION_BASE_PATH=${VALIDATION_BASE_PATH:-$ASSET_ROOT/data/libero_i2v_train}
VALIDATION_METADATA_PATH=${VALIDATION_METADATA_PATH:-$ASSET_ROOT/data/libero_i2v_train/metadata_dense_prompt.csv}
BASE_MODEL_DIR=${BASE_MODEL_DIR:-$ASSET_ROOT/checkpoints/Wan2.2-TI2V-5B}
CKPT_PATH=${CKPT_PATH:-$ASSET_ROOT/checkpoints/Wan2.2-5B-Robot/checkpoint.safetensors}

for required in "$CONFIG" "$CACHE_PATH" "$VALIDATION_BASE_PATH" "$VALIDATION_METADATA_PATH" "$CKPT_PATH" "$BASE_MODEL_DIR/google/umt5-xxl"; do
  if [[ ! -e "$required" ]]; then
    echo "[ERROR] required path not found: $required" >&2
    exit 1
  fi
done

MODEL_PATHS="[[\"$BASE_MODEL_DIR/diffusion_pytorch_model-00001-of-00003.safetensors\",\"$BASE_MODEL_DIR/diffusion_pytorch_model-00002-of-00003.safetensors\",\"$BASE_MODEL_DIR/diffusion_pytorch_model-00003-of-00003.safetensors\"],\"$BASE_MODEL_DIR/models_t5_umt5-xxl-enc-bf16.pth\",\"$BASE_MODEL_DIR/Wan2.2_VAE.pth\"]"

export LIGHTEWM_RUN_ID="$RUN_ID"
export WANDB_MODE=${WANDB_MODE:-disabled}

accelerate launch \
  --num_processes "$NUM_PROCESSES" \
  --num_machines 1 \
  --mixed_precision "$MIXED_PRECISION" \
  --dynamo_backend "$DYNAMO_BACKEND" \
  run.py \
  --config "$CONFIG" \
  --overrides \
  "model.params.tokenizer_path=$BASE_MODEL_DIR/google/umt5-xxl" \
  "model.params.model_paths=$MODEL_PATHS" \
  "dataset.params.dataset_base_path=$CACHE_PATH" \
  "dataset.params.max_data_items=$MAX_DATA_ITEMS" \
  "validation_dataset.params.base_path=$VALIDATION_BASE_PATH" \
  "validation_dataset.params.metadata_path=$VALIDATION_METADATA_PATH" \
  "validation_dataset.params.max_samples=1" \
  "runner.params.max_train_steps=$MAX_TRAIN_STEPS" \
  "runner.params.num_epochs=1" \
  "runner.params.save_steps=1" \
  "runner.params.validation_every_steps=999999" \
  "runner.params.validation_extra_steps=[]" \
  "runner.params.wandb_enabled=false" \
  "runner.params.dataset_num_workers=0" \
  "runtime.params.gradient_accumulation_steps=1" \
  --ckpt "$CKPT_PATH"

echo "[SmokeTrain] DONE: logs/LIBERO_train_full_ti2v_5b/$RUN_ID"
