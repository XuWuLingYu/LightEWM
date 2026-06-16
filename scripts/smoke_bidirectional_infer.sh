#!/usr/bin/env bash
set -euo pipefail

ASSET_ROOT=${ASSET_ROOT:-/pfs-verdent/zhangyu/robot-trial}
CONFIG=${CONFIG:-examples/LIBERO/infer_ti2v_5b.yaml}
DATASET_BASE_PATH=""
METADATA_PATH=""
CKPT_PATH=""
MAX_SAMPLES=1
INFER_STEPS=2
OUTPUT_ROOT=""

usage() {
  cat <<'EOF'
Usage: bash scripts/smoke_bidirectional_infer.sh [options]

Options:
  --asset-root PATH         Large-file root. Default: /pfs-verdent/zhangyu/robot-trial
  --config PATH             Inference config. Default: examples/LIBERO/infer_ti2v_5b.yaml
  --dataset-base-path PATH  Converted LIBERO data root. Default: <asset-root>/data/libero_i2v_train
  --metadata-path PATH      Metadata CSV. Default: <asset-root>/data/libero_i2v_train/metadata_dense_prompt.csv
  --ckpt PATH               DiT checkpoint. Default: <asset-root>/checkpoints/Wan2.2-5B-Libero/checkpoint.safetensors
  --max-samples N           Number of samples. Default: 1
  --infer-steps N           Diffusion steps. Default: 2
  --output-root PATH        Output root for smoke metrics/log pointers
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --asset-root) ASSET_ROOT="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --dataset-base-path) DATASET_BASE_PATH="$2"; shift 2 ;;
    --metadata-path) METADATA_PATH="$2"; shift 2 ;;
    --ckpt) CKPT_PATH="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --infer-steps) INFER_STEPS="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

DATASET_BASE_PATH=${DATASET_BASE_PATH:-$ASSET_ROOT/data/libero_i2v_train}
METADATA_PATH=${METADATA_PATH:-$ASSET_ROOT/data/libero_i2v_train/metadata_dense_prompt.csv}
CKPT_PATH=${CKPT_PATH:-$ASSET_ROOT/checkpoints/Wan2.2-5B-Libero/checkpoint.safetensors}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/smoke_bidirectional_infer/$(date -u +%Y%m%d_%H%M%S)}

python scripts/evaluate_video_quality.py \
  --config "$CONFIG" \
  --asset-root "$ASSET_ROOT" \
  --dataset-base-path "$DATASET_BASE_PATH" \
  --metadata-path "$METADATA_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --weight "bidirectional_smoke=$CKPT_PATH" \
  --max-samples "$MAX_SAMPLES" \
  --infer-steps "$INFER_STEPS" \
  --metrics ssim,psnr \
  --allow-missing-pairs
