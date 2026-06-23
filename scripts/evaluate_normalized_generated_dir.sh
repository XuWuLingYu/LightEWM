#!/usr/bin/env bash
set -euo pipefail

DATASET_BASE_PATH=${DATASET_BASE_PATH:-/pfs-verdent/zhangyu/robot-trial/data/libero_i2v_train}
SPLIT_ROOT=${SPLIT_ROOT:-/pfs-verdent/zhangyu/robot-trial/eval_splits/libero_80_original_prompt}
METADATA_PATH=${METADATA_PATH:-$SPLIT_ROOT/metadata.csv}
GENERATED_DIR=""
NORMALIZED_DIR=""
OUTPUT_ROOT=""
WEIGHT_NAME=${WEIGHT_NAME:-causal}
PAIR_HEIGHT=${PAIR_HEIGHT:-224}
PAIR_WIDTH=${PAIR_WIDTH:-224}
METRICS=${METRICS:-fvd,ssim,psnr,lpips}

usage() {
  cat <<'EOF'
Usage: bash scripts/evaluate_normalized_generated_dir.sh --generated-dir DIR --normalized-dir DIR --output-root DIR [options]

Options:
  --metadata-path PATH
  --dataset-base-path PATH
  --weight-name NAME
  --pair-height N
  --pair-width N
  --metrics CSV
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --generated-dir) GENERATED_DIR="$2"; shift 2 ;;
    --normalized-dir) NORMALIZED_DIR="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --metadata-path) METADATA_PATH="$2"; shift 2 ;;
    --dataset-base-path) DATASET_BASE_PATH="$2"; shift 2 ;;
    --weight-name) WEIGHT_NAME="$2"; shift 2 ;;
    --pair-height) PAIR_HEIGHT="$2"; shift 2 ;;
    --pair-width) PAIR_WIDTH="$2"; shift 2 ;;
    --metrics) METRICS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$GENERATED_DIR" || -z "$NORMALIZED_DIR" || -z "$OUTPUT_ROOT" ]]; then
  usage >&2
  exit 1
fi

python scripts/normalize_generated_videos.py \
  --metadata-path "$METADATA_PATH" \
  --generated-dir "$GENERATED_DIR" \
  --output-dir "$NORMALIZED_DIR"

python scripts/evaluate_video_quality.py \
  --skip-inference \
  --metadata-path "$METADATA_PATH" \
  --dataset-base-path "$DATASET_BASE_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --weight "$WEIGHT_NAME=none" \
  --generated-dir "$WEIGHT_NAME=$NORMALIZED_DIR" \
  --max-samples 80 \
  --pair-height "$PAIR_HEIGHT" \
  --pair-width "$PAIR_WIDTH" \
  --metrics "$METRICS"
