#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$ROOT"
source "$ROOT/examples/realbot-HDR/run_env.sh"

RUN_ID=${RUN_ID:-new100_3cam_suffix10_$(date +%Y%m%d_%H%M%S)}
START_STEP=${START_STEP:-30000}
FINAL_STEP=${FINAL_STEP:-50000}
TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3,4,5,6,7}
TRAIN_MASTER_PORT=${TRAIN_MASTER_PORT:-29671}
DATASET_ROOT=${DATASET_ROOT:-data/realbot_hdr_video_320x384_new100_3cam_61f_suffix10}
METADATA_PATH=${METADATA_PATH:-$DATASET_ROOT/metadata_train_preencoded.csv}
CKPT=${CKPT:?set CKPT=/path/to/checkpoint_model_xxxxxx/model.pt}
LOG_DIR=${LOG_DIR:-logs/realbot_new100_3cam_suffix_launch}
mkdir -p "$LOG_DIR"

if [ ! -f "$CKPT" ]; then
  echo "Missing checkpoint: $CKPT" >&2
  exit 1
fi
if [ ! -f "$METADATA_PATH" ]; then
  echo "Missing metadata: $METADATA_PATH" >&2
  exit 1
fi

export LIGHTEWM_RUN_ID="$RUN_ID"
export CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"

echo "[new100-suffix-train] $(date '+%F %T') run_id=$RUN_ID start=$START_STEP final=$FINAL_STEP ckpt=$CKPT gpus=$CUDA_VISIBLE_DEVICES dataset=$DATASET_ROOT metadata=$METADATA_PATH"
python3 run.py --config examples/realbot-HDR/train_suffix.yaml \
  --overrides \
    runner.params.master_port="$TRAIN_MASTER_PORT" \
    runner.params.causal_config_overrides.generator_ckpt="$CKPT" \
    runner.params.causal_config_overrides.resume_step="$START_STEP" \
    runner.params.causal_config_overrides.max_train_steps="$FINAL_STEP" \
    dataset.params.base_path="$DATASET_ROOT" \
    dataset.params.metadata_path="$METADATA_PATH"
