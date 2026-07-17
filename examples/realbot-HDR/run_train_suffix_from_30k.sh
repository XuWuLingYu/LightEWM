#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$ROOT"
source "$ROOT/examples/realbot-HDR/run_env.sh"

RUN_ID=${RUN_ID:-suffix_from_30k_$(date +%Y%m%d_%H%M%S)}
START_STEP=${START_STEP:-30000}
FINAL_STEP=${FINAL_STEP:-50000}
TRAIN_GPUS=${TRAIN_GPUS:-8,9,10,11,12,13,14,15}
TRAIN_MASTER_PORT=${TRAIN_MASTER_PORT:-29661}
CKPT=${CKPT:?set CKPT=/path/to/checkpoint_model_xxxxxx/model.pt}
LOG_DIR=logs/realbot_hdr_launch
mkdir -p "$LOG_DIR"

if [ ! -f "$CKPT" ]; then
  echo "Missing checkpoint: $CKPT" >&2
  exit 1
fi

export LIGHTEWM_RUN_ID="$RUN_ID"
export CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"

echo "[suffix-train] $(date '+%F %T') run_id=$RUN_ID start=$START_STEP final=$FINAL_STEP ckpt=$CKPT gpus=$CUDA_VISIBLE_DEVICES"
python3 run.py --config examples/realbot-HDR/train_suffix.yaml \
  --overrides \
    runner.params.master_port="$TRAIN_MASTER_PORT" \
    runner.params.causal_config_overrides.generator_ckpt="$CKPT" \
    runner.params.causal_config_overrides.resume_step="$START_STEP" \
    runner.params.causal_config_overrides.max_train_steps="$FINAL_STEP"
