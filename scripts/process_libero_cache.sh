#!/usr/bin/env bash

set -euo pipefail
START_TS=$(date +%s)

log() {
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"
}

# Use documented repo/env paths by default.
REPO_ROOT=${REPO_ROOT:-/mnt/world_foundational_model/wfm_ckp-fileset/qianzezhong/LightEWM}

# LIBERO converted dataset path (from docs/libero.md).
DATASET_BASE=${DATASET_BASE:-data/libero_i2v_train}
DATASET_META=${DATASET_META:-data/libero_i2v_train/metadata.csv}

# Fixed preprocessing options requested.
FPS=${FPS:-16}
RESIZE_MODE=${RESIZE_MODE:-letterbox}
CONTEXT_SHORT_MODE=${CONTEXT_SHORT_MODE:-drop}
CONTEXT_STRIDE=${CONTEXT_STRIDE:-81}
CONTEXT_TAIL_ALIGN=${CONTEXT_TAIL_ALIGN:-true}

OUTPUT_PATH=${OUTPUT_PATH:-./data/libero_i2v_train/latent_cache}
NUM_MACHINES=${NUM_MACHINES:-1}
MIXED_PRECISION=${MIXED_PRECISION:-no}
DYNAMO_BACKEND=${DYNAMO_BACKEND:-no}
CONTEXT_WINDOW_WAIT_TIMEOUT=${CONTEXT_WINDOW_WAIT_TIMEOUT:-7200}

MODEL_PATHS='["checkpoints/Wan2.1-I2V-1.3B/diffusion_pytorch_model.safetensors","checkpoints/Wan2.1-I2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth","checkpoints/Wan2.1-I2V-1.3B/Wan2.1_VAE.pth","checkpoints/Wan2.1-I2V-1.3B/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"]'

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "[ERROR] REPO_ROOT does not exist: $REPO_ROOT" >&2
  exit 1
fi

cd "$REPO_ROOT"

if [[ ! -f "$DATASET_META" ]]; then
  echo "[ERROR] metadata.csv not found: $DATASET_META" >&2
  exit 1
fi

if ! command -v accelerate >/dev/null 2>&1; then
  echo "[ERROR] 'accelerate' is not found in PATH." >&2
  exit 1
fi

tail_align_enabled="false"
if [[ "${CONTEXT_TAIL_ALIGN,,}" == "true" ]]; then
  tail_align_enabled="true"
fi

meta_lines=$(wc -l < "$DATASET_META" || echo 0)
if [[ "$meta_lines" -gt 0 ]]; then
  meta_rows=$((meta_lines - 1))
else
  meta_rows=0
fi

gpu_count="unknown"
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
fi

# Auto use all visible GPUs unless NUM_PROCESSES is explicitly set.
if [[ -z "${NUM_PROCESSES:-}" ]]; then
  if [[ "$gpu_count" =~ ^[0-9]+$ ]] && [[ "$gpu_count" -gt 0 ]]; then
    NUM_PROCESSES="$gpu_count"
  else
    NUM_PROCESSES="1"
  fi
fi

log "Starting LIBERO cache preprocessing."
log "Repo root: $REPO_ROOT"
log "Dataset base: $DATASET_BASE"
log "Metadata: $DATASET_META (rows=$meta_rows)"
log "Output path: $OUTPUT_PATH"
log "Detected GPUs: $gpu_count"
log "Accelerate config: num_processes=$NUM_PROCESSES, num_machines=$NUM_MACHINES, mixed_precision=$MIXED_PRECISION, dynamo_backend=$DYNAMO_BACKEND"
log "Preprocess config: fps=$FPS, resize_mode=$RESIZE_MODE, short_video_mode=$CONTEXT_SHORT_MODE, context_stride=$CONTEXT_STRIDE, tail_align=$tail_align_enabled, context_wait_timeout=$CONTEXT_WINDOW_WAIT_TIMEOUT"

cmd=(
  accelerate launch
  --num_processes "$NUM_PROCESSES"
  --num_machines "$NUM_MACHINES"
  --mixed_precision "$MIXED_PRECISION"
  --dynamo_backend "$DYNAMO_BACKEND"
  lightewm/wanvideo/model_training/train.py
  --dataset_base_path "$DATASET_BASE"
  --dataset_metadata_path "$DATASET_META"
  --height 480
  --width 832
  --dataset_repeat 1
  --model_paths "$MODEL_PATHS"
  --output_path "$OUTPUT_PATH"
  --trainable_models "dit"
  --extra_inputs "input_image,end_image"
  --fps "$FPS"
  --resize_mode "$RESIZE_MODE"
  --context_window_short_video_mode "$CONTEXT_SHORT_MODE"
  --context_window_stride "$CONTEXT_STRIDE"
  --context_window_wait_timeout "$CONTEXT_WINDOW_WAIT_TIMEOUT"
  --task "sft:data_process"
)

if [[ "$tail_align_enabled" == "true" ]]; then
  cmd+=(--context_window_tail_align)
fi

printf -v cmd_str '%q ' "${cmd[@]}"
log "Launch command: $cmd_str"

"${cmd[@]}"

end_ts=$(date +%s)
elapsed=$((end_ts - START_TS))
log "DONE: full cache generated at $OUTPUT_PATH"
log "Elapsed time: ${elapsed}s"
