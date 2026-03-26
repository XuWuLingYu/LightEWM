#!/usr/bin/env bash

set -euo pipefail
START_TS=$(date +%s)

log() {
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"
}

REPO_ROOT=${REPO_ROOT:-/mnt/world_foundational_model/wfm_ckp-fileset/qianzezhong/LightEWM}
CONFIG_PATH=${CONFIG_PATH:-configs/libero/cache.yaml}
NUM_MACHINES=${NUM_MACHINES:-1}
MIXED_PRECISION=${MIXED_PRECISION:-no}
DYNAMO_BACKEND=${DYNAMO_BACKEND:-no}

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "[ERROR] REPO_ROOT does not exist: $REPO_ROOT" >&2
  exit 1
fi

cd "$REPO_ROOT"

if ! command -v accelerate >/dev/null 2>&1; then
  echo "[ERROR] 'accelerate' is not found in PATH." >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERROR] config not found: $CONFIG_PATH" >&2
  exit 1
fi

gpu_count="unknown"
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
fi

if [[ -z "${NUM_PROCESSES:-}" ]]; then
  if [[ "$gpu_count" =~ ^[0-9]+$ ]] && [[ "$gpu_count" -gt 0 ]]; then
    NUM_PROCESSES="$gpu_count"
  else
    NUM_PROCESSES="1"
  fi
fi

log "Starting LIBERO cache preprocessing via config."
log "Repo root: $REPO_ROOT"
log "Config: $CONFIG_PATH"
log "Detected GPUs: $gpu_count"
log "Accelerate config: num_processes=$NUM_PROCESSES, num_machines=$NUM_MACHINES, mixed_precision=$MIXED_PRECISION, dynamo_backend=$DYNAMO_BACKEND"

cmd=(
  accelerate launch
  --num_processes "$NUM_PROCESSES"
  --num_machines "$NUM_MACHINES"
  --mixed_precision "$MIXED_PRECISION"
  --dynamo_backend "$DYNAMO_BACKEND"
  run.py
  --config "$CONFIG_PATH"
)

printf -v cmd_str '%q ' "${cmd[@]}"
log "Launch command: $cmd_str"

"${cmd[@]}"

end_ts=$(date +%s)
elapsed=$((end_ts - START_TS))
log "DONE: cache preprocessing completed"
log "Elapsed time: ${elapsed}s"
