#!/usr/bin/env bash

set -euo pipefail
START_TS=$(date +%s)

log() {
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"
}

usage() {
  cat <<'EOF'
Usage: bash scripts/train_full.sh --config <yaml> --dataset-base-path <path> [options]

Required:
  --config PATH             Training config YAML
  --dataset-base-path PATH  Latent cache directory used for training

Optional:
  --override KEY=VALUE      Extra run.py override; may be repeated
  --ckpt PATH               Optional .safetensors override for model.params.model_paths[0]
  --accelerate-config PATH  Optional accelerate config; defaults to configs/accelerate/deepspeed_zero3.yaml

Environment:
  ACCELERATE_CONFIG_FILE    accelerate --config_file; defaults to configs/accelerate/deepspeed_zero3.yaml
  NUM_PROCESSES             accelerate --num_processes; defaults to detected GPU count
  NUM_MACHINES              accelerate --num_machines (default: 1)
  MIXED_PRECISION           accelerate --mixed_precision (default: bf16)
  DYNAMO_BACKEND            accelerate --dynamo_backend (default: no)
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

CONFIG_PATH=""
DATASET_BASE_PATH=""
CKPT_PATH=""
DEFAULT_ACCELERATE_CONFIG="configs/accelerate/deepspeed_zero3.yaml"
ACCELERATE_CONFIG_FILE=${ACCELERATE_CONFIG_FILE:-$DEFAULT_ACCELERATE_CONFIG}
NUM_MACHINES=${NUM_MACHINES:-1}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
DYNAMO_BACKEND=${DYNAMO_BACKEND:-no}
EXTRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --dataset-base-path)
      DATASET_BASE_PATH="$2"
      shift 2
      ;;
    --ckpt)
      CKPT_PATH="$2"
      shift 2
      ;;
    --accelerate-config)
      ACCELERATE_CONFIG_FILE="$2"
      shift 2
      ;;
    --override)
      EXTRA_OVERRIDES+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$CONFIG_PATH" || -z "$DATASET_BASE_PATH" ]]; then
  echo "[ERROR] --config and --dataset-base-path are required." >&2
  usage >&2
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

if [[ -n "$ACCELERATE_CONFIG_FILE" && ! -f "$ACCELERATE_CONFIG_FILE" ]]; then
  echo "[ERROR] accelerate config not found: $ACCELERATE_CONFIG_FILE" >&2
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

log "Starting full training."
log "Repo root: $REPO_ROOT"
log "Config: $CONFIG_PATH"
log "Dataset base path: $DATASET_BASE_PATH"
if [[ -n "$CKPT_PATH" ]]; then
  log "Checkpoint override: $CKPT_PATH"
fi
log "Detected GPUs: $gpu_count"
log "Accelerate config: config_file=$ACCELERATE_CONFIG_FILE, num_processes=$NUM_PROCESSES, num_machines=$NUM_MACHINES, mixed_precision=$MIXED_PRECISION, dynamo_backend=$DYNAMO_BACKEND"
LIGHTEWM_RUN_ID=${LIGHTEWM_RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}
export LIGHTEWM_RUN_ID
log "Shared run id: $LIGHTEWM_RUN_ID"

cmd=(
  accelerate launch
  --config_file "$ACCELERATE_CONFIG_FILE"
  --num_processes "$NUM_PROCESSES"
  --num_machines "$NUM_MACHINES"
  --mixed_precision "$MIXED_PRECISION"
  --dynamo_backend "$DYNAMO_BACKEND"
  run.py
  --config "$CONFIG_PATH"
  --overrides
  "dataset.params.dataset_base_path=$DATASET_BASE_PATH"
)

if [[ -n "$CKPT_PATH" ]]; then
  cmd+=(--ckpt "$CKPT_PATH")
fi

if [[ ${#EXTRA_OVERRIDES[@]} -gt 0 ]]; then
  cmd+=("${EXTRA_OVERRIDES[@]}")
fi

printf -v cmd_str '%q ' "${cmd[@]}"
log "Launch command: $cmd_str"

"${cmd[@]}"

end_ts=$(date +%s)
elapsed=$((end_ts - START_TS))
log "DONE: training completed"
log "Elapsed time: ${elapsed}s"
