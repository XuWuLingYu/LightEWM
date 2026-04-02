#!/usr/bin/env bash

set -euo pipefail
START_TS=$(date +%s)

log() {
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"
}

usage() {
  cat <<'EOF'
Usage: bash scripts/process_cache.sh --config <yaml> --dataset-base-path <path> --metadata-path <path> --output-path <path> [options]

Required:
  --config PATH             Cache config YAML
  --dataset-base-path PATH  Dataset root used by the config
  --metadata-path PATH      Metadata CSV/JSON/JSONL for cache preprocessing
  --output-path PATH        Latent cache output path

Optional:
  --override KEY=VALUE      Extra run.py override; may be repeated
  --overwrite               Delete existing output path before cache preprocessing

Environment:
  NUM_PROCESSES             accelerate --num_processes; defaults to detected GPU count
  NUM_MACHINES              accelerate --num_machines (default: 1)
  MIXED_PRECISION           accelerate --mixed_precision (default: no)
  DYNAMO_BACKEND            accelerate --dynamo_backend (default: no)
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

CONFIG_PATH=""
DATASET_BASE_PATH=""
METADATA_PATH=""
OUTPUT_PATH=""
OVERWRITE=false
NUM_MACHINES=${NUM_MACHINES:-1}
MIXED_PRECISION=${MIXED_PRECISION:-no}
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
    --metadata-path)
      METADATA_PATH="$2"
      shift 2
      ;;
    --output-path)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=true
      shift 1
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

if [[ -z "$CONFIG_PATH" || -z "$DATASET_BASE_PATH" || -z "$METADATA_PATH" || -z "$OUTPUT_PATH" ]]; then
  echo "[ERROR] --config, --dataset-base-path, --metadata-path, and --output-path are required." >&2
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

if [[ ! -f "$METADATA_PATH" ]]; then
  echo "[ERROR] metadata file not found: $METADATA_PATH" >&2
  exit 1
fi

if [[ -e "$OUTPUT_PATH" ]]; then
  if [[ "$OVERWRITE" != "true" ]]; then
    echo "[ERROR] output path already exists: $OUTPUT_PATH" >&2
    echo "[ERROR] Refusing to reuse an existing cache directory. Pass --overwrite to delete it first." >&2
    exit 1
  fi

  output_abs=$(python - "$OUTPUT_PATH" <<'PY'
import os, sys
print(os.path.realpath(sys.argv[1]))
PY
)
  dataset_abs=$(python - "$DATASET_BASE_PATH" <<'PY'
import os, sys
print(os.path.realpath(sys.argv[1]))
PY
)
  repo_abs=$(python - "$REPO_ROOT" <<'PY'
import os, sys
print(os.path.realpath(sys.argv[1]))
PY
)

  if [[ "$output_abs" == "/" || "$output_abs" == "$repo_abs" || "$output_abs" == "$dataset_abs" ]]; then
    echo "[ERROR] refusing to delete unsafe output path: $OUTPUT_PATH" >&2
    echo "[ERROR] resolved path: $output_abs" >&2
    exit 1
  fi

  log "Removing existing cache output path because --overwrite was set: $OUTPUT_PATH"
  rm -rf -- "$OUTPUT_PATH"
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

log "Starting latent-cache preprocessing."
log "Repo root: $REPO_ROOT"
log "Config: $CONFIG_PATH"
log "Dataset base path: $DATASET_BASE_PATH"
log "Metadata path: $METADATA_PATH"
log "Output path: $OUTPUT_PATH"
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
  --overrides
  "dataset.params.dataset_base_path=$DATASET_BASE_PATH"
  "dataset.params.dataset_metadata_path=$METADATA_PATH"
  "runner.params.output_path=$OUTPUT_PATH"
)

if [[ ${#EXTRA_OVERRIDES[@]} -gt 0 ]]; then
  cmd+=("${EXTRA_OVERRIDES[@]}")
fi

printf -v cmd_str '%q ' "${cmd[@]}"
log "Launch command: $cmd_str"

"${cmd[@]}"

end_ts=$(date +%s)
elapsed=$((end_ts - START_TS))
log "DONE: cache preprocessing completed"
log "Elapsed time: ${elapsed}s"
