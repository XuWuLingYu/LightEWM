#!/usr/bin/env bash

set -euo pipefail
START_TS=$(date +%s)

log() {
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"
}

usage() {
  cat <<'EOF'
Usage: bash scripts/infer.sh --config <yaml> --dataset-base-path <path> --metadata-path <path> [options]

Required:
  --config PATH             Inference config YAML
  --dataset-base-path PATH  Dataset root used for inference inputs
  --metadata-path PATH      Metadata CSV/JSON/JSONL for inference

Optional:
  --ckpt PATH               Optional .safetensors override for model.params.model_paths[0]
  --override KEY=VALUE      Extra run.py override; may be repeated
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

CONFIG_PATH=""
DATASET_BASE_PATH=""
METADATA_PATH=""
CKPT_PATH=""
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
    --ckpt)
      CKPT_PATH="$2"
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

if [[ -z "$CONFIG_PATH" || -z "$DATASET_BASE_PATH" || -z "$METADATA_PATH" ]]; then
  echo "[ERROR] --config, --dataset-base-path, and --metadata-path are required." >&2
  usage >&2
  exit 1
fi

cd "$REPO_ROOT"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERROR] config not found: $CONFIG_PATH" >&2
  exit 1
fi

if [[ ! -f "$METADATA_PATH" ]]; then
  echo "[ERROR] metadata file not found: $METADATA_PATH" >&2
  exit 1
fi

log "Starting inference."
log "Repo root: $REPO_ROOT"
log "Config: $CONFIG_PATH"
log "Dataset base path: $DATASET_BASE_PATH"
log "Metadata path: $METADATA_PATH"
if [[ -n "$CKPT_PATH" ]]; then
  log "Checkpoint override: $CKPT_PATH"
fi

cmd=(
  python
  run.py
  --config "$CONFIG_PATH"
  --overrides
  "dataset.params.base_path=$DATASET_BASE_PATH"
  "dataset.params.metadata_path=$METADATA_PATH"
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
log "DONE: inference completed"
log "Elapsed time: ${elapsed}s"
