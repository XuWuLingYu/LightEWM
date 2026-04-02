#!/usr/bin/env bash

set -euo pipefail
START_TS=$(date +%s)

log() {
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"
}

usage() {
  cat <<'EOF'
Usage: bash scripts/process_dense_prompt.sh --metadata-path <path> --output-path <path> [options]

Required:
  --metadata-path PATH      Input metadata.csv path
  --output-path PATH        Output metadata_dense_prompt.csv path

Optional:
  --model-name NAME         Qwen2.5-VL model name or local path
  --cache-dir PATH          Hugging Face model cache directory
  --dtype TYPE              auto|bfloat16|float16|float32
  --num-frames N            Frames sampled per video
  --max-new-tokens N        Max generated tokens
  --temperature VALUE       Sampling temperature; <=0 disables sampling
  --top-p VALUE             Top-p when sampling
  --save-every N            Save shard CSV every N rows
  --overwrite               Rebuild output instead of resuming

Environment:
  NUM_PROCESSES             accelerate --num_processes; defaults to detected GPU count
  NUM_MACHINES              accelerate --num_machines (default: 1)
  MIXED_PRECISION           accelerate --mixed_precision (default: no)
  DYNAMO_BACKEND            accelerate --dynamo_backend (default: no)
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

METADATA_PATH=""
OUTPUT_PATH=""
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
CACHE_DIR="checkpoints"
DTYPE="bfloat16"
NUM_FRAMES="32"
MAX_NEW_TOKENS="96"
TEMPERATURE="0.0"
TOP_P="0.95"
SAVE_EVERY="20"
OVERWRITE="0"
NUM_MACHINES=${NUM_MACHINES:-1}
MIXED_PRECISION=${MIXED_PRECISION:-no}
DYNAMO_BACKEND=${DYNAMO_BACKEND:-no}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --metadata-path)
      METADATA_PATH="$2"
      shift 2
      ;;
    --output-path)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --cache-dir)
      CACHE_DIR="$2"
      shift 2
      ;;
    --dtype)
      DTYPE="$2"
      shift 2
      ;;
    --num-frames)
      NUM_FRAMES="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --top-p)
      TOP_P="$2"
      shift 2
      ;;
    --save-every)
      SAVE_EVERY="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE="1"
      shift
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

if [[ -z "$METADATA_PATH" || -z "$OUTPUT_PATH" ]]; then
  echo "[ERROR] --metadata-path and --output-path are required." >&2
  usage >&2
  exit 1
fi

cd "$REPO_ROOT"

if ! command -v accelerate >/dev/null 2>&1; then
  echo "[ERROR] 'accelerate' is not found in PATH." >&2
  exit 1
fi

if [[ ! -f "$METADATA_PATH" ]]; then
  echo "[ERROR] metadata file not found: $METADATA_PATH" >&2
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

log "Starting dense-prompt generation."
log "Repo root: $REPO_ROOT"
log "Metadata: $METADATA_PATH"
log "Output: $OUTPUT_PATH"
log "Model: $MODEL_NAME"
log "Model cache dir: $CACHE_DIR"
log "Detected GPUs: $gpu_count"
log "Accelerate config: num_processes=$NUM_PROCESSES, num_machines=$NUM_MACHINES, mixed_precision=$MIXED_PRECISION, dynamo_backend=$DYNAMO_BACKEND"
log "Generation config: dtype=$DTYPE, num_frames=$NUM_FRAMES, max_new_tokens=$MAX_NEW_TOKENS, temperature=$TEMPERATURE, top_p=$TOP_P, save_every=$SAVE_EVERY"

cmd=(
  accelerate launch
  --num_processes "$NUM_PROCESSES"
  --num_machines "$NUM_MACHINES"
  --mixed_precision "$MIXED_PRECISION"
  --dynamo_backend "$DYNAMO_BACKEND"
  scripts/generate_dense_prompt_qwen_vl.py
  --metadata-path "$METADATA_PATH"
  --output-path "$OUTPUT_PATH"
  --model-name "$MODEL_NAME"
  --cache-dir "$CACHE_DIR"
  --dtype "$DTYPE"
  --num-frames "$NUM_FRAMES"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --save-every "$SAVE_EVERY"
)

if [[ "$OVERWRITE" == "1" ]]; then
  cmd+=(--overwrite)
fi

printf -v cmd_str '%q ' "${cmd[@]}"
log "Launch command: $cmd_str"

"${cmd[@]}"

end_ts=$(date +%s)
elapsed=$((end_ts - START_TS))
log "DONE: dense-prompt generation completed"
log "Elapsed time: ${elapsed}s"
