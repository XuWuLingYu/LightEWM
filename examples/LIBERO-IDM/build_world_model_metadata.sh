#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/envs/lightewm/bin/python}"

SOURCE_METADATA_PATH="$REPO_ROOT/data/libero_idm_abs_action/metadata_abs_action.jsonl"
SOURCE_BASE_PATH="$REPO_ROOT/data/libero_idm_abs_action"
OUTPUT_DIR="$REPO_ROOT/data/libero_idm_abs_action_wm"
PROMPT_METADATA_PATH="$REPO_ROOT/data/libero_i2v_train/metadata_dense_prompt.csv"
VIDEO_CKPT="$REPO_ROOT/checkpoints/Wan2.2-5B-Libero/checkpoint.safetensors"
INFER_CONFIG="$REPO_ROOT/examples/LIBERO/infer_ti2v_5b.yaml"
TARGET_MIX_RATIO="${TARGET_MIX_RATIO:-0.2}"
NUM_GENERATED_ROWS="${NUM_GENERATED_ROWS:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_PROCESSES="${NUM_PROCESSES:-auto}"

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-metadata-path)
      SOURCE_METADATA_PATH="$2"
      shift 2
      ;;
    --source-base-path)
      SOURCE_BASE_PATH="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --prompt-metadata-path)
      PROMPT_METADATA_PATH="$2"
      shift 2
      ;;
    --video-ckpt)
      VIDEO_CKPT="$2"
      shift 2
      ;;
    --infer-config)
      INFER_CONFIG="$2"
      shift 2
      ;;
    --target-mix-ratio)
      TARGET_MIX_RATIO="$2"
      shift 2
      ;;
    --num-generated-rows)
      NUM_GENERATED_ROWS="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --num-processes)
      NUM_PROCESSES="$2"
      shift 2
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done
set -- "${POSITIONAL[@]}"

if [[ "$NUM_PROCESSES" == "auto" ]]; then
  NUM_PROCESSES="$("$PYTHON_BIN" -c 'import torch; n=torch.cuda.device_count() if torch.cuda.is_available() else 1; print(max(1, int(n)))')"
fi

if ! [[ "$NUM_PROCESSES" =~ ^[0-9]+$ ]] || [[ "$NUM_PROCESSES" -lt 1 ]]; then
  echo "Invalid --num-processes: $NUM_PROCESSES" >&2
  exit 1
fi

if [[ "$NUM_PROCESSES" -gt 1 ]]; then
  echo "[build_world_model_metadata] Auto multi-GPU mode: num_processes=$NUM_PROCESSES"
  "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$NUM_PROCESSES" \
    examples/LIBERO-IDM/build_world_model_metadata.py \
    --source-metadata-path "$SOURCE_METADATA_PATH" \
    --source-base-path "$SOURCE_BASE_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --prompt-metadata-path "$PROMPT_METADATA_PATH" \
    --video-ckpt "$VIDEO_CKPT" \
    --infer-config "$INFER_CONFIG" \
    --target-mix-ratio "$TARGET_MIX_RATIO" \
    --num-generated-rows "$NUM_GENERATED_ROWS" \
    --device "$DEVICE" \
    "$@"

  FINAL_METADATA_PATH="$OUTPUT_DIR/metadata_abs_action_wm.jsonl"
  : > "$FINAL_METADATA_PATH"
  for ((rank=0; rank<NUM_PROCESSES; rank++)); do
    printf -v SHARD_PATH "%s/metadata_abs_action_wm.rank%05d.jsonl" "$OUTPUT_DIR" "$rank"
    if [[ ! -f "$SHARD_PATH" ]]; then
      echo "Missing shard metadata: $SHARD_PATH" >&2
      exit 1
    fi
    cat "$SHARD_PATH" >> "$FINAL_METADATA_PATH"
  done
  echo "[build_world_model_metadata] Merged shard metadata into $FINAL_METADATA_PATH"
else
  "$PYTHON_BIN" examples/LIBERO-IDM/build_world_model_metadata.py \
    --source-metadata-path "$SOURCE_METADATA_PATH" \
    --source-base-path "$SOURCE_BASE_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --prompt-metadata-path "$PROMPT_METADATA_PATH" \
    --video-ckpt "$VIDEO_CKPT" \
    --infer-config "$INFER_CONFIG" \
    --target-mix-ratio "$TARGET_MIX_RATIO" \
    --num-generated-rows "$NUM_GENERATED_ROWS" \
    --device "$DEVICE" \
    "$@"
fi
