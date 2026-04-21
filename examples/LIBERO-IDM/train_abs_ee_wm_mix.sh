#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/envs/lightewm/bin/python}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"

REAL_METADATA_PATH="$REPO_ROOT/data/libero_idm_abs_action/metadata_abs_action.jsonl"
REAL_BASE_PATH="$REPO_ROOT/data/libero_idm_abs_action"
WM_OUTPUT_DIR="$REPO_ROOT/data/libero_idm_abs_action_wm"
WM_METADATA_PATH="$WM_OUTPUT_DIR/metadata_abs_action_wm.jsonl"
MIXED_METADATA_PATH="$REPO_ROOT/data/libero_idm_abs_action_mix/metadata_abs_action_mix.jsonl"

PROMPT_METADATA_PATH="$REPO_ROOT/data/libero_i2v_train/metadata_dense_prompt.csv"
VIDEO_CKPT="$REPO_ROOT/checkpoints/Wan2.2-5B-Libero/checkpoint.safetensors"
INFER_CONFIG="$REPO_ROOT/examples/LIBERO/infer_ti2v_5b.yaml"

TARGET_GENERATED_RATIO="${TARGET_GENERATED_RATIO:-0.2}"
NUM_GENERATED_ROWS="${NUM_GENERATED_ROWS:-0}"
RUN_NAME="${RUN_NAME:-libero_abs_ee_wm20}"
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/logs/libero_idm}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
WM_NUM_PROCESSES="${WM_NUM_PROCESSES:-auto}"
SKIP_BUILD_WM=0
SKIP_MIX=0

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --real-metadata-path)
      REAL_METADATA_PATH="$2"
      shift 2
      ;;
    --real-base-path)
      REAL_BASE_PATH="$2"
      shift 2
      ;;
    --wm-output-dir)
      WM_OUTPUT_DIR="$2"
      WM_METADATA_PATH="$WM_OUTPUT_DIR/metadata_abs_action_wm.jsonl"
      shift 2
      ;;
    --wm-metadata-path)
      WM_METADATA_PATH="$2"
      shift 2
      ;;
    --mixed-metadata-path)
      MIXED_METADATA_PATH="$2"
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
    --target-generated-ratio)
      TARGET_GENERATED_RATIO="$2"
      shift 2
      ;;
    --num-generated-rows)
      NUM_GENERATED_ROWS="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --save-dir)
      SAVE_DIR="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
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
    --wm-num-processes)
      WM_NUM_PROCESSES="$2"
      shift 2
      ;;
    --skip-build-wm)
      SKIP_BUILD_WM=1
      shift 1
      ;;
    --skip-mix)
      SKIP_MIX=1
      shift 1
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done
set -- "${POSITIONAL[@]}"

if [[ "$SKIP_BUILD_WM" == "1" ]]; then
  echo "[1/3] Skip world-model metadata generation."
else
  echo "[1/3] Build world-model metadata..."
  PYTHON_BIN="$PYTHON_BIN" bash examples/LIBERO-IDM/build_world_model_metadata.sh \
    --source-metadata-path "$REAL_METADATA_PATH" \
    --source-base-path "$REAL_BASE_PATH" \
    --output-dir "$WM_OUTPUT_DIR" \
    --prompt-metadata-path "$PROMPT_METADATA_PATH" \
    --video-ckpt "$VIDEO_CKPT" \
    --infer-config "$INFER_CONFIG" \
    --target-mix-ratio "$TARGET_GENERATED_RATIO" \
    --num-generated-rows "$NUM_GENERATED_ROWS" \
    --device "$DEVICE" \
    --num-processes "$WM_NUM_PROCESSES" \
    --seed "$SEED"
fi

if [[ "$SKIP_MIX" == "1" ]]; then
  echo "[2/3] Skip metadata mix."
else
  echo "[2/3] Mix metadata (real + world-model generated)..."
  "$PYTHON_BIN" examples/LIBERO-IDM/mix_metadata_with_world_model.py \
    --real-metadata-path "$REAL_METADATA_PATH" \
    --real-base-path "$REAL_BASE_PATH" \
    --wm-metadata-path "$WM_METADATA_PATH" \
    --wm-base-path "$WM_OUTPUT_DIR" \
    --output-metadata-path "$MIXED_METADATA_PATH" \
    --target-generated-ratio "$TARGET_GENERATED_RATIO" \
    --seed "$SEED"
fi

echo "[3/3] Launch IDM training..."
accelerate launch \
  --num_processes "$NUM_PROCESSES" \
  third_parties/AnyPos/train_metadata_abs_action.py \
  --metadata_path "$MIXED_METADATA_PATH" \
  --video_key video \
  --action_key abs_action \
  --model_name direction_aware \
  --learning_rate 1e-4 \
  --batch_size 32 \
  --eval_batch_size 64 \
  --num_workers 8 \
  --eval_interval 2000 \
  --save_interval 2000 \
  --save_dir "$SAVE_DIR" \
  --wandb_project IDM_LIBERO_abs_action \
  --run_name "$RUN_NAME" \
  "$@"
