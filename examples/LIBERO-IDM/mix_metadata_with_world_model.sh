#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/envs/lightewm/bin/python}"

REAL_METADATA_PATH="$REPO_ROOT/data/libero_idm_abs_action/metadata_abs_action.jsonl"
REAL_BASE_PATH="$REPO_ROOT/data/libero_idm_abs_action"
WM_METADATA_PATH="$REPO_ROOT/data/libero_idm_abs_action_wm/metadata_abs_action_wm.jsonl"
WM_BASE_PATH="$REPO_ROOT/data/libero_idm_abs_action_wm"
OUTPUT_METADATA_PATH="$REPO_ROOT/data/libero_idm_abs_action_mix/metadata_abs_action_mix.jsonl"
TARGET_GENERATED_RATIO="${TARGET_GENERATED_RATIO:-0.2}"
SEED="${SEED:-0}"

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
    --wm-metadata-path)
      WM_METADATA_PATH="$2"
      shift 2
      ;;
    --wm-base-path)
      WM_BASE_PATH="$2"
      shift 2
      ;;
    --output-metadata-path)
      OUTPUT_METADATA_PATH="$2"
      shift 2
      ;;
    --target-generated-ratio)
      TARGET_GENERATED_RATIO="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done
set -- "${POSITIONAL[@]}"

"$PYTHON_BIN" examples/LIBERO-IDM/mix_metadata_with_world_model.py \
  --real-metadata-path "$REAL_METADATA_PATH" \
  --real-base-path "$REAL_BASE_PATH" \
  --wm-metadata-path "$WM_METADATA_PATH" \
  --wm-base-path "$WM_BASE_PATH" \
  --output-metadata-path "$OUTPUT_METADATA_PATH" \
  --target-generated-ratio "$TARGET_GENERATED_RATIO" \
  --seed "$SEED" \
  "$@"
