#!/usr/bin/env bash
set -euo pipefail

ASSET_ROOT=${ASSET_ROOT:-/pfs-verdent/zhangyu/robot-trial}
DOWNLOAD_WEIGHTS=true
DOWNLOAD_DATA=true
DOWNLOAD_CAUSAL=false

usage() {
  cat <<'EOF'
Usage: bash scripts/download_eval_assets.sh [options]

Options:
  --asset-root PATH   Large-file root. Default: /pfs-verdent/zhangyu/robot-trial
  --skip-weights      Do not download Wan/Robot/LIBERO checkpoints
  --skip-data         Do not download raw LIBERO dataset
  --with-causal       Also download LIBERO-Causal model.pt
  -h, --help          Show this help

The converted LightEWM evaluation data is still produced with:
  python scripts/convert_libero_to_csv.py --libero-root <asset-root>/data/LIBERO-datasets \
    --output-dir <asset-root>/data/libero_i2v_train ...
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --asset-root)
      ASSET_ROOT="$2"
      shift 2
      ;;
    --skip-weights)
      DOWNLOAD_WEIGHTS=false
      shift 1
      ;;
    --skip-data)
      DOWNLOAD_DATA=false
      shift 1
      ;;
    --with-causal)
      DOWNLOAD_CAUSAL=true
      shift 1
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

if ! command -v hf >/dev/null 2>&1; then
  echo "[ERROR] 'hf' CLI is not found. Install with: pip install 'huggingface_hub[cli]'" >&2
  exit 1
fi

mkdir -p "$ASSET_ROOT/checkpoints" "$ASSET_ROOT/data"

echo "[Assets] root=$ASSET_ROOT"

if [[ "$DOWNLOAD_WEIGHTS" == "true" ]]; then
  hf download Wan-AI/Wan2.2-TI2V-5B \
    --local-dir "$ASSET_ROOT/checkpoints/Wan2.2-TI2V-5B"
  hf download XuWuLingYu/Wan2.2-5B-Robot \
    --local-dir "$ASSET_ROOT/checkpoints/Wan2.2-5B-Robot"
  hf download XuWuLingYu/Wan2.2-5B-Libero \
    --local-dir "$ASSET_ROOT/checkpoints/Wan2.2-5B-Libero"
fi

if [[ "$DOWNLOAD_CAUSAL" == "true" ]]; then
  hf download XuWuLingYu/LIBERO-Causal-Wan2.2-5BTI2V \
    model.pt \
    --local-dir "$ASSET_ROOT/checkpoints/LIBERO-Causal-Wan2.2-5BTI2V"
fi

if [[ "$DOWNLOAD_DATA" == "true" ]]; then
  hf download yifengzhu-hf/LIBERO-datasets \
    --repo-type dataset \
    --local-dir "$ASSET_ROOT/data/LIBERO-datasets"
fi

echo "[Assets] DONE"
echo "[Assets] Next: convert LIBERO to $ASSET_ROOT/data/libero_i2v_train, then run scripts/evaluate_video_quality.py --asset-root $ASSET_ROOT"
