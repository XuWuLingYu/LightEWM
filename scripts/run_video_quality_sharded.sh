#!/usr/bin/env bash
set -euo pipefail

ASSET_ROOT=${ASSET_ROOT:-/pfs-verdent/zhangyu/robot-trial}
DATASET_BASE_PATH=${DATASET_BASE_PATH:-$ASSET_ROOT/data/libero_i2v_train}
METADATA_PATH=${METADATA_PATH:-$DATASET_BASE_PATH/metadata.csv}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/video_quality_eval/sharded_$(date -u +%Y%m%d_%H%M%S)}
MAX_SAMPLES=${MAX_SAMPLES:-80}
INFER_STEPS=${INFER_STEPS:-50}
GPUS=${GPUS:-0,1,2,3}
METRICS=${METRICS:-fvd,ssim,psnr,lpips}
CONFIG=${CONFIG:-examples/LIBERO/infer_ti2v_5b.yaml}
FORCE=${FORCE:-0}

usage() {
  cat <<'EOF'
Usage: bash scripts/run_video_quality_sharded.sh [options]

Options can also be provided as environment variables.
  --asset-root PATH
  --dataset-base-path PATH
  --metadata-path PATH
  --output-root PATH
  --max-samples N
  --infer-steps N
  --gpus CSV             Default: 0,1,2,3
  --metrics CSV          Default: fvd,ssim,psnr,lpips
  --config PATH          Default: examples/LIBERO/infer_ti2v_5b.yaml
  --force                Re-run shard even when summary.csv exists

This launcher splits the first MAX_SAMPLES metadata rows round-robin across
GPUs, runs one evaluate_video_quality.py process per GPU, and writes one
independent shard output directory per process. Re-running resumes at shard
granularity by skipping shards that already have summary.csv unless --force is
provided.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --asset-root) ASSET_ROOT="$2"; shift 2 ;;
    --dataset-base-path) DATASET_BASE_PATH="$2"; shift 2 ;;
    --metadata-path) METADATA_PATH="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --infer-steps) INFER_STEPS="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --metrics) METRICS="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ ! -f "$METADATA_PATH" ]]; then
  echo "[ERROR] metadata file not found: $METADATA_PATH" >&2
  exit 1
fi

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
NUM_SHARDS=${#GPU_LIST[@]}
if [[ "$NUM_SHARDS" -lt 1 ]]; then
  echo "[ERROR] no GPUs provided" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT/shards" "$OUTPUT_ROOT/logs"

python - "$METADATA_PATH" "$OUTPUT_ROOT/shards" "$MAX_SAMPLES" "$NUM_SHARDS" <<'PY'
import csv
import sys
from pathlib import Path

metadata = Path(sys.argv[1])
shard_dir = Path(sys.argv[2])
max_samples = int(sys.argv[3])
num_shards = int(sys.argv[4])

with metadata.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fieldnames = reader.fieldnames or []

if max_samples > 0:
    rows = rows[:max_samples]

for rank in range(num_shards):
    shard_rows = rows[rank::num_shards]
    out = shard_dir / f"metadata_rank_{rank:03d}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(shard_rows)
    print(f"rank={rank} rows={len(shard_rows)} path={out}")
PY

pids=()
for rank in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$rank]}"
  shard_metadata="$OUTPUT_ROOT/shards/metadata_rank_$(printf '%03d' "$rank").csv"
  shard_output="$OUTPUT_ROOT/shard_$(printf '%03d' "$rank")"
  shard_log="$OUTPUT_ROOT/logs/rank_$(printf '%03d' "$rank").log"
  shard_rows=$(( $(wc -l < "$shard_metadata") - 1 ))

  if [[ "$FORCE" != "1" && -f "$shard_output/summary.csv" ]]; then
    echo "[Shard $rank] summary exists, skip: $shard_output/summary.csv"
    continue
  fi
  if [[ "$shard_rows" -le 0 ]]; then
    echo "[Shard $rank] no rows, skip: $shard_metadata"
    continue
  fi

  echo "[Shard $rank] gpu=$gpu rows=$shard_rows metadata=$shard_metadata output=$shard_output log=$shard_log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python scripts/evaluate_video_quality.py \
      --config "$CONFIG" \
      --asset-root "$ASSET_ROOT" \
      --dataset-base-path "$DATASET_BASE_PATH" \
      --metadata-path "$shard_metadata" \
      --output-root "$shard_output" \
      --max-samples "$shard_rows" \
      --infer-steps "$INFER_STEPS" \
      --metrics "$METRICS"
  ) >"$shard_log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

python - "$OUTPUT_ROOT" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
fieldnames = None
for path in sorted(root.glob("shard_*/summary.csv")):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if fieldnames is None:
            fieldnames = reader.fieldnames
        for row in reader:
            row["shard_summary"] = str(path)
            rows.append(row)

if rows and fieldnames:
    out = root / "summary_merged.csv"
    merged_fields = list(fieldnames)
    if "shard_summary" not in merged_fields:
        merged_fields.append("shard_summary")
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Merge] wrote {out} rows={len(rows)}")
else:
    print("[Merge] no shard summaries found yet")
PY

exit "$status"
