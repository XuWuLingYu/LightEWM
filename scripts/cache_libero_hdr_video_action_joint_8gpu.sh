#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-logs/LIBERO-HDR_train_video_action_joint_fastwam_local/20260621_135508/causal_forcing_config.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-data/libero_i2v_train/preencoded_hdr_video_action_joint_fastwam_local}
OUTPUT_JSONL=${OUTPUT_JSONL:-data/libero_i2v_train/metadata_preencoded_hdr_video_action_joint_fastwam_local.jsonl}
NUM_SHARDS=${NUM_SHARDS:-8}
PYTHON=${PYTHON:-/mnt/world_foundational_model/wfm_envs-fileset/qianzezhong/lightewm/bin/python3.10}
LOG_DIR=${LOG_DIR:-logs/run_outputs}

mkdir -p "${LOG_DIR}" "$(dirname "${OUTPUT_JSONL}")"
tmp_dir="${OUTPUT_JSONL}.shards"
mkdir -p "${tmp_dir}"

pids=()
for rank in $(seq 0 $((NUM_SHARDS - 1))); do
  shard_jsonl="${tmp_dir}/shard_${rank}.jsonl"
  log_path="${LOG_DIR}/cache_libero_hdr_preencoded_shard${rank}_$(date +%Y%m%d_%H%M%S).out"
  CUDA_VISIBLE_DEVICES="${rank}" "${PYTHON}" scripts/cache_libero_hdr_video_action_joint.py \
    --config "${CONFIG}" \
    --output-dir "${OUTPUT_DIR}" \
    --output-jsonl "${shard_jsonl}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-rank "${rank}" \
    > "${log_path}" 2>&1 &
  pids+=("$!")
  echo "[CacheLauncher] rank=${rank} pid=${pids[-1]} log=${log_path}"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" != "0" ]]; then
  echo "[CacheLauncher] one or more shards failed" >&2
  exit "${status}"
fi

"${PYTHON}" - "${tmp_dir}" "${OUTPUT_JSONL}" <<'PY'
import json
import re
import sys
from pathlib import Path

tmp_dir = Path(sys.argv[1])
output = Path(sys.argv[2])
rows = []
pattern = re.compile(r"sample_(\d+)\.pt$")
for shard in sorted(tmp_dir.glob("shard_*.jsonl")):
    with shard.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            match = pattern.search(str(row["preencoded_cache_path"]))
            if match is None:
                raise ValueError(f"Cannot parse sample index from {row['preencoded_cache_path']}")
            rows.append((int(match.group(1)), row))
rows.sort(key=lambda item: item[0])
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8") as f:
    for _, row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"[CacheLauncher] merged {len(rows)} rows -> {output}")
PY
