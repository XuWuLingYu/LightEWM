#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-6140}"
NUM_SERVERS="${NUM_SERVERS:-2}"
GPUS="${GPUS:-0,1}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-2}"
REPLAN_STEPS="${REPLAN_STEPS:-4}"
ACTION_HORIZON="${ACTION_HORIZON:-32}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_SAMPLES="${NUM_SAMPLES:-2}"
WAIT_SECONDS="${WAIT_SECONDS:-240}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python}"
KEEP_SERVERS="${KEEP_SERVERS:-false}"
CHECKPOINT="${CHECKPOINT:-}"
DATASET_STATS="${DATASET_STATS:-}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
ALLOW_DUMMY_POLICY="${ALLOW_DUMMY_POLICY:-false}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/remote_policy/fastwamhdr_smoke_${STAMP}}"
EVAL_ROOT="${EVAL_ROOT:-${ROOT_DIR}/logs/eval/RoboDojo-FASTWAMHDR/remote_policy_smoke_${STAMP}}"

export LD_LIBRARY_PATH="/usr/local/PPU_SDK/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

cleanup() {
  if [[ "${KEEP_SERVERS}" == "true" ]]; then
    echo "[smoke] KEEP_SERVERS=true; leaving servers running"
    return
  fi
  if compgen -G "${LOG_DIR}/*.pid" >/dev/null; then
    while IFS= read -r pid_file; do
      pid="$(cat "${pid_file}" 2>/dev/null || true)"
      if [[ -n "${pid}" ]]; then
        kill "${pid}" 2>/dev/null || true
      fi
    done < <(ls -1 "${LOG_DIR}"/*.pid)
  fi
}
trap cleanup EXIT

mkdir -p "${LOG_DIR}" "${EVAL_ROOT}"
echo "[smoke] log_dir=${LOG_DIR}"
echo "[smoke] eval_root=${EVAL_ROOT}"
echo "[smoke] launching num_servers=${NUM_SERVERS} gpus=${GPUS} base_port=${BASE_PORT}"

(
  cd "${ROOT_DIR}"
  HOST="${HOST}" \
  BASE_PORT="${BASE_PORT}" \
  NUM_SERVERS="${NUM_SERVERS}" \
  GPUS="${GPUS}" \
  LOG_DIR="${LOG_DIR}" \
  NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
  REPLAN_STEPS="${REPLAN_STEPS}" \
  ACTION_HORIZON="${ACTION_HORIZON}" \
  MIXED_PRECISION="${MIXED_PRECISION}" \
  ALLOW_DUMMY_POLICY="${ALLOW_DUMMY_POLICY}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  CHECKPOINT="${CHECKPOINT}" \
  DATASET_STATS="${DATASET_STATS}" \
  scripts/launch_remote_fastwamhdr_policy_servers.sh
)

echo "[smoke] waiting for servers"
deadline=$((SECONDS + WAIT_SECONDS))
for ((i=0; i<NUM_SERVERS; i++)); do
  port=$((BASE_PORT + i))
  log_glob=("${LOG_DIR}/server_${i}_port_${port}"_*.log)
  log="${log_glob[0]}"
  while ! grep -q "listening on ${HOST}:${port}" "${log}" 2>/dev/null; do
    pid_file_glob=("${LOG_DIR}/server_${i}_port_${port}"_*.pid)
    pid_file="${pid_file_glob[0]}"
    if [[ -f "${pid_file}" ]]; then
      pid="$(cat "${pid_file}" 2>/dev/null || true)"
      if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
        echo "[smoke] server ${i} pid ${pid} exited before listening" >&2
        tail -120 "${log}" >&2 || true
        exit 1
      fi
    fi
    if (( SECONDS >= deadline )); then
      echo "[smoke] timeout waiting for server ${i} port ${port}" >&2
      tail -120 "${log}" >&2 || true
      exit 1
    fi
    sleep 2
  done
  echo "[smoke] server ${i} ready on port ${port}"
done

echo "[smoke] running dataset batch clients"
for ((i=0; i<NUM_SERVERS; i++)); do
  port=$((BASE_PORT + i))
  out_dir="${EVAL_ROOT}/server_${i}_port_${port}"
  client_log="${EVAL_ROOT}/client_${i}_port_${port}.log"
  echo "[smoke] client ${i}: port=${port} batch_size=${BATCH_SIZE} samples=${NUM_SAMPLES}"
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/dataset_fastwamhdr_policy_client.py" \
    --host "${HOST}" \
    --port "${port}" \
    --num-samples "${NUM_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --plot \
    --output-dir "${out_dir}" | tee "${client_log}"
done

echo "[smoke] complete"
echo "[smoke] log_dir=${LOG_DIR}"
echo "[smoke] eval_root=${EVAL_ROOT}"
