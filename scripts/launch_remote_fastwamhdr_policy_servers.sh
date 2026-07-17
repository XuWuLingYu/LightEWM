#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-6120}"
NUM_SERVERS="${NUM_SERVERS:-8}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/remote_policy/fastwamhdr}"
CHECKPOINT="${CHECKPOINT:-}"
DATASET_STATS="${DATASET_STATS:-${ROOT_DIR}/data/robodojo_fastwam/robodojo-v21-video/dataset_stats.json}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
REPLAN_STEPS="${REPLAN_STEPS:-24}"
ACTION_HORIZON="${ACTION_HORIZON:-32}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
ALLOW_DUMMY_POLICY="${ALLOW_DUMMY_POLICY:-false}"

mkdir -p "${LOG_DIR}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS}"

if (( ${#GPU_LIST[@]} == 0 )); then
  echo "[launcher] GPUS must not be empty" >&2
  exit 1
fi

echo "[launcher] root=${ROOT_DIR}"
echo "[launcher] host=${HOST} base_port=${BASE_PORT} num_servers=${NUM_SERVERS} gpus=${GPUS}"
echo "[launcher] logs=${LOG_DIR}"

for ((i=0; i<NUM_SERVERS; i++)); do
  gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
  port="$((BASE_PORT + i))"
  log="${LOG_DIR}/server_${i}_port_${port}_gpu_${gpu}.log"
  pid_file="${LOG_DIR}/server_${i}_port_${port}_gpu_${gpu}.pid"
  runner="${LOG_DIR}/server_${i}_port_${port}_gpu_${gpu}.runner.sh"

  cmd=(
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/remote_fastwamhdr_policy_server.py"
    --host "${HOST}"
    --port "${port}"
    --dataset-stats "${DATASET_STATS}"
    --num-inference-steps "${NUM_INFERENCE_STEPS}"
    --replan-steps "${REPLAN_STEPS}"
    --action-horizon "${ACTION_HORIZON}"
    --mixed-precision "${MIXED_PRECISION}"
  )
  if [[ -n "${CHECKPOINT}" ]]; then
    cmd+=(--checkpoint "${CHECKPOINT}")
  fi
  if [[ "${ALLOW_DUMMY_POLICY}" == "true" ]]; then
    cmd+=(--allow-dummy-policy)
  fi

  echo "[launcher] starting server ${i}: gpu=${gpu} port=${port} log=${log}"
  {
    echo "#!/usr/bin/env bash"
    echo "set -u"
    printf "cd %q\n" "${ROOT_DIR}"
    echo "set +u"
    echo "source ~/.bashrc >/dev/null 2>&1 || true"
    echo "set -u"
    echo 'export LD_LIBRARY_PATH="/opt/accl-p:/usr/local/PPU_SDK/targets/x86_64-linux/lib:/usr/local/PPU_SDK/CUDA_SDK/lib64:${LD_LIBRARY_PATH:-}"'
    echo 'export PPU_SDK="/usr/local/PPU_SDK"'
    echo 'export PPU_HOME="/usr/local/PPU_SDK"'
    echo 'export DS_ACCELERATOR="cuda"'
    echo 'export PYTHONPATH="${EXTRA_PYTHONPATH:+${EXTRA_PYTHONPATH}:}'"${ROOT_DIR}"':'"${ROOT_DIR}"'/lightewm/vendor/fastwam:${PYTHONPATH:-}"'
    echo 'export DIFFSYNTH_MODEL_BASE_PATH="'"${ROOT_DIR}"'/checkpoints"'
    printf "echo %q\n" "[launcher-child] start gpu=${gpu} port=${port} pid=\$\$"
    printf "echo %q\n" "[launcher-child] command: CUDA_VISIBLE_DEVICES=${gpu} ${cmd[*]}"
    printf "exec env CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TOKENIZERS_PARALLELISM=false " "${gpu}"
    printf "%q " "${cmd[@]}"
    echo
  } > "${runner}"
  chmod +x "${runner}"
  nohup bash "${runner}" >"${log}" 2>&1 &
  child_pid="$!"
  echo "${child_pid}" > "${pid_file}"
done

echo "[launcher] started. PID files:"
ls -1 "${LOG_DIR}"/*.pid
