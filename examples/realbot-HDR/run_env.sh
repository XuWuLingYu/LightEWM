#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export PPU_SDK=/usr/local/PPU_SDK
export PPU_HOME=/usr/local/PPU_SDK
export LD_LIBRARY_PATH=/opt/accl-p:/usr/local/PPU_SDK/targets/x86_64-linux/lib:/usr/local/PPU_SDK/CUDA_SDK/lib64:${LD_LIBRARY_PATH:-}
export DS_ACCELERATOR=cuda
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE=1
export TORCHDYNAMO_SUPPRESS_ERRORS=1
export FASTWAM_PROCESS_GROUP_TIMEOUT_SEC=7200
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export NCCL_TIMEOUT=7200
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH=${ROOT}:${ROOT}/lightewm/vendor/causal_forcing:${ROOT}/data/python-packages:${PYTHONPATH:-}
export DIFFSYNTH_MODEL_BASE_PATH=$ROOT/checkpoints
cd "$ROOT"
