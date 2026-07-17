#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PY=${PY:-/usr/local/bin/python3}
GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-8}
LR=${LR:-1e-4}
REAL_EPOCHS=${REAL_EPOCHS:-30}
MODE=back_hdr
LAYOUT=realbot_top_wrist_front_320x384
SOURCE=${SOURCE:?set SOURCE=/path/to/new100/source_dataset}
DS_ROOT=${DS_ROOT:-$ROOT/data/realbot_painting/right_wrist_front_new100_pause_removed_lerobot_320x384}
TRAIN_DS=${TRAIN_DS:-$DS_ROOT/train}
HELDOUT_DS=${HELDOUT_DS:-$DS_ROOT/heldout}
TEXT_CACHE=${TEXT_CACHE:-$ROOT/data/text_embeds_cache/realbot_new100_3cam_back_hdr}
CACHE_ROOT=${CACHE_ROOT:-$ROOT/data/realbot_fastwamhdr_latent_cache_new100_3cam_back_hdr}
TRAIN_CACHE=${TRAIN_CACHE:-$CACHE_ROOT/train}
NORM_STATS=${NORM_STATS:-$TRAIN_CACHE/dataset_stats.json}
SKIP_PRECACHE=${SKIP_PRECACHE:-0}
RUN_ROOT=${RUN_ROOT:-$ROOT/logs/REALBOT-FASTWAMHDR_train_new100_3cam_back_hdr}
WAN_VIDEO=${WAN_VIDEO:-$ROOT/checkpoints/Wan2.2-5B-Robot/checkpoint.safetensors}
ACTION_DIT=${ACTION_DIT:-$ROOT/checkpoints/ActionDiT_linear_interp_Wan22Robot_alphascale_1024hdim.pt}
STAGE0=${STAGE0:?set STAGE0=/path/to/fastwamhdr_stage0_checkpoint.pt}
ACTION_ATTEND=${ACTION_ATTEND:-local_clean_first}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export LD_LIBRARY_PATH=/opt/accl-p:/usr/local/PPU_SDK/targets/x86_64-linux/lib:/usr/local/PPU_SDK/CUDA_SDK/lib64:${LD_LIBRARY_PATH:-}
export PPU_SDK=/usr/local/PPU_SDK
export PPU_HOME=/usr/local/PPU_SDK
export DS_ACCELERATOR=cuda
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export FASTWAM_PROCESS_GROUP_TIMEOUT_SEC=7200
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export NCCL_TIMEOUT=7200
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH=${ROOT}:${ROOT}/lightewm/vendor/fastwam:${ROOT}/data/python-packages/fastwam_pydeps:${ROOT}/third_parties/LIBERO:${PYTHONPATH:-}
export DIFFSYNTH_MODEL_BASE_PATH=$ROOT/checkpoints
export LIBERO_CONFIG_PATH=$ROOT/data/libero_config

cd "$ROOT"
mkdir -p "$TEXT_CACHE" "$CACHE_ROOT" "$RUN_ROOT"
[[ -d "$SOURCE" ]] || { echo "Missing source dataset: $SOURCE" >&2; exit 2; }
[[ -f "$STAGE0" ]] || { echo "Missing base checkpoint: $STAGE0" >&2; exit 2; }

COMMON_OVERRIDES=(
  task=libero_joint_2cam224_1e-4
  model=fastwam_joint
  data=libero_2cam
  model.model_id=Wan-AI/Wan2.2-TI2V-5B
  model.tokenizer_model_id=Wan-AI/Wan2.1-T2V-1.3B
  model.redirect_common_files=false
  model.mot_checkpoint_mixed_attn=false
  model.video_dit_pretrained_path="$WAN_VIDEO"
  model.action_dit_pretrained_path="$ACTION_DIT"
  model.action_attend_video="$ACTION_ATTEND"
  model.hdr_mode="$MODE"
  +data.train.hdr_enabled=true
  +data.train.hdr_local_rgb_frames=9
  +data.train.hdr_tree_rgb_frames=4
  +data.train.hdr_total_rgb_frames=13
  +data.train.hdr_tree_sampling=uniform_local_start_to_end
  +data.train.hdr_mode="$MODE"
  data.train.video_size="[384,320]"
  data.train.concat_multi_camera="$LAYOUT"
  "data.train.shape_meta.images=[{key:image,raw_shape:[3,512,512],shape:[3,224,224]},{key:wrist_image,raw_shape:[3,512,512],shape:[3,224,224]},{key:front_image,raw_shape:[3,512,512],shape:[3,224,224]}]"
  data.train.processor.num_output_cameras=3
  data.train.text_embedding_cache_dir="$TEXT_CACHE"
  data.train.num_frames=129
  data.train.action_video_freq_ratio=16
)

prepare_dataset() {
  if [[ -f "$TRAIN_DS/meta/info.json" && -f "$HELDOUT_DS/meta/info.json" ]]; then
    return
  fi
  if [[ -e "$DS_ROOT" ]]; then
    echo "Incomplete dataset exists at $DS_ROOT; remove it or rerun prepare with --overwrite explicitly." >&2
    exit 2
  fi
  "$PY" examples/realbot-HDR/prepare_new100_3cam_back_hdr.py \
    --input-root "$SOURCE" --output-root "$DS_ROOT"
}

precompute_text() {
  local ds=$1
  (cd lightewm/vendor/fastwam && "$PY" -m torch.distributed.run --standalone --nproc_per_node "$GPUS" scripts/precompute_text_embeds.py \
    "${COMMON_OVERRIDES[@]}" data.train.dataset_dirs="[$ds]")
}

precache() {
  if [[ "$SKIP_PRECACHE" == "1" ]]; then
    [[ -f "$NORM_STATS" ]] || { echo "Missing normalization stats: $NORM_STATS" >&2; exit 2; }
    return
  fi
  (cd "$ROOT" && "$PY" -m torch.distributed.run --standalone --nproc_per_node "$GPUS" scripts/precache_fastwamhdr_latents_episodewise.py \
    --task libero_joint_2cam224_1e-4 --model fastwam_joint --data libero_2cam \
    --output-dir "$TRAIN_CACHE" --encode-batch-size 8 --timing-report \
    data.train.dataset_dirs="[$TRAIN_DS]" "${COMMON_OVERRIDES[@]}")
}

steps_per_epoch() {
  "$PY" - <<PY
import math
from pathlib import Path
import pyarrow.parquet as pq
root = Path('$TRAIN_DS') / 'data/chunk-000'
frames = sum(pq.read_metadata(path).num_rows for path in root.glob('episode_*.parquet'))
print(max(1, math.ceil(frames / ($BATCH_SIZE * $GPUS))))
PY
}

train() {
  local save_every
  save_every=$(steps_per_epoch)
  local out="$RUN_ROOT/${MODE}_new100_3cam_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$out"
  echo "RUN_DIR=$out"
  echo "SAVE_EVERY=$save_every"
  (cd lightewm/vendor/fastwam && accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml --num_processes "$GPUS" scripts/train.py \
    "${COMMON_OVERRIDES[@]}" \
    +data.val.dataset_dirs="[$HELDOUT_DS]" \
    ++data.val.is_training_set=false \
    ++data.val.latent_cache_dir=null \
    ++data.val.pretrained_norm_stats="$NORM_STATS" \
    output_dir="$out" \
    resume="$STAGE0" \
    +reset_step_on_weight_load=true \
    data.train.dataset_dirs="[$TRAIN_DS]" \
    +data.train.latent_cache_dir="$TRAIN_CACHE" \
    +data.train.pretrained_norm_stats="$NORM_STATS" \
    batch_size="$BATCH_SIZE" num_workers="$NUM_WORKERS" learning_rate="$LR" \
    lr_scheduler_type=cosine num_epochs="$REAL_EPOCHS" max_steps=null log_every=10 save_every="$save_every" eval_every=200 \
    gradient_accumulation_steps=1 weight_decay=1e-2 mixed_precision=bf16 max_grad_norm=1.0 \
    wandb.enabled=true wandb.project=LightEWM-REALBOT-FASTWAMHDR wandb.name="realbot-back_hdr-new100-3cam") 2>&1 | tee "$out/train.log"
}

prepare_dataset
precompute_text "$TRAIN_DS"
precompute_text "$HELDOUT_DS"
precache
train
