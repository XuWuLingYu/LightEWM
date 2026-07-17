# LIBERO FastWAM HDR

This example starts from the vendored FastWAM implementation directly from
`lightewm/vendor/fastwam`. The model structure, MoT wrapper, data loader,
normalizer, action/video schedulers, optimizer, and training loop are part of
the LightEWM codebase and do not depend on an external FastWAM checkout.

This HDR variant keeps the original 9 local RGB frames first, then appends 4
special RGB frames sampled from the episode tail. The tail starts strictly
after the local window end and extends to the GT episode ending; it is split
into 4 equal parts and each part contributes its ending frame. If the local
window already reaches the episode tail, the local end frame is repeated for
all 4 HDR slots. Keeping local first preserves FastWAM's original clean first
latent, while the 13 RGB frames make the noisy latent length 3 frames instead
of the original 2.

For the joint action path, action tokens attend only to the local clean first
latent frame, not to the appended HDR frames.

FastWAM uses `224x448` RGB frames for LIBERO because each frame is a horizontal
two-camera concat: the left `224x224` half is the agent/external view and the
right `224x224` half is the wrist/hand view. The HDR frames use the same
camera layout and image transforms as the local frames.

## Environment

Install the FastWAM dependency set into the active LightEWM environment:

```bash
pip install -e ".[fastwam]"
```

If you keep using the shared `envs/lightewm` environment, at minimum make sure
the vendored runtime imports work:

```bash
PYTHONPATH=lightewm/vendor/fastwam \
python -c "from fastwam.runtime import create_fastwam_joint; print('ok')"
```

## Data

FastWAM expects the official LeRobot-format LIBERO dataset, not the LightEWM
HDF5/CSV metadata used by the HDR and Causal examples.

Download the FastWAM LIBERO dataset:

```bash
source /mnt/zezhong/scripts/source_download_proxy.sh
mkdir -p data/libero_mujoco3.3.2
huggingface-cli download yuanty/LIBERO-fastwam \
  --repo-type dataset \
  --local-dir data/libero_mujoco3.3.2 \
  --local-dir-use-symlinks False \
  --endpoint https://hf-mirror.com

cd data/libero_mujoco3.3.2
for f in *.tar.gz; do tar -xzf "$f"; done
cd -
```

Expected structure:

```text
data/libero_mujoco3.3.2/
├── libero_10_no_noops_lerobot/
├── libero_goal_no_noops_lerobot/
├── libero_object_no_noops_lerobot/
└── libero_spatial_no_noops_lerobot/
```

## Text Embeddings

FastWAM trains with cached T5 embeddings. Generate them once:

```bash
python run.py --config examples/LIBERO-FASTWAMHDR/precompute_text.yaml
```

## Latent Cache

HDR training can read precomputed Wan VAE latents instead of decoding video and
encoding 13 RGB frames in the dataloader. Build the cache with the same HDR
sampling overrides used by training:

```bash
torchrun --nproc_per_node=16 scripts/precache_fastwamhdr_latents.py \
  --output-dir data/fastwamhdr_latent_cache_13f \
  --encode-batch-size 4 \
  --sample-workers 1 \
  model.redirect_common_files=false \
  model.action_attend_video=local_clean_first \
  +data.train.hdr_enabled=true \
  +data.train.hdr_local_rgb_frames=9 \
  +data.train.hdr_tree_rgb_frames=4 \
  +data.train.hdr_total_rgb_frames=13 \
  +data.train.hdr_tree_sampling=uniform_local_start_to_end
```

The training config points `data.train.latent_cache_dir` at this cache. Train
time video metrics still bypass the latent cache for the sampled validation
item so the metric is computed on the full 13-frame RGB clip.

## Training

The training config matches FastWAM's LIBERO joint setup:

- `FastWAMJoint` MoT wrapper
- Wan2.2-TI2V-5B video expert
- ActionDiT initialized from `checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`
- local Wan2.2 common files from `checkpoints/Wan2.2-TI2V-5B` (`model.redirect_common_files=false`)
- two cameras: `image` and `wrist_image`
- FastWAM's `224x448` video frame is a horizontal camera concat:
  left `224x224` is agent/external view, right `224x224` is wrist/hand view
- 33 observations, 32 controller actions
- video stride 4, producing 9 local RGB frames, followed by 4 episode-HDR RGB frames
- per-GPU batch size 8, 16 GPUs, global batch 128
- learning rate `1e-4`, AdamW betas `(0.9, 0.95)`, weight decay `1e-2`
- cosine LR schedule with 5% warmup and 1% minimum LR
- checkpoint and eval cadence are both 500 optimizer steps

Run:

```bash
python run.py --config examples/LIBERO-FASTWAMHDR/train.yaml
```

FastWAM writes `dataset_stats.json` in the run directory. Keep it with the
checkpoint for evaluation.

## Evaluation

Set both checkpoint and dataset stats paths:

```bash
python run.py --config examples/LIBERO-FASTWAMHDR/eval.yaml --overrides \
  runner.params.checkpoint_path=logs/train/LIBERO-FASTWAMHDR/<run>/checkpoints/weights/step_XXXX.pt \
  runner.params.dataset_stats_path=logs/train/LIBERO-FASTWAMHDR/<run>/dataset_stats.json
```

The evaluation path calls FastWAM's LIBERO evaluator and uses its native action
post-processing and rollout code.

For the full LIBERO benchmark, run 40 tasks with 16-way task parallelism:

```bash
python run.py --config examples/LIBERO-FASTWAMHDR/eval.yaml --overrides \
  runner.params.checkpoint_path=logs/train/LIBERO-FASTWAMHDR/<run>/checkpoints/weights/step_XXXX.pt \
  runner.params.dataset_stats_path=logs/train/LIBERO-FASTWAMHDR/<run>/dataset_stats.json \
  runner.params.num_gpus=16 \
  runner.params.max_tasks_per_gpu=1 \
  runner.params.num_trials=50 \
  "runner.params.hydra_overrides=[seed=43,model.redirect_common_files=false,model.action_attend_video=local_clean_first,+data.train.hdr_enabled=true,+data.train.hdr_local_rgb_frames=9,+data.train.hdr_tree_rgb_frames=4,+data.train.hdr_total_rgb_frames=13,+data.train.hdr_tree_sampling=uniform_local_start_to_end,EVALUATION.replan_steps=10,EVALUATION.num_inference_steps=10,EVALUATION.text_cfg_scale=1.0,EVALUATION.visualize_future_video=false,EVALUATION.save_rollout_video=true,EVALUATION.save_failed_rollout_video=true,EVALUATION.save_success_rollout_video_per_task=1]"
```

The evaluator saves all failed rollout MP4s and at most one successful rollout
MP4 per task. When `MUJOCO_GL=egl`, the parallel launcher keeps the EGL render
device fixed at `MUJOCO_EGL_DEVICE_ID=0` and sends policy inference to
`EVALUATION.device=cuda:<gpu_id>` for each task.

Summarize a completed run:

```bash
python lightewm/vendor/fastwam/experiments/libero/summarize_results.py \
  --output_dir logs/eval/LIBERO-FASTWAMHDR/<eval-run>
```
