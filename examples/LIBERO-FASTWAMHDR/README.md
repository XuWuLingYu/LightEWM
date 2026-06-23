# LIBERO FastWAM HDR

This example starts from the vendored FastWAM implementation directly from
`lightewm/vendor/fastwam`. The model structure, MoT wrapper, data loader,
normalizer, action/video schedulers, optimizer, and training loop are part of
the LightEWM codebase and do not depend on an external FastWAM checkout.

This HDR variant keeps the original 9 local RGB frames first, then appends 4
special RGB frames sampled uniformly after the local window and through the GT
episode ending. The HDR sampling interval excludes the local end frame and
includes the episode tail frame. Keeping local first preserves the original
clean first latent, while the 13 RGB frames make the noisy latent tail 3 frames
instead of the original 2.

For the joint action path, action tokens attend only to the local clean first
latent frame, not to the appended HDR frames.

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
- per-GPU batch size 16, 8 GPUs, global batch 128
- learning rate `1e-4`, AdamW betas `(0.9, 0.95)`, weight decay `1e-2`
- cosine LR schedule with 5% warmup and 1% minimum LR

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
