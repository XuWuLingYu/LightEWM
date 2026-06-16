# Video Quality Evaluation

This pipeline compares downloadable Wan2.2 checkpoints on converted LIBERO videos by running
image-to-video inference and then pairing generated videos with the ground-truth demonstration
videos from metadata.

Metrics:
- FVD: Frechet distance over video features. The default backend is
  `torchvision_r3d18_kinetics400`; the backend name is written to every summary JSON/CSV.
- SSIM and PSNR: per-frame full-reference metrics from `scikit-image`.
- LPIPS: per-frame perceptual distance from the `lpips` package.

Large files can live outside the repository. The scripts default to:

```bash
/pfs-verdent/zhangyu/robot-trial
```

## 1. Download Assets

```bash
bash scripts/download_eval_assets.sh \
  --asset-root /pfs-verdent/zhangyu/robot-trial
```

This downloads:
- `Wan-AI/Wan2.2-TI2V-5B`
- `XuWuLingYu/Wan2.2-5B-Robot`
- `XuWuLingYu/Wan2.2-5B-Libero`
- `yifengzhu-hf/LIBERO-datasets`

Optional causal checkpoint:

```bash
bash scripts/download_eval_assets.sh \
  --asset-root /pfs-verdent/zhangyu/robot-trial \
  --skip-data \
  --with-causal
```

## 2. Convert LIBERO

```bash
python scripts/convert_libero_to_csv.py \
  --libero-root /pfs-verdent/zhangyu/robot-trial/data/LIBERO-datasets \
  --output-dir /pfs-verdent/zhangyu/robot-trial/data/libero_i2v_train \
  --suites libero_10,libero_90,libero_goal,libero_object,libero_spatial \
  --source-fps 16 \
  --fps 10 \
  --workers 8 \
  --camera-key agentview_rgb,eye_in_hand_rgb \
  --prompt-source attr_or_filename
```

Dense prompts are recommended for model parity with the training recipe:

```bash
bash scripts/process_dense_prompt.sh \
  --metadata-path /pfs-verdent/zhangyu/robot-trial/data/libero_i2v_train/metadata.csv \
  --output-path /pfs-verdent/zhangyu/robot-trial/data/libero_i2v_train/metadata_dense_prompt.csv
```

If dense prompt generation is unavailable, pass `--metadata-path .../metadata.csv` to the
evaluation script.

## 3. Run Three-Weight Evaluation

```bash
python scripts/evaluate_video_quality.py \
  --asset-root /pfs-verdent/zhangyu/robot-trial \
  --max-samples 16 \
  --infer-steps 50 \
  --metrics fvd,ssim,psnr,lpips
```

Default compared weights:
- `wan22_ti2v_5b_base`: base Wan2.2 TI2V 5B from `Wan-AI/Wan2.2-TI2V-5B`
- `wan22_5b_robot`: DiT override from `XuWuLingYu/Wan2.2-5B-Robot`
- `wan22_5b_libero`: DiT override from `XuWuLingYu/Wan2.2-5B-Libero`

Outputs:
- `outputs/video_quality_eval/<timestamp>/summary.csv`
- `outputs/video_quality_eval/<timestamp>/metrics/<weight>/summary.json`
- `outputs/video_quality_eval/<timestamp>/metrics/<weight>/pairs.csv`
- generated videos under `logs/LIBERO_infer_ti2v_5b/<timestamp>_<weight>/`

To compute metrics from already generated videos:

```bash
python scripts/evaluate_video_quality.py \
  --asset-root /pfs-verdent/zhangyu/robot-trial \
  --skip-inference \
  --generated-dir wan22_5b_libero=logs/LIBERO_infer_ti2v_5b/<run-id> \
  --weight wan22_5b_libero=/pfs-verdent/zhangyu/robot-trial/checkpoints/Wan2.2-5B-Libero/checkpoint.safetensors
```

## 4. Smoke Checks

Bidirectional diffusion inference, one sample and two denoising steps:

```bash
bash scripts/smoke_bidirectional_infer.sh \
  --asset-root /pfs-verdent/zhangyu/robot-trial
```

Short LIBERO finetuning, one cached item and one optimizer step:

```bash
bash scripts/smoke_libero_finetune.sh \
  --asset-root /pfs-verdent/zhangyu/robot-trial
```

The short finetuning script expects latent cache at:

```bash
/pfs-verdent/zhangyu/robot-trial/data/libero_i2v_train/latent_cache_ti2v_5b
```

Build it with the existing cache script, pointing paths at the same asset root.
