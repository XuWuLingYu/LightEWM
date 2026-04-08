---
license: apache-2.0
datasets:
- lerobot/libero
base_model:
- Wan-AI/Wan2.2-TI2V-5B
---

# Wan2.2-5B-Libero

This repository provides a Wan2.2 TI2V 5B checkpoint finetuned on the full LIBERO dataset for robotics video generation.

## Model Summary

- Base model: `Wan-AI/Wan2.2-TI2V-5B`
- Finetuning data: full LIBERO training set
- Training steps: `85000`
- Learning rate: `1e-5`
- Video length: `49` frames
- Frame rate: `10 FPS`
- Task setting: image-to-video generation for LIBERO-style robot trajectories

## Files

- `checkpoint.safetensors`: finetuned DiT checkpoint

## Intended Use

This checkpoint is intended to be used together with the original Wan2.2 TI2V 5B text encoder, tokenizer, and VAE.

In LightEWM, inference can be launched by keeping the standard `examples/LIBERO/infer_ti2v_5b.yaml` config and overriding the DiT checkpoint path with:

```bash
bash scripts/infer.sh \
  --config examples/LIBERO/infer_ti2v_5b.yaml \
  --dataset-base-path data/libero_i2v_train \
  --metadata-path data/libero_i2v_train/metadata_dense_prompt.csv \
  --ckpt checkpoints/Wan2.2-5B-Libero/checkpoint.safetensors
```

## Notes

- This repository stores only the finetuned DiT weights.
- The original Wan2.2 TI2V 5B base checkpoint is still required at inference time.
