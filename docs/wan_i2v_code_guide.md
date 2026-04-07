# Wan 1.3B I2V Code Guide (Minimal Maintenance)

This note documents the current minimal code path for Wan 1.3B I2V in LightEWM.

## 1) Where to modify first

- DiT backbone: `lightewm/model/wan/wan_video_dit.py`
- I2V pipeline logic: `lightewm/model/wan/pipeline.py`
- Training wrapper: `lightewm/runner/wan/wan_training.py`
- Data preprocessing runner: `lightewm/runner/wan/wan_data_preprocess.py`
- Inference runner: `lightewm/runner/wan/wan_infer.py`
- Model loading registry: `lightewm/configs/model_configs.py`

If you want to change model architecture behavior, start from `wan_video_dit.py`.
If you want to change training/inference input-output flow, start from `pipeline.py`.

## 2) Current default I2V runtime path

- Config entry:
  - LIBERO: `examples/LIBERO/*.yaml`
  - Custom dataset: `examples/CustomDataset/*.yaml`
- Model class:
  - `lightewm.model.wan.pipeline.WanVideoPipeline`
- Runner classes (orchestration only):
  - `lightewm.runner.wan.wan_training.WanTrainRunner`
  - `lightewm.runner.wan.wan_data_preprocess.WanCacheRunner`
  - `lightewm.runner.wan.wan_infer.WanInferRunner`

## 3) Core checkpoints used by default

Defined in `lightewm/configs/model_configs.py`:

- `wan_video_dit`  -> `lightewm.model.wan.wan_video_dit.WanModel`
- `wan_video_text_encoder` -> `lightewm.model.wan.wan_video_text_encoder.WanTextEncoder`
- `wan_video_vae` -> `lightewm.model.wan.wan_video_vae.WanVideoVAE`
- `wan_video_image_encoder` -> `lightewm.model.wan.wan_video_image_encoder.WanImageEncoder`

## 4) Minimalization status

The repository currently keeps only the default I2V core path in active use.
Optional extension modules (S2V/VACE/VAP/LongCat/etc.) were removed from active code files and are treated as non-core for this branch.

## 5) Quick sanity checks after changes

```bash
python -m py_compile \
  lightewm/model/wan/wan_video_dit.py \
  lightewm/model/wan/pipeline.py \
  lightewm/runner/wan/wan_training.py \
  lightewm/runner/wan/wan_data_preprocess.py \
  lightewm/runner/wan/wan_infer.py

python run.py --config examples/LIBERO/infer_1p3b.yaml --dry-run
```
