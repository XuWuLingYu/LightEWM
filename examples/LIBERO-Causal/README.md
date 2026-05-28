# LIBERO Causal Example

This example adapts LIBERO metadata to the Causal-Forcing AR diffusion backend.

Prepare `data/libero_i2v_train/metadata_dense_prompt.csv` with the regular LIBERO workflow, then run:

```bash
python run.py --config examples/LIBERO-Causal/train.yaml
```

For inference with a trained Causal-Forcing checkpoint:

```bash
python run.py --config examples/LIBERO-Causal/infer.yaml \
  --overrides runner.params.checkpoint_path=/path/to/checkpoint_model_xxxxxx/model.pt
```

## Use the Pretrained LIBERO-Causal Checkpoint

Download the pretrained checkpoint to `checkpoints/`:

```bash
hf download XuWuLingYu/LIBERO-Causal-Wan2.2-5BTI2V \
  model.pt \
  --local-dir checkpoints/LIBERO-Causal-Wan2.2-5BTI2V
```

Then run inference with the downloaded checkpoint:

```bash
python run.py --config examples/LIBERO-Causal/infer.yaml \
  --overrides runner.params.checkpoint_path=checkpoints/LIBERO-Causal-Wan2.2-5BTI2V/model.pt
```

The default inference config uses `data/libero_i2v_train/metadata_dense_prompt.csv`, `224 x 224` resolution, `49` RGB frames, and `13` latent output frames.
