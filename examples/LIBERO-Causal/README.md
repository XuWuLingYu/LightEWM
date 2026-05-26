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
