# LIBERO HDR Example

This example adapts LIBERO metadata to the Causal-Forcing HDR vertical hierarchy backend.

```bash
python run.py --config examples/LIBERO-HDR/train.yaml
```

For inference with a trained HDR checkpoint:

```bash
python run.py --config examples/LIBERO-HDR/infer.yaml \
  --overrides runner.params.checkpoint_path=/path/to/checkpoint_model_xxxxxx/model.pt
```
