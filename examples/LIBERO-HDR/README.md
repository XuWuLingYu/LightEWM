# LIBERO HDR Example

This example adapts LIBERO videos to the Causal-Forcing HDR vertical hierarchy backend. It is a video-only example for training and inference with the HDR video world model.

Action-policy and joint video-action workflows are intentionally not part of this example. Use `examples/LIBERO-FASTWAM` or `examples/LIBERO-FASTWAMHDR` for action training and evaluation.

## Prepare Data

Prepare the normal LIBERO video metadata under `./data/libero_i2v_train`:

- `data/libero_i2v_train/metadata_dense_prompt.csv`

## Train

Train the HDR video model with dense prompts, 49 video frames, and 13 HDR latent leaves:

```bash
python run.py --config examples/LIBERO-HDR/train.yaml
```

## Infer

For video inference with a trained HDR checkpoint:

```bash
python run.py --config examples/LIBERO-HDR/infer.yaml \
  --overrides runner.params.checkpoint_path=/path/to/checkpoint_model_xxxxxx/model.pt
```
