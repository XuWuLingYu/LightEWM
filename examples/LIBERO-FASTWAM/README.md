# LIBERO FastWAM

This example runs the vendored FastWAM implementation in
`lightewm/vendor/fastwam` against a LeRobot-format dataset.

## Environment

```bash
pip install -e ".[fastwam]"
PYTHONPATH=lightewm/vendor/fastwam \
  python -c "from fastwam.runtime import create_fastwam_joint; print('ok')"
```

## Prepare data

Download the public LeRobot-format dataset into a local directory outside the
repository, then supply its location through the selected FastWAM data config.
Generate text embeddings before training:

```bash
python run.py --config examples/LIBERO-FASTWAM/precompute_text.yaml
```

## Precache latents

Build cached Wan VAE latents before training to avoid repeated video decoding.
Choose task and data configs that describe the local dataset. Cache output is a
generated artifact and is ignored by git.

```bash
torchrun --standalone --nproc_per_node=<NUM_GPUS> \
  scripts/precache_fastwam_latents_episodewise.py \
  --task <task-config> \
  --data <data-config> \
  --output-dir <cache-directory> \
  --encode-batch-size <batch-size> \
  --timing-report
```

Set the training configuration's latent-cache directory to the same
`<cache-directory>`. Re-run with `--overwrite` only when the source data or
cache settings change.

## Train and evaluate

```bash
python run.py --config examples/LIBERO-FASTWAM/train.yaml
python run.py --config examples/LIBERO-FASTWAM/eval.yaml
```
