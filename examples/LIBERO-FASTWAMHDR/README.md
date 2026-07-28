# LIBERO FastWAM HDR

FastWAM HDR extends the local FastWAM video window with uniformly sampled
future frames from the same episode. It uses the vendored implementation in
`lightewm/vendor/fastwam` and a LeRobot-format dataset.

## Environment

```bash
pip install -e ".[fastwam]"
PYTHONPATH=lightewm/vendor/fastwam \
  python -c "from fastwam.runtime import create_fastwam_joint; print('ok')"
```

## Prepare data

Keep downloaded datasets and generated artifacts outside version control. Once
the chosen data config can load the local dataset, prepare text embeddings:

```bash
python run.py --config examples/LIBERO-FASTWAMHDR/precompute_text.yaml
```

## Precache HDR latents

Use the same task, data, and HDR sampling overrides that will be used for
training. The command below shards work across all launched processes.

```bash
torchrun --standalone --nproc_per_node=<NUM_GPUS> \
  scripts/precache_fastwamhdr_latents.py \
  --task <task-config> \
  --data <data-config> \
  --output-dir <cache-directory> \
  --encode-batch-size <batch-size> \
  --sample-workers <workers> \
  --timing-report \
  +data.train.hdr_enabled=true
```

Point the training configuration at `<cache-directory>`. To split cache work
between independent launchers, use `--shard-rank`, `--num-shards`, and one of
the documented shard modes. Use `--overwrite` only to rebuild stale entries.

## Train and evaluate

```bash
python run.py --config examples/LIBERO-FASTWAMHDR/train.yaml
python run.py --config examples/LIBERO-FASTWAMHDR/eval.yaml
```
