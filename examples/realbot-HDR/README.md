# HDR Video Latent Precache

This directory provides a standalone latent-cache tool for HDR robot-video
datasets. It reads a metadata CSV and writes one Wan VAE latent payload per
video while preserving the metadata needed by downstream training.

## Precache

```bash
python examples/realbot-HDR/precache_latents.py \
  --dataset-root <dataset-directory> \
  --model-root <wan-model-directory> \
  --input-metadata <input-metadata.csv> \
  --output-metadata <output-metadata.csv> \
  --cache-subdir <cache-directory> \
  --encode-batch-size <batch-size>
```

For distributed preprocessing, divide work with `--shard-rank` and
`--shard-world-size`; optionally use `--within-shard-rank` and
`--within-shard-world-size` for a second level of parallelism. Reuse the
generated output metadata in training so it can locate the cached latents.

The cache and generated media are local artifacts and are ignored by git.
