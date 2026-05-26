# CustomDataset Causal Example

This example adapts a custom `metadata.csv` to the Causal-Forcing AR diffusion backend.

Expected layout:

```text
data/your_dataset/
├── metadata.csv
└── videos/
```

Run training:

```bash
python run.py --config examples/CustomDataset-Causal/train.yaml
```
