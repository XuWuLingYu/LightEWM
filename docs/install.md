# Installation

## 1) Environment setup

```bash
git clone https://github.com/XuWuLingYu/LightEWM.git
cd LightEWM
conda create -n lightewm python=3.10
conda activate lightewm
pip install -e .
pip install flash-attn --no-build-isolation
```

## 2) Download checkpoint

```bash
hf download alibaba-pai/Wan2.1-Fun-1.3B-InP \
  --local-dir ./checkpoints/Wan2.1-I2V-1.3B
```
