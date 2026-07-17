#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
FASTWAM_ROOT = ROOT / 'lightewm' / 'vendor' / 'fastwam'
for path in (FASTWAM_ROOT, ROOT / 'data' / 'python-packages' / 'fastwam_pydeps', ROOT / 'third_parties' / 'LIBERO'):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


def _to_uint8(frame: torch.Tensor) -> np.ndarray:
    # frame: [C,H,W] in [-1, 1]
    frame = ((frame.detach().cpu().float().clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)
    return frame.permute(1, 2, 0).numpy()


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(frame).convert('RGB')
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 24), fill=(0, 0, 0))
    draw.text((6, 5), text, fill=(255, 255, 255))
    return np.asarray(image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--fastwam-root', type=Path, default=FASTWAM_ROOT)
    parser.add_argument('--task', default='robotwin_uncond_3cam_384_1e-4')
    parser.add_argument('--model', default='fastwam_joint')
    parser.add_argument('--data', default='robotwin')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--output', type=Path, default=Path('logs/visualizations/robodojo_fastwamhdr_smoke_sample0.mp4'))
    parser.add_argument('--fps', type=int, default=2)
    parser.add_argument('--label', action='store_true')
    parser.add_argument('overrides', nargs='*')
    args = parser.parse_args()

    misc.register_work_dir(str((ROOT / 'logs' / 'visualizations').resolve()))
    with initialize_config_dir(config_dir=str(args.fastwam_root.resolve() / 'configs'), version_base='1.3'):
        cfg = compose(
            config_name='train',
            overrides=[f'task={args.task}', f'model={args.model}', f'data={args.data}', *args.overrides],
        )
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    dataset = instantiate(cfg.data.train)
    sample = dataset._get(int(args.index))
    video = sample['video']
    if video.ndim != 4:
        raise ValueError(f'Expected sample video [C,T,H,W], got {tuple(video.shape)}')

    local_indices = sample.get('hdr_local_frame_indices')
    tree_indices = sample.get('hdr_tree_frame_indices')
    local_count = int(sample.get('local_video_frames', torch.tensor(video.shape[1])).item())
    frames = []
    for t in range(video.shape[1]):
        frame = _to_uint8(video[:, t])
        if args.label:
            kind = 'local' if t < local_count else 'hdr'
            src_idx = None
            if t < local_count and local_indices is not None:
                src_idx = int(local_indices[t].item())
            if t >= local_count and tree_indices is not None:
                src_idx = int(tree_indices[t - local_count].item())
            frame = _label(frame, f'{t:02d} {kind}' + (f' src={src_idx}' if src_idx is not None else ''))
        frames.append(frame)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.output, frames, fps=args.fps, codec='libx264', quality=8)
    sidecar = args.output.with_suffix('.json')
    sidecar.write_text(
        json.dumps(
            {
                'dataset_index': int(torch.as_tensor(sample.get('dataset_index', 0)).item()) if 'dataset_index' in sample else None,
                'episode_index': int(torch.as_tensor(sample.get('episode_index', -1)).item()) if 'episode_index' in sample else None,
                'frame_index': int(torch.as_tensor(sample.get('frame_index', -1)).item()) if 'frame_index' in sample else None,
                'prompt': sample.get('prompt'),
                'video_shape': list(video.shape),
                'layout': 'robotwin: cam_high top 256x320; cam_left_wrist/cam_right_wrist bottom 128x160 each; final 384x320',
                'local_frame_indices': local_indices.tolist() if isinstance(local_indices, torch.Tensor) else None,
                'hdr_tree_frame_indices': tree_indices.tolist() if isinstance(tree_indices, torch.Tensor) else None,
                'output': str(args.output),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding='utf-8',
    )
    print(f'saved {args.output}')
    print(f'saved {sidecar}')


if __name__ == '__main__':
    main()
