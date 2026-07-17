#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def sample_frames(path: Path, samples: int):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open {path}')
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    idxs = np.linspace(0, max(n - 1, 0), samples).round().astype(int).tolist() if n > 0 else []
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append((idx, frame))
    cap.release()
    return n, w, h, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('output_dir')
    ap.add_argument('--samples', type=int, default=3)
    ap.add_argument('--tile-width', type=int, default=160)
    ap.add_argument('--cols', type=int, default=6)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    mp4s = sorted([p for p in out_dir.glob('*.mp4') if not p.name.startswith('layer_')])
    if not mp4s:
        raise RuntimeError(f'No mp4 files found under {out_dir}')

    tiles = []
    summary = []
    for vid_idx, mp4 in enumerate(mp4s):
        n, w, h, frames = sample_frames(mp4, args.samples)
        summary.append({'path': str(mp4), 'frames': n, 'width': w, 'height': h, 'sampled': [i for i, _ in frames]})
        for frame_idx, frame in frames:
            scale = args.tile_width / max(frame.shape[1], 1)
            tile_h = max(1, int(round(frame.shape[0] * scale)))
            tile = cv2.resize(frame, (args.tile_width, tile_h), interpolation=cv2.INTER_AREA)
            label = f'{vid_idx:02d} f{frame_idx}'
            cv2.rectangle(tile, (0, 0), (args.tile_width, 18), (255, 255, 255), -1)
            cv2.putText(tile, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
            tiles.append(tile)

    if not tiles:
        raise RuntimeError(f'No frames sampled under {out_dir}')
    tile_h = max(t.shape[0] for t in tiles)
    cols = max(1, args.cols)
    rows = int(np.ceil(len(tiles) / cols))
    sheet = np.full((rows * tile_h, cols * args.tile_width, 3), 255, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = r * tile_h, c * args.tile_width
        sheet[y:y + tile.shape[0], x:x + tile.shape[1]] = tile
    sheet_path = out_dir / 'audit_contact_sheet.jpg'
    summary_path = out_dir / 'audit_summary.json'
    cv2.imwrite(str(sheet_path), sheet)
    summary_path.write_text(json.dumps({'count': len(mp4s), 'videos': summary, 'contact_sheet': str(sheet_path)}, indent=2), encoding='utf-8')
    print(json.dumps({'count': len(mp4s), 'contact_sheet': str(sheet_path), 'summary': str(summary_path)}, indent=2))


if __name__ == '__main__':
    main()
