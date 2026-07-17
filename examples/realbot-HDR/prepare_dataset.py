#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math, os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq

PROMPT = "A robot creates art painting with a brush."
Z_THRESHOLD = 0.415
RIGHT_CROP_FRAC = 0.7
OUT_W, OUT_H = 320, 384
TOP_H = 224
BOTTOM_H = 160
WRIST_W = 160
DRAW_S = 160
TARGET_FRAMES = 61
FPS = 10.0


def resize_exact(img, w, h):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

def crop_right_fraction(img, frac):
    h, w = img.shape[:2]
    cw = max(1, min(w, int(round(w * frac))))
    return img[:, w - cw:w]

def limits(vals):
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-6:
        mid = (lo + hi) * 0.5
        lo, hi = mid - 0.5, mid + 0.5
    pad = (hi - lo) * 0.08
    return lo - pad, hi + pad

def drawing_points(states):
    states = np.asarray(states, dtype=np.float32)
    x, y, z = states[:, 0], states[:, 1], states[:, 2]
    contact = z < Z_THRESHOLD
    # User-observed mapping: video down -> drawing left; video right -> drawing down.
    vx = -y
    vy = -x
    fit_x = vx[contact] if np.any(contact) else vx
    fit_y = vy[contact] if np.any(contact) else vy
    xlo, xhi = limits(fit_x)
    ylo, yhi = limits(fit_y)
    margin = max(10, int(DRAW_S * 0.07))
    inner = DRAW_S - 2 * margin
    pts = np.empty((len(states), 2), dtype=np.int32)
    pts[:, 0] = np.clip(np.round((vx - xlo) / (xhi - xlo) * inner + margin), 0, DRAW_S - 1).astype(np.int32)
    pts[:, 1] = np.clip(np.round((vy - ylo) / (yhi - ylo) * inner + margin), 0, DRAW_S - 1).astype(np.int32)
    return pts, contact

def read_video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video {path}')
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return n

def write_stitched_video(right_path: Path, wrist_path: Path, states: np.ndarray, out_path: Path):
    n_video = min(read_video_frame_count(right_path), read_video_frame_count(wrist_path))
    n = min(n_video, len(states))
    if n <= 0:
        raise RuntimeError(f'No frames for {right_path}')
    target_indices = np.linspace(0, n - 1, TARGET_FRAMES).round().astype(np.int64).tolist()
    target_set = set(target_indices)
    pts, contact = drawing_points(states[:n])
    cap_r = cv2.VideoCapture(str(right_path))
    cap_w = cv2.VideoCapture(str(wrist_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (OUT_W, OUT_H))
    if not writer.isOpened():
        raise RuntimeError(f'Could not open writer {out_path}')
    ink = np.full((DRAW_S, DRAW_S, 3), 255, np.uint8)
    last_pt = None
    last_contact = False
    written = 0
    i = 0
    while i < n:
        ok_r, fr = cap_r.read()
        ok_w, fw = cap_w.read()
        if not ok_r or not ok_w:
            break
        if contact[i]:
            p = tuple(int(v) for v in pts[i])
            if last_contact and last_pt is not None:
                cv2.line(ink, last_pt, p, (0, 0, 0), 2, lineType=cv2.LINE_AA)
            else:
                cv2.circle(ink, p, 2, (0, 0, 0), -1, lineType=cv2.LINE_AA)
            last_pt = p
            last_contact = True
        else:
            last_pt = None
            last_contact = False
        if i in target_set:
            canvas = np.full((OUT_H, OUT_W, 3), 255, np.uint8)
            canvas[0:TOP_H, 0:OUT_W] = resize_exact(crop_right_fraction(fr, RIGHT_CROP_FRAC), OUT_W, TOP_H)
            canvas[TOP_H:OUT_H, 0:WRIST_W] = resize_exact(fw, WRIST_W, BOTTOM_H)
            canvas[TOP_H:OUT_H, WRIST_W:OUT_W] = ink
            writer.write(canvas)
            written += 1
        i += 1
    cap_r.release(); cap_w.release(); writer.release()
    if written != TARGET_FRAMES:
        raise RuntimeError(f'Expected {TARGET_FRAMES} frames, wrote {written} for {out_path}')
    return {'source_frames': int(n), 'target_frames': written, 'contact_frames': int(np.sum(contact[:n]))}

def lerobot_episodes(root: Path, split: str, source_name: str):
    split_root = root / split
    for pq_path in sorted((split_root / 'data/chunk-000').glob('episode_*.parquet')):
        stem = pq_path.stem
        ep = stem.split('_')[-1]
        right = split_root / 'videos/chunk-000/observation.images.image' / f'{stem}.mp4'
        wrist = split_root / 'videos/chunk-000/observation.images.wrist_image' / f'{stem}.mp4'
        table = pq.read_table(pq_path, columns=['observation.state'])
        states = np.asarray(table.column('observation.state').to_pylist(), dtype=np.float32)
        yield {'source': source_name, 'split': split, 'episode_id': stem, 'right': right, 'wrist': wrist, 'states': states, 'source_file': pq_path}

def h5_episodes(root: Path, heldout_name: str):
    for ep_dir in sorted([p for p in root.iterdir() if p.is_dir() and (p / 'replay.hdf5').exists()]):
        split = 'heldout' if ep_dir.name == heldout_name else 'train'
        with h5py.File(ep_dir / 'replay.hdf5', 'r') as f:
            states = np.asarray(f['state/ee_pose_euler'][:], dtype=np.float32)
        yield {
            'source': 'custom_pause_removed',
            'split': split,
            'episode_id': ep_dir.name,
            'right': ep_dir / 'videos/observation.images.right.mp4',
            'wrist': ep_dir / 'videos/observation.images.wrist.mp4',
            'states': states,
            'source_file': ep_dir / 'replay.hdf5',
        }

def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['video','prompt','source','episode_id','split','num_frames','dense_prompt','sparse_prompt','source_file']
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-root', default=str(REPO_ROOT / 'data/realbot_hdr_video_320x384_z0415_61f'))
    ap.add_argument('--source1', required=True, help='First input dataset root.')
    ap.add_argument('--source2', required=True, help='Second input dataset root.')
    ap.add_argument('--source3', required=True, help='Third input dataset root.')
    ap.add_argument('--custom-heldout', default='custom_heldout')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    out_root = Path(args.output_root)
    videos_dir = out_root / 'videos'
    rows_train, rows_heldout, summary = [], [], []
    episodes = []
    episodes += list(lerobot_episodes(Path(args.source1), 'train', 'set1'))
    episodes += list(lerobot_episodes(Path(args.source1), 'heldout', 'set1'))
    episodes += list(lerobot_episodes(Path(args.source2), 'train', 'set2'))
    episodes += list(lerobot_episodes(Path(args.source2), 'heldout', 'set2'))
    episodes += list(h5_episodes(SRC3, args.custom_heldout))
    for idx, ep in enumerate(episodes):
        out_rel = Path('videos') / ep['source'] / ep['split'] / f"{ep['episode_id']}_realbot_hdr_320x384_61f.mp4"
        out_path = out_root / out_rel
        if args.force or not out_path.exists():
            stats = write_stitched_video(ep['right'], ep['wrist'], ep['states'], out_path)
        else:
            stats = {'source_frames': None, 'target_frames': TARGET_FRAMES, 'contact_frames': None, 'skipped_existing': True}
        row = {
            'video': str(out_rel),
            'prompt': PROMPT,
            'dense_prompt': PROMPT,
            'sparse_prompt': 'robot art painting',
            'source': ep['source'],
            'episode_id': ep['episode_id'],
            'split': ep['split'],
            'num_frames': TARGET_FRAMES,
            'source_file': str(ep['source_file']),
        }
        (rows_heldout if ep['split'] == 'heldout' else rows_train).append(row)
        summary.append({**row, **stats})
        print(f"[{idx+1}/{len(episodes)}] {ep['source']} {ep['split']} {ep['episode_id']} -> {out_path}", flush=True)
    write_csv(out_root / 'metadata_train.csv', rows_train)
    write_csv(out_root / 'metadata_heldout.csv', rows_heldout)
    (out_root / 'summary.json').write_text(json.dumps({
        'target_frames': TARGET_FRAMES,
        'canvas_wh': [OUT_W, OUT_H],
        'right_crop_frac': RIGHT_CROP_FRAC,
        'z_threshold': Z_THRESHOLD,
        'train_count': len(rows_train),
        'heldout_count': len(rows_heldout),
        'heldout_by_source': {s: [r['episode_id'] for r in rows_heldout if r['source']==s] for s in sorted(set(r['source'] for r in rows_heldout))},
        'rows': summary,
    }, indent=2), encoding='utf-8')
    print(json.dumps({'out_root': str(out_root), 'train': len(rows_train), 'heldout': len(rows_heldout)}, indent=2))

if __name__ == '__main__':
    main()
