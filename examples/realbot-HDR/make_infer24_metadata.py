#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_prepare_module():
    path = Path(__file__).with_name('prepare_suffix_dataset.py')
    spec = importlib.util.spec_from_file_location('realbot_hdr_prepare_suffix_dataset', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not load {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict]):
    fields = [
        'video', 'prompt', 'source', 'episode_id', 'parent_episode_id', 'split',
        'num_frames', 'clip_start', 'clip_end', 'clip_source_frames', 'start_fraction',
        'dense_prompt', 'sparse_prompt', 'source_file'
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, '') for k in fields})


def fraction_starts(n: int, target_frames: int, count: int) -> list[int]:
    if n < target_frames:
        return []
    max_start = n - target_frames
    starts = []
    for k in range(count):
        start = int(round(max_start * (k / count)))
        start = max(0, min(max_start, start))
        starts.append(start)
    return starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-root', default=str(REPO_ROOT / 'data/realbot_hdr_video_320x384_z0415_61f_suffix10'))
    ap.add_argument('--metadata-name', default='metadata_infer24.csv')
    ap.add_argument('--summary-name', default='summary_infer24.json')
    ap.add_argument('--custom-heldout', default='custom_heldout')
    ap.add_argument('--fractions', type=int, default=8)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    prep = load_prepare_module()
    out_root = Path(args.output_root)
    episodes = []
    episodes += list(prep.lerobot_episodes(prep.SRC1, 'heldout', 'set1'))
    episodes += list(prep.lerobot_episodes(prep.SRC2, 'heldout', 'set2'))
    episodes += [ep for ep in prep.h5_episodes(prep.SRC3, args.custom_heldout) if ep['split'] == 'heldout']

    rows = []
    summary = []
    for ep_idx, ep in enumerate(episodes):
        n_video = min(prep.read_video_frame_count(ep['right']), prep.read_video_frame_count(ep['wrist']))
        n = min(n_video, len(ep['states']))
        starts = fraction_starts(n, prep.TARGET_FRAMES, args.fractions)
        if len(starts) != args.fractions:
            raise RuntimeError(f"Heldout episode too short: {ep['source']} {ep['episode_id']} n={n}")
        tops, wrists = prep.read_processed_frames(ep['right'], ep['wrist'], n)
        for frac_idx, start in enumerate(starts):
            frac_label = f"f{frac_idx:02d}of{args.fractions:02d}"
            clip_id = f"{ep['episode_id']}_{frac_label}_s{start:06d}"
            out_rel = Path('videos') / ep['source'] / 'heldout_infer24' / f"{clip_id}_realbot_hdr_320x384_61f.mp4"
            out_path = out_root / out_rel
            if args.force or not out_path.exists():
                stats = prep.write_stitched_clip(tops, wrists, ep['states'], start, out_path)
            else:
                stats = {
                    'source_frames': int(n),
                    'clip_start': int(start),
                    'clip_end': int(n - 1),
                    'clip_source_frames': int(n - start),
                    'target_frames': prep.TARGET_FRAMES,
                    'skipped_existing': True,
                }
            row = {
                'video': str(out_rel),
                'prompt': prep.PROMPT,
                'dense_prompt': prep.PROMPT,
                'sparse_prompt': 'robot art painting',
                'source': ep['source'],
                'episode_id': clip_id,
                'parent_episode_id': ep['episode_id'],
                'split': 'heldout_infer24',
                'num_frames': prep.TARGET_FRAMES,
                'clip_start': int(start),
                'clip_end': int(n - 1),
                'clip_source_frames': int(n - start),
                'start_fraction': f'{frac_idx}/{args.fractions}',
                'source_file': str(ep['source_file']),
            }
            rows.append(row)
            summary.append({**row, **stats, 'source_frames': int(n)})
        print(f"[{ep_idx+1}/{len(episodes)}] {ep['source']} {ep['episode_id']} n={n} starts={starts}", flush=True)

    expected = len(episodes) * args.fractions
    if len(rows) != expected:
        raise RuntimeError(f'Expected {expected} rows, got {len(rows)}')
    write_csv(out_root / args.metadata_name, rows)
    (out_root / args.summary_name).write_text(json.dumps({
        'count': len(rows),
        'fractions': args.fractions,
        'target_frames': prep.TARGET_FRAMES,
        'rows': summary,
    }, indent=2), encoding='utf-8')
    print(json.dumps({
        'output_root': str(out_root),
        'metadata': str(out_root / args.metadata_name),
        'summary': str(out_root / args.summary_name),
        'count': len(rows),
    }, indent=2))


if __name__ == '__main__':
    main()
