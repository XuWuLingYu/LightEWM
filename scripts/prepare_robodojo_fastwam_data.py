#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SOURCE = None
DEFAULT_XPOLICY_STATS = None
VIDEO_PREFIX = 'observation.images.'


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + '\n')


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _feature_stats(array: np.ndarray) -> dict:
    array = np.asarray(array, dtype=np.float32)
    return {
        'min': array.min(axis=0),
        'max': array.max(axis=0),
        'mean': array.mean(axis=0),
        'std': array.std(axis=0),
        'count': np.asarray([array.shape[0]], dtype=np.int64),
    }


def _processor_stats(array: np.ndarray) -> dict:
    stats = _feature_stats(array)
    q01 = np.quantile(array, 0.01, axis=0).astype(np.float32)
    q99 = np.quantile(array, 0.99, axis=0).astype(np.float32)
    return {
        'global_min': stats['min'],
        'global_max': stats['max'],
        'global_mean': stats['mean'],
        'global_std': stats['std'],
        'global_q01': q01,
        'global_q99': q99,
        'global_count': stats['count'],
        'stepwise_min': stats['min'][None, :],
        'stepwise_max': stats['max'][None, :],
        'stepwise_mean': stats['mean'][None, :],
        'stepwise_std': stats['std'][None, :],
        'stepwise_q01': q01[None, :],
        'stepwise_q99': q99[None, :],
        'stepwise_count': stats['count'],
    }


def _stack_column(df: pd.DataFrame, column: str) -> np.ndarray:
    return np.stack(df[column].to_numpy()).astype(np.float32)


def compute_dataset_stats(lerobot_dir: Path, output: Path) -> None:
    parquet_files = sorted((lerobot_dir / 'data').glob('chunk-*/episode_*.parquet'))
    if not parquet_files:
        raise FileNotFoundError(f'No parquet episodes under {lerobot_dir /  data}')
    actions = []
    states = []
    total_frames = 0
    for parquet_path in parquet_files:
        df = pd.read_parquet(parquet_path)
        actions.append(_stack_column(df, 'action'))
        states.append(_stack_column(df, 'observation.state'))
        total_frames += len(df)
    all_actions = np.concatenate(actions, axis=0)
    all_states = np.concatenate(states, axis=0)
    payload = {
        'action': {'default': _processor_stats(all_actions)},
        'state': {'default': _processor_stats(all_states)},
        'num_episodes': len(parquet_files),
        'num_transition': int(total_frames),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, default=_json_default)
    print(f'[stats] episodes={len(parquet_files)} frames={total_frames} action_dim={all_actions.shape[-1]} -> {output}')


def _replace_symlink(link_path: Path, target: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.symlink_to(target.resolve(), target_is_directory=True)


def register_full(source: Path, output_root: Path, dataset_id: str, xpolicy_stats: Path | None, recompute_stats: bool) -> None:
    dest_root = output_root / dataset_id
    dest_lerobot = dest_root / 'lerobot'
    _replace_symlink(dest_lerobot, source)
    stats_path = dest_root / 'dataset_stats.json'
    if xpolicy_stats and xpolicy_stats.exists() and not recompute_stats:
        shutil.copy2(xpolicy_stats, stats_path)
        print(f'[full] copied stats {xpolicy_stats} -> {stats_path}')
    else:
        compute_dataset_stats(dest_lerobot, stats_path)
    print(f'[full] linked {dest_lerobot} -> {source.resolve()}')


def _load_source_index(source: Path):
    info = json.loads((source / 'meta' / 'info.json').read_text(encoding='utf-8'))
    tasks = _read_jsonl(source / 'meta' / 'tasks.jsonl')
    episodes = _read_jsonl(source / 'meta' / 'episodes.jsonl')
    episode_stats = {int(row['episode_index']): row for row in _read_jsonl(source / 'meta' / 'episodes_stats.jsonl')}
    task_by_text = {row['task']: int(row['task_index']) for row in tasks}
    return info, tasks, episodes, episode_stats, task_by_text


def _select_episodes(tasks: list[dict], episodes: list[dict], num_tasks: int, episodes_per_task: int) -> tuple[list[dict], list[dict]]:
    selected_tasks = sorted(tasks, key=lambda row: int(row['task_index']))[:num_tasks]
    selected_texts = {row['task'] for row in selected_tasks}
    by_task: dict[str, list[dict]] = defaultdict(list)
    for ep in sorted(episodes, key=lambda row: int(row['episode_index'])):
        ep_tasks = ep.get('tasks') or []
        if not ep_tasks:
            continue
        task = str(ep_tasks[0])
        if task in selected_texts and len(by_task[task]) < episodes_per_task:
            by_task[task].append(ep)
    missing = [row['task'] for row in selected_tasks if len(by_task[row['task']]) < episodes_per_task]
    if missing:
        detail = ', '.join(f'{task}: {len(by_task[task])}' for task in missing)
        raise RuntimeError(f'Not enough episodes for selected tasks: {detail}')
    selected_eps = []
    for row in selected_tasks:
        selected_eps.extend(by_task[row['task']])
    return selected_tasks, selected_eps


def _source_parquet(source: Path, info: dict, episode_index: int) -> Path:
    chunk = episode_index // int(info['chunks_size'])
    rel = info['data_path'].format(episode_chunk=chunk, episode_index=episode_index)
    return source / rel


def _source_video(source: Path, info: dict, episode_index: int, video_key: str) -> Path:
    chunk = episode_index // int(info['chunks_size'])
    rel = info['video_path'].format(episode_chunk=chunk, video_key=video_key, episode_index=episode_index)
    return source / rel


def _target_episode_paths(root: Path, episode_index: int, video_key: str | None = None) -> Path:
    if video_key is None:
        return root / 'data' / 'chunk-000' / f'episode_{episode_index:06d}.parquet'
    return root / 'videos' / 'chunk-000' / video_key / f'episode_{episode_index:06d}.mp4'


def build_smoke_subset(source: Path, output_root: Path, dataset_id: str, num_tasks: int, episodes_per_task: int, frames_per_episode: int) -> None:
    info, tasks, episodes, episode_stats, task_by_text = _load_source_index(source)
    selected_tasks, selected_eps = _select_episodes(tasks, episodes, num_tasks, episodes_per_task)
    dest_root = output_root / dataset_id
    dest_lerobot = dest_root / 'lerobot'
    if dest_lerobot.exists() or dest_lerobot.is_symlink():
        if dest_lerobot.is_dir() and not dest_lerobot.is_symlink():
            shutil.rmtree(dest_lerobot)
        else:
            dest_lerobot.unlink()
    (dest_lerobot / 'meta').mkdir(parents=True, exist_ok=True)
    (dest_lerobot / 'data' / 'chunk-000').mkdir(parents=True, exist_ok=True)

    task_remap = {int(row['task_index']): new_idx for new_idx, row in enumerate(selected_tasks)}
    text_remap = {row['task']: task_remap[int(row['task_index'])] for row in selected_tasks}
    video_keys = [key for key, feature in info['features'].items() if feature.get('dtype') == 'video']

    target_episodes = []
    target_episode_stats = []
    global_index = 0
    total_frames = 0
    for new_ep_idx, ep in enumerate(selected_eps):
        old_ep_idx = int(ep['episode_index'])
        src_df = pd.read_parquet(_source_parquet(source, info, old_ep_idx))
        keep = min(int(frames_per_episode), len(src_df))
        if keep <= 0:
            raise RuntimeError(f'Episode {old_ep_idx} has no frames')
        df = src_df.iloc[:keep].copy()
        task_text = str((ep.get('tasks') or [''])[0])
        df['episode_index'] = np.full(keep, new_ep_idx, dtype=np.int64)
        df['frame_index'] = np.arange(keep, dtype=np.int64)
        df['index'] = np.arange(global_index, global_index + keep, dtype=np.int64)
        df['task_index'] = np.full(keep, text_remap[task_text], dtype=np.int64)
        timestamps = np.arange(keep, dtype=np.float32) / float(info['fps'])
        df['timestamp'] = timestamps
        target_parquet = _target_episode_paths(dest_lerobot, new_ep_idx)
        df.to_parquet(target_parquet, index=False)

        for video_key in video_keys:
            src_video = _source_video(source, info, old_ep_idx, video_key)
            if not src_video.exists():
                continue
            dst_video = _target_episode_paths(dest_lerobot, new_ep_idx, video_key)
            dst_video.parent.mkdir(parents=True, exist_ok=True)
            dst_video.symlink_to(src_video.resolve())

        stats_row = episode_stats.get(old_ep_idx, {'stats': {}})
        target_episode_stats.append({'episode_index': new_ep_idx, 'stats': stats_row.get('stats', {})})
        target_episodes.append({'episode_index': new_ep_idx, 'tasks': [task_text], 'length': keep, 'source_episode_index': old_ep_idx})
        global_index += keep
        total_frames += keep

    target_info = dict(info)
    target_info['total_episodes'] = len(target_episodes)
    target_info['total_frames'] = int(total_frames)
    target_info['total_tasks'] = len(selected_tasks)
    target_info['total_videos'] = len(target_episodes) * len(video_keys)
    target_info['total_chunks'] = 1 if target_episodes else 0
    target_info['chunks_size'] = max(1000, len(target_episodes))
    target_info['splits'] = {'train': f'0:{len(target_episodes)}'}
    target_info['data_path'] = 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet'
    target_info['video_path'] = 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'

    (dest_lerobot / 'meta' / 'info.json').write_text(json.dumps(target_info, indent=2, ensure_ascii=False), encoding='utf-8')
    _write_jsonl(
        dest_lerobot / 'meta' / 'tasks.jsonl',
        [{'task_index': idx, 'task': row['task']} for idx, row in enumerate(selected_tasks)],
    )
    _write_jsonl(dest_lerobot / 'meta' / 'episodes.jsonl', target_episodes)
    _write_jsonl(dest_lerobot / 'meta' / 'episodes_stats.jsonl', target_episode_stats)
    modality_path = source / 'meta' / 'modality.json'
    if modality_path.exists():
        shutil.copy2(modality_path, dest_lerobot / 'meta' / 'modality.json')
    compute_dataset_stats(dest_lerobot, dest_root / 'dataset_stats.json')
    print(
        f'[smoke] dataset_id={dataset_id} tasks={len(selected_tasks)} '
        f'episodes={len(target_episodes)} frames={total_frames} -> {dest_lerobot}'
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-root', type=Path, default=Path('data/robodojo_fastwam'))
    parser.add_argument('--full-dataset-id', default='robodojo-v21-video')
    parser.add_argument('--smoke-dataset-id', default='robodojo-v21-video-smoke-10task-10ep')
    parser.add_argument('--num-tasks', type=int, default=10)
    parser.add_argument('--episodes-per-task', type=int, default=10)
    parser.add_argument('--frames-per-episode', type=int, default=65)
    parser.add_argument('--mode', choices=['full', 'smoke', 'both'], default='both')
    parser.add_argument('--xpolicy-stats', type=Path, default=DEFAULT_XPOLICY_STATS)
    parser.add_argument('--recompute-full-stats', action='store_true')
    args = parser.parse_args()

    source = args.source.resolve()
    if not (source / 'meta' / 'info.json').exists():
        raise FileNotFoundError(f'RoboDojo LeRobot source is missing meta/info.json: {source}')
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.mode in {'full', 'both'}:
        register_full(source, output_root, args.full_dataset_id, args.xpolicy_stats, args.recompute_full_stats)
    if args.mode in {'smoke', 'both'}:
        build_smoke_subset(
            source=source,
            output_root=output_root,
            dataset_id=args.smoke_dataset_id,
            num_tasks=args.num_tasks,
            episodes_per_task=args.episodes_per_task,
            frames_per_episode=args.frames_per_episode,
        )


if __name__ == '__main__':
    main()
