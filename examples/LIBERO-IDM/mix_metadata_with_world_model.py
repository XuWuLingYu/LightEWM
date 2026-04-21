#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REAL_METADATA = "data/libero_idm_abs_action/metadata_abs_action.jsonl"
DEFAULT_REAL_BASE = "data/libero_idm_abs_action"
DEFAULT_WM_METADATA = "data/libero_idm_abs_action_wm/metadata_abs_action_wm.jsonl"
DEFAULT_WM_BASE = "data/libero_idm_abs_action_wm"
DEFAULT_OUTPUT_METADATA = "data/libero_idm_abs_action_mix/metadata_abs_action_mix.jsonl"


def _resolve_path(path_str: str):
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def _load_rows(path: str):
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return data
    if path.endswith(".csv"):
        frame = []
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                frame.append(dict(row))
        return frame
    raise ValueError(f"Unsupported metadata extension: {path}")


def _sample_count(row: dict):
    actions = row.get("abs_action")
    if isinstance(actions, list) and len(actions) > 0 and isinstance(actions[0], list):
        return int(len(actions))
    return 1


def _truncate_row_to_samples(row: dict, keep_samples: int):
    if keep_samples <= 0:
        raise ValueError("keep_samples must be positive.")
    row_samples = _sample_count(row)
    if keep_samples >= row_samples:
        return dict(row)

    truncated = dict(row)
    actions = truncated.get("abs_action")
    if isinstance(actions, list) and len(actions) > 0 and isinstance(actions[0], list):
        truncated["abs_action"] = actions[:keep_samples]
    else:
        raise ValueError("Cannot truncate non-sequence abs_action row.")

    frame_indices = truncated.get("frame_indices")
    if isinstance(frame_indices, list):
        truncated["frame_indices"] = frame_indices[:keep_samples]
    if "num_frames" in truncated:
        truncated["num_frames"] = int(keep_samples)

    sample_id = str(truncated.get("sample_id", "wm"))
    truncated["sample_id"] = f"{sample_id}__slice{keep_samples}"
    demo_key = truncated.get("demo_key")
    if isinstance(demo_key, str) and demo_key:
        truncated["demo_key"] = f"{demo_key}__slice{keep_samples}"
    return truncated


def _resolve_media_path(value: str, metadata_path: str, base_path: str | None):
    if os.path.isabs(value):
        return os.path.abspath(value)
    if base_path:
        return os.path.abspath(os.path.join(base_path, value))
    metadata_dir = os.path.dirname(os.path.abspath(metadata_path))
    return os.path.abspath(os.path.join(metadata_dir, value))


def _normalize_media_paths(row: dict, metadata_path: str, base_path: str | None):
    out = dict(row)
    for key in ("video", "image"):
        value = out.get(key)
        if isinstance(value, str) and value.strip():
            out[key] = _resolve_media_path(value, metadata_path, base_path)
    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mix real IDM metadata with world-model-generated metadata by target train-sample ratio."
    )
    parser.add_argument("--real-metadata-path", type=str, default=DEFAULT_REAL_METADATA)
    parser.add_argument("--real-base-path", type=str, default=DEFAULT_REAL_BASE)
    parser.add_argument("--wm-metadata-path", type=str, default=DEFAULT_WM_METADATA)
    parser.add_argument("--wm-base-path", type=str, default=DEFAULT_WM_BASE)
    parser.add_argument("--output-metadata-path", type=str, default=DEFAULT_OUTPUT_METADATA)
    parser.add_argument("--target-generated-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle-train", action="store_true", default=True)
    parser.add_argument("--no-shuffle-train", action="store_false", dest="shuffle_train")
    return parser.parse_args()


def main():
    args = parse_args()
    if not (0.0 <= float(args.target_generated_ratio) < 1.0):
        raise ValueError("--target-generated-ratio must be in [0, 1).")

    real_metadata_path = _resolve_path(args.real_metadata_path)
    wm_metadata_path = _resolve_path(args.wm_metadata_path)
    real_base_path = _resolve_path(args.real_base_path) if args.real_base_path else None
    wm_base_path = _resolve_path(args.wm_base_path) if args.wm_base_path else None
    output_metadata_path = _resolve_path(args.output_metadata_path)
    os.makedirs(os.path.dirname(output_metadata_path), exist_ok=True)

    real_rows_raw = _load_rows(real_metadata_path)
    wm_rows_raw = _load_rows(wm_metadata_path)
    if len(real_rows_raw) == 0:
        raise RuntimeError(f"No rows in real metadata: {real_metadata_path}")
    if len(wm_rows_raw) == 0:
        raise RuntimeError(f"No rows in WM metadata: {wm_metadata_path}")

    real_rows = [_normalize_media_paths(row, real_metadata_path, real_base_path) for row in real_rows_raw]
    wm_rows = [_normalize_media_paths(row, wm_metadata_path, wm_base_path) for row in wm_rows_raw]

    real_train_rows = [row for row in real_rows if str(row.get("split", "")).strip() == "train"]
    real_non_train_rows = [row for row in real_rows if str(row.get("split", "")).strip() != "train"]
    wm_train_rows = [row for row in wm_rows if str(row.get("split", "")).strip() == "train"]

    if len(real_train_rows) == 0:
        raise RuntimeError("No train rows found in real metadata.")
    if len(wm_train_rows) == 0 and float(args.target_generated_ratio) > 0.0:
        raise RuntimeError("No train rows found in WM metadata.")

    real_train_samples = int(sum(_sample_count(row) for row in real_train_rows))
    target_ratio = float(args.target_generated_ratio)
    if target_ratio == 0.0:
        target_wm_samples = 0
    else:
        target_wm_samples = int(math.ceil(real_train_samples * target_ratio / (1.0 - target_ratio)))

    rng = random.Random(int(args.seed))
    wm_pool = list(wm_train_rows)
    rng.shuffle(wm_pool)

    selected_wm_rows = []
    selected_wm_samples = 0
    for row in wm_pool:
        if selected_wm_samples >= target_wm_samples:
            break
        remaining = target_wm_samples - selected_wm_samples
        row_samples = _sample_count(row)
        if row_samples > remaining:
            selected_row = _truncate_row_to_samples(row, remaining)
        else:
            selected_row = dict(row)
        selected_row["source_type"] = "wm_generated"
        selected_wm_rows.append(selected_row)
        selected_wm_samples += _sample_count(selected_row)

    if selected_wm_samples < target_wm_samples:
        raise RuntimeError(
            "WM metadata does not contain enough train samples for target ratio. "
            f"Need {target_wm_samples}, got {selected_wm_samples}. "
            "Generate more WM rows first."
        )

    final_train_rows = [dict(row) for row in real_train_rows] + selected_wm_rows
    for row in final_train_rows:
        row.setdefault("source_type", "real")
    if args.shuffle_train:
        rng.shuffle(final_train_rows)

    final_rows = real_non_train_rows + final_train_rows
    for row in final_rows:
        if "split" not in row or str(row["split"]).strip() == "":
            row["split"] = "train"

    with open(output_metadata_path, "w", encoding="utf-8") as handle:
        for row in final_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    final_train_samples = int(sum(_sample_count(row) for row in final_train_rows))
    final_wm_ratio = float(selected_wm_samples / max(1, final_train_samples))
    print(
        {
            "real_metadata_path": real_metadata_path,
            "wm_metadata_path": wm_metadata_path,
            "output_metadata_path": output_metadata_path,
            "real_train_rows": int(len(real_train_rows)),
            "real_train_samples": int(real_train_samples),
            "wm_train_rows_available": int(len(wm_train_rows)),
            "wm_rows_selected": int(len(selected_wm_rows)),
            "wm_samples_selected": int(selected_wm_samples),
            "target_generated_ratio": float(target_ratio),
            "final_generated_ratio": float(final_wm_ratio),
            "final_train_samples": int(final_train_samples),
            "final_total_rows": int(len(final_rows)),
            "shuffle_train": bool(args.shuffle_train),
            "note": "abs_action is treated as absolute action target. Do not mix relative actions.",
        }
    )


if __name__ == "__main__":
    main()
