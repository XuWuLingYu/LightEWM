#!/usr/bin/env python3
"""Generate dense prompts from metadata.csv videos with Qwen2.5-VL."""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from transformers.models.qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_: Any):
        return iterable


DEFAULT_PROMPT_TEMPLATE = """You are an expert in embodied manipulation.

<video> Given a video and a sparse task description, generate a moderately dense prompt describing the task execution.

Requirements:
- Use chronological order
- Focus on object interactions
- Use concise action phrases
- Avoid overly fine-grained details
- Produce a natural manipulation prompt
- Output exactly one single-line sentence
- Use this template style: "The robot ..., then ..., then ..."
- Start with "The robot"
- Separate action stages with "then"
- Do not use bullets, numbering, quotes, or line breaks
- Do not mention camera views, scene IDs, or dataset names

Sparse task:
{task_description}

Output:"""


def normalize_generated_prompt(text: str) -> str:
    text = " ".join(str(text).split())
    text = re.sub(r"^(output|dense prompt|prompt)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip("\"'").strip()
    if text.lower().startswith("robot "):
        text = "The " + text
    if not text.lower().startswith("the robot "):
        text = f"The robot {text.lstrip()}"
    text = re.sub(r"\s+", " ", text).strip()
    text = text.rstrip(" .")
    return f"{text}."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dense prompts from metadata.csv videos with Qwen2.5-VL."
    )
    parser.add_argument(
        "--metadata-path",
        required=True,
        help="Path to metadata.csv.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output CSV path. Defaults to <metadata_dir>/metadata_dense_prompt.csv.",
    )
    parser.add_argument(
        "--video-column",
        default="video",
        help="Column name containing video paths.",
    )
    parser.add_argument(
        "--prompt-column",
        default="prompt",
        help="Column name containing sparse task descriptions.",
    )
    parser.add_argument(
        "--dense-prompt-column",
        default="dense_prompt",
        help="Column name for generated dense prompts.",
    )
    parser.add_argument(
        "--sparse-prompt-column",
        default="sparse_prompt",
        help="Backup column used to preserve the original sparse prompt in the output CSV.",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Hugging Face model name or local path.",
    )
    parser.add_argument(
        "--cache-dir",
        default="checkpoints",
        help="Download/cache directory used when --model-name is a remote Hugging Face model.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Device to run on: "auto", "cuda", "cuda:0", or "cpu".',
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Model dtype.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=32,
        help="Uniformly sampled frames per video. Use <=0 to keep all frames.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=96,
        help="Max generated tokens per sample.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. <=0 disables sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p used only when sampling is enabled.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=20,
        help="Persist the output CSV every N processed rows.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap for quick tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output file instead of resuming from it.",
    )
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(dtype_name: str, device: str) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    if device.startswith("cuda"):
        return torch.bfloat16
    return torch.float32


def resolve_output_path(metadata_path: Path, output_path: str | None) -> Path:
    if output_path:
        return Path(output_path)
    return metadata_path.with_name("metadata_dense_prompt.csv")


def resolve_shard_dir(output_path: Path) -> Path:
    return output_path.parent / f".{output_path.stem}_shards"


def shard_output_path(shard_dir: Path, process_index: int) -> Path:
    return shard_dir / f"rank_{process_index:03d}.csv"


def resolve_video_path(metadata_path: Path, value: Any) -> Path:
    video_path = Path(str(value))
    if video_path.is_absolute():
        return video_path
    return (metadata_path.parent / video_path).resolve()


def get_total_frames(reader: Any, fallback_num_frames: Any) -> int:
    if pd.notna(fallback_num_frames):
        try:
            parsed = int(fallback_num_frames)
            if parsed > 0:
                return parsed
        except (TypeError, ValueError):
            pass

    meta = reader.get_meta_data()
    nframes = meta.get("nframes")
    if isinstance(nframes, (int, np.integer)) and int(nframes) > 0:
        return int(nframes)

    try:
        counted = int(reader.count_frames())
        if counted > 0:
            return counted
    except Exception:
        pass
    raise RuntimeError("Failed to determine video frame count.")


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    elif frame.ndim == 3 and frame.shape[-1] == 4:
        frame = frame[..., :3]
    return np.asarray(frame, dtype=np.uint8)


def sample_video_frames(video_path: Path, target_num_frames: int, fallback_num_frames: Any) -> list[np.ndarray]:
    reader = imageio.get_reader(str(video_path))
    try:
        total_frames = get_total_frames(reader, fallback_num_frames)
        if target_num_frames is None or target_num_frames <= 0 or target_num_frames >= total_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, target_num_frames, dtype=int).tolist()

        frames = [normalize_frame(reader.get_data(frame_idx)) for frame_idx in frame_indices]
    finally:
        reader.close()

    if not frames:
        raise RuntimeError(f"No frames were loaded from {video_path}.")
    return frames


def move_inputs_to_device(batch: dict[str, Any], device: str, dtype: torch.dtype) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def build_formatted_prompt(
    processor: Qwen2_5_VLProcessor,
    task_description: str,
    video_path: Path,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "path": str(video_path)},
                {
                    "type": "text",
                    "text": DEFAULT_PROMPT_TEMPLATE.format(task_description=task_description.strip()),
                },
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_existing_output(
    metadata: pd.DataFrame,
    output_path: Path,
    video_column: str,
    prompt_column: str,
    dense_prompt_column: str,
    sparse_prompt_column: str,
    overwrite: bool,
) -> pd.DataFrame:
    if overwrite or not output_path.exists():
        result = metadata.copy()
        if sparse_prompt_column not in result.columns:
            result[sparse_prompt_column] = result[prompt_column]
        if dense_prompt_column not in result.columns:
            result[dense_prompt_column] = ""
        return result

    existing = pd.read_csv(output_path)
    if len(existing) != len(metadata):
        raise RuntimeError(
            f"Existing output row count mismatch: expected {len(metadata)}, found {len(existing)}. "
            "Use --overwrite to rebuild."
        )
    if video_column not in existing.columns or not existing[video_column].astype(str).equals(metadata[video_column].astype(str)):
        raise RuntimeError(
            f"Existing output {output_path} does not align with the current metadata file. "
            "Use --overwrite to rebuild."
        )
    if dense_prompt_column not in existing.columns:
        existing[dense_prompt_column] = ""
    if sparse_prompt_column not in existing.columns:
        existing[sparse_prompt_column] = metadata[prompt_column]
    return existing


def merge_shard_output(
    base_df: pd.DataFrame,
    shard_df: pd.DataFrame,
    prompt_column: str,
    dense_prompt_column: str,
    sparse_prompt_column: str,
) -> pd.DataFrame:
    if "__row_index" not in shard_df.columns:
        raise RuntimeError("Shard output is missing '__row_index'.")

    merged = base_df.copy()
    shard_df = shard_df.dropna(subset=["__row_index"]).copy()
    if shard_df.empty:
        return merged

    shard_df["__row_index"] = shard_df["__row_index"].astype(int)
    for _, row in shard_df.iterrows():
        row_index = int(row["__row_index"])
        if row_index < 0 or row_index >= len(merged):
            continue
        dense_prompt = str(row.get(dense_prompt_column, "")).strip()
        if not dense_prompt:
            continue
        merged.at[row_index, dense_prompt_column] = dense_prompt
        merged.at[row_index, prompt_column] = str(row.get(prompt_column, dense_prompt))
        if sparse_prompt_column in row and pd.notna(row[sparse_prompt_column]):
            merged.at[row_index, sparse_prompt_column] = row[sparse_prompt_column]
    return merged


def load_existing_shards(
    base_df: pd.DataFrame,
    shard_dir: Path,
    prompt_column: str,
    dense_prompt_column: str,
    sparse_prompt_column: str,
) -> pd.DataFrame:
    merged = base_df
    if not shard_dir.is_dir():
        return merged

    for shard_path in sorted(shard_dir.glob("rank_*.csv")):
        shard_df = pd.read_csv(shard_path)
        merged = merge_shard_output(
            merged,
            shard_df,
            prompt_column=prompt_column,
            dense_prompt_column=dense_prompt_column,
            sparse_prompt_column=sparse_prompt_column,
        )
    return merged


def save_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def save_shard_output(shard_rows: list[dict[str, Any]], shard_path: Path) -> None:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_df = pd.DataFrame(shard_rows)
    shard_df.to_csv(shard_path, index=False)


def main() -> None:
    args = parse_args()
    accelerator = Accelerator()

    metadata_path = Path(args.metadata_path).resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    output_path = resolve_output_path(metadata_path, args.output_path).resolve()
    shard_dir = resolve_shard_dir(output_path).resolve()
    local_shard_path = shard_output_path(shard_dir, accelerator.process_index)
    device = str(accelerator.device) if args.device == "auto" else resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    do_sample = args.temperature > 0
    cache_dir = Path(args.cache_dir).resolve()

    metadata = pd.read_csv(metadata_path)
    if args.max_rows is not None:
        metadata = metadata.head(args.max_rows).copy()

    for required_column in (args.video_column, args.prompt_column):
        if required_column not in metadata.columns:
            raise KeyError(f"Required column '{required_column}' not found in {metadata_path}")

    results = load_existing_output(
        metadata=metadata,
        output_path=output_path,
        video_column=args.video_column,
        prompt_column=args.prompt_column,
        dense_prompt_column=args.dense_prompt_column,
        sparse_prompt_column=args.sparse_prompt_column,
        overwrite=args.overwrite,
    )
    results = load_existing_shards(
        results,
        shard_dir=shard_dir,
        prompt_column=args.prompt_column,
        dense_prompt_column=args.dense_prompt_column,
        sparse_prompt_column=args.sparse_prompt_column,
    )

    if accelerator.is_main_process:
        print(f"Metadata: {metadata_path}")
        print(f"Output: {output_path}")
        print(f"Shard dir: {shard_dir}")
        print(f"Rows: {len(results)}")
        print(f"Model: {args.model_name}")
        print(f"Model cache dir: {cache_dir}")
        print(f"Processes: {accelerator.num_processes}")
        print(f"Dtype: {dtype}")
        print(f"Frames per video: {args.num_frames}")
    print(f"[rank {accelerator.process_index}] device={device}")

    processor = Qwen2_5_VLProcessor.from_pretrained(
        args.model_name,
        cache_dir=str(cache_dir),
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        cache_dir=str(cache_dir),
        torch_dtype=dtype,
    )
    model = model.to(device)
    model.eval()

    shard_rows: list[dict[str, Any]] = []
    if local_shard_path.exists() and not args.overwrite:
        existing_shard = pd.read_csv(local_shard_path)
        shard_rows = existing_shard.to_dict(orient="records")

    completed_local_indices = {
        int(row["__row_index"])
        for row in shard_rows
        if str(row.get(args.dense_prompt_column, "")).strip()
    }
    completed_global_mask = results[args.dense_prompt_column].fillna("").astype(str).str.strip().ne("")
    local_indices = [
        int(idx)
        for idx in results.index
        if idx % accelerator.num_processes == accelerator.process_index
        and idx not in completed_local_indices
        and not bool(completed_global_mask.iloc[idx])
    ]

    processed_since_save = 0
    iterator = tqdm(local_indices, total=len(local_indices), desc=f"Dense prompt rank {accelerator.process_index}")
    for idx in iterator:
        row = results.loc[idx]
        sparse_prompt = str(row[args.prompt_column]).strip()
        video_path = resolve_video_path(metadata_path, row[args.video_column])
        if not video_path.is_file():
            print(f"[rank {accelerator.process_index}] [Skip] Missing video: {video_path}")
            continue

        try:
            frames = sample_video_frames(
                video_path=video_path,
                target_num_frames=args.num_frames,
                fallback_num_frames=row["num_frames"] if "num_frames" in row else None,
            )
            formatted_prompt = build_formatted_prompt(
                processor=processor,
                task_description=sparse_prompt,
                video_path=video_path,
            )
            model_inputs = processor(
                text=[formatted_prompt],
                videos=[frames],
                return_tensors="pt",
            )
            model_inputs = move_inputs_to_device(model_inputs, device=device, dtype=dtype)

            generate_kwargs = {
                "max_new_tokens": args.max_new_tokens,
            }
            generation_config = copy.deepcopy(model.generation_config)
            generation_config.do_sample = do_sample
            if do_sample:
                generation_config.temperature = args.temperature
                generation_config.top_p = args.top_p
            else:
                if hasattr(generation_config, "temperature"):
                    generation_config.temperature = None
                if hasattr(generation_config, "top_p"):
                    generation_config.top_p = None
                if hasattr(generation_config, "top_k"):
                    generation_config.top_k = None
            generate_kwargs["generation_config"] = generation_config

            with torch.inference_mode():
                generated_ids = model.generate(
                    **model_inputs,
                    **generate_kwargs,
                )

            prompt_length = model_inputs["input_ids"].shape[1]
            generated_text = processor.batch_decode(
                generated_ids[:, prompt_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            generated_text = normalize_generated_prompt(generated_text)
            shard_rows.append(
                {
                    "__row_index": idx,
                    args.prompt_column: generated_text,
                    args.dense_prompt_column: generated_text,
                    args.sparse_prompt_column: row.get(args.sparse_prompt_column, sparse_prompt),
                }
            )
        except Exception as exc:
            print(f"[rank {accelerator.process_index}] [Error] Row {idx} failed for {video_path}: {exc}")

        processed_since_save += 1
        if processed_since_save >= args.save_every:
            save_shard_output(shard_rows, local_shard_path)
            processed_since_save = 0

    save_shard_output(shard_rows, local_shard_path)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        merged_results = load_existing_output(
            metadata=metadata,
            output_path=output_path,
            video_column=args.video_column,
            prompt_column=args.prompt_column,
            dense_prompt_column=args.dense_prompt_column,
            sparse_prompt_column=args.sparse_prompt_column,
            overwrite=args.overwrite,
        )
        merged_results = load_existing_shards(
            merged_results,
            shard_dir=shard_dir,
            prompt_column=args.prompt_column,
            dense_prompt_column=args.dense_prompt_column,
            sparse_prompt_column=args.sparse_prompt_column,
        )
        save_output(merged_results, output_path)
        completed = merged_results[args.dense_prompt_column].fillna("").astype(str).str.strip().ne("").sum()
        print(f"Finished. Generated dense prompts for {completed}/{len(merged_results)} rows.")


if __name__ == "__main__":
    main()
