from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import torch

from lightewm.dataset.operators import ImageCropAndResize
from lightewm.utils.data import save_video


class PeriodicWanVideoValidator:
    def __init__(
        self,
        dataset,
        output_root: str,
        every_steps: int = 1000,
        extra_steps: list[int] | None = None,
        num_samples: int = 3,
        fps: int = 16,
        quality: int = 5,
        seed_base: int = 0,
        infer_kwargs: dict | None = None,
        input_image_resize_mode: str = "stretch",
        wandb_run=None,
    ):
        self.dataset = dataset
        self.output_root = output_root
        self.every_steps = max(int(every_steps), 0)
        self.extra_steps = sorted({int(step) for step in (extra_steps or []) if int(step) > 0})
        self.num_samples = max(int(num_samples), 0)
        self.fps = int(fps)
        self.quality = int(quality)
        self.seed_base = int(seed_base)
        self.infer_kwargs = {} if infer_kwargs is None else dict(infer_kwargs)
        self.input_image_resize_mode = input_image_resize_mode
        self.wandb_run = wandb_run

        self.output_dir = os.path.join(self.output_root, "validation")
        target_height = self.infer_kwargs.get("height", None)
        target_width = self.infer_kwargs.get("width", None)
        self.input_image_resizer = None
        if target_height is not None and target_width is not None:
            self.input_image_resizer = ImageCropAndResize(
                height=int(target_height),
                width=int(target_width),
                max_pixels=None,
                height_division_factor=16,
                width_division_factor=16,
                resize_mode=input_image_resize_mode,
            )

    def enabled(self) -> bool:
        return (self.every_steps > 0 or len(self.extra_steps) > 0) and self.num_samples > 0 and len(self.dataset) > 0

    def _step_dir(self, global_step: int) -> Path:
        return Path(self.output_dir) / f"step_{int(global_step)}"

    def maybe_run(self, accelerator, model, global_step: int):
        should_run = False
        if self.every_steps > 0 and global_step % self.every_steps == 0:
            should_run = True
        if global_step in self.extra_steps:
            should_run = True
        if not self.enabled() or not should_run:
            return

        accelerator.wait_for_everyone()

        training_module = accelerator.unwrap_model(model)
        pipe = training_module.pipe
        validation_units = getattr(training_module, "validation_pipe_units", None) or pipe.units
        previous_units = pipe.units
        prev_scheduler_steps = len(getattr(pipe.scheduler, "timesteps", []))
        prev_scheduler_training = bool(getattr(pipe.scheduler, "training", False))
        sample_idx = self._sample_index_for_rank(global_step, accelerator.process_index, accelerator.num_processes)

        try:
            pipe.units = validation_units
            with torch.inference_mode():
                item = self.dataset[sample_idx]
                input_image = item["input_image"]
                if self.input_image_resizer is not None:
                    input_image = self.input_image_resizer(input_image)
                video = pipe(
                    prompt=item["prompt"],
                    input_image=input_image,
                    seed=self.seed_base + int(global_step) + int(item.get("row_id", sample_idx)),
                    progress_bar_cmd=lambda x: x,
                    **self.infer_kwargs,
                )

                save_path = self._video_path(global_step, accelerator.process_index, item)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_video(video, save_path, fps=self.fps, quality=self.quality)
                self._write_sidecar(save_path, self._caption(item), accelerator.process_index, sample_idx)
                self._write_rank_manifest(global_step, accelerator.process_index, sample_idx, save_path)
                print(f"[Validation][rank{accelerator.process_index}] step={global_step} saved {save_path}")
                del video, input_image, item
        finally:
            pipe.units = previous_units
            if prev_scheduler_steps > 0:
                pipe.scheduler.set_timesteps(prev_scheduler_steps, training=prev_scheduler_training)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            saved_items = self._collect_saved_items(global_step)
            missing_ranks = self._find_missing_ranks(global_step, accelerator.num_processes)
            print(
                f"[Validation] step={global_step} generated {len(saved_items)} video(s) "
                f"at {self._step_dir(global_step)}"
            )
            if len(missing_ranks) > 0:
                print(f"[Validation] step={global_step} missing rank outputs: {missing_ranks}")
            self._log_to_wandb(saved_items, global_step)

    def _sample_index_for_rank(self, global_step: int, process_index: int, num_processes: int) -> int:
        sample_count = min(max(num_processes, self.num_samples), len(self.dataset))
        generator = torch.Generator(device="cpu").manual_seed(self.seed_base + int(global_step))
        indices = torch.randperm(len(self.dataset), generator=generator)[:sample_count].tolist()
        return indices[process_index % len(indices)]

    def _video_path(self, global_step: int, process_index: int, item: dict) -> str:
        row_id = int(item.get("row_id", process_index))
        demo_id = str(item.get("demo_id", row_id))
        camera_key = str(item.get("camera_key", "unknown"))
        name = f"row_{row_id:06d}__{demo_id}__{camera_key}.mp4"
        return str(self._step_dir(global_step) / f"rank{int(process_index)}" / name)

    def _sidecar_path(self, save_path: str) -> str:
        return str(Path(save_path).with_suffix(".json"))

    def _rank_manifest_path(self, global_step: int, process_index: int) -> str:
        return str(self._step_dir(global_step) / f"rank{int(process_index)}" / "_rank_manifest.json")

    def _write_sidecar(self, save_path: str, caption: str, process_index: int, sample_idx: int):
        sidecar = {
            "caption": caption,
            "process_index": int(process_index),
            "sample_idx": int(sample_idx),
            "video_path": save_path,
        }
        with open(self._sidecar_path(save_path), "w", encoding="utf-8") as f:
            json.dump(sidecar, f, ensure_ascii=False)

    def _write_rank_manifest(self, global_step: int, process_index: int, sample_idx: int, save_path: str):
        manifest = {
            "process_index": int(process_index),
            "sample_idx": int(sample_idx),
            "video_path": save_path,
        }
        manifest_path = self._rank_manifest_path(global_step, process_index)
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False)

    def _collect_saved_items(self, global_step: int) -> list[tuple[int, str, str]]:
        step_dir = self._step_dir(global_step)
        saved_items = []
        for sidecar_path in sorted(step_dir.glob("rank*/*.json")):
            if sidecar_path.name == "_rank_manifest.json":
                continue
            with open(sidecar_path, "r", encoding="utf-8") as f:
                sidecar = json.load(f)
            save_path = sidecar.get("video_path")
            if not save_path or not os.path.exists(save_path):
                continue
            process_index = int(sidecar.get("process_index", -1))
            caption = str(sidecar.get("caption", ""))
            saved_items.append((process_index, save_path, caption))
        return saved_items

    def _find_missing_ranks(self, global_step: int, num_processes: int) -> list[int]:
        step_dir = self._step_dir(global_step)
        present = set()
        for manifest_path in step_dir.glob("rank*/_rank_manifest.json"):
            rank_name = manifest_path.parent.name
            if rank_name.startswith("rank"):
                try:
                    present.add(int(rank_name[4:]))
                except Exception:
                    pass
        return [rank for rank in range(int(num_processes)) if rank not in present]

    def _caption(self, item: dict) -> str:
        row_id = int(item.get("row_id", -1))
        demo_id = str(item.get("demo_id", "unknown"))
        camera_key = str(item.get("camera_key", "unknown"))
        prompt = str(item.get("prompt", ""))
        return f"row_id={row_id} demo_id={demo_id} camera_key={camera_key} prompt={prompt}"

    def _log_to_wandb(self, saved_items: list[tuple[int, str, str]], global_step: int):
        if self.wandb_run is None or len(saved_items) == 0:
            return
        try:
            import wandb
        except Exception as exc:
            print(f"[Validation] wandb is unavailable, skip video upload. Error: {exc}")
            return

        log_data = {}
        for process_index, save_path, caption in saved_items:
            log_data[f"val/rank_{process_index}"] = wandb.Video(
                save_path,
                fps=self.fps,
                format="mp4",
                caption=caption,
            )
        self.wandb_run.log(log_data, step=global_step)
