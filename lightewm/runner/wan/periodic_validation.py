from __future__ import annotations

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
        selected_indices = self._sample_indices(global_step)
        saved_items = []

        try:
            pipe.units = validation_units
            with torch.inference_mode():
                for slot_id, sample_idx in enumerate(selected_indices):
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

                    if accelerator.is_main_process:
                        save_path = self._video_path(global_step, slot_id, item)
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        save_video(video, save_path, fps=self.fps, quality=self.quality)
                        saved_items.append((slot_id, save_path, self._caption(item)))
        finally:
            pipe.units = previous_units
            if prev_scheduler_steps > 0:
                pipe.scheduler.set_timesteps(prev_scheduler_steps, training=prev_scheduler_training)
            accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            print(
                f"[Validation] step={global_step} generated {len(saved_items)} video(s) "
                f"at {os.path.join(self.output_dir, f'step_{int(global_step):08d}')}"
            )
            self._log_to_wandb(saved_items, global_step)

    def _sample_indices(self, global_step: int) -> list[int]:
        sample_count = min(self.num_samples, len(self.dataset))
        generator = torch.Generator(device="cpu").manual_seed(self.seed_base + int(global_step))
        return torch.randperm(len(self.dataset), generator=generator)[:sample_count].tolist()

    def _video_path(self, global_step: int, slot_id: int, item: dict) -> str:
        row_id = int(item.get("row_id", slot_id))
        demo_id = str(item.get("demo_id", row_id))
        camera_key = str(item.get("camera_key", "unknown"))
        name = f"{slot_id:02d}__row_{row_id:06d}__{demo_id}__{camera_key}.mp4"
        return str(Path(self.output_dir) / f"step_{int(global_step):08d}" / name)

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
        for slot_id, save_path, caption in saved_items:
            log_data[f"val/video_{slot_id}"] = wandb.Video(
                save_path,
                fps=self.fps,
                format="mp4",
                caption=caption,
            )
        self.wandb_run.log(log_data, step=global_step)
