import os
from pathlib import Path

from tqdm import tqdm

from lightewm.dataset.operators import ImageCropAndResize
from lightewm.runner.runner_util.instantiation import instantiate_component_from_section
from lightewm.runner.runner_util.wan_runtime import build_wan_i2v_pipeline_from_params
from lightewm.utils.data import save_video


class WanInferRunner:
    def __init__(self, config):
        self.config = config

    def run(self):
        full_config = self.config.full_config
        dataset, _ = instantiate_component_from_section(
            full_config.dataset,
            full_config,
            section_name="dataset",
        )
        model_params = (
            full_config.model.params.to_dict()
            if hasattr(full_config.model.params, "to_dict")
            else dict(full_config.model.params)
        )
        model_params["pipeline_class_path"] = full_config.model.class_path
        model = build_wan_i2v_pipeline_from_params(model_params)

        output_dir = getattr(self.config, "output_dir", "./outputs/libero_infer")
        os.makedirs(output_dir, exist_ok=True)

        fps = int(getattr(self.config, "fps", 16))
        quality = int(getattr(self.config, "quality", 5))
        seed_base = int(getattr(self.config, "seed", 0))
        infer_kwargs = dict(getattr(self.config, "infer_kwargs", {}))
        input_image_resize_mode = getattr(self.config, "input_image_resize_mode", "stretch")
        target_height = infer_kwargs.get("height", None)
        target_width = infer_kwargs.get("width", None)
        input_image_resizer = None
        if target_height is not None and target_width is not None:
            input_image_resizer = ImageCropAndResize(
                height=int(target_height),
                width=int(target_width),
                max_pixels=None,
                height_division_factor=16,
                width_division_factor=16,
                resize_mode=input_image_resize_mode,
            )

        for item in tqdm(dataset, total=len(dataset), desc="Infer"):
            input_image = item["input_image"]
            if input_image_resizer is not None:
                input_image = input_image_resizer(input_image)
            video = model(
                prompt=item["prompt"],
                input_image=input_image,
                seed=seed_base + int(item["row_id"]),
                **infer_kwargs,
            )
            name = f"{item['row_id']:06d}__{item['demo_id']}__{item['camera_key']}.mp4"
            save_path = str(Path(output_dir) / name)
            save_video(video, save_path, fps=fps, quality=quality)
