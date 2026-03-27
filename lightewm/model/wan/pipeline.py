import torch
from PIL import Image
from tqdm import tqdm
from typing import Optional, Union
from typing_extensions import Literal

from ...model.flow_match import FlowMatchScheduler
from ...utils.device.npu_compatible_device import get_device_type
from ...utils.gradient import gradient_checkpoint_forward
from ...utils.loader.config import ModelConfig
from ...runner.base_pipeline import BasePipeline, PipelineUnit

from .wan_video_dit import WanModel, sinusoidal_embedding_1d
from .wan_video_image_encoder import WanImageEncoder
from .wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder
from .wan_video_vae import WanVideoVAE


class WanVideoPipeline(BasePipeline):
    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=16,
            width_division_factor=16,
            time_division_factor=4,
            time_division_remainder=1,
        )
        self.scheduler = FlowMatchScheduler("Wan")
        self.tokenizer: HuggingfaceTokenizer = None
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.dit2: WanModel = None
        self.vae: WanVideoVAE = None

        self.in_iteration_models = ("dit",)
        self.in_iteration_models_2 = ("dit2",)
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
            WanVideoUnit_ImageEmbedderFused(),
            WanVideoUnit_CfgMerger(),
        ]
        self.post_units = []
        self.model_fn = model_fn_wan_video
        self.compilable_models = ["dit", "dit2"]

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/umt5-xxl/",
        ),
        audio_processor_config: ModelConfig = None,
        redirect_common_files: bool = True,
        use_usp: bool = False,
        vram_limit: float = None,
    ):
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": (
                    "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
                    "models_t5_umt5-xxl-enc-bf16.safetensors",
                ),
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": (
                    "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
                    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors",
                ),
                "Wan2.1_VAE.pth": (
                    "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
                    "Wan2.1_VAE.safetensors",
                ),
                "Wan2.2_VAE.pth": (
                    "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
                    "Wan2.2_VAE.safetensors",
                ),
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if (
                    model_config.origin_file_pattern in redirect_dict
                    and model_config.model_id
                    != redirect_dict[model_config.origin_file_pattern][0]
                ):
                    print(
                        "To avoid repeatedly downloading model files, "
                        f"({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to "
                        f"{redirect_dict[model_config.origin_file_pattern]}. "
                        "You can use `redirect_common_files=False` to disable file redirection."
                    )
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern][0]
                    model_config.origin_file_pattern = redirect_dict[model_config.origin_file_pattern][1]

        if use_usp:
            raise RuntimeError("Unified sequence parallel is not supported in Wan 1.3B I2V minimal pipeline.")
        if audio_processor_config is not None:
            print("[WanI2V] audio_processor_config is ignored in the minimal I2V pipeline.")

        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
        dit = model_pool.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_pool.fetch_model("wan_video_vae")
        pipe.image_encoder = model_pool.fetch_model("wan_video_image_encoder")

        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = HuggingfaceTokenizer(
                name=tokenizer_config.path,
                seq_len=512,
                clean="whitespace",
            )

        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = "",
        input_image: Optional[Image.Image] = None,
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames: int = 81,
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        switch_DiT_boundary: Optional[float] = 0.875,
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        progress_bar_cmd=tqdm,
        output_type: Optional[Literal["quantized", "floatpoint"]] = "quantized",
    ):
        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=denoising_strength,
            shift=sigma_shift,
        )

        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "input_image": input_image,
            "input_video": input_video,
            "denoising_strength": denoising_strength,
            "seed": seed,
            "rand_device": rand_device,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "cfg_scale": cfg_scale,
            "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }

        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit,
                self,
                inputs_shared,
                inputs_posi,
                inputs_nega,
            )

        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            if (
                timestep.item() < switch_DiT_boundary * 1000
                and self.dit2 is not None
                and models["dit"] is not self.dit2
            ):
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2

            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = self.model_fn(
                        **models,
                        **inputs_shared,
                        **inputs_nega,
                        timestep=timestep,
                    )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            inputs_shared["latents"] = self.scheduler.step(
                noise_pred,
                self.scheduler.timesteps[progress_id],
                inputs_shared["latents"],
            )
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]

        self.load_models_to_device(["vae"])
        video = self.vae.decode(
            inputs_shared["latents"],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        if output_type == "quantized":
            video = self.vae_output_to_video(video)
        elif output_type == "floatpoint":
            pass
        self.load_models_to_device([])
        return video


class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}


class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device"),
            output_params=("noise",),
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device):
        length = (num_frames - 1) // 4 + 1
        shape = (
            1,
            pipe.vae.model.z_dim,
            length,
            height // pipe.vae.upsampling_factor,
            width // pipe.vae.upsampling_factor,
        )
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        return {"noise": noise}


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",),
        )

    def process(self, pipe: WanVideoPipeline, input_video, noise, tiled, tile_size, tile_stride):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(self.onload_model_names)
        input_video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(
            input_video,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
        return {"latents": latents}


class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            output_params=("context",),
            onload_model_names=("text_encoder",),
        )

    def encode_prompt(self, pipe: WanVideoPipeline, prompt):
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = self.encode_prompt(pipe, prompt)
        return {"context": prompt_emb}


class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "height", "width"),
            output_params=("clip_feature",),
            onload_model_names=("image_encoder",),
        )

    def process(self, pipe: WanVideoPipeline, input_image, height, width):
        if input_image is None or pipe.image_encoder is None or not pipe.dit.require_clip_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context}


class WanVideoUnit_ImageEmbedderVAE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("y",),
            onload_model_names=("vae",),
        )

    def process(self, pipe: WanVideoPipeline, input_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.require_vae_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        msk = torch.ones(1, num_frames, height // 8, width // 8, device=pipe.device)
        msk[:, 1:] = 0
        vae_input = torch.concat(
            [image.transpose(0, 1), torch.zeros(3, num_frames - 1, height, width).to(image.device)],
            dim=1,
        )

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
        msk = msk.transpose(1, 2)[0]

        y = pipe.vae.encode(
            [vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)],
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y]).unsqueeze(0)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"y": y}


class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "latents", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "fuse_vae_embedding_in_latents", "first_frame_latents"),
            onload_model_names=("vae",),
        )

    def process(self, pipe: WanVideoPipeline, input_image, latents, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).transpose(0, 1)
        z = pipe.vae.encode([image], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        latents[:, :, 0:1] = z
        return {
            "latents": latents,
            "fuse_vae_embedding_in_latents": True,
            "first_frame_latents": z,
        }


class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y"]

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega


def model_fn_wan_video(
    dit: WanModel,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    fuse_vae_embedding_in_latents: bool = False,
    **kwargs,
):
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat(
            [
                torch.zeros(
                    (1, latents.shape[3] * latents.shape[4] // 4),
                    dtype=latents.dtype,
                    device=latents.device,
                ),
                torch.ones(
                    (latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4),
                    dtype=latents.dtype,
                    device=latents.device,
                )
                * timestep,
            ]
        ).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    context = dit.text_embedding(context)

    x = latents
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    x, (f, h, w) = dit.patchify(x)

    freqs = torch.cat(
        [
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1).to(x.device)

    for block in dit.blocks:
        x = gradient_checkpoint_forward(
            block,
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            x,
            context,
            t_mod,
            freqs,
        )

    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    return x
