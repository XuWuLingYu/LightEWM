import torch
from typing import Union

from ...utils.device.npu_compatible_device import get_device_type
from ...utils.loader.config import ModelConfig

from .pipeline import WanVideoPipeline
from .wan_video_vae import WanVideoVAE38


class WanTI2V5BPipeline(WanVideoPipeline):
    """Minimal pipeline for Wan2.2-TI2V-5B cache, training, and inference."""

    @classmethod
    def from_pretrained(
        cls,
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
            raise RuntimeError("Unified sequence parallel is not supported in Wan2.2-TI2V-5B pipeline.")
        if audio_processor_config is not None:
            print("[WanTI2V5B] audio_processor_config is ignored in the TI2V pipeline.")

        pipe = cls(device=device, torch_dtype=torch_dtype)
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
            from .wan_video_text_encoder import HuggingfaceTokenizer

            pipe.tokenizer = HuggingfaceTokenizer(
                name=tokenizer_config.path,
                seq_len=512,
                clean="whitespace",
            )

        if pipe.dit is not None and not pipe.dit.fuse_vae_embedding_in_latents:
            raise ValueError("WanTI2V5BPipeline expects a TI2V model with fused VAE embeddings in latents.")
        if pipe.vae is not None and not isinstance(pipe.vae, WanVideoVAE38):
            raise ValueError("WanTI2V5BPipeline expects Wan2.2 VAE38.")

        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe
