flux_general_vram_config = {
    "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
    "torch.nn.Embedding": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    "torch.nn.LayerNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    "torch.nn.Conv2d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    "torch.nn.GroupNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    "lightewm.diffsynth.models.general_modules.RMSNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    "lightewm.diffsynth.models.flux_lora_encoder.LoRALayerBlock": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    "lightewm.diffsynth.models.flux_lora_patcher.LoraMerger": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
}

VRAM_MANAGEMENT_MODULE_MAPS = {
    "lightewm.diffsynth.models.qwen_image_dit.QwenImageDiT": {
        "lightewm.diffsynth.models.qwen_image_dit.RMSNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.Embedding": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.qwen_image_text_encoder.QwenImageTextEncoder": {
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.Embedding": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLRotaryEmbedding": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2RMSNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionPatchEmbed": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionRotaryEmbedding": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.qwen_image_vae.QwenImageVAE": {
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.Conv3d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Conv2d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "lightewm.diffsynth.models.qwen_image_vae.QwenImageRMS_norm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.qwen_image_controlnet.BlockWiseControlBlock": {
        "lightewm.diffsynth.models.qwen_image_dit.RMSNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
    },
    "lightewm.diffsynth.models.siglip2_image_encoder.Siglip2ImageEncoder": {
        "transformers.models.siglip.modeling_siglip.SiglipVisionEmbeddings": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "transformers.models.siglip.modeling_siglip.SiglipMultiheadAttentionPoolingHead": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Conv2d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Embedding": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.LayerNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
    },
    "lightewm.diffsynth.models.dinov3_image_encoder.DINOv3ImageEncoder": {
        "transformers.models.dinov3_vit.modeling_dinov3_vit.DINOv3ViTLayerScale": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "transformers.models.dinov3_vit.modeling_dinov3_vit.DINOv3ViTRopePositionEmbedding": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "transformers.models.dinov3_vit.modeling_dinov3_vit.DINOv3ViTEmbeddings": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Conv2d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.LayerNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
    },
    "lightewm.diffsynth.models.qwen_image_image2lora.QwenImageImage2LoRAModel": {
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
    },
    "lightewm.diffsynth.models.wan_video_animate_adapter.WanAnimateAdapter": {
        "lightewm.diffsynth.models.wan_video_animate_adapter.FaceEncoder": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "lightewm.diffsynth.models.wan_video_animate_adapter.EqualLinear": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "lightewm.diffsynth.models.wan_video_animate_adapter.ConvLayer": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "lightewm.diffsynth.models.wan_video_animate_adapter.FusedLeakyReLU": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "lightewm.diffsynth.models.wan_video_animate_adapter.RMSNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.LayerNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Conv1d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Conv2d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Conv3d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.wan_video_dit_s2v.WanS2VModel": {
        "lightewm.diffsynth.models.wan_video_dit.Head": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "lightewm.diffsynth.models.wan_video_dit_s2v.WanS2VDiTBlock": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "lightewm.diffsynth.models.wan_video_dit_s2v.CausalAudioEncoder": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Embedding": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.Conv3d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.LayerNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "lightewm.diffsynth.models.wan_video_dit.RMSNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Conv2d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.wan_video_dit.WanModel": {
        "lightewm.diffsynth.models.wan_video_dit.MLP": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "lightewm.diffsynth.models.wan_video_dit.DiTBlock": "lightewm.diffsynth.core.vram.layers.AutoWrappedNonRecurseModule",
        "lightewm.diffsynth.models.wan_video_dit.Head": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.Conv3d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.LayerNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "lightewm.diffsynth.models.wan_video_dit.RMSNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Conv2d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.wan_video_image_encoder.WanImageEncoder": {
        "lightewm.diffsynth.models.wan_video_image_encoder.VisionTransformer": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.Conv2d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.LayerNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.wan_video_mot.MotWanModel": {
        "lightewm.diffsynth.models.wan_video_mot.MotWanAttentionBlock": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Conv3d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.LayerNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.wan_video_motion_controller.WanMotionControllerModel": {
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
    },
    "lightewm.diffsynth.models.wan_video_text_encoder.WanTextEncoder": {
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.Embedding": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "lightewm.diffsynth.models.wan_video_text_encoder.T5RelativeEmbedding": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "lightewm.diffsynth.models.wan_video_text_encoder.T5LayerNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.wan_video_vace.VaceWanModel": {
        "lightewm.diffsynth.models.wan_video_dit.DiTBlock": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.Conv3d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.LayerNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "lightewm.diffsynth.models.wan_video_dit.RMSNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.wan_video_vae.WanVideoVAE": {
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.Conv2d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "lightewm.diffsynth.models.wan_video_vae.RMS_norm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "lightewm.diffsynth.models.wan_video_vae.CausalConv3d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "lightewm.diffsynth.models.wan_video_vae.Upsample": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.SiLU": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Dropout": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.wan_video_vae.WanVideoVAE38": {
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.Conv2d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "lightewm.diffsynth.models.wan_video_vae.RMS_norm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "lightewm.diffsynth.models.wan_video_vae.CausalConv3d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "lightewm.diffsynth.models.wan_video_vae.Upsample": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.SiLU": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Dropout": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.wav2vec.WanS2VAudioEncoder": {
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.LayerNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Conv1d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.longcat_video_dit.LongCatVideoTransformer3DModel": {
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.Conv3d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "lightewm.diffsynth.models.longcat_video_dit.RMSNorm_FP32": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "lightewm.diffsynth.models.longcat_video_dit.LayerNorm_FP32": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.flux_dit.FluxDiT": {
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
        "lightewm.diffsynth.models.flux_dit.RMSNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.flux_text_encoder_clip.FluxTextEncoderClip": flux_general_vram_config,
    "lightewm.diffsynth.models.flux_vae.FluxVAEEncoder": flux_general_vram_config,
    "lightewm.diffsynth.models.flux_vae.FluxVAEDecoder": flux_general_vram_config,
    "lightewm.diffsynth.models.flux_controlnet.FluxControlNet": flux_general_vram_config,
    "lightewm.diffsynth.models.flux_infiniteyou.InfiniteYouImageProjector": flux_general_vram_config,
    "lightewm.diffsynth.models.flux_ipadapter.FluxIpAdapter": flux_general_vram_config,
    "lightewm.diffsynth.models.flux_lora_patcher.FluxLoraPatcher": flux_general_vram_config,
    "lightewm.diffsynth.models.step1x_connector.Qwen2Connector": flux_general_vram_config,
    "lightewm.diffsynth.models.flux_lora_encoder.FluxLoRAEncoder": flux_general_vram_config,
    "lightewm.diffsynth.models.flux_text_encoder_t5.FluxTextEncoderT5": {
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.Embedding": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "transformers.models.t5.modeling_t5.T5LayerNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "transformers.models.t5.modeling_t5.T5DenseActDense": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "transformers.models.t5.modeling_t5.T5DenseGatedActDense": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.flux_ipadapter.SiglipVisionModelSO400M": {
        "transformers.models.siglip.modeling_siglip.SiglipVisionEmbeddings": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "transformers.models.siglip.modeling_siglip.SiglipEncoder": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "transformers.models.siglip.modeling_siglip.SiglipMultiheadAttentionPoolingHead": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.MultiheadAttention": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.LayerNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.flux2_dit.Flux2DiT": {
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.LayerNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.RMSNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.flux2_text_encoder.Flux2TextEncoder": {
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.Conv2d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Embedding": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "transformers.models.mistral.modeling_mistral.MistralRMSNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.flux2_vae.Flux2VAE": {
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.Conv2d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.GroupNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.z_image_text_encoder.ZImageTextEncoder": {
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
        "transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Embedding": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.z_image_dit.ZImageDiT": {
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
        "lightewm.diffsynth.models.z_image_dit.RMSNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.z_image_controlnet.ZImageControlNet": {
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
        "lightewm.diffsynth.models.z_image_dit.RMSNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.z_image_image2lora.ZImageImage2LoRAModel": {
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
    },
    "lightewm.diffsynth.models.siglip2_image_encoder.Siglip2ImageEncoder428M": {
        "transformers.models.siglip2.modeling_siglip2.Siglip2VisionEmbeddings": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "transformers.models.siglip2.modeling_siglip2.Siglip2MultiheadAttentionPoolingHead": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Conv2d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Embedding": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.LayerNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
    },
    "lightewm.diffsynth.models.ltx2_dit.LTXModel": {
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.RMSNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.ltx2_upsampler.LTX2LatentUpsampler": {
        "torch.nn.Conv2d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Conv3d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.GroupNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.ltx2_video_vae.LTX2VideoEncoder": {
        "torch.nn.Conv3d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.ltx2_video_vae.LTX2VideoDecoder": {
        "torch.nn.Conv3d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.ltx2_audio_vae.LTX2AudioDecoder": {
        "torch.nn.Conv2d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.ltx2_audio_vae.LTX2Vocoder": {
        "torch.nn.Conv1d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.ConvTranspose1d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.ltx2_text_encoder.LTX2TextEncoderPostModules": {
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.RMSNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "lightewm.diffsynth.models.ltx2_text_encoder.Embeddings1DConnector": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.ltx2_text_encoder.LTX2TextEncoder": {
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
        "transformers.models.gemma3.modeling_gemma3.Gemma3MultiModalProjector": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "transformers.models.gemma3.modeling_gemma3.Gemma3RMSNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "transformers.models.gemma3.modeling_gemma3.Gemma3TextScaledWordEmbedding": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.anima_dit.AnimaDiT": {
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.LayerNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.RMSNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Embedding": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.mova_audio_dit.MovaAudioDit": {
        "lightewm.diffsynth.models.wan_video_dit.DiTBlock": "lightewm.diffsynth.core.vram.layers.AutoWrappedNonRecurseModule",
        "lightewm.diffsynth.models.wan_video_dit.Head": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.Conv1d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.LayerNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "lightewm.diffsynth.models.wan_video_dit.RMSNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.mova_dual_tower_bridge.DualTowerConditionalBridge": {
        "torch.nn.Linear": "lightewm.diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.LayerNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "lightewm.diffsynth.models.wan_video_dit.RMSNorm": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "lightewm.diffsynth.models.mova_audio_vae.DacVAE": {
        "lightewm.diffsynth.models.mova_audio_vae.Snake1d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Conv1d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.ConvTranspose1d": "lightewm.diffsynth.core.vram.layers.AutoWrappedModule",
    },
}

def QwenImageTextEncoder_Module_Map_Updater():
    current = VRAM_MANAGEMENT_MODULE_MAPS["lightewm.diffsynth.models.qwen_image_text_encoder.QwenImageTextEncoder"]
    from packaging import version
    import transformers
    if version.parse(transformers.__version__) >= version.parse("5.2.0"):
        # The Qwen2RMSNorm in transformers 5.2.0+ has been renamed to Qwen2_5_VLRMSNorm, so we need to update the module map accordingly
        current.pop("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2RMSNorm", None)
        current["transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLRMSNorm"] = "lightewm.diffsynth.core.vram.layers.AutoWrappedModule"
    return current

VERSION_CHECKER_MAPS = {
    "lightewm.diffsynth.models.qwen_image_text_encoder.QwenImageTextEncoder": QwenImageTextEncoder_Module_Map_Updater,
}