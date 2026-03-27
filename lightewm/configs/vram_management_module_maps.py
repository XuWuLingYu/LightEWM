VRAM_MANAGEMENT_MODULE_MAPS = {
    "lightewm.model.wan.wan_video_dit.WanModel": {
        "lightewm.model.wan.wan_video_dit.MLP": "lightewm.utils.vram.layers.AutoWrappedModule",
        "lightewm.model.wan.wan_video_dit.DiTBlock": "lightewm.utils.vram.layers.AutoWrappedNonRecurseModule",
        "lightewm.model.wan.wan_video_dit.Head": "lightewm.utils.vram.layers.AutoWrappedModule",
        "torch.nn.Linear": "lightewm.utils.vram.layers.AutoWrappedLinear",
        "torch.nn.Conv3d": "lightewm.utils.vram.layers.AutoWrappedModule",
        "torch.nn.LayerNorm": "lightewm.utils.vram.layers.AutoWrappedModule",
        "lightewm.model.wan.wan_video_dit.RMSNorm": "lightewm.utils.vram.layers.AutoWrappedModule",
        "torch.nn.Conv2d": "lightewm.utils.vram.layers.AutoWrappedModule",
    },
    "lightewm.model.wan.wan_video_text_encoder.WanTextEncoder": {
        "torch.nn.Linear": "lightewm.utils.vram.layers.AutoWrappedLinear",
        "torch.nn.Embedding": "lightewm.utils.vram.layers.AutoWrappedModule",
        "lightewm.model.wan.wan_video_text_encoder.T5RelativeEmbedding": "lightewm.utils.vram.layers.AutoWrappedModule",
        "lightewm.model.wan.wan_video_text_encoder.T5LayerNorm": "lightewm.utils.vram.layers.AutoWrappedModule",
    },
    "lightewm.model.wan.wan_video_vae.WanVideoVAE": {
        "torch.nn.Linear": "lightewm.utils.vram.layers.AutoWrappedLinear",
        "torch.nn.Conv2d": "lightewm.utils.vram.layers.AutoWrappedModule",
        "lightewm.model.wan.wan_video_vae.RMS_norm": "lightewm.utils.vram.layers.AutoWrappedModule",
        "lightewm.model.wan.wan_video_vae.CausalConv3d": "lightewm.utils.vram.layers.AutoWrappedModule",
        "lightewm.model.wan.wan_video_vae.Upsample": "lightewm.utils.vram.layers.AutoWrappedModule",
        "torch.nn.SiLU": "lightewm.utils.vram.layers.AutoWrappedModule",
        "torch.nn.Dropout": "lightewm.utils.vram.layers.AutoWrappedModule",
    },
    "lightewm.model.wan.wan_video_image_encoder.WanImageEncoder": {
        "lightewm.model.wan.wan_video_image_encoder.VisionTransformer": "lightewm.utils.vram.layers.AutoWrappedModule",
        "torch.nn.Linear": "lightewm.utils.vram.layers.AutoWrappedLinear",
        "torch.nn.Conv2d": "lightewm.utils.vram.layers.AutoWrappedModule",
        "torch.nn.LayerNorm": "lightewm.utils.vram.layers.AutoWrappedModule",
    },
}

VERSION_CHECKER_MAPS = {}
