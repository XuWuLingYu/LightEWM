MODEL_CONFIGS = [
    {
        # Wan2.1-Fun-1.3B-InP diffusion model
        "model_hash": "6d6ccde6845b95ad9114ab993d917893",
        "model_name": "wan_video_dit",
        "model_class": "lightewm.model.wan.wan_video_dit.WanModel",
        "extra_kwargs": {
            "has_image_input": True,
            "patch_size": [1, 2, 2],
            "in_dim": 36,
            "dim": 1536,
            "ffn_dim": 8960,
            "freq_dim": 256,
            "text_dim": 4096,
            "out_dim": 16,
            "num_heads": 12,
            "num_layers": 30,
            "eps": 1e-06,
        },
    },
    {
        # Shared text encoder
        "model_hash": "9c8818c2cbea55eca56c7b447df170da",
        "model_name": "wan_video_text_encoder",
        "model_class": "lightewm.model.wan.wan_video_text_encoder.WanTextEncoder",
    },
    {
        # Shared VAE
        "model_hash": "ccc42284ea13e1ad04693284c7a09be6",
        "model_name": "wan_video_vae",
        "model_class": "lightewm.model.wan.wan_video_vae.WanVideoVAE",
        "state_dict_converter": "lightewm.model.wan.wan_video_vae.WanVideoVAEStateDictConverter",
    },
    {
        # Shared CLIP image encoder
        "model_hash": "5941c53e207d62f20f9025686193c40b",
        "model_name": "wan_video_image_encoder",
        "model_class": "lightewm.model.wan.wan_video_image_encoder.WanImageEncoder",
        "state_dict_converter": "lightewm.model.wan.wan_video_image_encoder.WanImageEncoderStateDictConverter",
    },
]
