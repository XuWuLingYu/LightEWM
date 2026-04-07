MODEL_CONFIGS = [
    {
        # Wan2.2-TI2V-5B diffusion model
        "model_hash": "1f5ab7703c6fc803fdded85ff040c316",
        "model_name": "wan_video_dit",
        "model_class": "lightewm.model.wan.wan_video_dit.WanModel",
        "extra_kwargs": {
            "has_image_input": False,
            "patch_size": [1, 2, 2],
            "in_dim": 48,
            "dim": 3072,
            "ffn_dim": 14336,
            "freq_dim": 256,
            "text_dim": 4096,
            "out_dim": 48,
            "num_heads": 24,
            "num_layers": 30,
            "eps": 1e-06,
            "seperated_timestep": True,
            "require_clip_embedding": False,
            "require_vae_embedding": False,
            "fuse_vae_embedding_in_latents": True,
        },
    },
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
        # Wan2.1-I2V-14B-480P / Wan2.1-Fun-14B-InP diffusion-family model
        "model_hash": "6bfcfb3b342cb286ce886889d519a77e",
        "model_name": "wan_video_dit",
        "model_class": "lightewm.model.wan.wan_video_dit.WanModel",
        "extra_kwargs": {
            "has_image_input": True,
            "patch_size": [1, 2, 2],
            "in_dim": 36,
            "dim": 5120,
            "ffn_dim": 13824,
            "freq_dim": 256,
            "text_dim": 4096,
            "out_dim": 16,
            "num_heads": 40,
            "num_layers": 40,
            "eps": 1e-06,
        },
    },
    {
        # Wan2.1-I2V-14B-720P / FLF2V-14B-720P diffusion-family model
        "model_hash": "3ef3b1f8e1dab83d5b71fd7b617f859f",
        "model_name": "wan_video_dit",
        "model_class": "lightewm.model.wan.wan_video_dit.WanModel",
        "extra_kwargs": {
            "has_image_input": True,
            "patch_size": [1, 2, 2],
            "in_dim": 36,
            "dim": 5120,
            "ffn_dim": 13824,
            "freq_dim": 256,
            "text_dim": 4096,
            "out_dim": 16,
            "num_heads": 40,
            "num_layers": 40,
            "eps": 1e-06,
            "has_image_pos_emb": True,
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
        # Wan2.2-TI2V-5B VAE
        "model_hash": "e1de6c02cdac79f8b739f4d3698cd216",
        "model_name": "wan_video_vae",
        "model_class": "lightewm.model.wan.wan_video_vae.WanVideoVAE38",
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
