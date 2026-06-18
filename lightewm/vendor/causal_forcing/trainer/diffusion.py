import gc
import logging

from model import CausalDiffusion
from model.action_mot import HDRVideoActionJointMoT
from utils.dataset import cycle, LatentLMDBDataset, TextVideoDataset
from utils.misc import set_seed
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import wandb
import time
import os
import math
from utils.distributed import EMA_FSDP, barrier, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)


def _is_valid_wandb_value(value):
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if not stripped:
        return False
    placeholder_markers = ("your key", "your entity", "your project")
    if any(marker in stripped.lower() for marker in placeholder_markers):
        return False
    if stripped.startswith("{") and stripped.endswith("}"):
        return False
    return True


def _resolve_wandb_value(config_value, env_name, default=None):
    if _is_valid_wandb_value(config_value):
        return config_value.strip()
    env_value = os.environ.get(env_name, default)
    if _is_valid_wandb_value(env_value):
        return env_value.strip()
    return None


def _to_plain_config(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _load_generator_checkpoint(path):
    state_dict = torch.load(path, map_location="cpu")
    if "generator" in state_dict:
        state_dict = state_dict["generator"]
        fixed = {}
        for k, v in state_dict.items():
            if k.startswith("model._fsdp_wrapped_module."):
                k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
            fixed[k] = v
        state_dict = fixed
    elif "model" in state_dict:
        state_dict = state_dict["model"]
    elif "generator_ema" in state_dict:
        gen_sd = state_dict["generator_ema"]
        fixed = {}
        for k, v in gen_sd.items():
            if k.startswith("model._fsdp_wrapped_module."):
                k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
            fixed[k] = v
        state_dict = fixed
    return state_dict


def _maybe_fsdp_state_dict(model):
    if hasattr(model, "module"):
        return fsdp_state_dict(model)
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _clip_module_grad_norm(model, max_norm: float):
    if hasattr(model, "clip_grad_norm_"):
        return model.clip_grad_norm_(max_norm)
    return torch.nn.utils.clip_grad_norm_(
        [param for param in model.parameters() if param.grad is not None],
        max_norm,
    )


def _tensor_to_float(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu().item())
    return float(value)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.world_size = dist.get_world_size()
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb
        self.wandb_host = _resolve_wandb_value(getattr(config, "wandb_host", None), "WANDB_BASE_URL")
        self.wandb_key = _resolve_wandb_value(getattr(config, "wandb_key", None), "WANDB_API_KEY")
        self.wandb_entity = _resolve_wandb_value(getattr(config, "wandb_entity", None), "WANDB_ENTITY")
        self.wandb_project = _resolve_wandb_value(getattr(config, "wandb_project", None), "WANDB_PROJECT", "LightEWM")

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            init_kwargs = dict(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                project=self.wandb_project,
                dir=config.wandb_save_dir
            )
            if self.wandb_entity is not None:
                init_kwargs["entity"] = self.wandb_entity
            if self.wandb_host is not None:
                init_kwargs["settings"] = wandb.Settings(base_url=self.wandb_host)
            if self.wandb_key is not None:
                wandb.login(key=self.wandb_key)
            wandb.init(**init_kwargs)

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        self.model = CausalDiffusion(config, device=self.device)
        self.action_training = bool(getattr(config, "action_training", False))
        self.video_action_joint_training = bool(getattr(config, "video_action_joint_training", False))
        self.action_video_fsdp = bool(getattr(config, "action_video_fsdp", False))
        if self.action_training and self.action_video_fsdp and not self.video_action_joint_training:
            raise NotImplementedError(
                "action_video_fsdp=True is not supported yet. HDRActionMoT reads per-layer video Q/K/V directly, "
                "which bypasses FSDP forward all-gather. Keep action_video_fsdp=false for correct action training."
            )
        if self.video_action_joint_training and not self.action_video_fsdp:
            raise ValueError("video_action_joint_training requires action_video_fsdp=true so the whole MoT is FSDP-owned.")
        self.model.joint_mot = None
        if self.video_action_joint_training:
            action_cfg = getattr(config, "action_dit_config", {}) or {}
            if self.is_main_process and "action_attend_video" in action_cfg:
                print(
                    "video_action_joint_training ignores action_dit_config.action_attend_video="
                    f"{action_cfg['action_attend_video']}; using joint_action_attend_video="
                    f"{getattr(config, 'joint_action_attend_video', 'local_start')} instead.",
                    flush=True,
                )
            self.model.generator = self.model.generator.to(device=self.device, dtype=self.dtype)
            self.model.generator.train().requires_grad_(True)
            if self.action_training and getattr(config, "generator_ckpt", False):
                print(f"Loading pretrained generator from {config.generator_ckpt}")
                self.model.generator.load_state_dict(_load_generator_checkpoint(config.generator_ckpt), strict=True)
            if self.action_training and bool(getattr(config, "action_dit_init_from_video", False)):
                self.model.action_dit.bind_video_model(self.model.generator.model)
                backbone_state, init_summary = self.model.action_dit.build_interpolated_video_backbone_state_dict(
                    apply_alpha_scaling=bool(getattr(config, "action_dit_alpha_scaling", True)),
                )
                load_info = self.model.action_dit.load_state_dict(backbone_state, strict=False)
                if self.is_main_process:
                    print(
                        "Initialized action_dit backbone from loaded video expert with "
                        f"{init_summary}; missing={list(load_info.missing_keys)} "
                        f"unexpected={list(load_info.unexpected_keys)}",
                        flush=True,
                    )
            self.model.joint_mot = HDRVideoActionJointMoT(
                generator=self.model.generator,
                action_dit=self.model.action_dit,
            ).to(device=self.device, dtype=self.dtype)
            self.model.joint_mot.action_dit.debug_action_video_kv = bool(
                getattr(config, "debug_action_video_kv", False)
            )
            self.model.joint_mot.action_dit.disable_action_text_cross_attn = bool(
                getattr(config, "disable_action_text_cross_attn", False)
            )
            self.model.joint_mot = fsdp_wrap(
                self.model.joint_mot,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.generator_fsdp_wrap_strategy,
                cpu_offload=getattr(config, "generator_cpu_offload", False),
                reduce_dtype=getattr(config, "fsdp_reduce_dtype", "float32"),
                buffer_dtype=getattr(config, "fsdp_buffer_dtype", "float32"),
            )
            self.model.generator = None
            self.model.action_dit = None
        elif self.action_training and not self.action_video_fsdp:
            self.model.generator = self.model.generator.to(device=self.device, dtype=self.dtype)
            self.model.generator.eval().requires_grad_(False)
        else:
            if self.action_training and not self.video_action_joint_training:
                self.model.generator.eval().requires_grad_(False)
            self.model.generator = fsdp_wrap(
                self.model.generator,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.generator_fsdp_wrap_strategy,
                cpu_offload=getattr(config, "generator_cpu_offload", False),
                reduce_dtype=getattr(config, "fsdp_reduce_dtype", "float32"),
                buffer_dtype=getattr(config, "fsdp_buffer_dtype", "float32"),
            )
        if self.action_training and (not self.video_action_joint_training) and getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            self.model.generator.load_state_dict(_load_generator_checkpoint(config.generator_ckpt), strict=True)

        if self.action_training and (not self.video_action_joint_training) and bool(getattr(config, "action_dit_init_from_video", False)):
            self.model.action_dit.bind_video_model(self._video_expert_for_action())
            backbone_state, init_summary = self.model.action_dit.build_interpolated_video_backbone_state_dict(
                apply_alpha_scaling=bool(getattr(config, "action_dit_alpha_scaling", True)),
            )
            load_info = self.model.action_dit.load_state_dict(backbone_state, strict=False)
            if self.is_main_process:
                print(
                    "Initialized action_dit backbone from loaded video expert with "
                    f"{init_summary}; missing={list(load_info.missing_keys)} "
                    f"unexpected={list(load_info.unexpected_keys)}",
                    flush=True,
                )

        if self.action_training and not self.video_action_joint_training:
            self.model.action_dit.bind_video_model(self._video_expert_for_action())
            self.model.action_dit = fsdp_wrap(
                self.model.action_dit,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=getattr(config, "action_fsdp_wrap_strategy", "size"),
                cpu_offload=getattr(config, "action_cpu_offload", False),
                reduce_dtype=getattr(config, "fsdp_reduce_dtype", "float32"),
                buffer_dtype=getattr(config, "fsdp_buffer_dtype", "float32"),
            )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False),
            reduce_dtype=getattr(config, "fsdp_reduce_dtype", "float32"),
            buffer_dtype=getattr(config, "fsdp_buffer_dtype", "float32"),
        )

        if not config.no_visualize or config.load_raw_video or self.video_action_joint_training:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        if self.video_action_joint_training:
            action_new_lr = getattr(config, "action_new_lr", None)
            if action_new_lr is not None:
                joint_video_params = []
                joint_action_pretrained_params = []
                joint_action_new_params = []
                joint_other_params = []
                for name, param in self.model.joint_mot.named_parameters():
                    if not param.requires_grad:
                        continue
                    clean_name = name.replace("_fsdp_wrapped_module.", "")
                    if clean_name.startswith("module.generator.") or clean_name.startswith("generator."):
                        joint_video_params.append(param)
                    elif clean_name.startswith("module.action_dit.") or clean_name.startswith("action_dit."):
                        action_name = clean_name
                        if action_name.startswith("module.action_dit."):
                            action_name = action_name.removeprefix("module.action_dit.")
                        elif action_name.startswith("action_dit."):
                            action_name = action_name.removeprefix("action_dit.")
                        if (
                            "action_encoder" in action_name
                            or action_name.startswith("head.")
                            or action_name.startswith("proprio_encoder.")
                        ):
                            joint_action_new_params.append(param)
                        else:
                            joint_action_pretrained_params.append(param)
                    else:
                        joint_other_params.append(param)
                trainable_params = []
                if joint_video_params:
                    trainable_params.append({"params": joint_video_params, "lr": config.lr})
                if joint_action_pretrained_params:
                    trainable_params.append({
                        "params": joint_action_pretrained_params,
                        "lr": float(getattr(config, "action_backbone_lr", config.lr)),
                    })
                if joint_action_new_params:
                    trainable_params.append({"params": joint_action_new_params, "lr": float(action_new_lr)})
                if joint_other_params:
                    trainable_params.append({"params": joint_other_params, "lr": config.lr})
            else:
                trainable_params = [param for param in self.model.joint_mot.parameters() if param.requires_grad]
        else:
            trainable_params = [param for param in self.model.generator.parameters() if param.requires_grad]
        if self.action_training and not self.video_action_joint_training:
            action_new_lr = getattr(config, "action_new_lr", None)
            if action_new_lr is not None:
                action_new_params = []
                action_pretrained_params = []
                for name, param in self.model.action_dit.named_parameters():
                    if not param.requires_grad:
                        continue
                    clean_name = name.replace("_fsdp_wrapped_module.", "")
                    if "action_encoder" in clean_name or clean_name.startswith("head."):
                        action_new_params.append(param)
                    else:
                        action_pretrained_params.append(param)
                trainable_params.extend([
                    {"params": action_pretrained_params, "lr": config.lr},
                    {"params": action_new_params, "lr": float(action_new_lr)},
                ])
            else:
                trainable_params.extend([param for param in self.model.action_dit.parameters() if param.requires_grad])
        if not trainable_params:
            raise RuntimeError("No trainable parameters found for diffusion training.")
        self.generator_optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
        self.lr_schedule = getattr(config, "lr_schedule", "constant")
        self.lr_warmup_steps = int(getattr(config, "lr_warmup_steps", 0) or 0)
        self.lr_total_steps = int(getattr(config, "lr_total_steps", 0) or 0)
        self.lr_min_ratio = float(getattr(config, "lr_min_ratio", 0.0))
        self._base_lrs = [float(group["lr"]) for group in self.generator_optimizer.param_groups]

        # Step 3: Initialize the dataloader
        data_backend = getattr(config, "data_backend", "lmdb_latent")
        if data_backend == "jsonl_video":
            variable_num_frames_train = getattr(config, "variable_num_frames_train", True)
            if variable_num_frames_train and config.batch_size != 1:
                raise ValueError(
                    "Variable-length jsonl_video training currently requires batch_size=1. "
                    "The VAE encode path stacks equal-length videos only."
                )
            dataset = TextVideoDataset(
                metadata_path=config.data_path,
                height=config.height,
                width=config.width,
                num_frames=config.num_frames,
                variable_num_frames=variable_num_frames_train,
                max_num_frames=getattr(config, "max_training_video_frames", 253),
                video_action_joint=bool(getattr(config, "video_action_joint_training", False)),
                joint_window_frames=int(getattr(config, "joint_window_frames", 13)),
                joint_source_fps=float(getattr(config, "joint_source_fps", 16.0)),
                joint_target_fps=float(getattr(config, "joint_target_fps", 10.0)),
                joint_video_frame_stride=int(getattr(config, "joint_video_frame_stride", 1)),
                joint_camera_key=getattr(config, "joint_camera_key", "agentview_rgb"),
                joint_include_terminal_video_frame=bool(getattr(config, "joint_include_terminal_video_frame", False)),
                joint_norm_clip=float(getattr(config, "joint_norm_clip", 1.0)),
                joint_drop_tree_tokens=bool(getattr(config, "joint_drop_tree_tokens", False)),
                joint_proprio_stats_path=getattr(config, "joint_proprio_stats_path", None),
            )
        else:
            dataset = LatentLMDBDataset(config.data_path, max_pair=int(1e8))

        self.dataset = dataset
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=getattr(config, "num_workers", 8))

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)
        total_batch_size = getattr(config, "total_batch_size", None)
        if total_batch_size is None:
            self.gradient_accumulation_steps = 1
        else:
            micro_batch_size = int(config.batch_size) * self.world_size
            if int(total_batch_size) % micro_batch_size != 0:
                raise ValueError(
                    f"total_batch_size={total_batch_size} must be divisible by "
                    f"batch_size * world_size = {config.batch_size} * {self.world_size}."
                )
            self.gradient_accumulation_steps = max(1, int(total_batch_size) // micro_batch_size)
        self.micro_step = 0
        if self.is_main_process:
            print(
                "Gradient accumulation steps: "
                f"{self.gradient_accumulation_steps} "
                f"(batch_size={config.batch_size}, world_size={self.world_size}, "
                f"total_batch_size={total_batch_size})",
                flush=True,
            )

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        if self.video_action_joint_training:
            for n, p in self.model.joint_mot.named_parameters():
                if not p.requires_grad:
                    continue
                renamed_n = rename_param(n)
                self.name_to_trainable_params[renamed_n] = p
        elif not self.action_training:
            for n, p in self.model.generator.named_parameters():
                if not p.requires_grad:
                    continue

                renamed_n = rename_param(n)
                self.name_to_trainable_params[renamed_n] = p
        if self.action_training and not self.video_action_joint_training:
            for n, p in self.model.action_dit.named_parameters():
                if not p.requires_grad:
                    continue
                renamed_n = "action_dit." + rename_param(n)
                self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (not self.action_training) and (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False) and not self.action_training:
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            self.model.generator.load_state_dict(_load_generator_checkpoint(config.generator_ckpt), strict=True)

        if self.action_training and getattr(config, "action_dit_ckpt", None):
            print(f"Loading pretrained action_dit from {config.action_dit_ckpt}")
            action_state = torch.load(config.action_dit_ckpt, map_location="cpu")
            if "action_dit" in action_state:
                action_state = action_state["action_dit"]
            elif "backbone_state_dict" in action_state:
                action_state = action_state["backbone_state_dict"]
            action_target = (
                self.model.joint_mot.module.action_dit
                if self.video_action_joint_training
                else self.model.action_dit
            )
            load_info = action_target.load_state_dict(action_state, strict=False)
            if self.is_main_process:
                print(
                    "Loaded action_dit with "
                    f"missing={list(load_info.missing_keys)} "
                    f"unexpected={list(load_info.unexpected_keys)}",
                    flush=True,
                )

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm = float(getattr(config, "max_grad_norm", 10.0))
        self.previous_time = None
        self.delta_mean = None
        self.rtf_ema_ratio = getattr(self.config, "rtf_ema_ratio", 0.9)
        self.eval_interval = getattr(self.config, "eval_interval", 0)      # 0 => disable
        self.eval_frames = getattr(self.config, "eval_num_output_frames", 21)
        self.eval_init = getattr(self.config, "eval_num_init_frames", 3)
        self.rtf_single_gpu_batch = getattr(self.config, "rtf_single_gpu_batch", 1)
        self.given_first_chunk = getattr(self.config, "given_first_chunk", True)
        self.train_timer = getattr(self.config, "train_timer", False)
        self.train_timer_interval = int(getattr(self.config, "train_timer_interval", 10))
        self.train_timer_first_steps = int(getattr(self.config, "train_timer_first_steps", 5))
        self.train_timer_sync_cuda = getattr(self.config, "train_timer_sync_cuda", True)
        if self.eval_interval and self.video_action_joint_training:
            raise NotImplementedError("video_action_joint_training does not support the legacy eval pipeline yet.")
        if self.eval_interval:
            self.pipeline = CausalDiffusionInferencePipeline(config, device=self.device)
            self.pipeline.generator = self.model.generator
            self.pipeline.text_encoder = self.model.text_encoder

    def _video_expert_for_action(self):
        if getattr(self.model, "joint_mot", None) is not None:
            wrapper = getattr(self.model.joint_mot, "module", self.model.joint_mot)
            return wrapper.generator.model
        generator = self.model.generator
        wrapper = getattr(generator, "module", generator)
        return wrapper.model

    def _should_profile_step(self) -> bool:
        if not self.train_timer:
            return False
        next_step = self.step + 1
        if next_step <= self.train_timer_first_steps:
            return True
        return self.train_timer_interval > 0 and next_step % self.train_timer_interval == 0

    def _timer_now(self):
        if self.train_timer_sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def _log_train_timer(self, timer: dict):
        if not self.is_main_process or not timer:
            return
        ordered_keys = [
            "data_wait",
            "empty_cache",
            "host_to_device",
            "vae_encode",
            "text_cond",
            "text_uncond",
            "generator_loss",
            "model_prepare",
            "model_generator",
            "model_loss",
            "zero_grad",
            "backward",
            "clip_grad",
            "optimizer_step",
            "wandb_log",
            "gc",
            "save",
            "barrier",
            "loop_total",
        ]
        items = []
        for key in ordered_keys:
            if key in timer:
                items.append(f"{key}={timer[key]:.3f}s")
        for key in sorted(timer):
            if key not in ordered_keys:
                items.append(f"{key}={timer[key]:.3f}s")
        print(f"[TrainTimer] step={self.step} " + " ".join(items), flush=True)

    def _action_grad_norms(self) -> dict[str, float]:
        if not self.action_training:
            return {}
        action_module = self.model.action_dit
        if self.video_action_joint_training:
            action_module = self.model.joint_mot.module.action_dit
        buckets = {
            "action_encoder": 0.0,
            "head": 0.0,
            "blocks": 0.0,
            "time": 0.0,
            "text": 0.0,
            "other": 0.0,
        }
        for name, param in action_module.named_parameters():
            if param.grad is None:
                continue
            grad_norm = float(param.grad.detach().float().norm().item())
            squared = grad_norm * grad_norm
            clean_name = name.replace("_fsdp_wrapped_module.", "")
            if "action_encoder" in clean_name:
                key = "action_encoder"
            elif "head" in clean_name:
                key = "head"
            elif "blocks" in clean_name:
                key = "blocks"
            elif "time_embedding" in clean_name or "time_projection" in clean_name:
                key = "time"
            elif "text_embedding" in clean_name:
                key = "text"
            else:
                key = "other"
            buckets[key] += squared
        return {f"grad_norm/{key}": math.sqrt(value) for key, value in buckets.items()}

    def _joint_grad_norms(self) -> dict[str, float]:
        if not self.video_action_joint_training:
            return {}
        wrapper = getattr(self.model.joint_mot, "module", self.model.joint_mot)
        buckets = {
            "video_dit": 0.0,
            "action_dit": 0.0,
            "action_encoder": 0.0,
            "action_head": 0.0,
            "action_blocks": 0.0,
        }
        for name, param in wrapper.named_parameters():
            if param.grad is None:
                continue
            grad_norm = float(param.grad.detach().float().norm().item())
            squared = grad_norm * grad_norm
            clean_name = name.replace("_fsdp_wrapped_module.", "")
            if clean_name.startswith("generator."):
                buckets["video_dit"] += squared
            elif clean_name.startswith("action_dit."):
                buckets["action_dit"] += squared
                if ".action_encoder" in clean_name:
                    buckets["action_encoder"] += squared
                elif ".head" in clean_name:
                    buckets["action_head"] += squared
                elif ".blocks" in clean_name:
                    buckets["action_blocks"] += squared
        if dist.is_initialized():
            reduced = torch.tensor(
                [buckets[key] for key in buckets],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            for key, value in zip(buckets.keys(), reduced.tolist()):
                buckets[key] = value
        return {key: math.sqrt(value) for key, value in buckets.items()}

    def _lr_scale_for_step(self, step: int) -> float:
        if self.lr_schedule in (None, "constant"):
            return 1.0
        if self.lr_schedule != "linear_warmup_cosine":
            raise ValueError(f"Unsupported lr_schedule: {self.lr_schedule}")
        if step <= 0:
            return 0.0 if self.lr_warmup_steps > 0 else 1.0
        if self.lr_warmup_steps > 0 and step <= self.lr_warmup_steps:
            return float(step) / float(self.lr_warmup_steps)
        if self.lr_total_steps <= self.lr_warmup_steps:
            return 1.0
        progress = (step - self.lr_warmup_steps) / float(self.lr_total_steps - self.lr_warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.lr_min_ratio + (1.0 - self.lr_min_ratio) * cosine

    def _set_optimizer_lr(self, step: int) -> float:
        scale = self._lr_scale_for_step(step)
        for group, base_lr in zip(self.generator_optimizer.param_groups, self._base_lrs):
            group["lr"] = base_lr * scale
        return scale

    def save(self):
        print("Start gathering distributed model states...")
        if self.video_action_joint_training:
            joint_state_dict = fsdp_state_dict(self.model.joint_mot)
            generator_state_dict = {
                key.removeprefix("generator."): value
                for key, value in joint_state_dict.items()
                if key.startswith("generator.")
            }
            action_state_dict = {
                key.removeprefix("action_dit."): value
                for key, value in joint_state_dict.items()
                if key.startswith("action_dit.")
            }
        else:
            generator_state_dict = (
                None
                if (self.action_training and not self.video_action_joint_training)
                else _maybe_fsdp_state_dict(self.model.generator)
            )
            action_state_dict = fsdp_state_dict(self.model.action_dit) if self.action_training else None

        if self.action_training:
            state_dict = {
                "checkpoint_type": "hdr_video_action_joint" if self.video_action_joint_training else "hdr_action_mot",
                "video_generator_ckpt": getattr(self.config, "generator_ckpt", None),
                "action_dit_config": _to_plain_config(getattr(self.config, "action_dit_config", {})),
                "actions_per_leaf": getattr(self.config, "actions_per_leaf", None),
                "action_stats_path": getattr(self.config, "action_stats_path", None),
                "vertical_leaf_frames": getattr(self.config, "vertical_leaf_frames", None),
                "vertical_level_sizes": _to_plain_config(getattr(self.config, "vertical_level_sizes", [])),
                "vertical_step_budgets": _to_plain_config(getattr(self.config, "vertical_step_budgets", [])),
            }
        elif self.config.ema_start_step < self.step:
            state_dict = {
                "generator": generator_state_dict,
                "generator_ema": self.generator_ema.full_state_dict(self.model.generator),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
            }
        if action_state_dict is not None:
            state_dict["action_dit"] = action_state_dict
        if generator_state_dict is not None and "generator" not in state_dict:
            state_dict["generator"] = generator_state_dict

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def train_one_step(self, batch, data_wait_time: float | None = None, profile_step: bool = False):
        self.log_iters = 1
        timer = {"data_wait": data_wait_time} if profile_step and data_wait_time is not None else {}
        last_time = self._timer_now() if profile_step else None

        def record_time(name: str):
            nonlocal last_time
            if not profile_step:
                return
            now = self._timer_now()
            timer[name] = now - last_time
            last_time = now

        if self.step % 20 == 0:
            torch.cuda.empty_cache()
        record_time("empty_cache")

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        cached_action_video_latents = batch.get("video_latents")
        cached_action_video_leaf_latents = batch.get("video_leaf_latents")
        cached_action_video_leaf_k = batch.get("video_leaf_k")
        cached_action_video_leaf_v = batch.get("video_leaf_v")
        cached_first_frame_latent = batch.get("first_frame_latent")
        cached_video_timestep = batch.get("video_timestep")
        joint_local_start_latent = None
        joint_local_video_latents = None
        if self.video_action_joint_training:
            joint_local_frames = batch["joint_local_frames"].to(device=self.device, dtype=self.dtype)
            record_time("host_to_device")
            with torch.no_grad():
                joint_local_start_latent = self.model.vae.encode_to_latent(
                    joint_local_frames[:, :, :1]
                ).to(device=self.device, dtype=self.dtype)
                joint_local_video_latents = self.model.vae.encode_to_latent(
                    joint_local_frames
                ).to(device=self.device, dtype=self.dtype)[:, 1:]
                if bool(getattr(self.config, "joint_drop_tree_tokens", False)):
                    clean_latent = joint_local_start_latent
                else:
                    frames = batch["frames"].to(device=self.device, dtype=self.dtype)
                    clean_latent = self.model.vae.encode_to_latent(frames).to(device=self.device, dtype=self.dtype)
            record_time("vae_encode")
            image_latent = clean_latent[:, 0:1]
            image_or_video_shape = list(clean_latent.shape)
            action_video_latents = None
            action_video_leaf_k = None
            action_video_leaf_v = None
        elif self.action_training and (
            cached_action_video_latents is not None or cached_action_video_leaf_latents is not None
        ):
            action_video_latents = (
                cached_action_video_latents
                if cached_action_video_latents is not None
                else cached_action_video_leaf_latents
            ).to(device=self.device, dtype=self.dtype)
            action_video_leaf_k = (
                cached_action_video_leaf_k.to(device=self.device, dtype=self.dtype)
                if cached_action_video_leaf_k is not None
                else None
            )
            action_video_leaf_v = (
                cached_action_video_leaf_v.to(device=self.device, dtype=self.dtype)
                if cached_action_video_leaf_v is not None
                else None
            )
            if cached_first_frame_latent is not None:
                image_latent = cached_first_frame_latent.to(device=self.device, dtype=self.dtype)
                clean_latent = image_latent
                record_time("host_to_device")
                record_time("vae_encode")
            else:
                frames = batch["frames"][:, :, :1].to(device=self.device, dtype=self.dtype)
                record_time("host_to_device")
                with torch.no_grad():
                    image_latent = self.model.vae.encode_to_latent(frames).to(device=self.device, dtype=self.dtype)
                clean_latent = image_latent
                record_time("vae_encode")
            image_or_video_shape = [
                action_video_latents.shape[0],
                getattr(self.config, "vertical_leaf_frames", action_video_latents.shape[1]),
                *list(action_video_latents.shape[2:]),
            ]
        elif not self.config.load_raw_video:  # precomputed latent
            clean_latent = batch["clean_latent"].to(
                device=self.device, dtype=self.dtype)
            record_time("host_to_device")
            record_time("vae_encode")
            image_latent = clean_latent[:, 0:1, ]
            image_or_video_shape = list(clean_latent.shape)
            action_video_latents = None
            action_video_leaf_k = None
            action_video_leaf_v = None
        else:  # encode raw video to latent
            frames = batch["frames"].to(
                device=self.device, dtype=self.dtype)
            record_time("host_to_device")

            with torch.no_grad():
                clean_latent = self.model.vae.encode_to_latent(
                    frames).to(device=self.device, dtype=self.dtype)
            record_time("vae_encode")
            image_latent = clean_latent[:, 0:1, ]
            image_or_video_shape = list(clean_latent.shape)
            action_video_latents = None
            action_video_leaf_k = None
            action_video_leaf_v = None

        batch_size = len(text_prompts)
        if cached_video_timestep is not None:
            cached_video_timestep = cached_video_timestep.to(device=self.device, dtype=self.dtype)

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)
            record_time("text_cond")
            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
                record_time("text_uncond")
            else:
                unconditional_dict = self.unconditional_dict
                if profile_step:
                    timer["text_uncond"] = 0.0
        if (
            self.is_main_process
            and bool(getattr(self.config, "debug_prompt_embed_stats", False))
            and self.step < 5
        ):
            prompt_embeds_debug = conditional_dict["prompt_embeds"].detach().float()
            print(
                "[PromptEmbedStats] "
                f"mean={float(prompt_embeds_debug.mean().cpu()):.6f} "
                f"std={float(prompt_embeds_debug.std().cpu()):.6f} "
                f"absmax={float(prompt_embeds_debug.abs().max().cpu()):.6f} "
                f"shape={tuple(prompt_embeds_debug.shape)}",
                flush=True,
            )

        # Step 3: Train the generator
        self.model._profile_train_timer = profile_step
        generator_loss, log_dict = self.model.generator_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent,
            actions=batch.get("actions"),
            action_is_pad=batch.get("action_is_pad"),
            action_video_latents=action_video_latents,
            action_video_timestep=cached_video_timestep,
            action_video_leaf_k=action_video_leaf_k,
            action_video_leaf_v=action_video_leaf_v,
            joint_actions=batch.get("joint_actions"),
            joint_proprio=batch.get("joint_proprio"),
            joint_local_start_latent=joint_local_start_latent,
            joint_local_video_latents=joint_local_video_latents,
        )
        self.model._profile_train_timer = False
        if profile_step:
            for key, value in log_dict.items():
                if key.startswith("timer/"):
                    timer[key.removeprefix("timer/")] = float(value)
        record_time("generator_loss")
        if self.micro_step == 0:
            self.generator_optimizer.zero_grad()
        record_time("zero_grad")
        (generator_loss / self.gradient_accumulation_steps).backward()
        record_time("backward")
        self.micro_step += 1
        should_step = self.micro_step >= self.gradient_accumulation_steps
        if should_step:
            lr_scale = self._set_optimizer_lr(self.step + 1)
            joint_grad_norms = self._joint_grad_norms() if self.video_action_joint_training else {}
            action_grad_norms = (
                {}
                if self.video_action_joint_training
                else (self._action_grad_norms() if self.action_training else {})
            )
            if self.video_action_joint_training:
                generator_grad_norm = _clip_module_grad_norm(self.model.joint_mot, self.max_grad_norm)
            elif self.action_training:
                generator_grad_norm = _clip_module_grad_norm(self.model.action_dit, self.max_grad_norm)
            else:
                generator_grad_norm = _clip_module_grad_norm(self.model.generator, self.max_grad_norm)
            record_time("clip_grad")
            self.generator_optimizer.step()
            self.micro_step = 0
            record_time("optimizer_step")

            # Increment the step since we finished gradient update
            self.step += 1
        else:
            lr_scale = self._lr_scale_for_step(self.step + 1)
            joint_grad_norms = {}
            action_grad_norms = {}
            generator_grad_norm = torch.tensor(0.0, device=self.device)
            record_time("clip_grad")
            record_time("optimizer_step")
        self._last_optimizer_step = should_step

        loss_video_tree = log_dict.get("loss_video_tree")
        loss_video_local = log_dict.get("loss_video_local")
        loss_action = log_dict.get("loss_action")
        wandb_loss_dict = {
            "generator_loss": generator_loss.item(),
            "generator_grad_norm": _tensor_to_float(generator_grad_norm),
            "lr": self.generator_optimizer.param_groups[0]["lr"],
            "lr_scale": lr_scale,
            "train/loss_total": generator_loss.item(),
            "train/lr": self.generator_optimizer.param_groups[0]["lr"],
            "train/lr_scale": lr_scale,
            "train/grad_norm_total": _tensor_to_float(generator_grad_norm),
        }
        if loss_video_tree is not None:
            wandb_loss_dict["train/loss_video_leaf_tree"] = _tensor_to_float(loss_video_tree)
        if loss_video_local is not None:
            wandb_loss_dict["train/loss_video_local"] = _tensor_to_float(loss_video_local)
        if loss_action is not None:
            wandb_loss_dict["train/loss_action"] = _tensor_to_float(loss_action)
        for key, value in joint_grad_norms.items():
            wandb_loss_dict[f"train/grad_norm_{key}"] = value
        for key, value in action_grad_norms.items():
            wandb_loss_dict[key] = value
        for metric_key in ("loss_action", "loss_video_tree", "loss_video_local"):
            if metric_key in log_dict:
                value = log_dict[metric_key]
                wandb_loss_dict[metric_key] = float(value.detach().float().cpu())

        # Step 4: Logging
        if self.is_main_process and should_step:
            if self.action_training and (self.step <= 5 or self.step % 20 == 0):
                debug_items = {
                    "generator_loss": wandb_loss_dict["generator_loss"],
                    "generator_grad_norm": wandb_loss_dict["generator_grad_norm"],
                    "lr": wandb_loss_dict["lr"],
                }
                for grad_key in (
                    "train/grad_norm_video_dit",
                    "train/grad_norm_action_dit",
                    "train/grad_norm_action_encoder",
                    "train/grad_norm_action_head",
                    "train/grad_norm_action_blocks",
                ):
                    if grad_key in wandb_loss_dict:
                        debug_items[grad_key.removeprefix("train/")] = wandb_loss_dict[grad_key]
                if "action_video_condition_source" in log_dict:
                    debug_items["action_video_condition_source"] = log_dict["action_video_condition_source"]
                print(f"[ActionDebug] step={self.step} {debug_items}", flush=True)
            if not self.disable_wandb:
                wandb.log(wandb_loss_dict, step=self.step)
        record_time("wandb_log")

        if self.step % self.config.gc_interval == 0:
            if dist.get_rank() == 0:
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()
        record_time("gc")
        return timer if profile_step else None


    def train(self):

        while True:
            profile_step = self._should_profile_step()
            loop_start = self._timer_now() if profile_step else None
            data_start = self._timer_now() if profile_step else None
            batch = next(self.dataloader)
            data_wait_time = self._timer_now() - data_start if profile_step else None
            timer = self.train_one_step(
                batch,
                data_wait_time=data_wait_time,
                profile_step=profile_step,
            )

            save_start = self._timer_now() if profile_step else None
            did_optimizer_step = bool(getattr(self, "_last_optimizer_step", True))
            if did_optimizer_step and (not self.config.no_save) and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()
            if profile_step:
                timer["save"] = self._timer_now() - save_start

            barrier_start = self._timer_now() if profile_step else None
            barrier()
            if profile_step:
                timer["barrier"] = self._timer_now() - barrier_start
                timer["loop_total"] = self._timer_now() - loop_start
                self._log_train_timer(timer)
            if self.is_main_process and did_optimizer_step:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per_iteration_time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time
            max_train_steps = int(getattr(self.config, "max_train_steps", 0) or 0)
            if max_train_steps > 0 and self.step >= max_train_steps:
                if self.is_main_process:
                    print(f"Reached max_train_steps={max_train_steps}; stopping training.", flush=True)
                break
