from typing import Tuple
import json
import time
from pathlib import Path
import torch

from model.base import BaseModel
from model.action_mot import HDRActionMoT
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.vertical_hierarchy import (
    CONDITION_TOKEN_ID,
    build_vertical_hierarchy,
    get_dynamic_vertical_level_avg_step_budgets,
    gather_vertical_latents,
    get_vertical_token_step_budgets,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.scheduler import WanContinuousFlowMatchScheduler


def build_fixed_vertical_retained_timesteps(
    *,
    token_step_budgets: list[int],
    num_train_timesteps: int,
    timestep_shift: float,
    fixed_denoise_steps: int,
    preserve_budget_ratio: bool,
    reference_total_steps: int,
    default_sampling_steps: int,
    device,
    dtype,
) -> torch.Tensor:
    retained = []
    for original_budget in token_step_budgets:
        if original_budget < 0:
            raise ValueError(f"vertical step budget must be non-negative, got {original_budget}.")
        if fixed_denoise_steps <= 0:
            current_budget = int(original_budget)
            current_total_steps = int(default_sampling_steps)
        elif original_budget == 0:
            current_budget = 0
            current_total_steps = int(default_sampling_steps)
        else:
            current_budget = int(fixed_denoise_steps)
            if preserve_budget_ratio:
                current_total_steps = int(
                    round(current_budget * int(reference_total_steps) / float(original_budget))
                )
                current_total_steps = max(current_budget, current_total_steps)
            else:
                current_total_steps = int(default_sampling_steps)
                current_budget = min(current_budget, current_total_steps)

        if current_budget <= 0:
            retained.append(0.0)
            continue
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=int(num_train_timesteps),
            shift=1,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(
            current_total_steps,
            device=device,
            shift=float(timestep_shift),
        )
        if current_budget < len(scheduler.timesteps):
            retained.append(float(scheduler.timesteps[current_budget].detach().float().cpu().item()))
        else:
            retained.append(0.0)
    return torch.tensor(retained, device=device, dtype=dtype)


class CausalDiffusion(BaseModel):
    def __init__(self, args, device):
        """
        Initialize the Diffusion loss module.
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True

        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
        self.generator_patch_size_hw = tuple(self.generator.model.patch_size[-2:])

        # Step 2: Initialize all hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.guidance_scale = args.guidance_scale
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.teacher_forcing = getattr(args, "teacher_forcing", False)
        self.condition_first_frame = getattr(args, "condition_first_frame", False)
        self.vertical_hierarchy = getattr(args, "vertical_hierarchy", False)
        self.start_from_st_end = getattr(args, "start_from_st_end", False)
        self.st_end_plus = getattr(args, "st_end_plus", False)
        if self.start_from_st_end and self.st_end_plus:
            raise ValueError("start_from_st_end and st_end_plus are mutually exclusive; enable at most one.")
        self.vertical_allow_condition_for_all_frames = getattr(
            args, "vertical_allow_condition_for_all_frames", False
        )
        self.dynamic_vertical_hierarchy = getattr(args, "dynamic_vertical_hierarchy", False)
        self.vertical_use_representative_rope = getattr(args, "vertical_use_representative_rope", False)
        self.vertical_leaf_frames = getattr(args, "vertical_leaf_frames", args.image_or_video_shape[1])
        self.vertical_level_sizes = list(getattr(args, "vertical_level_sizes", [1, 2, 4, 8, 16, 21]))
        self.vertical_step_budgets = list(getattr(args, "vertical_step_budgets", [8, 16, 24, 24, 24, 50]))
        self.dynamic_vertical_max_step_budget = getattr(args, "dynamic_vertical_max_step_budget", 50)
        self.sampling_steps = int(getattr(args, "sampling_steps", self.vertical_step_budgets[-1]))
        self.vertical_infer_fixed_denoise_steps = int(getattr(args, "vertical_infer_fixed_denoise_steps", 0))
        self.vertical_infer_preserve_budget_ratio = bool(
            getattr(args, "vertical_infer_preserve_budget_ratio", False)
        )
        self.vertical_infer_reference_total_steps = int(
            getattr(args, "vertical_infer_reference_total_steps", self.sampling_steps)
        )
        self.vertical_num_levels = len(self.vertical_level_sizes)
        if self.vertical_hierarchy:
            if self.dynamic_vertical_hierarchy:
                self.vertical_info = None
                self.vertical_token_step_budgets = None
            else:
                self.vertical_info = build_vertical_hierarchy(
                    num_leaf_frames=self.vertical_leaf_frames,
                    num_levels=self.vertical_num_levels,
                    level_sizes=self.vertical_level_sizes,
                    start_from_st_end=self.start_from_st_end,
                    st_end_plus=self.st_end_plus,
                    allow_condition_for_all_frames=self.vertical_allow_condition_for_all_frames,
                )
                if self.st_end_plus:
                    if len(self.vertical_info["level_sizes"]) != len(self.vertical_level_sizes):
                        raise ValueError(
                            f"st_end_plus hierarchy levels={self.vertical_info['level_sizes']} "
                            f"does not match configured level count={len(self.vertical_level_sizes)}."
                        )
                elif self.vertical_info["level_sizes"] != self.vertical_level_sizes:
                    raise ValueError(
                        f"vertical_level_sizes={self.vertical_level_sizes} does not match constructed hierarchy "
                        f"{self.vertical_info['level_sizes']}."
                    )
                self.vertical_token_step_budgets = get_vertical_token_step_budgets(
                    self.vertical_info,
                    self.vertical_step_budgets,
                )
            self.vertical_reference_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timestep,
                shift=1,
                use_dynamic_shifting=False,
            )
            self.vertical_reference_scheduler.set_timesteps(
                self.dynamic_vertical_max_step_budget if self.dynamic_vertical_hierarchy else self.vertical_step_budgets[-1],
                device=self.device,
                shift=self.timestep_shift,
            )
            self.vertical_reference_timesteps = self.vertical_reference_scheduler.timesteps.to(
                device=self.device,
                dtype=self.dtype,
            )
        else:
            self.vertical_info = None
            self.vertical_token_step_budgets = None
            self.vertical_reference_scheduler = None
            self.vertical_reference_timesteps = None
        
        # Noise augmentation in teacher forcing, we add small noise to clean context latents
        self.noise_augmentation_max_timestep = getattr(args, "noise_augmentation_max_timestep", 0)
        self.action_training = bool(getattr(args, "action_training", False))
        self.video_action_joint_training = bool(getattr(args, "video_action_joint_training", False))
        self.action_loss_weight = float(getattr(args, "action_loss_weight", 1.0))
        self.joint_tree_video_loss_weight = float(getattr(args, "joint_tree_video_loss_weight", 1.0))
        self.joint_local_video_loss_weight = float(getattr(args, "joint_local_video_loss_weight", 1.0))
        self.joint_action_loss_weight = float(getattr(args, "joint_action_loss_weight", self.action_loss_weight))
        self.joint_local_video_tokens = int(getattr(args, "joint_local_video_tokens", 5))
        self.joint_detach_action_video_kv = bool(getattr(args, "joint_detach_action_video_kv", False))
        self.joint_action_attend_video = str(getattr(args, "joint_action_attend_video", "local_start"))
        self.joint_action_video_kv_scale = float(getattr(args, "joint_action_video_kv_scale", 1.0))
        self.joint_action_fixed_timestep = getattr(args, "joint_action_fixed_timestep", None)
        self.joint_drop_tree_tokens = bool(getattr(args, "joint_drop_tree_tokens", False))
        self.joint_tree_num_levels = int(getattr(args, "joint_tree_num_levels", 0) or 0)
        self.actions_per_leaf = int(getattr(args, "actions_per_leaf", 8))
        self.action_train_shift = float(getattr(args, "action_train_shift", 5.0))
        self.action_infer_shift = float(getattr(args, "action_infer_shift", self.action_train_shift))
        self.action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=self.num_train_timestep,
            shift=self.action_train_shift,
        )
        self.local_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=self.num_train_timestep,
            shift=self.action_train_shift,
        )
        self._init_action_metric_stats(getattr(args, "action_stats_path", None))
        if self.action_training and not self.vertical_hierarchy:
            raise ValueError("action_training currently requires vertical_hierarchy=True.")
        if self.action_training and not self.video_action_joint_training:
            self.generator.model.requires_grad_(False)
        if self.video_action_joint_training:
            self.generator.model.requires_grad_(True)
            if not bool(getattr(args, "action_training", False)):
                raise ValueError("video_action_joint_training=True requires action_training=True.")
            if self.actions_per_leaf != int(getattr(args, "joint_window_frames", 13)):
                self.actions_per_leaf = int(getattr(args, "joint_window_frames", 13))
        if self.action_training:
            action_cfg = getattr(args, "action_dit_config", {}) or {}
            self.action_dit = HDRActionMoT(
                video_model=self.generator.model,
                action_dim=int(action_cfg.get("action_dim", getattr(args, "action_dim", 7))),
                hidden_dim=int(action_cfg.get("hidden_dim", 1024)),
                ffn_dim=int(action_cfg.get("ffn_dim", 4096)),
                freq_dim=int(action_cfg.get("freq_dim", 256)),
                eps=float(action_cfg.get("eps", 1e-6)),
                actions_per_leaf=int(action_cfg.get("actions_per_leaf", self.actions_per_leaf)),
                action_attend_video=str(action_cfg.get("action_attend_video", "parents")),
                use_gradient_checkpointing=bool(action_cfg.get("use_gradient_checkpointing", args.gradient_checkpointing)),
                proprio_dim=action_cfg.get("proprio_dim", None),
            )

    def _init_action_metric_stats(self, stats_path: str | None) -> None:
        self.action_metric_eps = 1e-6
        self.action_metric_has_stats = False
        self.register_buffer("action_metric_min", torch.empty(0, dtype=torch.float32), persistent=False)
        self.register_buffer("action_metric_max", torch.empty(0, dtype=torch.float32), persistent=False)
        if not stats_path:
            return
        path = Path(str(stats_path))
        candidates = [path, Path.cwd() / path]
        cwd_parents = list(Path.cwd().parents)
        if len(cwd_parents) >= 3:
            candidates.append(cwd_parents[2] / path)
        resolved = next((candidate for candidate in candidates if candidate.exists()), None)
        if resolved is None:
            return
        with resolved.open("r", encoding="utf-8") as f:
            stats = json.load(f)
        min_v = torch.tensor(stats["min"], dtype=torch.float32, device=self.device)
        max_v = torch.tensor(stats["max"], dtype=torch.float32, device=self.device)
        self.action_metric_min = min_v
        self.action_metric_max = max_v
        self.action_metric_eps = float(stats.get("eps", 1e-6))
        self.action_metric_has_stats = True

    def _denormalize_action_for_metrics(self, actions: torch.Tensor) -> torch.Tensor:
        if not self.action_metric_has_stats:
            return actions.float()
        min_v = self.action_metric_min.to(device=actions.device, dtype=torch.float32)
        max_v = self.action_metric_max.to(device=actions.device, dtype=torch.float32)
        return (actions.float() + 1.0) * 0.5 * (max_v - min_v + self.action_metric_eps) + min_v

    def _build_action_cache_video_timestep(self, vertical_latents: torch.Tensor) -> torch.Tensor:
        if self.vertical_token_step_budgets is None:
            raise ValueError("Cannot build vertical retained timesteps without vertical_token_step_budgets.")
        timestep = build_fixed_vertical_retained_timesteps(
            token_step_budgets=list(self.vertical_token_step_budgets),
            num_train_timesteps=int(self.num_train_timestep),
            timestep_shift=float(self.timestep_shift),
            fixed_denoise_steps=int(self.vertical_infer_fixed_denoise_steps),
            preserve_budget_ratio=bool(self.vertical_infer_preserve_budget_ratio),
            reference_total_steps=int(self.vertical_infer_reference_total_steps),
            default_sampling_steps=int(self.sampling_steps),
            device=self.device,
            dtype=self.dtype,
        )
        return timestep.unsqueeze(0).expand(vertical_latents.shape[0], -1)

    def _initialize_models(self, args, device):
        model_kwargs = getattr(args, "model_kwargs", {})
        model_name = getattr(model_kwargs, "model_name", "Wan2.1-T2V-1.3B")
        model_root = getattr(model_kwargs, "model_root", "wan_models")
        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        self.generator.model.requires_grad_(
            (not bool(getattr(args, "action_training", False)))
            or bool(getattr(args, "video_action_joint_training", False))
        )
        self.action_dit = None

        self.text_encoder = WanTextEncoder(model_name=model_name, model_root=model_root)
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper(model_name=model_name, model_root=model_root)
        self.vae.requires_grad_(False)
        
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _sample_vertical_timesteps(self, batch_size: int, token_step_budgets: list[int]) -> torch.Tensor:
        max_reference_timestep = self.vertical_reference_timesteps[0]
        sampled = torch.empty(
            [batch_size, len(token_step_budgets)],
            device=self.device,
            dtype=self.dtype,
        )
        for token_index, budget in enumerate(token_step_budgets):
            min_reference_timestep = self.vertical_reference_timesteps[budget - 1]
            if torch.allclose(min_reference_timestep, max_reference_timestep):
                sampled[:, token_index] = max_reference_timestep
            else:
                random_value = torch.rand([batch_size], device=self.device, dtype=self.dtype)
                sampled[:, token_index] = min_reference_timestep + random_value * (
                    max_reference_timestep - min_reference_timestep
                )
        return sampled

    def _get_runtime_vertical(self, num_leaf_frames: int):
        if not self.dynamic_vertical_hierarchy:
            if num_leaf_frames != self.vertical_leaf_frames:
                raise ValueError(
                    "Variable-length raw-video training is currently disabled for this vertical config. "
                    f"Expected {self.vertical_leaf_frames} leaf latents, got {num_leaf_frames}."
                )
            return self.vertical_info, self.vertical_token_step_budgets

        vertical_info = build_vertical_hierarchy(
            num_leaf_frames=num_leaf_frames,
            num_levels=None,
            level_sizes=None,
            start_from_st_end=self.start_from_st_end,
            st_end_plus=self.st_end_plus,
            allow_condition_for_all_frames=self.vertical_allow_condition_for_all_frames,
        )
        token_step_budgets = get_dynamic_vertical_level_avg_step_budgets(
            vertical_info,
            max_step_budget=self.dynamic_vertical_max_step_budget,
        )
        return vertical_info, token_step_budgets

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        actions: torch.Tensor = None,
        action_is_pad: torch.Tensor = None,
        action_video_latents: torch.Tensor = None,
        action_video_timestep: torch.Tensor = None,
        action_video_leaf_k: torch.Tensor = None,
        action_video_leaf_v: torch.Tensor = None,
        joint_actions: torch.Tensor = None,
        joint_proprio: torch.Tensor = None,
        joint_local_start_latent: torch.Tensor = None,
        joint_local_video_latents: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        profile_timer = getattr(self, "_profile_train_timer", False)
        timer = {}
        if profile_timer and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        last_time = time.perf_counter() if profile_timer else None

        def record_time(name: str):
            nonlocal last_time
            if not profile_timer:
                return
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            now = time.perf_counter()
            timer[f"timer/{name}"] = now - last_time
            last_time = now

        noise = torch.randn_like(clean_latent)
        batch_size, num_frame = image_or_video_shape[:2]
        first_frame_latent = initial_latent if initial_latent is not None else clean_latent[:, :1]
        frame_seq_len = (
            clean_latent.shape[-2] // self.generator_patch_size_hw[0]
        ) * (
            clean_latent.shape[-1] // self.generator_patch_size_hw[1]
        )
        record_time("model_prepare")

        if self.vertical_hierarchy:
            runtime_leaf_frames = clean_latent.shape[1]
            if self.video_action_joint_training and self.joint_drop_tree_tokens:
                runtime_leaf_frames = int(self.vertical_leaf_frames)
            if self.action_training and action_video_latents is not None:
                runtime_leaf_frames = int(self.vertical_leaf_frames)
            runtime_vertical_info, runtime_vertical_token_step_budgets = self._get_runtime_vertical(
                runtime_leaf_frames
            )
            seq_len_override = runtime_vertical_info["num_tokens"] * frame_seq_len

            prefix_t = torch.zeros([batch_size, 1], device=self.device, dtype=self.dtype)

            if self.video_action_joint_training:
                if joint_actions is None:
                    raise KeyError("video_action_joint_training=True requires batch['joint_actions'].")
                if joint_local_start_latent is None or joint_local_video_latents is None:
                    raise KeyError("video_action_joint_training=True requires encoded joint local video latents.")

                local_start_clean = joint_local_start_latent.to(device=self.device, dtype=self.dtype)
                local_video_clean = joint_local_video_latents.to(device=self.device, dtype=self.dtype)
                if local_start_clean.ndim != 5 or local_start_clean.shape[1] != 1:
                    raise ValueError(
                        f"Expected joint_local_start_latent [B, 1, C, H, W], got {tuple(local_start_clean.shape)}."
                    )
                if local_video_clean.ndim != 5:
                    raise ValueError(
                        f"Expected joint_local_video_latents [B, T, C, H, W], got {tuple(local_video_clean.shape)}."
                    )
                if local_video_clean.shape[1] < self.joint_local_video_tokens:
                    pad = local_video_clean[:, -1:].expand(
                        -1,
                        self.joint_local_video_tokens - local_video_clean.shape[1],
                        -1,
                        -1,
                        -1,
                    )
                    local_video_clean = torch.cat([local_video_clean, pad], dim=1)
                elif local_video_clean.shape[1] > self.joint_local_video_tokens:
                    local_video_clean = local_video_clean[:, : self.joint_local_video_tokens]

                local_video_noise = torch.randn_like(local_video_clean)
                local_video_timestep_sample = self.local_video_scheduler.sample_training_t(
                    batch_size=batch_size,
                    device=self.device,
                    dtype=self.dtype,
                )
                local_video_timestep = local_video_timestep_sample[:, None].expand(
                    -1, local_video_clean.shape[1]
                )
                local_video_noisy = self.local_video_scheduler.add_noise(
                    local_video_clean.flatten(0, 1),
                    local_video_noise.flatten(0, 1),
                    local_video_timestep.flatten(0, 1),
                ).unflatten(0, (batch_size, local_video_clean.shape[1]))
                local_video_target = self.local_video_scheduler.training_target(
                    local_video_clean,
                    local_video_noise,
                    local_video_timestep,
                )

                zero_start_timestep = torch.zeros([batch_size, 1], device=self.device, dtype=self.dtype)
                if self.joint_drop_tree_tokens:
                    tree_clean_latent = clean_latent.new_zeros(
                        batch_size,
                        0,
                        clean_latent.shape[2],
                        clean_latent.shape[3],
                        clean_latent.shape[4],
                    )
                    tree_target = tree_clean_latent
                    tree_timestep = clean_latent.new_zeros(batch_size, 0)
                    tree_token_ids = []
                    prefix_token_ids = []
                    prefix_for_joint = local_start_clean[:, :0]
                    prefix_t_for_joint = prefix_t[:, :0]
                    noisy_video_latents = torch.cat(
                        [local_start_clean, local_video_noisy],
                        dim=1,
                    )
                    video_timestep = torch.cat(
                        [zero_start_timestep, local_video_timestep],
                        dim=1,
                    )
                else:
                    full_tree_clean_latent = gather_vertical_latents(clean_latent, runtime_vertical_info)
                    full_tree_token_ids = list(range(runtime_vertical_info["num_tokens"]))
                    full_tree_token_budgets = list(runtime_vertical_token_step_budgets)
                    if self.joint_tree_num_levels > 0:
                        if self.joint_tree_num_levels > len(runtime_vertical_info["level_sizes"]):
                            raise ValueError(
                                f"joint_tree_num_levels={self.joint_tree_num_levels} exceeds "
                                f"runtime hierarchy levels={runtime_vertical_info['level_sizes']}."
                            )
                        tree_count = int(sum(runtime_vertical_info["level_sizes"][: self.joint_tree_num_levels]))
                        tree_clean_latent = full_tree_clean_latent[:, :tree_count]
                        tree_token_ids = full_tree_token_ids[:tree_count]
                        tree_token_budgets = full_tree_token_budgets[:tree_count]
                    else:
                        tree_clean_latent = full_tree_clean_latent
                        tree_token_ids = full_tree_token_ids
                        tree_token_budgets = full_tree_token_budgets
                    tree_noise = torch.randn_like(tree_clean_latent)
                    tree_timestep = self._sample_vertical_timesteps(batch_size, tree_token_budgets)
                    tree_noisy_latents = self.scheduler.add_noise(
                        tree_clean_latent.flatten(0, 1),
                        tree_noise.flatten(0, 1),
                        tree_timestep.flatten(0, 1),
                    ).unflatten(0, (batch_size, tree_clean_latent.shape[1]))
                    tree_target = self.scheduler.training_target(tree_clean_latent, tree_noise, tree_timestep)
                    prefix_token_ids = [CONDITION_TOKEN_ID]
                    prefix_for_joint = first_frame_latent
                    prefix_t_for_joint = prefix_t
                    noisy_video_latents = torch.cat(
                        [tree_noisy_latents, local_start_clean, local_video_noisy],
                        dim=1,
                    )
                    video_timestep = torch.cat(
                        [tree_timestep, zero_start_timestep, local_video_timestep],
                        dim=1,
                    )
                joint_video_token_count = len(tree_token_ids) + 1 + self.joint_local_video_tokens

                actions = joint_actions.to(device=self.device, dtype=self.dtype)
                if actions.ndim != 3:
                    raise ValueError(f"Expected joint_actions [B, T, D], got {tuple(actions.shape)}.")
                action_noise = torch.randn_like(actions)
                if self.joint_action_fixed_timestep is None:
                    action_timestep_sample = self.action_scheduler.sample_training_t(
                        batch_size=batch_size,
                        device=self.device,
                        dtype=self.dtype,
                    )
                else:
                    action_timestep_sample = torch.full(
                        [batch_size],
                        float(self.joint_action_fixed_timestep),
                        device=self.device,
                        dtype=self.dtype,
                    )
                action_timestep = action_timestep_sample[:, None].expand(-1, actions.shape[1])
                noisy_actions = self.action_scheduler.add_noise(
                    actions.flatten(0, 1).unsqueeze(-1).unsqueeze(-1),
                    action_noise.flatten(0, 1).unsqueeze(-1).unsqueeze(-1),
                    action_timestep.flatten(0, 1),
                ).squeeze(-1).squeeze(-1).unflatten(0, actions.shape[:2])
                action_target = self.action_scheduler.training_target(actions, action_noise, action_timestep)

                joint_mot = getattr(self, "joint_mot", None)
                if joint_mot is None:
                    flow_pred_video, flow_pred_action = self.action_dit.forward_video_action_joint(
                        noisy_actions=noisy_actions,
                        action_timestep=action_timestep_sample,
                        video_latents=noisy_video_latents,
                        video_timestep=video_timestep,
                        conditional_dict=conditional_dict,
                        prefix_x=prefix_for_joint,
                        prefix_t=prefix_t_for_joint,
                        prefix_token_ids=prefix_token_ids,
                        tree_token_ids=tree_token_ids,
                        vertical_info=runtime_vertical_info,
                        vertical_use_representative_rope=self.vertical_use_representative_rope,
                        local_start_count=1,
                        local_video_count=self.joint_local_video_tokens,
                        detach_action_video_kv=self.joint_detach_action_video_kv,
                        action_attend_video=self.joint_action_attend_video,
                        action_video_kv_scale=self.joint_action_video_kv_scale,
                        joint_proprio=joint_proprio.to(device=self.device, dtype=self.dtype) if joint_proprio is not None else None,
                        seq_len_override=(
                            joint_video_token_count * frame_seq_len
                        ),
                    )
                else:
                    flow_pred_video, flow_pred_action = joint_mot(
                        noisy_actions=noisy_actions,
                        action_timestep=action_timestep_sample,
                        video_latents=noisy_video_latents,
                        video_timestep=video_timestep,
                        conditional_dict=conditional_dict,
                        prefix_x=prefix_for_joint,
                        prefix_t=prefix_t_for_joint,
                        prefix_token_ids=prefix_token_ids,
                        tree_token_ids=tree_token_ids,
                        vertical_info=runtime_vertical_info,
                        vertical_use_representative_rope=self.vertical_use_representative_rope,
                        local_start_count=1,
                        local_video_count=self.joint_local_video_tokens,
                        detach_action_video_kv=self.joint_detach_action_video_kv,
                        action_attend_video=self.joint_action_attend_video,
                        action_video_kv_scale=self.joint_action_video_kv_scale,
                        joint_proprio=joint_proprio.to(device=self.device, dtype=self.dtype) if joint_proprio is not None else None,
                        seq_len_override=(
                            joint_video_token_count * frame_seq_len
                        ),
                    )
                record_time("model_joint_mot")

                tree_count = len(tree_token_ids)
                flow_pred_tree = flow_pred_video[:, :tree_count]
                flow_pred_local = flow_pred_video[:, tree_count + 1: tree_count + 1 + self.joint_local_video_tokens]

                if tree_count == 0:
                    tree_loss = flow_pred_video.sum() * 0.0
                else:
                    tree_loss = torch.nn.functional.mse_loss(
                        flow_pred_tree.float(), tree_target.float(), reduction="none"
                    ).mean(dim=(2, 3, 4))
                    tree_loss = tree_loss * self.scheduler.training_weight(tree_timestep).unflatten(
                        0, (batch_size, tree_clean_latent.shape[1])
                    )
                    tree_loss = tree_loss.mean()

                local_video_loss = torch.nn.functional.mse_loss(
                    flow_pred_local.float(), local_video_target.float(), reduction="none"
                ).mean(dim=(2, 3, 4))
                local_video_loss = local_video_loss * self.local_video_scheduler.training_weight(
                    local_video_timestep_sample
                )[:, None]
                local_video_loss = local_video_loss.mean()

                action_loss = torch.nn.functional.mse_loss(
                    flow_pred_action.float(), action_target.float(), reduction="none"
                ).mean(dim=2).mean(dim=1)
                action_weight = self.action_scheduler.training_weight(action_timestep_sample).to(
                    device=action_loss.device,
                    dtype=action_loss.dtype,
                )
                action_loss = (action_loss * action_weight).mean()

                loss = (
                    tree_loss * self.joint_tree_video_loss_weight
                    + local_video_loss * self.joint_local_video_loss_weight
                    + action_loss * self.joint_action_loss_weight
                )
                record_time("model_loss")
                tree_timestep_stats = tree_timestep.detach().float()
                if tree_timestep_stats.numel() == 0:
                    tree_timestep_min = tree_timestep_stats.new_tensor(0.0)
                    tree_timestep_max = tree_timestep_stats.new_tensor(0.0)
                else:
                    tree_timestep_min = tree_timestep_stats.min()
                    tree_timestep_max = tree_timestep_stats.max()
                log_dict = {
                    "x0": tree_clean_latent.detach(),
                    "x0_pred": tree_clean_latent.detach(),
                    "loss_video_tree": tree_loss.detach(),
                    "loss_video_leaf_tree": tree_loss.detach(),
                    "loss_video_local": local_video_loss.detach(),
                    "loss_action": action_loss.detach(),
                    "joint_action_pred_abs_mean": flow_pred_action.detach().float().abs().mean(),
                    "joint_action_target_abs_mean": action_target.detach().float().abs().mean(),
                    "joint_action_clean_abs_mean": actions.detach().float().abs().mean(),
                    "joint_action_timestep_mean": action_timestep_sample.detach().float().mean(),
                    "joint_local_video_timestep_mean": local_video_timestep_sample.detach().float().mean(),
                    "joint_tree_timestep_min": tree_timestep_min,
                    "joint_tree_timestep_max": tree_timestep_max,
                }
                log_dict.update(timer)
                return loss, log_dict

            if self.action_training:
                if actions is None:
                    raise KeyError("action_training=True requires batch['actions'].")
                if (
                    action_video_latents is not None
                    and (
                        action_video_leaf_k is not None
                        or action_video_latents.shape[1] == runtime_vertical_info["num_leaf_frames"]
                    )
                ):
                    vertical_clean_latent = action_video_latents.to(device=self.device, dtype=self.dtype)
                    if vertical_clean_latent.ndim != 5:
                        raise ValueError(
                            f"Expected cached leaf latents [B, T, C, H, W], got {tuple(vertical_clean_latent.shape)}."
                        )
                    if vertical_clean_latent.shape[1] != runtime_vertical_info["num_leaf_frames"]:
                        raise ValueError(
                            f"Expected {runtime_vertical_info['num_leaf_frames']} cached leaf latents, "
                            f"got {vertical_clean_latent.shape[1]}."
                    )
                    noisy_latents = vertical_clean_latent
                    timestep = torch.zeros(
                        [batch_size, runtime_vertical_info["num_tokens"]],
                        device=self.device,
                        dtype=self.dtype,
                    )
                    action_video_leaf_k = (
                        action_video_leaf_k.to(device=self.device, dtype=self.dtype)
                        if action_video_leaf_k is not None
                        else None
                    )
                    action_video_leaf_v = (
                        action_video_leaf_v.to(device=self.device, dtype=self.dtype)
                        if action_video_leaf_v is not None
                        else None
                    )
                    video_condition_source = "leaf_kv_cache"
                elif action_video_latents is not None:
                    vertical_clean_latent = action_video_latents.to(device=self.device, dtype=self.dtype)
                    if vertical_clean_latent.ndim != 5:
                        raise ValueError(
                            f"Expected action_video_latents [B, T, C, H, W], got {tuple(vertical_clean_latent.shape)}."
                        )
                    if vertical_clean_latent.shape[1] != runtime_vertical_info["num_tokens"]:
                        raise ValueError(
                            f"Expected {runtime_vertical_info['num_tokens']} cached vertical tokens, "
                            f"got {vertical_clean_latent.shape[1]}."
                        )
                    noisy_latents = vertical_clean_latent
                    if action_video_timestep is None or bool(getattr(self, "action_rebuild_video_cache_timestep", True)):
                        timestep = self._build_action_cache_video_timestep(vertical_clean_latent)
                    else:
                        timestep = action_video_timestep.to(device=self.device, dtype=self.dtype)
                        if timestep.ndim == 1:
                            timestep = timestep[:, None].expand(-1, vertical_clean_latent.shape[1])
                    video_condition_source = "cache_clean_tree_fixed5_timestep"
                    action_video_leaf_k = None
                    action_video_leaf_v = None
                else:
                    vertical_clean_latent = gather_vertical_latents(clean_latent, runtime_vertical_info)
                    vertical_noise = torch.randn_like(vertical_clean_latent)
                    timestep = self._sample_vertical_timesteps(batch_size, runtime_vertical_token_step_budgets)
                    noisy_latents = self.scheduler.add_noise(
                        vertical_clean_latent.flatten(0, 1),
                        vertical_noise.flatten(0, 1),
                        timestep.flatten(0, 1),
                    ).unflatten(0, (batch_size, vertical_clean_latent.shape[1]))
                    video_condition_source = "gt_noisy"
                    action_video_leaf_k = None
                    action_video_leaf_v = None
                record_time("model_vertical_prepare")
                actions = actions.to(device=self.device, dtype=self.dtype)
                if actions.ndim != 3:
                    raise ValueError(f"Expected actions [B, T, D], got {tuple(actions.shape)}.")
                expected_action_steps = runtime_vertical_info["num_leaf_frames"] * self.actions_per_leaf
                if actions.shape[1] != expected_action_steps:
                    raise ValueError(
                        f"Expected {expected_action_steps} action steps for {runtime_vertical_info['num_leaf_frames']} latent leaves "
                        f"and actions_per_leaf={self.actions_per_leaf}, got {actions.shape[1]}."
                    )
                action_noise = torch.randn_like(actions)
                action_timestep_sample = self.action_scheduler.sample_training_t(
                    batch_size=batch_size,
                    device=self.device,
                    dtype=self.dtype,
                )
                action_timestep = action_timestep_sample[:, None].expand(-1, actions.shape[1])
                noisy_actions = self.action_scheduler.add_noise(
                    actions.flatten(0, 1).unsqueeze(-1).unsqueeze(-1),
                    action_noise.flatten(0, 1).unsqueeze(-1).unsqueeze(-1),
                    action_timestep.flatten(0, 1),
                ).squeeze(-1).squeeze(-1).unflatten(0, actions.shape[:2])
                action_target = self.action_scheduler.training_target(actions, action_noise, action_timestep)
                flow_pred_action = self.action_dit(
                    noisy_actions=noisy_actions,
                    action_timestep=action_timestep_sample,
                    video_latents=noisy_latents,
                    video_timestep=timestep,
                    video_leaf_k=action_video_leaf_k,
                    video_leaf_v=action_video_leaf_v,
                    conditional_dict=conditional_dict,
                    prefix_x=first_frame_latent,
                    prefix_t=prefix_t,
                    prefix_token_ids=[CONDITION_TOKEN_ID],
                    noisy_token_ids=list(range(runtime_vertical_info["num_tokens"])),
                    vertical_info=runtime_vertical_info,
                    vertical_use_representative_rope=self.vertical_use_representative_rope,
                    seq_len_override=seq_len_override,
                )
                action_loss = torch.nn.functional.mse_loss(
                    flow_pred_action.float(), action_target.float(), reduction="none"
                ).mean(dim=2)
                if action_is_pad is not None:
                    valid = (~action_is_pad.to(device=self.device)).to(dtype=action_loss.dtype)
                    action_loss = (action_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
                else:
                    action_loss = action_loss.mean(dim=1)
                action_weight = self.action_scheduler.training_weight(action_timestep_sample).to(
                    device=action_loss.device,
                    dtype=action_loss.dtype,
                )
                loss = (action_loss * action_weight).mean() * self.action_loss_weight
                record_time("model_action_mot")
                log_dict = {
                    "x0": vertical_clean_latent.detach(),
                    "x0_pred": vertical_clean_latent.detach(),
                    "loss_action": loss.detach(),
                    "action_video_condition_source": video_condition_source,
                    "action_pred_abs_mean": flow_pred_action.detach().float().abs().mean(),
                    "action_pred_abs_max": flow_pred_action.detach().float().abs().max(),
                    "action_target_abs_mean": action_target.detach().float().abs().mean(),
                    "action_target_abs_max": action_target.detach().float().abs().max(),
                    "action_noisy_abs_mean": noisy_actions.detach().float().abs().mean(),
                    "action_clean_abs_mean": actions.detach().float().abs().mean(),
                    "action_weight_mean": action_weight.detach().float().mean(),
                    "action_timestep_mean": action_timestep_sample.detach().float().mean(),
                    "action_video_timestep_min": timestep.detach().float().min(),
                    "action_video_timestep_max": timestep.detach().float().max(),
                }
                log_dict.update(timer)
                return loss, log_dict

            vertical_clean_latent = gather_vertical_latents(clean_latent, runtime_vertical_info)
            vertical_noise = torch.randn_like(vertical_clean_latent)
            timestep = self._sample_vertical_timesteps(batch_size, runtime_vertical_token_step_budgets)
            noisy_latents = self.scheduler.add_noise(
                vertical_clean_latent.flatten(0, 1),
                vertical_noise.flatten(0, 1),
                timestep.flatten(0, 1),
            ).unflatten(0, (batch_size, vertical_clean_latent.shape[1]))
            training_target = self.scheduler.training_target(vertical_clean_latent, vertical_noise, timestep)
            record_time("model_vertical_prepare")

            flow_pred, x0_pred = self.generator(
                noisy_image_or_video=noisy_latents,
                conditional_dict=conditional_dict,
                timestep=timestep,
                prefix_x=first_frame_latent,
                prefix_t=prefix_t,
                prefix_token_ids=[CONDITION_TOKEN_ID],
                noisy_token_ids=list(range(runtime_vertical_info["num_tokens"])),
                vertical_info=runtime_vertical_info,
                vertical_use_representative_rope=self.vertical_use_representative_rope,
                seq_len_override=seq_len_override,
            )
            record_time("model_generator")
            loss = torch.nn.functional.mse_loss(
                flow_pred.float(), training_target.float(), reduction='none'
            ).mean(dim=(2, 3, 4))
            loss = loss * self.scheduler.training_weight(timestep).unflatten(
                0, (batch_size, vertical_clean_latent.shape[1])
            )
            loss = loss.mean()
            record_time("model_loss")

            log_dict = {
                "x0": vertical_clean_latent.detach(),
                "x0_pred": x0_pred.detach()
            }
            log_dict.update(timer)
            return loss, log_dict

        # Step 2: Randomly sample a timestep and add noise to denoiser inputs
        index = self._get_timestep(
            0,
            self.scheduler.num_train_timesteps,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=False
        )
        timestep = self.scheduler.timesteps[index].to(dtype=self.dtype, device=self.device)
        noisy_latents = self.scheduler.add_noise(
            clean_latent.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))
        training_target = self.scheduler.training_target(clean_latent, noise, timestep)
        if self.condition_first_frame:
            noisy_latents[:, :1] = first_frame_latent
            timestep[:, :1] = 0
        record_time("model_noise_prepare")

        # Step 3: Noise augmentation, also add small noise to clean context latents
        if self.noise_augmentation_max_timestep > 0:
            index_clean_aug = self._get_timestep(
                self.noise_augmentation_max_timestep,
                1000,
                image_or_video_shape[0],
                image_or_video_shape[1],
                self.num_frame_per_block,
                uniform_timestep=False
            )
            timestep_clean_aug = self.scheduler.timesteps[index_clean_aug].to(dtype=self.dtype, device=self.device)
            clean_latent_aug = self.scheduler.add_noise(
                clean_latent.flatten(0, 1),
                noise.flatten(0, 1),
                timestep_clean_aug.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frame))
        else:
            clean_latent_aug = clean_latent
            timestep_clean_aug = None

        if self.condition_first_frame:
            clean_latent_aug[:, :1] = first_frame_latent
            if timestep_clean_aug is not None:
                timestep_clean_aug[:, :1] = 0
        record_time("model_tf_prepare")
        # Compute loss
        seq_len_override = num_frame * frame_seq_len
        flow_pred, x0_pred = self.generator(
            noisy_image_or_video=noisy_latents,
            conditional_dict=conditional_dict,
            timestep=timestep,
            seq_len_override=seq_len_override,
            clean_x=clean_latent_aug if self.teacher_forcing else None,
            aug_t=timestep_clean_aug if self.teacher_forcing else None
        )
        record_time("model_generator")
        # loss = torch.nn.functional.mse_loss(flow_pred.float(), training_target.float())
        loss = torch.nn.functional.mse_loss(
            flow_pred.float(), training_target.float(), reduction='none'
        ).mean(dim=(2, 3, 4))
        loss = loss * self.scheduler.training_weight(timestep).unflatten(0, (batch_size, num_frame))
        if self.condition_first_frame:
            loss = loss[:, 1:]
        loss = loss.mean()
        record_time("model_loss")

        log_dict = {
            "x0": clean_latent.detach(),
            "x0_pred": x0_pred.detach()
        }
        log_dict.update(timer)
        return loss, log_dict

    @torch.no_grad()
    def open_loop_action_val(
        self,
        *,
        conditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor,
        joint_actions: torch.Tensor,
        joint_local_start_latent: torch.Tensor,
        joint_local_video_latents: torch.Tensor,
        joint_proprio: torch.Tensor = None,
        joint_steps: int = 10,
    ) -> dict:
        if not self.video_action_joint_training:
            raise ValueError("open_loop_action_val requires video_action_joint_training=True.")
        batch_size = int(joint_actions.shape[0])
        first_frame_latent = initial_latent if initial_latent is not None else clean_latent[:, :1]
        frame_seq_len = (
            clean_latent.shape[-2] // self.generator_patch_size_hw[0]
        ) * (
            clean_latent.shape[-1] // self.generator_patch_size_hw[1]
        )
        runtime_leaf_frames = int(self.vertical_leaf_frames) if self.joint_drop_tree_tokens else clean_latent.shape[1]
        runtime_vertical_info, _ = self._get_runtime_vertical(runtime_leaf_frames)

        local_start_clean = joint_local_start_latent.to(device=self.device, dtype=self.dtype)
        local_video_clean = joint_local_video_latents.to(device=self.device, dtype=self.dtype)
        if local_video_clean.shape[1] < self.joint_local_video_tokens:
            pad = local_video_clean[:, -1:].expand(
                -1,
                self.joint_local_video_tokens - local_video_clean.shape[1],
                -1,
                -1,
                -1,
            )
            local_video_clean = torch.cat([local_video_clean, pad], dim=1)
        elif local_video_clean.shape[1] > self.joint_local_video_tokens:
            local_video_clean = local_video_clean[:, : self.joint_local_video_tokens]

        prefix_t = torch.zeros([batch_size, 1], device=self.device, dtype=self.dtype)
        zero_start_timestep = torch.zeros([batch_size, 1], device=self.device, dtype=self.dtype)
        if self.joint_drop_tree_tokens:
            tree_clean_latent = clean_latent.new_zeros(
                batch_size,
                0,
                clean_latent.shape[2],
                clean_latent.shape[3],
                clean_latent.shape[4],
            )
            tree_timestep = clean_latent.new_zeros(batch_size, 0)
            tree_token_ids = []
            prefix_token_ids = []
            prefix_for_joint = local_start_clean[:, :0]
            prefix_t_for_joint = prefix_t[:, :0]
        else:
            full_tree_clean_latent = gather_vertical_latents(clean_latent, runtime_vertical_info)
            full_tree_token_ids = list(range(runtime_vertical_info["num_tokens"]))
            if self.joint_tree_num_levels > 0:
                tree_count = int(sum(runtime_vertical_info["level_sizes"][: self.joint_tree_num_levels]))
                tree_clean_latent = full_tree_clean_latent[:, :tree_count]
                tree_token_ids = full_tree_token_ids[:tree_count]
            else:
                tree_clean_latent = full_tree_clean_latent
                tree_token_ids = full_tree_token_ids
            tree_timestep = torch.zeros(
                batch_size,
                tree_clean_latent.shape[1],
                device=self.device,
                dtype=self.dtype,
            )
            prefix_token_ids = [CONDITION_TOKEN_ID]
            prefix_for_joint = first_frame_latent
            prefix_t_for_joint = prefix_t

        video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(self.num_train_timestep),
            shift=float(self.action_infer_shift),
        )
        action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(self.num_train_timestep),
            shift=float(self.action_infer_shift),
        )
        video_timesteps, video_deltas = video_scheduler.build_inference_schedule(
            int(joint_steps), device=self.device, dtype=self.dtype
        )
        action_timesteps, action_deltas = action_scheduler.build_inference_schedule(
            int(joint_steps), device=self.device, dtype=self.dtype
        )

        local_video = torch.randn_like(local_video_clean)
        actions = torch.randn_like(joint_actions.to(device=self.device, dtype=self.dtype))
        joint_video_token_count = len(tree_token_ids) + 1 + self.joint_local_video_tokens
        joint_proprio = joint_proprio.to(device=self.device, dtype=self.dtype) if joint_proprio is not None else None
        joint_mot = getattr(self, "joint_mot", None)
        for video_t, video_delta, action_t, action_delta in zip(
            video_timesteps, video_deltas, action_timesteps, action_deltas
        ):
            video_latents = torch.cat([tree_clean_latent, local_start_clean, local_video], dim=1)
            video_timestep = torch.cat(
                [
                    tree_timestep,
                    zero_start_timestep,
                    torch.full(
                        (batch_size, self.joint_local_video_tokens),
                        float(video_t),
                        device=self.device,
                        dtype=self.dtype,
                    ),
                ],
                dim=1,
            )
            kwargs = dict(
                noisy_actions=actions,
                action_timestep=torch.full([batch_size], float(action_t), device=self.device, dtype=self.dtype),
                video_latents=video_latents,
                video_timestep=video_timestep,
                conditional_dict=conditional_dict,
                prefix_x=prefix_for_joint,
                prefix_t=prefix_t_for_joint,
                prefix_token_ids=prefix_token_ids,
                tree_token_ids=tree_token_ids,
                vertical_info=runtime_vertical_info,
                vertical_use_representative_rope=self.vertical_use_representative_rope,
                local_start_count=1,
                local_video_count=self.joint_local_video_tokens,
                detach_action_video_kv=True,
                action_attend_video=self.joint_action_attend_video,
                action_video_kv_scale=self.joint_action_video_kv_scale,
                joint_proprio=joint_proprio,
                seq_len_override=joint_video_token_count * frame_seq_len,
            )
            if joint_mot is None:
                flow_pred_video, flow_pred_action = self.action_dit.forward_video_action_joint(**kwargs)
            else:
                flow_pred_video, flow_pred_action = joint_mot(**kwargs)
            local_flow = flow_pred_video[
                :,
                len(tree_token_ids) + 1: len(tree_token_ids) + 1 + self.joint_local_video_tokens,
            ]
            local_video = video_scheduler.step(local_flow, video_delta, local_video)
            actions = action_scheduler.step(flow_pred_action, action_delta, actions)

        gt = joint_actions.to(device=self.device, dtype=torch.float32)
        pred = actions.float()
        diff = pred - gt
        raw_gt = self._denormalize_action_for_metrics(gt)
        raw_pred = self._denormalize_action_for_metrics(pred)
        raw_diff = raw_pred - raw_gt
        return {
            "val_action_l1_raw_sum": raw_diff.abs().sum(),
            "val_action_l2_raw_sum": raw_diff.square().sum(),
            "val_action_l1_norm_sum": diff.abs().sum(),
            "val_action_l2_norm_sum": torch.linalg.vector_norm(diff, dim=-1).sum(),
            "val_action_mse_norm_sum": diff.square().sum(),
            "val_action_count": torch.tensor(float(diff.numel()), device=self.device),
            "val_action_step_count": torch.tensor(float(diff.shape[0] * diff.shape[1]), device=self.device),
        }
