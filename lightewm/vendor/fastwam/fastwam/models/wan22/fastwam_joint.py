from typing import Any, Optional, Sequence

import torch

from fastwam.utils.logging_config import get_logger

from .fastwam import FastWAM

logger = get_logger(__name__)


class FastWAMJoint(FastWAM):
    """FastWAM variant with configurable action-to-video attention."""

    @classmethod
    def from_wan22_pretrained(cls, **kwargs):
        video_dit_config = kwargs.get("video_dit_config", None)
        if not isinstance(video_dit_config, dict):
            raise ValueError(
                "`video_dit_config` must be provided as dict for FastWAMJoint."
            )
        if bool(video_dit_config.get("action_conditioned", False)):
            raise ValueError(
                "FastWAMJoint requires `video_dit_config['action_conditioned']=false`."
            )
        return super().from_wan22_pretrained(**kwargs)

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        if getattr(self, "hdr_mode", "back_hdr") in {"episode_first_hdr", "episode_back_mixed", "episode_first_special_latent"}:
            if video_seq_len % video_tokens_per_frame != 0:
                raise ValueError(
                    "episode_first_hdr requires video_seq_len divisible by video_tokens_per_frame, "
                    f"got {video_seq_len} and {video_tokens_per_frame}."
                )
            num_latent_frames = video_seq_len // video_tokens_per_frame
            if num_latent_frames in {6, 7}:
                frame_mask = torch.zeros((num_latent_frames, num_latent_frames), dtype=torch.bool, device=device)
                # [episode first clean], [episode planning noisy x2], [local first clean], [local/back noisy]
                frame_mask[0, 0] = True
                frame_mask[1, [0, 1]] = True
                frame_mask[2, [0, 2]] = True
                frame_mask[3, [0, 3]] = True
                for frame_idx in range(4, num_latent_frames):
                    frame_mask[frame_idx, [0, 3, frame_idx]] = True
                action_video_frames = [0, 3]
            elif num_latent_frames == 2:
                frame_mask = torch.zeros((2, 2), dtype=torch.bool, device=device)
                # Closed-loop action-only path keeps only [episode first clean], [local first clean].
                frame_mask[0, 0] = True
                frame_mask[1, [0, 1]] = True
                action_video_frames = [0, 1]
            else:
                raise ValueError(
                    "episode_first/mixed HDR expects 6 or 7 latent frames for joint/video training or "
                    f"2 clean latent frames for action-only inference, got {num_latent_frames}."
                )
            mask[:video_seq_len, :video_seq_len] = frame_mask.repeat_interleave(
                video_tokens_per_frame, dim=0
            ).repeat_interleave(video_tokens_per_frame, dim=1)
            mask[video_seq_len:, video_seq_len:] = True
            for frame_idx in action_video_frames:
                frame_tokens = slice(frame_idx * video_tokens_per_frame, (frame_idx + 1) * video_tokens_per_frame)
                mask[video_seq_len:, frame_tokens] = True
            return mask

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        if self.action_attend_video == "full":
            mask[video_seq_len:, :video_seq_len] = True
        elif self.action_attend_video == "local_clean_first":
            first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
            mask[video_seq_len:, :first_frame_tokens] = True
        else:
            raise ValueError(f"Unsupported action_attend_video: {self.action_attend_video}")
        return mask

    @torch.no_grad()
    def _episode_first_hdr_action_only_video_freqs(self, video_pre: dict[str, Any]) -> torch.Tensor:
        f, h, w = video_pre["meta"]["grid_size"]
        if int(f) != 2:
            raise ValueError(f"episode_first_hdr action-only RoPE remap expects 2 latent frames, got {f}.")
        frame_indices = torch.tensor([0, 3], device=self.video_expert.freqs[0].device, dtype=torch.long)
        freqs = torch.cat([
            self.video_expert.freqs[0][frame_indices].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.video_expert.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.video_expert.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(f * h * w, 1, -1)
        return freqs.to(video_pre["tokens"].device)

    def _prepare_action_abs6_first_frame_condition(
        self,
        action_abs6_state: Optional[torch.Tensor],
        latents_action: torch.Tensor,
        generator: Optional[torch.Generator],
        rand_device: str,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if action_abs6_state is None:
            return None, None
        if self.action_expert.action_dim < 13:
            raise ValueError(
                "`action_abs6_state` requires action_dim >= 13, "
                f"got {self.action_expert.action_dim}."
            )
        if action_abs6_state.ndim == 1:
            action_abs6_state = action_abs6_state.unsqueeze(0)
        if action_abs6_state.ndim != 2 or action_abs6_state.shape != (1, 6):
            raise ValueError(
                "`action_abs6_state` must have shape [6] or [1,6], "
                f"got {tuple(action_abs6_state.shape)}."
            )
        state = action_abs6_state.to(device=self.device, dtype=latents_action.dtype)
        noise = torch.randn(
            (1, 6),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=latents_action.dtype)
        return state, noise

    def _overwrite_first_action_abs6_with_state_noise(
        self,
        latents_action: torch.Tensor,
        action_abs6_state: Optional[torch.Tensor],
        action_abs6_noise: Optional[torch.Tensor],
        timestep_action: torch.Tensor,
    ) -> torch.Tensor:
        if action_abs6_state is None or action_abs6_noise is None:
            return latents_action
        sigma = (timestep_action / float(self.infer_action_scheduler.num_train_timesteps)).to(
            device=latents_action.device, dtype=latents_action.dtype
        ).view(-1, 1)
        latents_action[:, 0, 7:13] = action_abs6_state + sigma * action_abs6_noise
        return latents_action

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
        episode_image: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        action_abs6_state: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        action_adapter: Optional[str | Sequence[str]] = None,
        test_action_with_infer_action: bool = True,
    ) -> dict[str, Any]:
        if getattr(self, "hdr_mode", "back_hdr") not in {"episode_first_hdr", "episode_back_mixed", "episode_first_special_latent"}:
            if test_action_with_infer_action:
                logger.warning(
                    "`FastWAMJoint.infer_joint` ignores `test_action_with_infer_action=True` "
                    "and always runs with `test_action_with_infer_action=False`."
                )
            return super().infer_joint(
                prompt=prompt,
                input_image=input_image,
                num_video_frames=num_video_frames,
                action_horizon=action_horizon,
                action=action,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                negative_prompt=negative_prompt,
                text_cfg_scale=text_cfg_scale,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                action_adapter=action_adapter,
                test_action_with_infer_action=False,
            )

        self.eval()
        if test_action_with_infer_action:
            logger.warning(
                "`FastWAMJoint.infer_joint` ignores `test_action_with_infer_action=True` "
                "for episode_first_hdr and always runs joint video/action inference."
            )
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}")

        if episode_image is None:
            episode_image = input_image
        elif episode_image.ndim == 5:
            if episode_image.shape[0] != 1 or episode_image.shape[1] != 3:
                raise ValueError(f"`episode_image` clip must be [1,3,T,H,W], got {tuple(episode_image.shape)}")
            episode_image = episode_image[:, :, 0]
        elif episode_image.ndim == 4 and episode_image.shape[0] == 3:
            episode_image = episode_image[:, 0].unsqueeze(0)
        elif episode_image.ndim == 3:
            episode_image = episode_image.unsqueeze(0)
        if episode_image.ndim != 4 or episode_image.shape[0] != 1 or episode_image.shape[1] != 3:
            raise ValueError(
                f"`episode_image` must be [1,3,H,W], [3,H,W], [3,T,H,W], or [1,3,T,H,W], got {tuple(episode_image.shape)}"
            )
        if tuple(episode_image.shape[-2:]) != (height, width):
            raise ValueError(
                "`episode_image` spatial size must match `input_image`, "
                f"got {tuple(episode_image.shape[-2:])} vs {(height, width)}"
            )

        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} "
                    f"with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        local_latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        mode = getattr(self, "hdr_mode", "back_hdr")
        expected_local_latent_t = 4 if mode == "episode_back_mixed" else 3
        if local_latent_t != expected_local_latent_t:
            expected_frames = (expected_local_latent_t - 1) * self.vae.temporal_downsample_factor + 1
            raise ValueError(
                f"{mode} joint video inference expects {expected_frames} RGB frames / "
                f"{expected_local_latent_t} local latents, got {num_video_frames}."
            )
        episode_latent_t = 3
        local_start = episode_latent_t
        latent_t = episode_latent_t + local_latent_t
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        action_abs6_state_cond, action_abs6_fixed_noise = self._prepare_action_abs6_first_frame_condition(
            action_abs6_state=action_abs6_state,
            latents_action=latents_action,
            generator=action_generator,
            rand_device=rand_device,
        )

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        episode_image = episode_image.to(device=self.device, dtype=self.torch_dtype)
        episode_first_latents = self._encode_input_image_latents_tensor(input_image=episode_image, tiled=tiled)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = episode_first_latents.clone()
        latents_video[:, :, local_start:local_start + 1] = first_frame_latents.clone()
        clean_latent_indices = torch.tensor([0, local_start], dtype=torch.long, device=self.device)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            latents_action = self._overwrite_first_action_abs6_with_state_noise(
                latents_action=latents_action,
                action_abs6_state=action_abs6_state_cond,
                action_abs6_noise=action_abs6_fixed_noise,
                timestep_action=timestep_action,
            )
            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
                clean_latent_indices=clean_latent_indices,
                action_adapter=action_adapter,
            )
            latents_video = self.infer_video_scheduler.step(pred_video_posi, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action_posi, step_delta_action, latents_action)
            latents_action = self._overwrite_first_action_abs6_with_state_noise(
                latents_action=latents_action,
                action_abs6_state=action_abs6_state_cond,
                action_abs6_noise=action_abs6_fixed_noise,
                timestep_action=step_t_action.new_zeros((1,)),
            )
            latents_video[:, :, 0:1] = episode_first_latents.clone()
            latents_video[:, :, local_start:local_start + 1] = first_frame_latents.clone()

        return {
            "episode_video": self._decode_latents(latents_video[:, :, :episode_latent_t], tiled=tiled),
            "video": self._decode_latents(latents_video[:, :, local_start:], tiled=tiled),
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        episode_image: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        action_abs6_state: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        action_adapter: Optional[str | Sequence[str]] = None,
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            episode_image=episode_image,
            proprio=proprio,
            action_abs6_state=action_abs6_state,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            action_adapter=action_adapter,
            test_action_with_infer_action=False,
        )

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
        episode_image: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        action_abs6_state: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        action_adapter: Optional[str | Sequence[str]] = None,
    ) -> dict[str, Any]:
        self.eval()

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        episode_first_hdr_action_only = getattr(self, "hdr_mode", "back_hdr") in {"episode_first_hdr", "episode_back_mixed", "episode_first_special_latent"}
        latent_t = 2 if episode_first_hdr_action_only else (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        back_hdr_clean_kv_action_only = (
            not episode_first_hdr_action_only
            and latent_t == 1
            and self.action_attend_video == "local_clean_first"
        )
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor
        if episode_first_hdr_action_only:
            if episode_image is None:
                episode_image = input_image
            if episode_image.ndim == 3:
                episode_image = episode_image.unsqueeze(0)
            if episode_image.ndim != 4 or episode_image.shape[0] != 1 or episode_image.shape[1] != 3:
                raise ValueError(
                    f"`episode_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(episode_image.shape)}"
                )
            if tuple(episode_image.shape[-2:]) != (height, width):
                raise ValueError(
                    "`episode_image` spatial size must match `input_image`, "
                    f"got {tuple(episode_image.shape[-2:])} vs {(height, width)}"
                )

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        action_abs6_state_cond, action_abs6_fixed_noise = self._prepare_action_abs6_first_frame_condition(
            action_abs6_state=action_abs6_state,
            latents_action=latents_action,
            generator=action_generator,
            rand_device=rand_device,
        )

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        if episode_first_hdr_action_only:
            episode_image = episode_image.to(device=self.device, dtype=self.torch_dtype)
            episode_first_latents = self._encode_input_image_latents_tensor(input_image=episode_image, tiled=tiled)
            latents_video[:, :, 0:1] = episode_first_latents.clone()
            latents_video[:, :, 1:2] = first_frame_latents.clone()
        else:
            latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        if episode_first_hdr_action_only or back_hdr_clean_kv_action_only:
            timestep_video = torch.zeros(
                (latents_video.shape[0],),
                dtype=latents_video.dtype,
                device=self.device,
            )
            clean_latent_indices = torch.arange(latent_t, dtype=torch.long, device=self.device)
            video_pre = self.video_expert.pre_dit(
                x=latents_video,
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_flag,
                clean_latent_indices=clean_latent_indices,
            )
            if episode_first_hdr_action_only:
                video_pre["freqs"] = self._episode_first_hdr_action_only_video_freqs(video_pre)
            video_seq_len = int(video_pre["tokens"].shape[1])
            attention_mask = self._build_mot_attention_mask(
                video_seq_len=video_seq_len,
                action_seq_len=latents_action.shape[1],
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=video_pre["tokens"].device,
            )
            video_kv_cache = self.mot.prefill_video_cache(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
            )
            infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
            for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
                timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
                latents_action = self._overwrite_first_action_abs6_with_state_noise(
                    latents_action=latents_action,
                    action_abs6_state=action_abs6_state_cond,
                    action_abs6_noise=action_abs6_fixed_noise,
                    timestep_action=timestep_action,
                )
                pred_action_posi = self._predict_action_noise_with_cache(
                    latents_action=latents_action,
                    timestep_action=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    video_kv_cache=video_kv_cache,
                    attention_mask=attention_mask,
                    video_seq_len=video_seq_len,
                    action_adapter=action_adapter,
                )
                latents_action = self.infer_action_scheduler.step(pred_action_posi, step_delta_action, latents_action)
                latents_action = self._overwrite_first_action_abs6_with_state_noise(
                    latents_action=latents_action,
                    action_abs6_state=action_abs6_state_cond,
                    action_abs6_noise=action_abs6_fixed_noise,
                    timestep_action=step_t_action.new_zeros((1,)),
                )
            return {
                "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
            }

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            latents_action = self._overwrite_first_action_abs6_with_state_noise(
                latents_action=latents_action,
                action_abs6_state=action_abs6_state_cond,
                action_abs6_noise=action_abs6_fixed_noise,
                timestep_action=timestep_action,
            )

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=None,
                action_adapter=action_adapter,
            )

            latents_video = self.infer_video_scheduler.step(pred_video_posi, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action_posi, step_delta_action, latents_action)
            latents_action = self._overwrite_first_action_abs6_with_state_noise(
                latents_action=latents_action,
                action_abs6_state=action_abs6_state_cond,
                action_abs6_noise=action_abs6_fixed_noise,
                timestep_action=step_t_action.new_zeros((1,)),
            )
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }
