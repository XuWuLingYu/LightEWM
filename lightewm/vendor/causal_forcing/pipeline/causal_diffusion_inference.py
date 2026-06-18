from tqdm import tqdm
from typing import Callable, List, Optional
import torch

from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.vertical_hierarchy import (
    CONDITION_TOKEN_ID,
    build_vertical_hierarchy,
    get_dynamic_vertical_level_avg_step_budgets,
    get_vertical_leaf_latents,
    get_vertical_token_step_budgets,
)


class CausalDiffusionInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None,
            need_vae = True
    ):
        super().__init__()
        model_kwargs = getattr(args, "model_kwargs", {})
        model_name = getattr(model_kwargs, "model_name", "Wan2.1-T2V-1.3B")
        model_root = getattr(model_kwargs, "model_root", "wan_models")
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder(model_name=model_name, model_root=model_root) if text_encoder is None else text_encoder
        if need_vae:
            self.vae = WanVAEWrapper(model_name=model_name, model_root=model_root) if vae is None else vae

        # Step 2: Initialize scheduler
        self.num_train_timesteps = args.num_train_timestep
        self.sampling_steps = int(getattr(args, "sampling_steps", 50))
        if self.sampling_steps <= 0:
            raise ValueError(f"sampling_steps must be positive, got {self.sampling_steps}.")
        self.sample_solver = 'unipc'
        self.shift = args.timestep_shift

        self.num_transformer_blocks = self.generator.model.num_layers
        patch_h, patch_w = self.generator.model.patch_size[1], self.generator.model.patch_size[2]
        latent_h = args.image_or_video_shape[-2]
        latent_w = args.image_or_video_shape[-1]
        self.frame_seq_length = (latent_h // patch_h) * (latent_w // patch_w)
        self.max_latent_frames = args.image_or_video_shape[1]

        self.kv_cache_pos = None
        self.kv_cache_neg = None
        self.vertical_kv_cache_pos = None
        self.vertical_kv_cache_neg = None
        self.crossattn_cache_pos = None
        self.crossattn_cache_neg = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size
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
        self.vertical_infer_fixed_denoise_steps = int(getattr(args, "vertical_infer_fixed_denoise_steps", 0))
        self.vertical_infer_preserve_budget_ratio = bool(
            getattr(args, "vertical_infer_preserve_budget_ratio", False)
        )
        self.vertical_infer_reference_total_steps = int(
            getattr(args, "vertical_infer_reference_total_steps", self.sampling_steps)
        )
        if self.vertical_infer_fixed_denoise_steps < 0:
            raise ValueError(
                f"vertical_infer_fixed_denoise_steps must be non-negative, got {self.vertical_infer_fixed_denoise_steps}."
            )
        if self.vertical_infer_reference_total_steps <= 0:
            raise ValueError(
                f"vertical_infer_reference_total_steps must be positive, got {self.vertical_infer_reference_total_steps}."
            )
        if self.vertical_hierarchy:
            self.configure_vertical_runtime(self.vertical_leaf_frames)
        else:
            self.vertical_info = None
            self.vertical_token_step_budgets = None

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def configure_vertical_runtime(self, num_leaf_frames: int) -> None:
        if not self.vertical_hierarchy:
            return
        if self.dynamic_vertical_hierarchy:
            self.vertical_info = build_vertical_hierarchy(
                num_leaf_frames=num_leaf_frames,
                num_levels=None,
                level_sizes=None,
                start_from_st_end=self.start_from_st_end,
                st_end_plus=self.st_end_plus,
                allow_condition_for_all_frames=self.vertical_allow_condition_for_all_frames,
            )
            self.vertical_token_step_budgets = get_dynamic_vertical_level_avg_step_budgets(
                self.vertical_info,
                max_step_budget=self.dynamic_vertical_max_step_budget,
            )
        else:
            if num_leaf_frames != self.vertical_leaf_frames:
                raise ValueError(
                    "Dynamic vertical inference is disabled for this config. "
                    f"Expected num_output_frames={self.vertical_leaf_frames}, got {num_leaf_frames}."
                )
            self.vertical_info = build_vertical_hierarchy(
                num_leaf_frames=self.vertical_leaf_frames,
                num_levels=len(self.vertical_level_sizes),
                level_sizes=self.vertical_level_sizes,
                start_from_st_end=self.start_from_st_end,
                st_end_plus=self.st_end_plus,
                allow_condition_for_all_frames=self.vertical_allow_condition_for_all_frames,
            )
            self.vertical_token_step_budgets = get_vertical_token_step_budgets(
                self.vertical_info,
                self.vertical_step_budgets,
            )
        self.vertical_leaf_frames = num_leaf_frames
        self.vertical_kv_cache_pos = None
        self.vertical_kv_cache_neg = None

    def _get_vertical_token_sampling_plan(self, original_budget: int) -> tuple[int, int]:
        if original_budget < 0:
            raise ValueError(f"vertical step budget must be non-negative, got {original_budget}.")
        if self.vertical_infer_fixed_denoise_steps <= 0:
            return original_budget, self.sampling_steps
        if original_budget == 0:
            return 0, self.sampling_steps

        denoise_steps = self.vertical_infer_fixed_denoise_steps
        if self.vertical_infer_preserve_budget_ratio:
            # Keep per-level noise ratio roughly aligned with training:
            # original_budget / reference_total_steps ~= denoise_steps / runtime_total_steps
            runtime_total_steps = int(
                round(
                    denoise_steps * self.vertical_infer_reference_total_steps / float(original_budget)
                )
            )
            runtime_total_steps = max(denoise_steps, runtime_total_steps)
        else:
            runtime_total_steps = self.sampling_steps
            denoise_steps = min(denoise_steps, runtime_total_steps)
        return denoise_steps, runtime_total_steps

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        return_vertical_layer_videos: bool = False,
        return_vertical_detail_logs: bool = False,
        vertical_layer_callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        start_frame_index: Optional[int] = 0,
        return_video=True
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
            start_frame_index (int): In long video generation, where does the current window start?
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        if self.vertical_hierarchy:
            return self._inference_vertical(
                noise=noise,
                text_prompts=text_prompts,
                initial_latent=initial_latent,
                return_latents=return_latents,
                return_vertical_layer_videos=return_vertical_layer_videos,
                return_vertical_detail_logs=return_vertical_detail_logs,
                vertical_layer_callback=vertical_layer_callback,
                return_video=return_video,
            )

        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        elif self.independent_first_frame and initial_latent is None:
            # Using a [1, 4, 4, 4, 4, 4] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )
        unconditional_dict = self.text_encoder(
            text_prompts=[self.args.negative_prompt] * len(text_prompts)
        )

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache_pos is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache_pos[block_index]["is_init"] = False
                self.crossattn_cache_neg[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache_pos)):
                self.kv_cache_pos[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_pos[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_neg[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_neg[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = start_frame_index
        cache_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                current_start_frame += 1
                cache_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for block_index in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, cache_start_frame:cache_start_frame + self.num_frame_per_block]
                output[:, cache_start_frame:cache_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                current_start_frame += self.num_frame_per_block
                cache_start_frame += self.num_frame_per_block

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for current_num_frames in all_num_frames:
            noisy_input = noise[
                :, cache_start_frame - num_input_frames:cache_start_frame + current_num_frames - num_input_frames]
            latents = noisy_input

            # Step 3.1: Spatial denoising loop
            sample_scheduler = self._initialize_sample_scheduler(noise)
            for _, t in enumerate(tqdm(sample_scheduler.timesteps)):
                latent_model_input = latents
                timestep = t * torch.ones(
                    [batch_size, current_num_frames], device=noise.device, dtype=torch.float32
                )

                flow_pred_cond, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                flow_pred_uncond, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    conditional_dict=unconditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )

                flow_pred = flow_pred_uncond + self.args.guidance_scale * (
                    flow_pred_cond - flow_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    flow_pred,
                    t,
                    latents,
                    return_dict=False)[0]
                latents = temp_x0

            # Step 3.2: record the model's output
            output[:, cache_start_frame:cache_start_frame + current_num_frames] = latents

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=conditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length
            )
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=unconditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_neg,
                crossattn_cache=self.crossattn_cache_neg,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length
            )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames
            cache_start_frame += current_num_frames

        # Step 4: Decode the output
        if return_video:
            video = self.vae.decode_to_pixel(output)
            video = (video * 0.5 + 0.5).clamp(0, 1)

            if return_latents:
                return video, output
            else:
                return video
        else:
            return output


    def inference_for_cd(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        record_step_indices: List[int],
        initial_latent: Optional[torch.Tensor] = None,
        start_frame_index: int = 0
    ):
        """
        Causal-forcing inference + record selected diffusion steps (per-chunk) for consistency distillation data.
        Record semantics: record xt BEFORE scheduler.step() at the specified progress_id (index in timesteps list).
        Also record the final latent of each chunk after the denoising loop.

        Returns:
            if return_video:
                (video, output_latents, cd_pack)
            else:
                (output_latents, cd_pack)

        cd_pack:
            {
            "record_step_indices": [...],
            "record_t_values": [t_i ...]  # same for all chunks
            "chunks": [
                {
                    "frame_start": int,
                    "frame_len": int,
                    "latents": Tensor [B, R, T, C, H, W]  (R = len(record_step_indices)+1, last one is final)
                }, ...
            ]
            }
        """
        self.sampling_steps = 48
        batch_size, num_frames, num_channels, height, width = noise.shape

        # ---- block counting (same logic as inference) ----
        if (not self.independent_first_frame) or (self.independent_first_frame and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block

        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames

        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        unconditional_dict = self.text_encoder(text_prompts=[self.args.negative_prompt] * len(text_prompts))

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # ---- Step 1: init/reset caches (same as inference) ----
        if self.kv_cache_pos is None:
            self._initialize_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
            self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache_pos[block_index]["is_init"] = False
                self.crossattn_cache_neg[block_index]["is_init"] = False
            for block_index in range(len(self.kv_cache_pos)):
                self.kv_cache_pos[block_index]["global_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache_pos[block_index]["local_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache_neg[block_index]["global_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache_neg[block_index]["local_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)

        # ---- validate record indices against scheduler length ----
        sample_scheduler_probe = self._initialize_sample_scheduler(noise)
        T = len(sample_scheduler_probe.timesteps)
        record_step_indices = sorted(set(int(i) for i in record_step_indices))
        if len(record_step_indices) == 0:
            raise ValueError("record_step_indices must be non-empty")
        if record_step_indices[0] < 0 or record_step_indices[-1] >= T:
            raise ValueError(f"record_step_indices out of range: valid=[0,{T-1}], got={record_step_indices}")
        record_set = set(record_step_indices)

        # ---- Step 2: cache context from initial_latent (same as inference) ----
        current_start_frame = start_frame_index
        cache_start_frame = 0

        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0

            # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
            assert num_input_frames % self.num_frame_per_block == 0
            num_input_blocks = num_input_frames // self.num_frame_per_block

            for block_index in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, cache_start_frame:cache_start_frame + self.num_frame_per_block]
                output[:, cache_start_frame:cache_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                current_start_frame += self.num_frame_per_block
                cache_start_frame += self.num_frame_per_block

        # ---- Step 3: causal-forcing denoising per chunk + record ----
        all_num_frames = [self.num_frame_per_block] * num_blocks

        full_chunk_record = []
        for current_num_frames in all_num_frames:
            # noise slice for current window (same as inference)
            noisy_input = noise[:, cache_start_frame - num_input_frames:cache_start_frame + current_num_frames - num_input_frames]
            latents = noisy_input

            # record list for this chunk
            chunk_records = []

            sample_scheduler = self._initialize_sample_scheduler(noise)
            for progress_id, t in enumerate(tqdm(sample_scheduler.timesteps)):
                if progress_id in record_set:
                    print(f'{progress_id}: {t} saved')
                    chunk_records.append(latents.detach().clone())

                timestep = t * torch.ones([batch_size, current_num_frames], device=noise.device, dtype=torch.float32)

                flow_pred_cond, _ = self.generator(
                    noisy_image_or_video=latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                flow_pred_uncond, _ = self.generator(
                    noisy_image_or_video=latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )

                flow_pred = flow_pred_uncond + self.args.guidance_scale * (flow_pred_cond - flow_pred_uncond)
                latents = sample_scheduler.step(flow_pred, t, latents, return_dict=False)[0]

            # always append final latent of this chunk (like "-2")
            chunk_records.append(latents.detach().clone())
            chunk_records = torch.stack(chunk_records, dim=1)  # [B, R, T, C, H, W]

            full_chunk_record.append(chunk_records)
            # write output
            output[:, cache_start_frame:cache_start_frame + current_num_frames] = latents

            # rerun at t=0 to update cache using clean context (same as inference)
            timestep0 = torch.zeros([batch_size, current_num_frames], device=noise.device, dtype=torch.float32)
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=conditional_dict,
                timestep=timestep0,
                kv_cache=self.kv_cache_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length
            )
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=unconditional_dict,
                timestep=timestep0,
                kv_cache=self.kv_cache_neg,
                crossattn_cache=self.crossattn_cache_neg,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length
            )


            current_start_frame += current_num_frames
            cache_start_frame += current_num_frames


        full_chunk_record = torch.cat(full_chunk_record, dim=2)
        # ---- Step 4: decode if needed ----

        return full_chunk_record


    def inference_for_genuine_cd(
        self,
        noisy_input: torch.Tensor,
        conditional_dict = None,
        unconditional_dict = None,
        text_prompts = None,
        initial_latent: Optional[torch.Tensor] = None,
        timestep_idx=0,
        sampling_steps=48,
        chunksize = 3
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noisy_input.shape
        assert num_frames == chunksize
        if initial_latent is not None:
            num_input_frames = initial_latent.shape[1]
            assert num_input_frames % chunksize == 0
            num_output_frames = num_frames + num_input_frames
        else:
            num_output_frames = num_frames

        if conditional_dict is None:
            assert text_prompts is not None
            conditional_dict = self.text_encoder(
                text_prompts=text_prompts
            )
            unconditional_dict = self.text_encoder(
                text_prompts=[self.args.negative_prompt] * len(text_prompts)
            )

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noisy_input.device,
            dtype=noisy_input.dtype
        )

        if self.kv_cache_pos is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noisy_input.dtype,
                device=noisy_input.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noisy_input.dtype,
                device=noisy_input.device
            )
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache_pos[block_index]["is_init"] = False
                self.crossattn_cache_neg[block_index]["is_init"] = False
            for block_index in range(len(self.kv_cache_pos)):
                self.kv_cache_pos[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noisy_input.device)
                self.kv_cache_pos[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noisy_input.device)
                self.kv_cache_neg[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noisy_input.device)
                self.kv_cache_neg[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noisy_input.device)

        current_start_frame = 0
        cache_start_frame = 0
        timestep = torch.ones([batch_size, 1], device=noisy_input.device, dtype=torch.int64) * 0


        if initial_latent is not None:
            num_input_blocks = num_input_frames // chunksize
            for block_index in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, cache_start_frame:cache_start_frame + chunksize]
                output[:, cache_start_frame:cache_start_frame + chunksize] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                current_start_frame += chunksize
                cache_start_frame += chunksize


        latents = noisy_input
        sample_scheduler = self._initialize_sample_scheduler(noisy_input, sampling_steps=sampling_steps)
        t = sample_scheduler.timesteps[timestep_idx]
        latent_model_input = latents
        timestep = t * torch.ones(
            [batch_size, chunksize], device=noisy_input.device, dtype=torch.float32
        )
        flow_pred_cond, _ = self.generator(
            noisy_image_or_video=latent_model_input,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=self.kv_cache_pos,
            crossattn_cache=self.crossattn_cache_pos,
            current_start=current_start_frame * self.frame_seq_length,
            cache_start=cache_start_frame * self.frame_seq_length
        )
        flow_pred_uncond, _ = self.generator(
            noisy_image_or_video=latent_model_input,
            conditional_dict=unconditional_dict,
            timestep=timestep,
            kv_cache=self.kv_cache_neg,
            crossattn_cache=self.crossattn_cache_neg,
            current_start=current_start_frame * self.frame_seq_length,
            cache_start=cache_start_frame * self.frame_seq_length
        )
        flow_pred = flow_pred_uncond + self.args.guidance_scale * (
            flow_pred_cond - flow_pred_uncond)

        latents = sample_scheduler.step(
            flow_pred,
            t,
            latents,
            return_dict=False)[0]

        return latents



    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache_pos = []
        kv_cache_neg = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = self.max_latent_frames * self.frame_seq_length

        num_heads = self.generator.model.num_heads
        head_dim = self.generator.model.dim // self.generator.model.num_heads

        for _ in range(self.num_transformer_blocks):
            kv_cache_pos.append({
                "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
            kv_cache_neg.append({
                "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache_pos = kv_cache_pos  # always store the clean cache
        self.kv_cache_neg = kv_cache_neg  # always store the clean cache

    def _initialize_vertical_kv_cache(self, batch_size, dtype, device):
        vertical_kv_cache_pos = []
        vertical_kv_cache_neg = []
        vertical_cache_size = self.vertical_info["num_tokens"] + 1
        num_heads = self.generator.model.num_heads
        head_dim = self.generator.model.dim // self.generator.model.num_heads

        for _ in range(self.num_transformer_blocks):
            vertical_kv_cache_pos.append({
                "k": torch.zeros(
                    [batch_size, vertical_cache_size, self.frame_seq_length, num_heads, head_dim],
                    dtype=dtype,
                    device=device,
                ),
                "v": torch.zeros(
                    [batch_size, vertical_cache_size, self.frame_seq_length, num_heads, head_dim],
                    dtype=dtype,
                    device=device,
                ),
                "valid": torch.zeros([vertical_cache_size], dtype=torch.bool, device=device),
            })
            vertical_kv_cache_neg.append({
                "k": torch.zeros(
                    [batch_size, vertical_cache_size, self.frame_seq_length, num_heads, head_dim],
                    dtype=dtype,
                    device=device,
                ),
                "v": torch.zeros(
                    [batch_size, vertical_cache_size, self.frame_seq_length, num_heads, head_dim],
                    dtype=dtype,
                    device=device,
                ),
                "valid": torch.zeros([vertical_cache_size], dtype=torch.bool, device=device),
            })

        self.vertical_kv_cache_pos = vertical_kv_cache_pos
        self.vertical_kv_cache_neg = vertical_kv_cache_neg

    def _reset_vertical_kv_cache(self):
        if self.vertical_kv_cache_pos is None:
            return
        for block_index in range(self.num_transformer_blocks):
            self.vertical_kv_cache_pos[block_index]["valid"].zero_()
            self.vertical_kv_cache_neg[block_index]["valid"].zero_()

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache_pos = []
        crossattn_cache_neg = []
        num_heads = self.generator.model.num_heads
        head_dim = self.generator.model.dim // self.generator.model.num_heads
        for _ in range(self.num_transformer_blocks):
            crossattn_cache_pos.append({
                "k": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
                "is_init": False
            })
            crossattn_cache_neg.append({
                "k": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
                "is_init": False
            })

        self.crossattn_cache_pos = crossattn_cache_pos  # always store the clean cache
        self.crossattn_cache_neg = crossattn_cache_neg  # always store the clean cache

    def _initialize_sample_scheduler(self, noise, sampling_steps=-1):
        if sampling_steps == -1:
            sampling_steps = self.sampling_steps
        if self.sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                sampling_steps, device=noise.device, shift=self.shift)
            self.timesteps = sample_scheduler.timesteps
        elif self.sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, self.shift)
            self.timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=noise.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")
        return sample_scheduler

    def _inference_vertical(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        return_vertical_layer_videos: bool = False,
        return_vertical_detail_logs: bool = False,
        vertical_layer_callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        return_video: bool = True,
    ) -> torch.Tensor:
        if initial_latent is None:
            raise ValueError("Vertical TI2V inference requires an initial_latent.")
        if noise.shape[1] != self.vertical_info["num_tokens"]:
            raise ValueError(
                f"Vertical inference expects {self.vertical_info['num_tokens']} noisy tokens, got {noise.shape[1]}."
            )

        batch_size, _, num_channels, height, width = noise.shape
        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        unconditional_dict = self.text_encoder(
            text_prompts=[self.args.negative_prompt] * len(text_prompts)
        )

        if self.vertical_kv_cache_pos is None:
            self._initialize_vertical_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
            )
        else:
            self._reset_vertical_kv_cache()

        if self.crossattn_cache_pos is None:
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
            )
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache_pos[block_index]["is_init"] = False
                self.crossattn_cache_neg[block_index]["is_init"] = False

        condition_timestep = torch.zeros([batch_size, 1], device=noise.device, dtype=torch.float32)
        condition_rope_frame_index = 0 if self.vertical_use_representative_rope else None
        self.generator(
            noisy_image_or_video=initial_latent[:, :1],
            conditional_dict=conditional_dict,
            timestep=condition_timestep,
            vertical_kv_cache=self.vertical_kv_cache_pos,
            vertical_current_token_id=CONDITION_TOKEN_ID,
            vertical_info=self.vertical_info,
            vertical_cache_write=True,
            vertical_rope_frame_index=condition_rope_frame_index,
            crossattn_cache=self.crossattn_cache_pos,
            current_start=0,
        )
        self.generator(
            noisy_image_or_video=initial_latent[:, :1],
            conditional_dict=unconditional_dict,
            timestep=condition_timestep,
            vertical_kv_cache=self.vertical_kv_cache_neg,
            vertical_current_token_id=CONDITION_TOKEN_ID,
            vertical_info=self.vertical_info,
            vertical_cache_write=True,
            vertical_rope_frame_index=condition_rope_frame_index,
            crossattn_cache=self.crossattn_cache_neg,
            current_start=0,
        )

        generated_tokens = []
        x0_pred_tokens = []
        detail_x0_preds = None
        if return_vertical_detail_logs:
            detail_x0_preds = []
            for level_offset, level_size in zip(
                self.vertical_info["level_offsets"],
                self.vertical_info["level_sizes"],
            ):
                max_level_budget = 0
                for token_index in range(level_offset, level_offset + level_size):
                    level_budget, _ = self._get_vertical_token_sampling_plan(
                        self.vertical_token_step_budgets[token_index]
                    )
                    if level_budget > max_level_budget:
                        max_level_budget = level_budget
                detail_x0_preds.append([[] for _ in range(max_level_budget)])

        level_offsets = self.vertical_info["level_offsets"]
        level_sizes = self.vertical_info["level_sizes"]
        level_end_indices = [
            level_offset + level_size - 1
            for level_offset, level_size in zip(level_offsets, level_sizes)
        ]

        for token_index in range(self.vertical_info["num_tokens"]):
            current_latent = noise[:, token_index:token_index + 1].clone()
            original_budget = self.vertical_token_step_budgets[token_index]
            current_budget, current_total_steps = self._get_vertical_token_sampling_plan(original_budget)
            sample_scheduler = self._initialize_sample_scheduler(noise, sampling_steps=current_total_steps)
            current_timesteps = sample_scheduler.timesteps[:current_budget]
            current_start = (token_index + 1) * self.frame_seq_length
            current_rope_frame_index = (
                self.vertical_info["representative_indices"][token_index]
                if self.vertical_use_representative_rope
                else None
            )
            current_level = self.vertical_info["token_to_level"][token_index]
            final_x0_pred = None

            for step_index, timestep_value in enumerate(tqdm(current_timesteps)):
                current_timestep = timestep_value * torch.ones(
                    [batch_size, 1],
                    device=noise.device,
                    dtype=torch.float32,
                )

                flow_pred_cond, _ = self.generator(
                    noisy_image_or_video=current_latent,
                    conditional_dict=conditional_dict,
                    timestep=current_timestep,
                    vertical_kv_cache=self.vertical_kv_cache_pos,
                    vertical_current_token_id=token_index,
                    vertical_info=self.vertical_info,
                    vertical_rope_frame_index=current_rope_frame_index,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start,
                )
                flow_pred_uncond, _ = self.generator(
                    noisy_image_or_video=current_latent,
                    conditional_dict=unconditional_dict,
                    timestep=current_timestep,
                    vertical_kv_cache=self.vertical_kv_cache_neg,
                    vertical_current_token_id=token_index,
                    vertical_info=self.vertical_info,
                    vertical_rope_frame_index=current_rope_frame_index,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start,
                )
                flow_pred = flow_pred_uncond + self.args.guidance_scale * (
                    flow_pred_cond - flow_pred_uncond
                )
                current_x0_pred = None
                if return_vertical_detail_logs or step_index == current_budget - 1:
                    if sample_scheduler.step_index is None:
                        sample_scheduler._init_step_index(timestep_value)
                    current_x0_pred = sample_scheduler.convert_model_output(
                        flow_pred,
                        sample=current_latent,
                    )
                    if return_vertical_detail_logs:
                        detail_x0_preds[current_level][step_index].append(current_x0_pred)
                    if step_index == current_budget - 1:
                        final_x0_pred = current_x0_pred
                current_latent = sample_scheduler.step(
                    flow_pred,
                    timestep_value,
                    current_latent,
                    return_dict=False,
                )[0]

            if current_budget == 0:
                final_x0_pred = current_latent.clone()
            elif final_x0_pred is None:
                raise RuntimeError("Vertical inference failed to capture the final x0 prediction for a token.")

            if current_budget > 0:
                if current_budget < len(sample_scheduler.timesteps):
                    retained_timestep_value = sample_scheduler.timesteps[current_budget]
                    retained_timestep = retained_timestep_value * torch.ones(
                        [batch_size, 1],
                        device=noise.device,
                        dtype=torch.float32,
                    )
                else:
                    retained_timestep = torch.zeros([batch_size, 1], device=noise.device, dtype=torch.float32)

                self.generator(
                    noisy_image_or_video=current_latent,
                    conditional_dict=conditional_dict,
                    timestep=retained_timestep,
                    vertical_kv_cache=self.vertical_kv_cache_pos,
                    vertical_current_token_id=token_index,
                    vertical_info=self.vertical_info,
                    vertical_cache_write=True,
                    vertical_rope_frame_index=current_rope_frame_index,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start,
                )
                self.generator(
                    noisy_image_or_video=current_latent,
                    conditional_dict=unconditional_dict,
                    timestep=retained_timestep,
                    vertical_kv_cache=self.vertical_kv_cache_neg,
                    vertical_current_token_id=token_index,
                    vertical_info=self.vertical_info,
                    vertical_cache_write=True,
                    vertical_rope_frame_index=current_rope_frame_index,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start,
                )
            generated_tokens.append(current_latent)
            x0_pred_tokens.append(final_x0_pred)

            if vertical_layer_callback is not None and token_index == level_end_indices[current_level]:
                level_offset = level_offsets[current_level]
                level_size = level_sizes[current_level]
                level_latents = torch.cat(
                    x0_pred_tokens[level_offset:level_offset + level_size],
                    dim=1,
                )
                level_video = self.vae.decode_to_pixel(level_latents)
                level_video = (level_video * 0.5 + 0.5).clamp(0, 1)
                vertical_layer_callback(current_level, level_size, level_video)

        vertical_latents = torch.cat(generated_tokens, dim=1)
        vertical_x0_preds = torch.cat(x0_pred_tokens, dim=1)
        leaf_latents = get_vertical_leaf_latents(vertical_latents, self.vertical_info)

        vertical_payload = None
        if return_vertical_layer_videos:
            layer_videos = []
            for level_offset, level_size in zip(
                self.vertical_info["level_offsets"],
                self.vertical_info["level_sizes"],
            ):
                level_latents = vertical_x0_preds[:, level_offset:level_offset + level_size]
                level_video = self.vae.decode_to_pixel(level_latents)
                level_video = (level_video * 0.5 + 0.5).clamp(0, 1)
                layer_videos.append(level_video)
            vertical_payload = {
                "layer_videos": layer_videos,
                "level_sizes": list(self.vertical_info["level_sizes"]),
            }
            if return_vertical_detail_logs:
                detail_layer_step_videos = []
                for level_step_latents in detail_x0_preds:
                    step_videos = []
                    for step_latents in level_step_latents:
                        if not step_latents:
                            continue
                        step_latents = torch.cat(step_latents, dim=1)
                        step_video = self.vae.decode_to_pixel(step_latents)
                        step_video = (step_video * 0.5 + 0.5).clamp(0, 1)
                        step_videos.append(step_video)
                    detail_layer_step_videos.append(step_videos)
                vertical_payload["detail_layer_step_videos"] = detail_layer_step_videos

        if not return_video:
            if return_vertical_layer_videos:
                if return_latents:
                    return leaf_latents, vertical_latents, vertical_payload
                return leaf_latents, vertical_payload
            if return_latents:
                return leaf_latents, vertical_latents
            return leaf_latents

        video = self.vae.decode_to_pixel(leaf_latents)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_vertical_layer_videos:
            if return_latents:
                return video, leaf_latents, vertical_payload
            return video, vertical_payload
        if return_latents:
            return video, leaf_latents
        return video
