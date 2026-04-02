import inspect
import os

import accelerate
import torch
from peft import LoraConfig, inject_adapter_in_model

from lightewm.model.loss import DirectDistillLoss, FlowMatchSFTLoss
from lightewm.runner.loops import launch_training_task
from lightewm.runner.base_pipeline import PipelineUnit
from lightewm.runner.runner_util.instantiation import instantiate_component_from_section
from lightewm.runner.runner_util.wan_runtime import (
    build_wan_i2v_pipeline_from_params,
    build_wan_i2v_runtime_args,
    build_wan_training_dataset,
)
from lightewm.utils.loader import load_state_dict
from lightewm.utils.logger import ModelLogger

from .periodic_validation import PeriodicWanVideoValidator

class GeneralUnit_RemoveCache(PipelineUnit):
    def __init__(
        self,
        required_params=tuple(),
        force_remove_params_shared=tuple(),
        force_remove_params_posi=tuple(),
        force_remove_params_nega=tuple(),
    ):
        super().__init__(take_over=True)
        self.required_params = required_params
        self.force_remove_params_shared = force_remove_params_shared
        self.force_remove_params_posi = force_remove_params_posi
        self.force_remove_params_nega = force_remove_params_nega

    def process_params(self, inputs, required_params, force_remove_params):
        inputs_ = {}
        for name, param in inputs.items():
            if name in required_params and name not in force_remove_params:
                inputs_[name] = param
        return inputs_

    def process(self, pipe, inputs_shared, inputs_posi, inputs_nega):
        inputs_shared = self.process_params(inputs_shared, self.required_params, self.force_remove_params_shared)
        inputs_posi = self.process_params(inputs_posi, self.required_params, self.force_remove_params_posi)
        inputs_nega = self.process_params(inputs_nega, self.required_params, self.force_remove_params_nega)
        return inputs_shared, inputs_posi, inputs_nega


class WanTrainingModule(torch.nn.Module):
    def __init__(
        self,
        pipe,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        self.pipe = pipe
        self.validation_pipe_units = list(pipe.units)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint,
            preset_lora_path,
            preset_lora_model,
            task=task,
        )
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    def to(self, *args, **kwargs):
        for _, model in self.named_children():
            model.to(*args, **kwargs)
        return self

    def trainable_modules(self):
        return filter(lambda p: p.requires_grad, self.parameters())

    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        return set([named_param[0] for named_param in trainable_param_names])

    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        if isinstance(target_modules, list) and len(target_modules) == 1:
            target_modules = target_modules[0]
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        if upcast_dtype is not None:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)
        return model

    def mapping_lora_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                new_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                new_state_dict[key] = value
        return new_state_dict

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix) :]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict

    def transfer_data_to_device(self, data, device, torch_float_dtype=None):
        if data is None:
            return data
        if isinstance(data, torch.Tensor):
            data = data.to(device)
            if torch_float_dtype is not None and data.dtype in [torch.float, torch.float16, torch.bfloat16]:
                data = data.to(torch_float_dtype)
            return data
        if isinstance(data, tuple):
            return tuple(self.transfer_data_to_device(x, device, torch_float_dtype) for x in data)
        if isinstance(data, list):
            return list(self.transfer_data_to_device(x, device, torch_float_dtype) for x in data)
        if isinstance(data, dict):
            return {i: self.transfer_data_to_device(data[i], device, torch_float_dtype) for i in data}
        return data

    def auto_detect_lora_target_modules(
        self,
        model: torch.nn.Module,
        search_for_linear=False,
        linear_detector=lambda x: min(x.weight.shape) >= 512,
        block_list_detector=lambda x: isinstance(x, torch.nn.ModuleList) and len(x) > 1,
        name_prefix="",
    ):
        lora_target_modules = []
        if search_for_linear:
            for name, module in model.named_modules():
                module_name = name_prefix + ["", "."][name_prefix != ""] + name
                if isinstance(module, torch.nn.Linear) and linear_detector(module):
                    lora_target_modules.append(module_name)
        else:
            for name, module in model.named_children():
                module_name = name_prefix + ["", "."][name_prefix != ""] + name
                lora_target_modules += self.auto_detect_lora_target_modules(
                    module,
                    search_for_linear=block_list_detector(module),
                    linear_detector=linear_detector,
                    block_list_detector=block_list_detector,
                    name_prefix=module_name,
                )
        return lora_target_modules

    def parse_lora_target_modules(self, model, lora_target_modules):
        if lora_target_modules == "":
            print("No LoRA target modules specified. The framework will automatically search for them.")
            lora_target_modules = self.auto_detect_lora_target_modules(model)
            print(f"LoRA will be patched at {lora_target_modules}.")
        else:
            lora_target_modules = lora_target_modules.split(",")
        return lora_target_modules

    def switch_pipe_to_training_mode(
        self,
        pipe,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        task="sft",
    ):
        pipe.scheduler.set_timesteps(1000, training=True)
        pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))

        if preset_lora_path is not None:
            pipe.load_lora(getattr(pipe, preset_lora_model), preset_lora_path)

        if lora_base_model is not None and not task.endswith(":data_process"):
            if (not hasattr(pipe, lora_base_model)) or getattr(pipe, lora_base_model) is None:
                print(
                    f"No {lora_base_model} models in the pipeline. "
                    "We cannot patch LoRA on the model. If this occurs during the data processing stage, it is normal."
                )
                return
            model = self.add_lora_to_model(
                getattr(pipe, lora_base_model),
                target_modules=self.parse_lora_target_modules(getattr(pipe, lora_base_model), lora_target_modules),
                lora_rank=lora_rank,
                upcast_dtype=pipe.torch_dtype,
            )
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                state_dict = self.mapping_lora_state_dict(state_dict)
                load_result = model.load_state_dict(state_dict, strict=False)
                print(f"LoRA checkpoint loaded: {lora_checkpoint}, total {len(state_dict)} keys")
                if len(load_result[1]) > 0:
                    print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")
            setattr(pipe, lora_base_model, model)

    def split_pipeline_units(
        self,
        task,
        pipe,
        trainable_models=None,
        lora_base_model=None,
        remove_unnecessary_params=False,
        loss_required_params=(
            "input_latents",
            "max_timestep_boundary",
            "min_timestep_boundary",
            "first_frame_latents",
            "video_latents",
            "audio_input_latents",
            "num_inference_steps",
        ),
        force_remove_params_shared=tuple(),
        force_remove_params_posi=tuple(),
        force_remove_params_nega=tuple(),
    ):
        models_require_backward = []
        if trainable_models is not None:
            models_require_backward += trainable_models.split(",")
        if lora_base_model is not None:
            models_require_backward += [lora_base_model]
        if task.endswith(":data_process"):
            other_units, pipe.units = pipe.split_pipeline_units(models_require_backward)
            if remove_unnecessary_params:
                required_params = list(loss_required_params) + [i for i in inspect.signature(self.pipe.model_fn).parameters]
                for unit in other_units:
                    required_params.extend(unit.fetch_input_params())
                required_params = sorted(list(set(required_params)))
                pipe.units.append(
                    GeneralUnit_RemoveCache(
                        required_params,
                        force_remove_params_shared,
                        force_remove_params_posi,
                        force_remove_params_nega,
                    )
                )
        elif task.endswith(":train"):
            pipe.units, _ = pipe.split_pipeline_units(models_require_backward)
        return pipe

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            else:
                raise ValueError(
                    f"Unsupported extra input '{extra_input}' in current Wan training pipeline. "
                    "Only 'input_image' is supported."
                )
        return inputs_shared

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        elif isinstance(inputs, (tuple, list)) and len(inputs) == 3 and isinstance(inputs[0], dict):
            inputs_shared = dict(inputs[0])
            inputs_shared["use_gradient_checkpointing"] = self.use_gradient_checkpointing
            inputs_shared["use_gradient_checkpointing_offload"] = self.use_gradient_checkpointing_offload
            inputs = (inputs_shared, inputs[1], inputs[2])
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


class WanTrainRunner:
    def __init__(self, config):
        self.config = config

    def run(self):
        args = build_wan_i2v_runtime_args(self.config.full_config)
        if not args.task:
            args.task = "sft"

        accelerator = accelerate.Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            dataloader_config=accelerate.DataLoaderConfiguration(
                use_seedable_sampler=True,
                data_seed=args.data_seed,
            ),
            kwargs_handlers=[
                accelerate.DistributedDataParallelKwargs(
                    find_unused_parameters=args.find_unused_parameters
                )
            ],
        )
        dataset = build_wan_training_dataset(args, use_data_process_controls=False)
        pipe = build_wan_i2v_pipeline_from_params(
            {
                "pipeline_class_path": self.config.full_config.model.class_path,
                "model_paths": args.model_paths,
                "model_id_with_origin_paths": args.model_id_with_origin_paths,
                "tokenizer_path": args.tokenizer_path,
                "audio_processor_path": args.audio_processor_path,
                "fp8_models": args.fp8_models,
                "offload_models": args.offload_models,
                "device": "cpu" if args.initialize_model_on_cpu else accelerator.device,
                "torch_dtype": "bfloat16",
            },
            device_override="cpu" if args.initialize_model_on_cpu else accelerator.device,
        )
        model = WanTrainingModule(
            pipe=pipe,
            trainable_models=args.trainable_models,
            lora_base_model=args.lora_base_model,
            lora_target_modules=args.lora_target_modules,
            lora_rank=args.lora_rank,
            lora_checkpoint=args.lora_checkpoint,
            preset_lora_path=args.preset_lora_path,
            preset_lora_model=args.preset_lora_model,
            use_gradient_checkpointing=args.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
            extra_inputs=args.extra_inputs,
            task=args.task,
            max_timestep_boundary=args.max_timestep_boundary,
            min_timestep_boundary=args.min_timestep_boundary,
        )
        model_logger = ModelLogger(
            args.output_path,
            remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        )
        validator = None
        validation_dataset_section = getattr(self.config.full_config, "validation_dataset", None)
        if validation_dataset_section is not None:
            validation_dataset, _ = instantiate_component_from_section(
                validation_dataset_section,
                self.config.full_config,
                section_name="validation_dataset",
            )
            validator = PeriodicWanVideoValidator(
                dataset=validation_dataset,
                output_root=args.output_path,
                every_steps=int(getattr(args, "validation_every_steps", 1000)),
                extra_steps=getattr(args, "validation_extra_steps", []),
                num_samples=int(getattr(args, "validation_num_samples", 3)),
                fps=int(getattr(args, "validation_fps", 16)),
                quality=int(getattr(args, "validation_quality", 5)),
                seed_base=int(getattr(args, "validation_seed_base", 0)),
                infer_kwargs=getattr(args, "validation_infer_kwargs", {}),
                input_image_resize_mode=getattr(args, "validation_input_image_resize_mode", "stretch"),
            )
        launch_training_task(accelerator, dataset, model, model_logger, validator=validator, args=args)
