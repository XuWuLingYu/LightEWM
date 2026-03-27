import os, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from ..utils.logger import ModelLogger


def _maybe_init_wandb(args, accelerator: Accelerator):
    enabled = bool(getattr(args, "wandb_enabled", False)) if args is not None else False
    if not enabled or not accelerator.is_main_process:
        return None
    try:
        import wandb
    except Exception as exc:
        print(f"[WandB] wandb is not available, skip logging. Error: {exc}")
        return None
    project = getattr(args, "wandb_project", "LightEWM")
    run_name = getattr(args, "wandb_run_name", None)
    mode = getattr(args, "wandb_mode", "online")
    cfg = {}
    if args is not None:
        for key, value in vars(args).items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                cfg[key] = value
    return wandb.init(project=project, name=run_name, mode=mode, config=cfg)


def _validate_cached_sample(sample, expected_latent_frames: int = 21):
    if not (isinstance(sample, (tuple, list)) and len(sample) == 3 and all(isinstance(x, dict) for x in sample)):
        return False, "cache tuple format is invalid"
    shared = sample[0]
    input_latents = shared.get("input_latents")
    y = shared.get("y")
    if not isinstance(input_latents, torch.Tensor):
        return False, f"input_latents is {type(input_latents).__name__}"
    if not isinstance(y, torch.Tensor):
        return False, f"y is {type(y).__name__}"
    if input_latents.ndim != 5 or y.ndim != 5:
        return False, f"ndim mismatch input_latents={input_latents.ndim}, y={y.ndim}"
    if input_latents.shape[2] != expected_latent_frames or y.shape[2] != expected_latent_frames:
        return False, f"time mismatch input_latents={input_latents.shape[2]}, y={y.shape[2]}, expected={expected_latent_frames}"
    for dim in (0, 2, 3, 4):
        if input_latents.shape[dim] != y.shape[dim]:
            return False, f"shape mismatch at dim={dim}: input_latents={tuple(input_latents.shape)}, y={tuple(y.shape)}"
    return True, ""


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    batch_size: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        batch_size = max(1, int(getattr(args, "batch_size", 1)))
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=lambda x: x,
        num_workers=num_workers,
    )
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    wandb_run = _maybe_init_wandb(args, accelerator)
    wandb_log_every = int(getattr(args, "wandb_log_every", 10)) if args is not None else 10
    global_step = 0
    initialize_deepspeed_gradient_checkpointing(accelerator)
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                batch_items = data if isinstance(data, list) else [data]
                losses = []
                for sample in batch_items:
                    if dataset.load_from_cache:
                        loss_sample = model({}, inputs=sample)
                    else:
                        loss_sample = model(sample)
                    losses.append(loss_sample)
                loss = torch.stack(losses).mean()
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                scheduler.step()
                global_step += 1
                should_log = global_step % max(1, wandb_log_every) == 0
                if should_log:
                    # Distributed collective must be called on all ranks.
                    gathered_loss = accelerator.gather(loss.detach())
                if wandb_run is not None and should_log:
                    loss_value = gathered_loss.mean().item()
                    lr = optimizer.param_groups[0]["lr"]
                    wandb_run.log({"train/loss": loss_value, "train/lr": lr, "train/epoch": epoch_id}, step=global_step)
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
            if wandb_run is not None:
                wandb_run.log({"train/epoch_end": epoch_id}, step=global_step)
    model_logger.on_training_end(accelerator, model, save_steps)
    if wandb_run is not None:
        wandb_run.finish()


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
    expected_latent_frames = int(getattr(args, "expected_latent_frames", 21)) if args is not None else 21
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, dataloader = accelerator.prepare(model, dataloader)
    saved_count = 0
    skipped_count = 0
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                data = model(data)
                valid, reason = _validate_cached_sample(data, expected_latent_frames=expected_latent_frames)
                if not valid:
                    skipped_count += 1
                    if skipped_count <= 20:
                        print(f"[DataProcess][rank{accelerator.process_index}] Skip sample {data_id}: {reason}")
                    continue
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{saved_count}.pth")
                torch.save(data, save_path)
                saved_count += 1

    local_stats = torch.tensor([saved_count, skipped_count], device=accelerator.device, dtype=torch.long)
    all_stats = accelerator.gather(local_stats).reshape(-1, 2)
    if accelerator.is_main_process:
        total_saved = int(all_stats[:, 0].sum().item())
        total_skipped = int(all_stats[:, 1].sum().item())
        print(
            f"[DataProcess] Completed cache generation. "
            f"Saved={total_saved}, SkippedInvalid={total_skipped}, ExpectedLatentFrames={expected_latent_frames}"
        )


def initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator):
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if "activation_checkpointing" in ds_config:
            import deepspeed
            act_config = ds_config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None, 
                partition_activations=act_config.get("partition_activations", False),
                checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False)
            )
        else:
            print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")
