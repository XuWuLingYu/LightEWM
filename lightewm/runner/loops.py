import json
import os, torch
from tqdm import tqdm
from accelerate import Accelerator
from ..utils.logger import ModelLogger


def _maybe_init_wandb(args, accelerator: Accelerator):
    enabled = bool(getattr(args, "wandb_enabled", False)) if args is not None else False
    env_rank = os.environ.get("RANK", "0")
    is_global_rank_zero = str(env_rank) == "0"
    if not enabled:
        print(f"[WandB] rank={env_rank} disabled by config")
        return None
    if not accelerator.is_main_process:
        print(f"[WandB] rank={env_rank} skipped because accelerator.is_main_process is false")
        return None
    if not is_global_rank_zero:
        print(f"[WandB] rank={env_rank} skipped because only global rank 0 initializes wandb")
        return None
    try:
        import wandb
    except Exception as exc:
        print(f"[WandB] wandb is not available, skip logging. Error: {exc}")
        return None
    project = getattr(args, "wandb_project", "LightEWM")
    run_name = getattr(args, "wandb_run_name", None)
    mode = getattr(args, "wandb_mode", "online")
    print(f"[WandB] rank={env_rank} initializing. project={project} run_name={run_name} mode={mode}")
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
    sample_num_frames = shared.get("num_frames")
    if sample_num_frames is not None:
        try:
            sample_num_frames = int(sample_num_frames)
            if sample_num_frames > 0:
                expected_latent_frames = (sample_num_frames - 1) // 4 + 1
        except Exception:
            pass
    input_latents = shared.get("input_latents")
    y = shared.get("y")
    first_frame_latents = shared.get("first_frame_latents")
    if not isinstance(input_latents, torch.Tensor):
        return False, f"input_latents is {type(input_latents).__name__}"
    if input_latents.ndim != 5:
        return False, f"input_latents ndim is {input_latents.ndim}"
    if input_latents.shape[2] != expected_latent_frames:
        return False, f"time mismatch input_latents={input_latents.shape[2]}, expected={expected_latent_frames}"
    if isinstance(y, torch.Tensor):
        if y.ndim != 5:
            return False, f"y ndim is {y.ndim}"
        if y.shape[2] != expected_latent_frames:
            return False, f"time mismatch y={y.shape[2]}, expected={expected_latent_frames}"
        for dim in (0, 2, 3, 4):
            if input_latents.shape[dim] != y.shape[dim]:
                return False, f"shape mismatch at dim={dim}: input_latents={tuple(input_latents.shape)}, y={tuple(y.shape)}"
    elif isinstance(first_frame_latents, torch.Tensor):
        if first_frame_latents.ndim != 5:
            return False, f"first_frame_latents ndim is {first_frame_latents.ndim}"
        if first_frame_latents.shape[2] != 1:
            return False, f"first_frame_latents time dim is {first_frame_latents.shape[2]}, expected=1"
        for dim in (0, 1, 3, 4):
            if input_latents.shape[dim] != first_frame_latents.shape[dim]:
                return (
                    False,
                    "shape mismatch between input_latents and first_frame_latents: "
                    f"{tuple(input_latents.shape)} vs {tuple(first_frame_latents.shape)}",
                )
    else:
        return False, "cache sample misses both y and first_frame_latents"
    return True, ""


def _extract_data_error(sample):
    if isinstance(sample, dict) and "__load_error__" in sample:
        return sample
    return None


def _distributed_rank_id(accelerator: Accelerator) -> int:
    world_size_env = os.environ.get("WORLD_SIZE")
    rank_env = os.environ.get("RANK")
    if world_size_env is not None and rank_env is not None:
        try:
            world_size = int(world_size_env)
            rank = int(rank_env)
            if world_size > 0 and 0 <= rank < world_size:
                return rank
        except Exception:
            pass
    return int(accelerator.process_index)


def _write_bad_sample_record(output_root: str, process_index: int, data_id: int, sample: dict):
    quarantine_dir = os.path.join(output_root, "_bad_quarantine", f"rank{int(process_index)}")
    os.makedirs(quarantine_dir, exist_ok=True)
    quarantine_path = os.path.join(quarantine_dir, f"{int(data_id):08d}.json")
    payload = {
        "data_id": int(data_id),
        "process_index": int(process_index),
        "error_type": sample.get("__load_error_type__", "UnknownError"),
        "error": sample.get("__load_error__", ""),
        "traceback": sample.get("__load_error_traceback__", ""),
        "load_from_cache": bool(sample.get("__load_from_cache__", False)),
        "source_item": sample.get("__source_item__", None),
    }
    with open(quarantine_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, default=str, indent=2)
    return quarantine_path


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    model_logger: ModelLogger,
    validator = None,
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
    if getattr(dataset, "load_from_cache_local_shard", False):
        deepspeed_plugin = getattr(accelerator.state, "deepspeed_plugin", None)
        if deepspeed_plugin is not None:
            deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = int(batch_size)
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
        if accelerator.is_main_process:
            print("[Train] Using local cache shard mode. Skip accelerate dataloader sharding.")
    else:
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
                    load_error = _extract_data_error(sample)
                    if load_error is not None:
                        quarantine_path = _write_bad_sample_record(model_logger.output_path, accelerator.process_index, global_step, load_error)
                        print(
                            f"[Train][ALERT][rank{accelerator.process_index}] "
                            f"Skip broken sample at step={global_step}. "
                            f"error={load_error.get('__load_error__', '')} "
                            f"quarantine={quarantine_path}"
                        )
                        continue
                    if dataset.load_from_cache:
                        loss_sample = model({}, inputs=sample)
                    else:
                        loss_sample = model(sample)
                    losses.append(loss_sample)
                if len(losses) == 0:
                    continue
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
                if validator is not None:
                    validator.wandb_run = wandb_run
                    validator.maybe_run(accelerator, model, global_step)
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
    model: torch.nn.Module,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
    configured_num_frames = int(getattr(args, "num_frames", 0)) if args is not None else 0
    if configured_num_frames > 0:
        expected_latent_frames = (configured_num_frames - 1) // 4 + 1
    else:
        expected_latent_frames = int(getattr(args, "expected_latent_frames", 21)) if args is not None else 21
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, dataloader = accelerator.prepare(model, dataloader)
    saved_count = 0
    skipped_invalid_count = 0
    skipped_broken_count = 0
    rank_id = _distributed_rank_id(accelerator)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(rank_id))
                os.makedirs(folder, exist_ok=True)
                load_error = _extract_data_error(data)
                if load_error is not None:
                    skipped_broken_count += 1
                    quarantine_path = _write_bad_sample_record(model_logger.output_path, rank_id, data_id, load_error)
                    print(
                        f"[DataProcess][ALERT][rank{rank_id}] "
                        f"Broken sample {data_id} skipped. "
                        f"error={load_error.get('__load_error__', '')} "
                        f"quarantine={quarantine_path}"
                    )
                    continue
                data = model(data)
                valid, reason = _validate_cached_sample(data, expected_latent_frames=expected_latent_frames)
                if not valid:
                    skipped_invalid_count += 1
                    if skipped_invalid_count <= 20:
                        print(f"[DataProcess][rank{rank_id}] Skip sample {data_id}: {reason}")
                    continue
                save_path = os.path.join(model_logger.output_path, str(rank_id), f"{saved_count}.pth")
                torch.save(data, save_path)
                saved_count += 1

    local_stats = torch.tensor([saved_count, skipped_invalid_count, skipped_broken_count], device=accelerator.device, dtype=torch.long)
    all_stats = accelerator.gather(local_stats).reshape(-1, 3)
    if accelerator.is_main_process:
        total_saved = int(all_stats[:, 0].sum().item())
        total_skipped_invalid = int(all_stats[:, 1].sum().item())
        total_skipped_broken = int(all_stats[:, 2].sum().item())
        print(
            f"[DataProcess] Completed cache generation. "
            f"Saved={total_saved}, SkippedInvalid={total_skipped_invalid}, SkippedBroken={total_skipped_broken}, "
            f"ConfigNumFrames={configured_num_frames if configured_num_frames > 0 else 'unknown'}, "
            f"ExpectedLatentFrames={expected_latent_frames}"
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
