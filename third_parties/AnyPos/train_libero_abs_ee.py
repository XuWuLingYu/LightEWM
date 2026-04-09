import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from idm.idm import IDM
from idm.libero_abs_ee_dataset import (
    DEFAULT_SUITES,
    LiberoAbsoluteEEDataset,
    extract_single_arm_regions,
    extract_official_split_regions,
)
from idm.loss import WeightedSmoothL1Loss
from idm.preprocessor import DinoPreprocessor


def parse_box(box_str):
    values = [float(x) for x in box_str.split(",")]
    if len(values) != 4:
        raise ValueError(f"Expected 4 comma-separated values, got: {box_str}")
    return tuple(values)


def parse_args():
    parser = argparse.ArgumentParser(description="Train AnyPos on LIBERO absolute end-effector states.")
    parser.add_argument("--libero_root", type=str, required=True, help="Path to raw LIBERO hdf5 root.")
    parser.add_argument("--suites", type=str, default=",".join(DEFAULT_SUITES), help="Comma-separated LIBERO suites.")
    parser.add_argument("--camera_key", type=str, default="agentview_rgb", help="RGB key inside obs.")
    parser.add_argument("--target_key", type=str, default="ee_states", help="Target key inside obs.")
    parser.add_argument("--model_name", type=str, default="direction_aware", choices=["direction_aware", "direction_aware_with_single_arm_split"])
    parser.add_argument("--load_from", type=str, default=None, help="Checkpoint path.")
    parser.add_argument("--wandb_mode", type=str, default="online", help="Wandb mode.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=32, help="Train batch size per GPU.")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Eval batch size per GPU.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Prefetch factor.")
    parser.add_argument("--num_iterations", type=int, default=100000, help="Number of iterations.")
    parser.add_argument("--eval_interval", type=int, default=2000, help="Evaluation interval.")
    parser.add_argument("--save_interval", type=int, default=2000, help="Checkpoint interval.")
    parser.add_argument("--run_name", type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), help="Run name.")
    parser.add_argument("--save_dir", type=str, default="logs/libero_idm", help="Output directory.")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Episode-level validation ratio.")
    parser.add_argument("--test_ratio", type=float, default=0.05, help="Episode-level test ratio.")
    parser.add_argument("--frame_stride", type=int, default=1, help="Sample every N-th frame.")
    parser.add_argument("--min_episode_len", type=int, default=30, help="Skip episodes shorter than this.")
    parser.add_argument("--use_transform", action="store_true", default=False, help="Enable image augmentation.")
    parser.add_argument("--freeze_dinov2", action="store_true", default=False, help="Freeze DINOv2 backbone.")
    parser.add_argument("--dinov2_name", type=str, default="facebook/dinov2-with-registers-base", help="Backbone name.")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay.")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["constant", "cosine"], help="LR scheduler type.")
    parser.add_argument("--use_normalization", action="store_true", default=True, help="Train in normalized target space.")
    parser.add_argument("--eval_only", action="store_true", default=False, help="Only run evaluation.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--workspace_box", type=str, default="0.16,0.08,0.88,0.84", help="Normalized crop box x0,y0,x1,y1 for workspace crop.")
    parser.add_argument("--arm_box", type=str, default="0.42,0.18,0.98,0.98", help="Normalized crop box x0,y0,x1,y1 for arm crop.")
    parser.add_argument("--close_thresholds", type=str, default="0.03,0.03,0.03,0.12,0.12,0.12", help="Per-dim thresholds for correct-rate.")
    return parser.parse_args()


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_data_generator(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def _process_images(images, preprocessor, model_name, workspace_box, arm_box):
    if model_name == "direction_aware_with_single_arm_split":
        left_arm_images = []
        right_arm_images = []
        for image in images:
            _, _, regions = extract_official_split_regions(image)
            left_arm_images.append(regions[0])
            right_arm_images.append(regions[2])
        left_tensor = preprocessor.process_batch(left_arm_images)
        right_tensor = preprocessor.process_batch(right_arm_images)
        return torch.stack([left_tensor, right_tensor], dim=0)
    return preprocessor.process_batch(images)


def collate_fn(batch, preprocessor, model_name, workspace_box, arm_box):
    targets, images = zip(*batch)
    images = _process_images(images, preprocessor, model_name, workspace_box, arm_box)
    targets = torch.stack(targets)
    return images, targets


def save_model(accelerator, net, optimizer, step, save_path):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": accelerator.unwrap_model(net).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
            },
            save_path,
        )
    accelerator.wait_for_everyone()


def compute_is_close(target, output, thresholds):
    limits = thresholds.to(target.device)
    if target.dim() == 1:
        return torch.all(torch.abs(target - output) < limits)
    return torch.all(torch.abs(target - output) < limits, dim=1)


def eval_loop(accelerator, net, dataloader, loss_fn, step, use_normalization, thresholds, mode="val"):
    accelerator.wait_for_everyone()
    net.eval()
    total_loss = 0.0
    total_l1 = 0.0
    total_correct = 0.0
    total_samples = 0
    with torch.no_grad():
        for images, targets in tqdm(dataloader, disable=not accelerator.is_main_process):
            gathered_targets = accelerator.gather(targets)
            outputs = net(images)
            gathered_outputs = accelerator.gather(outputs)
            if accelerator.is_main_process:
                if use_normalization:
                    loss = loss_fn(net.normalize(gathered_outputs), net.normalize(gathered_targets))
                else:
                    loss = loss_fn(gathered_outputs, gathered_targets)
                total_loss += loss.item() * len(gathered_targets)
                total_l1 += torch.abs(gathered_targets - gathered_outputs).mean(dim=1).sum().item()
                total_correct += compute_is_close(gathered_targets, gathered_outputs, thresholds).float().sum().item()
                total_samples += len(gathered_targets)
    if accelerator.is_main_process and total_samples > 0:
        metrics = {
            f"{mode}_loss": total_loss / total_samples,
            f"{mode}_l1_error": total_l1 / total_samples,
            f"{mode}_correct_rate": total_correct / total_samples,
        }
        print(f"{mode}: loss={metrics[f'{mode}_loss']:.4f}, l1={metrics[f'{mode}_l1_error']:.4f}, correct={metrics[f'{mode}_correct_rate']:.4f}")
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    net.train()
    accelerator.wait_for_everyone()


def main():
    args = parse_args()
    seed_everything(args.seed)
    workspace_box = parse_box(args.workspace_box)
    arm_box = parse_box(args.arm_box)
    thresholds = torch.tensor([float(x) for x in args.close_thresholds.split(",")], dtype=torch.float32)
    suites = [suite.strip() for suite in args.suites.split(",") if suite.strip()]

    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    save_dir = os.path.join(args.save_dir, args.run_name)
    if accelerator.is_main_process and not args.eval_only:
        os.makedirs(save_dir, exist_ok=True)
        wandb.init(
            project="IDM_LIBERO_abs_ee",
            mode=args.wandb_mode,
            config=args.__dict__,
            name=args.run_name,
        )

    preprocessor = DinoPreprocessor(args)
    train_dataset = LiberoAbsoluteEEDataset(
        libero_root=args.libero_root,
        split="train",
        suites=suites,
        camera_key=args.camera_key,
        target_key=args.target_key,
        frame_stride=args.frame_stride,
        min_episode_len=args.min_episode_len,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    val_dataset = LiberoAbsoluteEEDataset(
        libero_root=args.libero_root,
        split="val",
        suites=suites,
        camera_key=args.camera_key,
        target_key=args.target_key,
        frame_stride=args.frame_stride,
        min_episode_len=args.min_episode_len,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    test_dataset = LiberoAbsoluteEEDataset(
        libero_root=args.libero_root,
        split="test",
        suites=suites,
        camera_key=args.camera_key,
        target_key=args.target_key,
        frame_stride=args.frame_stride,
        min_episode_len=args.min_episode_len,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    train_mean, train_std = train_dataset.compute_target_stats()
    output_dim = train_dataset.target_dim
    if thresholds.numel() != output_dim:
        raise ValueError(f"close_thresholds expects {output_dim} values, got {thresholds.numel()}")

    if accelerator.is_main_process:
        print(
            {
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "test_samples": len(test_dataset),
                "output_dim": output_dim,
                "train_mean": train_mean.tolist(),
                "train_std": train_std.tolist(),
                "workspace_box": workspace_box,
                "arm_box": arm_box,
            }
        )

    use_collate_fn = lambda batch: collate_fn(batch, preprocessor, args.model_name, workspace_box, arm_box)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=use_collate_fn,
        drop_last=True,
        prefetch_factor=args.prefetch_factor,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=use_collate_fn,
        drop_last=False,
        prefetch_factor=args.prefetch_factor,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=use_collate_fn,
        drop_last=False,
        prefetch_factor=args.prefetch_factor,
    )

    net = IDM(
        model_name=args.model_name,
        dinov2_name=args.dinov2_name,
        freeze_dinov2=args.freeze_dinov2,
        output_dim=output_dim,
        train_mean=train_mean,
        train_std=train_std,
    )

    base_params = []
    dino_params = []
    for name, param in net.named_parameters():
        if "dino_model" in name:
            dino_params.append(param)
        else:
            base_params.append(param)
    optimizer = AdamW(
        [
            {"params": dino_params, "lr": args.learning_rate * 0.1},
            {"params": base_params, "lr": args.learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    loss_fn = WeightedSmoothL1Loss(beta=0.1, output_dim=output_dim)

    if args.lr_scheduler == "cosine":
        num_gpus = max(torch.cuda.device_count(), 1)
        warmup_steps = int(0.1 * args.num_iterations)

        def lr_lambda(step):
            step = step // num_gpus
            eta_min = 1e-9
            if step < warmup_steps:
                return eta_min + float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, args.num_iterations - warmup_steps))
            return 0.5 * (np.cos(progress * np.pi) + 1)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    start_step = 0
    if args.load_from:
        loaded = torch.load(args.load_from, map_location="cpu")
        net.load_state_dict(loaded["model_state_dict"])
        if not args.eval_only and "optimizer_state_dict" in loaded:
            optimizer.load_state_dict(loaded["optimizer_state_dict"])
            start_step = int(loaded.get("step", 0))
            if scheduler is not None:
                for _ in range(start_step):
                    scheduler.step()
        if accelerator.is_main_process:
            print(f"Loaded checkpoint from {args.load_from}")

    net, optimizer, train_dataloader, val_dataloader, test_dataloader = accelerator.prepare(
        net, optimizer, train_dataloader, val_dataloader, test_dataloader
    )
    net.normalize = accelerator.unwrap_model(net).normalize
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)

    if args.eval_only:
        preprocessor.use_transform = False
        eval_loop(accelerator, net, val_dataloader, loss_fn, 0, args.use_normalization, thresholds, mode="val")
        eval_loop(accelerator, net, test_dataloader, loss_fn, 0, args.use_normalization, thresholds, mode="test")
        return

    train_generator = get_data_generator(train_dataloader)
    pbar = tqdm(range(start_step, args.num_iterations), disable=not accelerator.is_main_process)
    for step in pbar:
        images, targets = next(train_generator)
        outputs = net(images)
        if args.use_normalization:
            loss = loss_fn(net.normalize(outputs), net.normalize(targets))
        else:
            loss = loss_fn(outputs, targets)
        batch_accuracy = compute_is_close(targets, outputs, thresholds.to(targets.device)).float().mean().item()

        optimizer.zero_grad()
        accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if accelerator.is_main_process:
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}", batch_acc=f"{batch_accuracy:.4f}")
            if wandb.run is not None and step % 10 == 0:
                wandb.log(
                    {
                        "loss": loss.item(),
                        "learning_rate": current_lr,
                        "batch_accuracy": batch_accuracy,
                    },
                    step=step,
                )

        if (step + 1) % args.eval_interval == 0:
            preprocessor.use_transform = False
            eval_loop(accelerator, net, val_dataloader, loss_fn, step + 1, args.use_normalization, thresholds, mode="val")
            eval_loop(accelerator, net, test_dataloader, loss_fn, step + 1, args.use_normalization, thresholds, mode="test")
            preprocessor.use_transform = args.use_transform

        if (step + 1) % args.save_interval == 0:
            save_model(accelerator, net, optimizer, step + 1, os.path.join(save_dir, f"{step + 1}.pt"))

    if accelerator.is_main_process and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
