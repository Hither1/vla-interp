"""Train Diffusion Policy from scratch on LIBERO.

Usage:
    # 8 GPUs:
    torchrun --nproc_per_node=8 -m dp_scratch.train --data_dir /path/to/libero_data

    # Single GPU:
    python -m dp_scratch.train --data_dir /path/to/libero_data

    # Train on specific suites:
    torchrun --nproc_per_node=8 -m dp_scratch.train --suite libero_spatial libero_goal --data_dir /path/to/libero_data
"""

import argparse
import os

import numpy as np
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import wandb

from dp_scratch.model import DiffusionPolicy, create_ema, update_ema
from dp_scratch.dataset import LiberoDataset, compute_stats, ALL_SUITES


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", nargs="+", default=ALL_SUITES,
                        choices=ALL_SUITES)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="checkpoints/dp")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--tensorboard", action="store_true", help="Enable tensorboard logging")
    parser.add_argument("--log_interval", type=int, default=100, help="Log every N batches")
    parser.add_argument("--n_gpus", type=int, default=8, help="Number of GPUs to use for data parallel")
    parser.add_argument("--num_workers", type=int, default=16, help="DataLoader workers, 0 if HDF5 multi-read hangs")
    parser.add_argument("--save_interval", type=int, default=10, help="Save checkpoint every N epochs")
    args = parser.parse_args()

    rank, world_size, local_rank = setup_ddp()
    is_main = rank == 0

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    if torch.cuda.is_available():
        n_gpus = world_size
        device = torch.device(f"cuda:{local_rank}")
        if is_main:
            print(f"Training with {n_gpus} GPUs (DDP)")
    else:
        n_gpus = 1
        device = torch.device(args.device)
        if is_main:
            print("Training on CPU")

    # Dataset (loads all specified suites)
    if is_main:
        print("Loading dataset...")
    dataset = LiberoDataset(args.data_dir, args.suite, verbose=is_main)

    # Only rank 0 computes stats, then broadcast to other processes (avoid redundant computation)
    if n_gpus > 1:
        if is_main:
            stats = compute_stats(dataset, verbose=True)
            stats_tensors = [torch.from_numpy(v).float().cuda() for v in [stats["action_mean"], stats["action_std"], stats["state_mean"], stats["state_std"]]]
            for t in stats_tensors:
                dist.broadcast(t, src=0)
        else:
            stats_tensors = [torch.zeros(7).cuda(), torch.ones(7).cuda(), torch.zeros(8).cuda(), torch.ones(8).cuda()]
            for t in stats_tensors:
                dist.broadcast(t, src=0)
            stats = {"action_mean": stats_tensors[0].cpu().numpy(), "action_std": stats_tensors[1].cpu().numpy(),
                     "state_mean": stats_tensors[2].cpu().numpy(), "state_std": stats_tensors[3].cpu().numpy()}
        dist.barrier()
    else:
        if is_main:
            print("Computing normalization stats...")
        stats = compute_stats(dataset, verbose=is_main)

    num_workers = args.num_workers
    if n_gpus > 1:
        sampler = DistributedSampler(dataset, shuffle=True)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )
    else:
        sampler = None
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )

    # Model
    model = DiffusionPolicy(task_descs=dataset.task_descs).to(device)
    model.set_norm_stats(**stats)

    if n_gpus > 1:
        model = DDP(model, device_ids=[local_rank])
        trainable_model = model.module
    else:
        trainable_model = model

    ema_model = create_ema(trainable_model)

    optimizer = torch.optim.AdamW(trainable_model.parameters(), lr=args.lr, weight_decay=1e-6)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Logging (only rank 0)
    run_name = "dp_" + "+".join(args.suite)
    writer = None
    if is_main and args.tensorboard:
        log_dir = os.path.join(args.output_dir, "+".join(args.suite), "logs")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs saved to: {log_dir}")

    if is_main and args.wandb:
        wandb.init(project="dp_libero", name=run_name, config=vars(args))
        print("WandB logging enabled.")

    save_dir = os.path.join(args.output_dir, "+".join(args.suite))
    if is_main:
        os.makedirs(save_dir, exist_ok=True)

    if n_gpus > 1:
        dist.barrier()

    # Training loop
    global_step = 0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not is_main, leave=True)
        for batch_idx, batch in enumerate(pbar):
            images = batch["images"].to(device)
            state = batch["state"].to(device)
            actions = batch["actions"].to(device)
            task_idx = batch["task_idx"].to(device)

            loss = model.module.compute_loss(images, state, task_idx, actions) if n_gpus > 1 else model.compute_loss(images, state, task_idx, actions)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_model.parameters(), 1.0)
            optimizer.step()

            update_ema(ema_model, trainable_model, decay=0.995)

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if is_main:
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            if is_main and global_step % args.log_interval == 0:
                if writer:
                    writer.add_scalar("train/batch_loss", loss.item(), global_step)
                    writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                if args.wandb:
                    wandb.log({"train/batch_loss": loss.item(), "train/learning_rate": optimizer.param_groups[0]["lr"]}, step=global_step)

        lr_scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        lr_now = lr_scheduler.get_last_lr()[0]
        if is_main:
            print(f"Epoch {epoch+1}/{args.epochs} | loss={avg_loss:.4f} | lr={lr_now:.2e}")

        if is_main:
            if writer:
                writer.add_scalar("train/epoch_loss", avg_loss, epoch + 1)
                writer.add_scalar("train/learning_rate", lr_now, epoch + 1)
            if args.wandb:
                wandb.log({"train/epoch_loss": avg_loss, "train/learning_rate": lr_now, "train/epoch": epoch + 1}, step=global_step)

        if is_main and ((epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1):
            ckpt = {
                "epoch": epoch + 1,
                "model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "task_descs": dataset.task_descs,
                "stats": stats,
                "suites": args.suite,
            }
            path = os.path.join(save_dir, f"ckpt_{epoch+1}.pt")
            torch.save(ckpt, path)
            print(f"  -> saved {path}")

    if writer:
        writer.close()
    cleanup_ddp()


if __name__ == "__main__":
    main()