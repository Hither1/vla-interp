"""Train Diffusion Policy from scratch on LIBERO.

Usage:
    # Auto-detect all GPUs, no torchrun needed:
    python -m dp_scratch.train --data_dir /path/to/libero_data

    # Use 4 of 8 GPUs:
    python -m dp_scratch.train --data_dir /path/to/libero_data --n_gpus 4

    # Single GPU:
    python -m dp_scratch.train --data_dir /path/to/libero_data --n_gpus 1 --batch_size 64 --lr 1e-4

    # With sim eval:
    python -m dp_scratch.train --data_dir /path/to/libero_data \
        --eval_interval 2000 --eval_suite libero_spatial
"""

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import wandb

from dp_scratch.model import DiffusionPolicy, create_ema, update_ema
from dp_scratch.dataset import LiberoDataset, compute_stats, ALL_SUITES


# ── Training worker (one per GPU) ───────────────────────────────────────────


def train_worker(rank, world_size, args):
    is_main = rank == 0

    # eval_on_save: sys.path + LIBERO_CONFIG_PATH (avoid input prompt on import -> EOFError)
    if getattr(args, "eval_on_save", False):
        import importlib.util
        proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        libero_dir = getattr(args, "libero_path", None) or os.environ.get("LIBERO_PATH")
        if not libero_dir:
            parent = os.path.dirname(proj_root)
            for name in ["LIBERO", "LIBERO-master"]:
                p = os.path.join(parent, name)
                if os.path.isdir(p) and os.path.exists(os.path.join(p, "libero")):
                    libero_dir = p
                    break
        if libero_dir:
            libero_dir = os.path.abspath(os.path.expanduser(libero_dir))
            if libero_dir not in sys.path:
                sys.path.insert(0, libero_dir)
        if importlib.util.find_spec("libero") is None:
            raise RuntimeError(
                "libero not found. Set --libero_path /path/to/LIBERO or env LIBERO_PATH"
            )
        if not libero_dir:
            spec = importlib.util.find_spec("libero")
            if spec and spec.origin:
                libero_dir = os.path.dirname(os.path.dirname(spec.origin))
        if libero_dir:
            lb = os.path.join(libero_dir, "libero", "libero")
            cfg_dir = os.path.join(proj_root, ".libero_eval_config")
            os.makedirs(cfg_dir, exist_ok=True)
            with open(os.path.join(cfg_dir, "config.yaml"), "w") as f:
                f.write(f"benchmark_root: {lb}\nbddl_files: {lb}/bddl_files\ninit_states: {lb}/init_files\ndatasets: {os.path.join(libero_dir, 'libero', 'datasets')}\nassets: {lb}/assets\n")
            os.environ["LIBERO_CONFIG_PATH"] = cfg_dir

    if world_size > 1:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", str(args.port))
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    effective_bs = args.batch_size * world_size
    if is_main:
        print(f"GPUs: {world_size} | Per-GPU BS: {args.batch_size} | Effective BS: {effective_bs}")
        print(f"lr: {args.lr} | warmup: {args.warmup_steps} steps | wd: {args.weight_decay}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    if is_main:
        print("Loading dataset...")
    dataset = LiberoDataset(args.data_dir, args.suite, verbose=is_main)

    if world_size > 1:
        if is_main:
            stats = compute_stats(dataset, verbose=True)
            stats_tensors = [torch.from_numpy(v).float().to(device) for v in [
                stats["action_mean"], stats["action_std"],
                stats["state_mean"], stats["state_std"]]]
            for t in stats_tensors:
                dist.broadcast(t, src=0)
        else:
            stats_tensors = [torch.zeros(7, device=device), torch.ones(7, device=device),
                             torch.zeros(8, device=device), torch.ones(8, device=device)]
            for t in stats_tensors:
                dist.broadcast(t, src=0)
            stats = {
                "action_mean": stats_tensors[0].cpu().numpy(),
                "action_std": stats_tensors[1].cpu().numpy(),
                "state_mean": stats_tensors[2].cpu().numpy(),
                "state_std": stats_tensors[3].cpu().numpy(),
            }
        dist.barrier()
    else:
        stats = compute_stats(dataset, verbose=is_main)

    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=True)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
            prefetch_factor=2 if args.num_workers > 0 else None,
            persistent_workers=args.num_workers > 0,
        )
    else:
        sampler = None
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
            prefetch_factor=2 if args.num_workers > 0 else None,
            persistent_workers=args.num_workers > 0,
        )

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    if is_main:
        print(f"Samples: {len(dataset)} | Steps/epoch: {steps_per_epoch} | Total steps: {total_steps}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = DiffusionPolicy(task_descs=dataset.task_descs).to(device)
    model.set_norm_stats(**stats)

    if world_size > 1:
        model = DDP(model, device_ids=[rank])
        trainable_model = model.module
    else:
        trainable_model = model

    ema_model = create_ema(trainable_model)

    if is_main:
        n_params = sum(p.numel() for p in trainable_model.parameters() if p.requires_grad)
        print(f"Trainable params: {n_params / 1e6:.1f}M")

    # ── Optimizer + LR (per-step warmup → cosine) ────────────────────────────
    optimizer = torch.optim.AdamW(
        trainable_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-2, total_iters=args.warmup_steps)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - args.warmup_steps))
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[args.warmup_steps])

    # ── Logging ──────────────────────────────────────────────────────────────
    run_name = "dp_" + "+".join(args.suite)
    writer = None
    if is_main and args.tensorboard:
        log_dir = os.path.join(args.output_dir, "+".join(args.suite), "logs")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
    if is_main and args.wandb:
        wandb.init(project="dp_libero", name=run_name, config=vars(args))

    save_dir = os.path.join(args.output_dir, "+".join(args.suite))
    if is_main:
        os.makedirs(save_dir, exist_ok=True)

    # ── Eval envs (rank 0 only, first eval_max_tasks per suite) ──
    all_eval_envs = {}
    last_eval_sr = float("nan")
    if is_main and args.eval_on_save:
        from dp_scratch.eval import setup_suite_envs, evaluate_with_envs, close_envs, get_suite_max_steps
        for s in args.suite:
            print(f"Setting up eval envs: {s} ...")
            all_eval_envs[s] = {
                "envs": setup_suite_envs(s, args.seed, max_tasks=args.eval_max_tasks),
                "max_steps": get_suite_max_steps(s),
            }
            print(f"  {len(all_eval_envs[s]['envs'])} tasks, max_steps={all_eval_envs[s]['max_steps']}")

    if world_size > 1:
        dist.barrier()

    # ── Helper: run eval on all suites, log to wandb with video ──────────────
    def run_eval(step_label, record_video=False):
        nonlocal last_eval_sr
        ema_model.eval()
        suite_results = {}
        all_videos = []
        for s, sdata in all_eval_envs.items():
            per_task, avg_sr, videos = evaluate_with_envs(
                ema_model, sdata["envs"],
                n_episodes=args.eval_episodes, max_steps=sdata["max_steps"],
                record_video=record_video,
            )
            suite_results[s] = avg_sr
            print(f"  [{s}] SR={avg_sr:.3f}")
            for tname, tsr in per_task.items():
                print(f"    {tsr:.2f} | {tname}")
            for tname, frames, done in videos:
                all_videos.append((s, tname, frames, done))
        overall = np.mean(list(suite_results.values()))
        last_eval_sr = overall
        print(f"  Overall SR={overall:.3f}")
        if writer:
            for s, sr in suite_results.items():
                writer.add_scalar(f"eval/{s}", sr, step_label)
            writer.add_scalar("eval/overall", overall, step_label)
        if args.wandb:
            log_dict = {f"eval/{s}": sr for s, sr in suite_results.items()}
            log_dict["eval/overall"] = overall
            if record_video and all_videos:
                for i, (suite, tname, frames, done) in enumerate(all_videos):
                    tag = "OK" if done else "FAIL"
                    short_name = tname[:40].replace(" ", "_")
                    vid = np.stack(frames)  # (T, H, W, 3)
                    log_dict[f"video/{suite}/{short_name}_{tag}"] = wandb.Video(
                        vid.transpose(0, 3, 1, 2), fps=20, format="mp4")
            wandb.log(log_dict, step=step_label)

    # ── Step 0 eval (before any training) ────────────────────────────────────
    if is_main and args.eval_on_save and all_eval_envs:
        print("\n[Step 0 Eval] evaluating random init model ...")
        run_eval(step_label=0, record_video=args.wandb)

    # ── Training loop ────────────────────────────────────────────────────────
    global_step = 0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not is_main, leave=True)
        for batch in pbar:
            images = batch["images"].to(device)
            state = batch["state"].to(device)
            actions = batch["actions"].to(device)
            task_idx = batch["task_idx"].to(device)

            if world_size > 1:
                loss = model.module.compute_loss(images, state, task_idx, actions)
            else:
                loss = model.compute_loss(images, state, task_idx, actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_model.parameters(), args.grad_clip)
            optimizer.step()
            lr_scheduler.step()

            update_ema(ema_model, trainable_model, decay=args.ema_decay)

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if is_main:
                cur_lr = optimizer.param_groups[0]["lr"]
                postfix = {"loss": f"{loss.item():.4f}", "lr": f"{cur_lr:.1e}"}
                if args.eval_on_save:
                    postfix["eval_sr"] = f"{last_eval_sr:.3f}"
                pbar.set_postfix(postfix)

            if is_main and global_step % args.log_interval == 0:
                cur_lr = optimizer.param_groups[0]["lr"]
                if writer:
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar("train/lr", cur_lr, global_step)
                if args.wandb:
                    wandb.log({"train/loss": loss.item(), "train/lr": cur_lr}, step=global_step)

        # ── Epoch end ──
        avg_loss = epoch_loss / max(n_batches, 1)
        if is_main:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1}/{args.epochs} | loss={avg_loss:.4f} | lr={cur_lr:.2e}")
            if writer:
                writer.add_scalar("train/epoch_loss", avg_loss, epoch + 1)
            if args.wandb:
                wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch + 1}, step=global_step)

        # ── Eval (on eval_interval_epochs or save) ──
        do_eval = is_main and args.eval_on_save and all_eval_envs
        is_save_epoch = (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1
        is_eval_epoch = args.eval_interval_epochs > 0 and (epoch + 1) % args.eval_interval_epochs == 0

        if do_eval and (is_eval_epoch or is_save_epoch):
            record_vid = args.wandb and is_save_epoch
            print(f"\n[Eval] epoch {epoch+1} ...")
            run_eval(step_label=global_step, record_video=record_vid)

        if is_main and is_save_epoch:
            ckpt = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "task_descs": dataset.task_descs,
                "stats": stats,
                "suites": args.suite,
            }
            path = os.path.join(save_dir, f"ckpt_{epoch+1}.pt")
            torch.save(ckpt, path)
            path_eval = os.path.join(save_dir, f"ckpt_{epoch+1}_eval.pt")
            torch.save({k: ckpt[k] for k in ["model", "task_descs", "stats", "suites"]}, path_eval)
            print(f"  -> saved {path} & {path_eval}")

    # ── Cleanup ──
    if is_main and all_eval_envs:
        for sdata in all_eval_envs.values():
            close_envs(sdata["envs"])
    if writer:
        writer.close()
    if world_size > 1:
        dist.destroy_process_group()


# ── Entry point ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", nargs="+", default=ALL_SUITES, choices=ALL_SUITES)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="checkpoints/dp")
    # ── Hyperparameters (defaults for 8×A100, effective batch 2048) ──
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--n_gpus", type=int, default=-1,
                        help="Number of GPUs (-1 = all available)")
    parser.add_argument("--port", type=int, default=29500,
                        help="DDP communication port")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    # ── Logging & saving ──
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers per GPU. 8x8=64 workers may exhaust /dev/shm")
    parser.add_argument("--save_interval", type=int, default=100,
                        help="Save checkpoint every N epochs")
    # ── Sim eval (lightweight during training) ──
    parser.add_argument("--eval_on_save", action="store_true",
                        help="Eval on all training suites")
    parser.add_argument("--eval_interval_epochs", type=int, default=0,
                        help="Eval every N epochs. 0=only when saving ckpt")
    parser.add_argument("--eval_max_tasks", type=int, default=2,
                        help="Eval first N tasks per suite (for speed)")
    parser.add_argument("--eval_episodes", type=int, default=2,
                        help="Episodes per task during train-time eval")
    parser.add_argument("--libero_path", default=None,
                        help="Path to LIBERO repo (for eval). Or set env LIBERO_PATH.")
    args = parser.parse_args()

    if torch.cuda.is_available():
        n_available = torch.cuda.device_count()
        world_size = n_available if args.n_gpus < 0 else min(args.n_gpus, n_available)
        print(f"Detected {n_available} GPU(s), using {world_size} for training")
    else:
        world_size = 1

    if world_size > 1:
        mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        train_worker(0, 1, args)


if __name__ == "__main__":
    main()
