#!/usr/bin/env python3
"""
Evaluate training loss of a DreamZero finetuned checkpoint on training data.

Loads the finetuned LoRA checkpoint, recreates the training dataset exactly as
during training (same transforms, normalisation, collator), runs forward passes,
and prints / plots the per-batch loss breakdown.

Usage (must be launched with torchrun because the dataset requires dist):
    torchrun --standalone --nproc_per_node=2 analysis/dreamzero_ckpt_loss.py \
        --ckpt-path /n/netscratch/sham_lab/Lab/chloe00/libero/dreamzero_libero_all_lora/DreamZero_libero/checkpoint-3400 \
        --num-batches 100

    # quick smoke-test (1 GPU, 5 batches):
    torchrun --standalone --nproc_per_node=1 analysis/dreamzero_ckpt_loss.py \
        --ckpt-path /n/netscratch/sham_lab/Lab/chloe00/libero/dreamzero_libero_all_lora/DreamZero_libero/checkpoint-3400 \
        --num-batches 5

Outputs (rank-0 only):
    analysis/loss_results/loss_results.json   - raw per-batch numbers
    analysis/loss_results/loss_curves.png     - time-series + histogram plots
"""

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import OmegaConf

# ── path setup ───────────────────────────────────────────────────────────────
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
_DREAMZERO_DIR = str(_ROOT_DIR / "dreamzero")
if _DREAMZERO_DIR not in sys.path:
    sys.path.insert(0, _DREAMZERO_DIR)

from groot.vla.model.dreamzero.base_vla import VLA


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-path", required=True,
                   help="Checkpoint directory (contains model.safetensors + experiment_cfg/)")
    p.add_argument("--num-batches", type=int, default=50,
                   help="Number of training batches to evaluate (default 50)")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Batch size per process (default 1)")
    p.add_argument("--num-workers", type=int, default=1,
                   help="DataLoader workers (default 1)")
    p.add_argument("--output-dir", default=str(_THIS_DIR / "loss_results"),
                   help="Directory for JSON results + PNG plot")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device=device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def main():
    args = parse_args()

    # ── distributed init ─────────────────────────────────────────────────────
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)

    ckpt_path = pathlib.Path(args.ckpt_path)
    exp_cfg_dir = ckpt_path / "experiment_cfg"
    train_cfg = OmegaConf.load(exp_cfg_dir / "conf.yaml")

    # ── 1. Load model ────────────────────────────────────────────────────────
    # The checkpoint was saved with save_lora_only=True, so model.safetensors
    # holds only the LoRA delta weights.  VLA.load_lora instantiates the full
    # base model (loading pretrained Wan weights from the paths stored in
    # config.json) and then overlays the LoRA parameters.
    if rank == 0:
        print(f"\n[model] Loading LoRA checkpoint from {ckpt_path} …")

    model = VLA.load_lora(str(ckpt_path))
    model = model.to(device=device, dtype=torch.bfloat16)

    # Move RoPE frequency buffers to the correct device (required after .to())
    model.post_initialize()

    # eval mode: disables dropout; does NOT affect the stochastic flow-matching
    # timestep sampling used for loss computation.
    model.eval()

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[model] Loaded – {n_params/1e9:.2f}B parameters on {device}")

    # ── 2. Create training dataset ───────────────────────────────────────────
    # Replicate the exact dataset used during training (same paths, modality
    # configs, transforms, normalisation stats).
    if rank == 0:
        print("\n[data] Initialising training dataset …")

    train_dataset = instantiate(train_cfg.train_dataset)

    if rank == 0:
        print(f"[data] Dataset ready: {type(train_dataset).__name__}")

    # ── 3. Data collator ─────────────────────────────────────────────────────
    data_collator = instantiate(train_cfg.data_collator)

    # ── 4. DataLoader ────────────────────────────────────────────────────────
    # ShardedLeRobotMixtureDataset is an IterableDataset that already handles
    # per-rank / per-worker shard assignment internally.
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ── 5. Compute loss ──────────────────────────────────────────────────────
    total_losses: list[float] = []
    action_losses: list[float] = []
    dynamics_losses: list[float] = []

    if rank == 0:
        print(f"\n[eval] Running {args.num_batches} batches (batch_size={args.batch_size}) …")

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if step >= args.num_batches:
                break

            batch = move_batch_to_device(batch, device)

            outputs = model(batch)

            total_losses.append(outputs["loss"].item())
            if "action_loss" in outputs:
                action_losses.append(outputs["action_loss"].item())
            if "dynamics_loss" in outputs:
                dynamics_losses.append(outputs["dynamics_loss"].item())

            if rank == 0 and (step + 1) % max(1, args.num_batches // 10) == 0:
                print(f"  step {step+1:>4}/{args.num_batches}  "
                      f"loss={total_losses[-1]:.4f}"
                      + (f"  action={action_losses[-1]:.4f}" if action_losses else "")
                      + (f"  dynamics={dynamics_losses[-1]:.4f}" if dynamics_losses else ""))

    # ── 6. Report & save (rank-0 only) ───────────────────────────────────────
    if rank == 0:
        tl = np.array(total_losses)
        al = np.array(action_losses) if action_losses else None
        dl = np.array(dynamics_losses) if dynamics_losses else None

        print(f"\n{'='*55}")
        print(f" Loss statistics over {len(tl)} batches")
        print(f"{'='*55}")
        print(f"  Total loss    : {tl.mean():.4f}  ±{tl.std():.4f}"
              f"  [min={tl.min():.4f}, max={tl.max():.4f}]")
        if al is not None:
            print(f"  Action loss   : {al.mean():.4f}  ±{al.std():.4f}"
                  f"  [min={al.min():.4f}, max={al.max():.4f}]")
        if dl is not None:
            print(f"  Dynamics loss : {dl.mean():.4f}  ±{dl.std():.4f}"
                  f"  [min={dl.min():.4f}, max={dl.max():.4f}]")
        print(f"{'='*55}\n")

        # Save JSON
        out_dir = pathlib.Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "ckpt_path": str(ckpt_path),
            "num_batches": len(tl),
            "total_loss": tl.tolist(),
            "action_loss": al.tolist() if al is not None else [],
            "dynamics_loss": dl.tolist() if dl is not None else [],
            "summary": {
                "total_loss_mean": float(tl.mean()),
                "total_loss_std": float(tl.std()),
                "action_loss_mean": float(al.mean()) if al is not None else None,
                "dynamics_loss_mean": float(dl.mean()) if dl is not None else None,
            },
        }
        json_path = out_dir / "loss_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[save] JSON results → {json_path}")

        # Plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            n_panels = 1 + (al is not None) + (dl is not None)
            fig, axes = plt.subplots(2, n_panels, figsize=(5 * n_panels, 8))
            if n_panels == 1:
                axes = axes.reshape(2, 1)

            col = 0
            for label, arr, colour in [
                ("Total loss", tl, "steelblue"),
                ("Action loss", al, "darkorange"),
                ("Dynamics loss", dl, "seagreen"),
            ]:
                if arr is None:
                    continue
                # time-series
                axes[0, col].plot(arr, alpha=0.8, color=colour, linewidth=0.9)
                axes[0, col].axhline(arr.mean(), color="black", linestyle="--",
                                     linewidth=1.2, label=f"mean={arr.mean():.4f}")
                axes[0, col].set_title(label)
                axes[0, col].set_xlabel("Batch")
                axes[0, col].set_ylabel("Loss")
                axes[0, col].legend(fontsize=8)
                axes[0, col].grid(True, alpha=0.3)
                # histogram
                axes[1, col].hist(arr, bins=20, color=colour, alpha=0.75, edgecolor="white")
                axes[1, col].axvline(arr.mean(), color="black", linestyle="--", linewidth=1.2)
                axes[1, col].set_xlabel("Loss")
                axes[1, col].set_ylabel("Count")
                axes[1, col].grid(True, alpha=0.3)
                col += 1

            fig.suptitle(
                f"DreamZero training loss  |  ckpt: …/{ckpt_path.name}"
                f"\n{len(tl)} batches, batch_size={args.batch_size}",
                fontsize=10,
            )
            plt.tight_layout()
            plot_path = out_dir / "loss_curves.png"
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[save] Plot          → {plot_path}")
        except ImportError as e:
            print(f"[warn] matplotlib unavailable, skipping plot ({e})")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
