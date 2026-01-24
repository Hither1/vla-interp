import os
import glob
import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")  # safe on headless SLURM
import matplotlib.pyplot as plt

# overcomplete SAE libs
from overcomplete.sae import BatchTopKSAE


# -------------------------
# Config
# -------------------------
SEED = 42
npy_dir = "/n/netscratch/sham_lab/Lab/chloe00/pi0_activations"
layer_indices = [11]

batch_size = 1024
lr = 3e-4
nb_epochs = 200
device = "cuda" if torch.cuda.is_available() else "cpu"

NB_CONCEPTS_LIST = [512, 1024, 2048, 4096]  # sweep n
TOPK_LIST = [16]                             # sweep k (per-sample notion)

# Combined training across subsets
LIBERO_SUBSETS = [
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_spatial",
]

ckpt_dir = "./checkpoints/BatchTopKSAE"
os.makedirs(ckpt_dir, exist_ok=True)

# When selecting the "best" checkpoint, only consider epochs >= this
MIN_EPOCH_FOR_BEST = 10


# -------------------------
# Seeding
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic-ish
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Data helpers
# -------------------------
def list_subset_files(npy_dir: str, subset: str):
    pattern = os.path.join(npy_dir, "*.npy")
    all_paths = sorted(glob.glob(pattern))
    subset_paths = [p for p in all_paths if os.path.basename(p).startswith(subset)]
    return subset_paths


def load_concat(paths):
    assert len(paths) > 0, "No paths provided."
    chunks = []
    for p in paths:
        print(f"Loading {p}")
        arr = np.load(p).astype(np.float32)
        chunks.append(arr)
    return np.concatenate(chunks, axis=0)


def collect_all_subset_paths(npy_dir: str, subsets):
    per_subset = {}
    all_paths = []
    for s in subsets:
        paths = list_subset_files(npy_dir, s)
        if len(paths) == 0:
            raise RuntimeError(f"No .npy files found for subset='{s}' in {npy_dir}")
        per_subset[s] = paths
        all_paths.extend(paths)
    all_paths = sorted(all_paths)
    return all_paths, per_subset


# -------------------------
# Plotting
# -------------------------
def plot_training_curves(logs, out_path, title_prefix="", min_epoch_for_best=10):
    # Pick loss series
    if "avg_loss" in logs and len(logs["avg_loss"]) > 0:
        loss = logs["avg_loss"]
        loss_name = "avg_loss"
    else:
        loss = logs.get("step_loss", [])
        loss_name = "step_loss"

    r2 = logs.get("r2", [])
    dead = logs.get("dead_features", [])

    L = min(len(loss), len(r2), len(dead))
    if L == 0:
        print("No log data to plot.")
        return

    loss = np.asarray(loss[:L])
    r2 = np.asarray(r2[:L])
    dead = np.asarray(dead[:L])

    epochs = np.arange(1, L + 1)

    start_idx = min(min_epoch_for_best - 1, L - 1)
    mask = np.arange(L) >= start_idx

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    # Loss
    axes[0].plot(epochs, loss, label="loss")
    axes[0].scatter(epochs[-1], loss[-1], zorder=3, label="final")
    best_loss_idx = start_idx + np.argmin(loss[mask])
    axes[0].scatter(epochs[best_loss_idx], loss[best_loss_idx], marker="*", s=120, zorder=4, label=f"best ≥{min_epoch_for_best}")
    axes[0].set_ylabel(loss_name)
    axes[0].set_title(f"{title_prefix} Loss")
    axes[0].annotate(f"final = {loss[-1]:.4g}", (epochs[-1], loss[-1]), xytext=(5, 5), textcoords="offset points", fontsize=9)
    axes[0].annotate(f"best@≥{min_epoch_for_best} = {loss[best_loss_idx]:.4g}", (epochs[best_loss_idx], loss[best_loss_idx]), xytext=(5, -12), textcoords="offset points", fontsize=9)
    axes[0].legend(loc="best", fontsize=9)

    # R2
    axes[1].plot(epochs, r2, label="R2")
    axes[1].scatter(epochs[-1], r2[-1], zorder=3, label="final")
    best_r2_idx = start_idx + np.argmax(r2[mask])
    axes[1].scatter(epochs[best_r2_idx], r2[best_r2_idx], marker="*", s=120, zorder=4, label=f"best ≥{min_epoch_for_best}")
    axes[1].set_ylabel("R2")
    axes[1].set_title(f"{title_prefix} R2")
    axes[1].annotate(f"final = {r2[-1]:.4f}", (epochs[-1], r2[-1]), xytext=(5, 5), textcoords="offset points", fontsize=9)
    axes[1].annotate(f"best@≥{min_epoch_for_best} = {r2[best_r2_idx]:.4f}", (epochs[best_r2_idx], r2[best_r2_idx]), xytext=(5, -12), textcoords="offset points", fontsize=9)
    axes[1].legend(loc="best", fontsize=9)

    # Dead features
    axes[2].plot(epochs, dead, label="dead")
    axes[2].scatter(epochs[-1], dead[-1], zorder=3, label="final")
    best_dead_idx = start_idx + np.argmin(dead[mask])
    axes[2].scatter(epochs[best_dead_idx], dead[best_dead_idx], marker="*", s=120, zorder=4, label=f"best ≥{min_epoch_for_best}")
    axes[2].set_ylabel("Dead features")
    axes[2].set_xlabel("Epoch")
    axes[2].set_title(f"{title_prefix} Dead Features")
    axes[2].annotate(f"final = {dead[-1]}", (epochs[-1], dead[-1]), xytext=(5, 5), textcoords="offset points", fontsize=9)
    axes[2].annotate(f"best@≥{min_epoch_for_best} = {dead[best_dead_idx]}", (epochs[best_dead_idx], dead[best_dead_idx]), xytext=(5, -12), textcoords="offset points", fontsize=9)
    axes[2].legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved training curves to: {out_path}")


# -------------------------
# SAE training loop that saves BEST checkpoint by min dead features
# -------------------------
def train_sae_with_best_dead_checkpoint(
    sae: torch.nn.Module,
    dataloader: DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    nb_epochs: int,
    device: str,
    min_epoch_for_best: int = 10,
):
    """
    Returns:
      logs: dict with keys ['avg_loss', 'r2', 'dead_features']
      best: dict with keys ['epoch', 'dead', 'state_dict', 'optimizer_state_dict']
    """
    sae.to(device)
    logs = {"avg_loss": [], "r2": [], "dead_features": []}

    best = {
        "epoch": None,  # 1-based epoch number
        "dead": float("inf"),
        "state_dict": None,
        "optimizer_state_dict": None,
    }

    for epoch in range(1, nb_epochs + 1):
        sae.train()

        total_loss = 0.0
        total_batches = 0

        # For dead feature counting across the *whole epoch*
        fire_counts = None  # [C] on CPU
        r2_sum = 0.0

        for (x,) in dataloader:
            x = x.to(device, non_blocking=True)

            out = sae(x)
            # Expect something like: x_hat, pre_codes, codes, dictionary
            if not isinstance(out, (tuple, list)) or len(out) < 3:
                raise RuntimeError(
                    "SAE forward() must return (x_hat, pre_codes, codes, ...) "
                    "for this training loop. Adapt unpacking if your API differs."
                )

            x_hat = out[0]
            pre_codes = out[1] if len(out) > 1 else None
            codes = out[2]
            dictionary = out[3] if len(out) > 3 else None

            loss = criterion(x, x_hat, pre_codes, codes, dictionary)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().item())
            total_batches += 1

            # Dead features: accumulate fires across the epoch
            fires = (codes.detach() > 0).sum(dim=0).to("cpu")  # [C]
            if fire_counts is None:
                fire_counts = fires
            else:
                fire_counts += fires

            # R2: compute per-batch reconstruction R2 and average over batches
            with torch.no_grad():
                ss_res = (x - x_hat).pow(2).sum()
                x_mean = x.mean(dim=0, keepdim=True)
                ss_tot = (x - x_mean).pow(2).sum() + 1e-12
                r2_batch = 1.0 - (ss_res / ss_tot)
                r2_sum += float(r2_batch.detach().item())

        avg_loss = total_loss / max(total_batches, 1)
        avg_r2 = r2_sum / max(total_batches, 1)

        if fire_counts is None:
            dead_features = 0
        else:
            dead_features = int((fire_counts == 0).sum().item())

        logs["avg_loss"].append(avg_loss)
        logs["r2"].append(avg_r2)
        logs["dead_features"].append(dead_features)

        print(
            f"Epoch {epoch:4d}/{nb_epochs} | loss={avg_loss:.6g} | "
            f"r2={avg_r2:.4f} | dead={dead_features}"
        )

        # Update BEST checkpoint by MIN dead features (ties -> keep earlier best)
        if epoch >= min_epoch_for_best and dead_features < best["dead"]:
            best["epoch"] = epoch
            best["dead"] = dead_features
            # Deep-copy weights to CPU to avoid later mutation
            best["state_dict"] = {k: v.detach().cpu().clone() for k, v in sae.state_dict().items()}
            best["optimizer_state_dict"] = optimizer.state_dict()
            print(f"  -> New BEST by dead features: epoch={epoch}, dead={dead_features}")

    return logs, best


# -------------------------
# Criterion (BatchTopKSAE "reanimation" trick)
# -------------------------
def batchtopk_reanim_criterion(x, x_hat, pre_codes, codes, dictionary):
    loss = (x - x_hat).square().mean()

    # dead feature = no fires in batch
    is_dead = ((codes > 0).sum(dim=0) == 0).float().detach()
    # push pre-codes for dead feats positive to "reanimate"
    reanim_loss = (pre_codes * is_dead[None, :]).mean()

    loss -= reanim_loss * 1e-3
    return loss


# -------------------------
# Main: combined training on all subsets
# -------------------------
def train_sae_on_all_subsets():
    set_seed(SEED)

    # 1) Collect all files across subsets
    all_paths, per_subset_paths = collect_all_subset_paths(npy_dir, LIBERO_SUBSETS)
    print("\n=== Combined training ===")
    for s, ps in per_subset_paths.items():
        print(f"  {s}: {len(ps)} files")
    print(f"TOTAL files: {len(all_paths)}\n")

    # 2) Load activations
    Activations_np = load_concat(all_paths)  # [N, L, ...]
    Activations = torch.from_numpy(Activations_np)  # CPU tensor

    print("[ALL] Torch activations shape:", Activations.shape)
    N_total = Activations.shape[0]

    # 3) Select layers and flatten
    layer_acts_list = []
    per_layer_dims = []
    for li in layer_indices:
        acts_li = Activations[:, li, ...].reshape(N_total, -1)  # [N, d_li]
        layer_acts_list.append(acts_li)
        per_layer_dims.append(acts_li.shape[1])

    layer_acts = torch.cat(layer_acts_list, dim=1)  # [N, sum(d_li)]
    d = layer_acts.shape[1]

    print(
        f"[ALL] Training on layers {layer_indices} "
        f"with per-layer dims={per_layer_dims}, total d={d}, N={layer_acts.shape[0]}"
    )

    # 4) DataLoader
    g = torch.Generator()
    g.manual_seed(SEED)
    dataloader = DataLoader(
        TensorDataset(layer_acts),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        generator=g,
        drop_last=False,
    )

    combined_tag = "libero_all"
    layers_tag = "-".join(map(str, layer_indices))

    for nb_concepts in NB_CONCEPTS_LIST:
        for top_k_per_sample in TOPK_LIST:
            set_seed(SEED)

            # BatchTopKSAE expects top_k as "total number of nonzeros per batch"
            sae = BatchTopKSAE(
                d,
                nb_concepts=nb_concepts,
                top_k=top_k_per_sample * batch_size,
                device=device,
            )
            optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

            logs, best = train_sae_with_best_dead_checkpoint(
                sae=sae,
                dataloader=dataloader,
                criterion=batchtopk_reanim_criterion,
                optimizer=optimizer,
                nb_epochs=nb_epochs,
                device=device,
                min_epoch_for_best=MIN_EPOCH_FOR_BEST,
            )

            # If we never set a best (e.g. nb_epochs < MIN_EPOCH_FOR_BEST), fall back to final
            if best["state_dict"] is None:
                best["epoch"] = nb_epochs
                best["dead"] = logs["dead_features"][-1] if len(logs["dead_features"]) else None
                best["state_dict"] = {k: v.detach().cpu().clone() for k, v in sae.state_dict().items()}
                best["optimizer_state_dict"] = optimizer.state_dict()

            best_dead = best["dead"]
            best_epoch = best["epoch"]

            ckpt_path = os.path.join(
                ckpt_dir,
                f"sae_{combined_tag}_layer{layers_tag}_k{top_k_per_sample}_c{nb_concepts}"
                f"_bestdead{int(best_dead)}_at{int(best_epoch)}.pt"
            )

            ckpt = {
                "subset": combined_tag,
                "subsets_included": list(LIBERO_SUBSETS),
                "layer_idx": layers_tag,
                "d": d,
                "nb_concepts": nb_concepts,
                "top_k_per_sample": top_k_per_sample,
                "top_k_batch": top_k_per_sample * batch_size,
                "lr": lr,
                "nb_epochs": nb_epochs,
                "best_epoch": best_epoch,
                "best_dead_features": best_dead,
                "model_state_dict": best["state_dict"],  # <-- BEST by min dead features
                "optimizer_state_dict": best["optimizer_state_dict"],
                "logs": logs,
                "npy_paths_all": all_paths,
                "npy_paths_per_subset": per_subset_paths,
            }
            torch.save(ckpt, ckpt_path)
            print(f"[ALL] Saved BEST checkpoint to: {ckpt_path}")

            plot_path = os.path.join(
                ckpt_dir,
                f"curves_{combined_tag}_layer{layers_tag}_k{top_k_per_sample}_c{nb_concepts}.png"
            )
            plot_training_curves(
                logs,
                plot_path,
                title_prefix=f"{combined_tag} | layer {layers_tag} | k={top_k_per_sample} | c={nb_concepts}",
                min_epoch_for_best=MIN_EPOCH_FOR_BEST,
            )


if __name__ == "__main__":
    train_sae_on_all_subsets()