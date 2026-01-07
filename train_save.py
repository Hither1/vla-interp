import os
import glob
import re
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import random


from typing import Tuple, Optional
import matplotlib
matplotlib.use("Agg")  # safe on headless SLURM
if not hasattr(np, "Inf"):
    np.Inf = np.inf
import matplotlib.pyplot as plt
from overcomplete.sae import TopKSAE, JumpSAE, BatchTopKSAE, train_sae

# -------------------------
# Config
# -------------------------
SEED = 42
npy_dir = "/n/netscratch/sham_lab/Lab/chloe00/pi0_activations"
layer_indices = [ 11]
batch_size = 1024
lr = 3e-4
nb_epochs = 200
device = "cuda" if torch.cuda.is_available() else "cpu"

nb_concepts = 40
NB_CONCEPTS_LIST = [512, 1024, 2048, 4096]     # <-- sweep n
TOPK_LIST        = [1, 2, 4, 8, 16]            # <-- sweep k (per sample notion)

top_k = 10

# Train a separate SAE per subset:
# Put whatever LIBERO subsets 
LIBERO_SUBSETS = [
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_spatial",
]

ckpt_dir = "./checkpoints/BatchTopKSAE"
os.makedirs(ckpt_dir, exist_ok=True)



# -------------------------
# Targets / labels loading
# -------------------------
def list_subset_label_files(y_dir: str, subset: str):
    """
    Return .npy paths whose basename starts with subset, same as activations.
    Assumes labels are stored as npy files parallel to activations.
    """
    pattern = os.path.join(y_dir, "*.npy")
    all_paths = sorted(glob.glob(pattern))
    subset_paths = [p for p in all_paths if os.path.basename(p).startswith(subset)]
    return subset_paths

def load_concat_labels(paths):
    """Load and concatenate label npys. Ensures float32 (or int64 if you prefer)."""
    assert len(paths) > 0, "No label paths provided."
    chunks = []
    for p in paths:
        print(f"Loading labels {p}")
        arr = np.load(p)
        # Allow (N,), (N,1), (N,k). We'll standardize later.
        chunks.append(arr)
    y = np.concatenate(chunks, axis=0)
    return y

# -------------------------
# SAE code extraction
# -------------------------
@torch.no_grad()
def sae_get_codes(sae, X: torch.Tensor, batch_size: int, device: str) -> torch.Tensor:
    """
    Returns SAE sparse codes Z with shape [N, nb_concepts].
    Tries common APIs:
      - sae.encode(X)
      - sae(X) returning (x_hat, pre_codes, codes, dictionary?) or similar
    """
    sae.eval()
    Z_list = []
    N = X.shape[0]
    for start in range(0, N, batch_size):
        xb = X[start:start + batch_size].to(device, non_blocking=True)

        if hasattr(sae, "encode"):
            codes = sae.encode(xb)
        else:
            out = sae(xb)
            # Handle different return signatures:
            # expected in your criterion: x_hat, pre_codes, codes, dictionary
            if isinstance(out, (tuple, list)) and len(out) >= 3:
                codes = out[2]
            else:
                raise RuntimeError("Could not infer codes from sae forward(). "
                                   "Please adapt sae_get_codes() to your SAE API.")

        Z_list.append(codes.detach().cpu())

    return torch.cat(Z_list, dim=0)

# -------------------------
# Least squares (linear regression) with bias
# -------------------------
def fit_least_squares(Z_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit y ~= Z W + b using least squares.
    Returns (W, b) where:
      W shape: [d, k] if y is [N, k], else [d, 1]
      b shape: [k] or [1]
    """
    if y_train.ndim == 1:
        y_train = y_train[:, None]  # [N,1]

    N, d = Z_train.shape
    k = y_train.shape[1]

    # Add bias column
    A = np.concatenate([Z_train, np.ones((N, 1), dtype=Z_train.dtype)], axis=1)  # [N, d+1]

    # Solve min ||A theta - y||_2
    # theta: [d+1, k]
    theta, *_ = np.linalg.lstsq(A, y_train, rcond=None)

    W = theta[:d, :]            # [d,k]
    b = theta[d, :]             # [k]
    return W, b

def predict_linear(Z: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    if W.ndim == 1:
        W = W[:, None]
    yhat = Z @ W + b[None, :]
    return yhat

def mse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean((yhat - y) ** 2))

def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    # Supports y as [N] or [N,k]; returns mean over outputs
    if y.ndim == 1:
        y = y[:, None]
    if yhat.ndim == 1:
        yhat = yhat[:, None]
    ss_res = np.sum((y - yhat) ** 2, axis=0)
    y_mean = np.mean(y, axis=0, keepdims=True)
    ss_tot = np.sum((y - y_mean) ** 2, axis=0) + 1e-12
    return float(np.mean(1.0 - ss_res / ss_tot))


def binarize_from_score(score: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (score.reshape(-1) >= threshold).astype(np.int64)

# -------------------------
# Train/val split
# -------------------------
def train_val_split_indices(N: int, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = int(round(val_frac * N))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------
# Helpers
# -------------------------
def list_subset_files(npy_dir: str, subset: str):
    """Return .npy paths whose basename starts with e.g. 'libero_90'."""
    pattern = os.path.join(npy_dir, "*.npy")
    all_paths = sorted(glob.glob(pattern))
    subset_paths = [
        p for p in all_paths
        if os.path.basename(p).startswith(subset)
    ]
    return subset_paths

def load_concat(paths):
    """Load and concatenate a list of .npy files into one big float32 array."""
    assert len(paths) > 0, "No paths provided."
    chunks = []
    for p in paths:
        print(f"Loading {p}")
        arr = np.load(p).astype(np.float32)
        chunks.append(arr)
    return np.concatenate(chunks, axis=0)



def plot_training_curves(logs, out_path, title_prefix="", min_epoch_for_best=30):
    """
    Plots:
      - Loss (avg_loss or step_loss)
      - R2
      - Dead features

    Annotates:
      - final value
      - best value after min_epoch_for_best
    """
    # Pick loss series
    if "avg_loss" in logs and len(logs["avg_loss"]) > 0:
        loss = logs["avg_loss"]
        loss_name = "avg_loss"
    else:
        loss = logs.get("step_loss", [])
        loss_name = "step_loss"

    r2 = logs.get("r2", [])
    dead = logs.get("dead_features", [])

    # Ensure consistent length
    L = min(len(loss), len(r2), len(dead))
    if L == 0:
        print("No log data to plot.")
        return

    loss = np.asarray(loss[:L])
    r2 = np.asarray(r2[:L])
    dead = np.asarray(dead[:L])

    epochs = np.arange(1, L + 1)

    # Epoch mask for "after epoch 100"
    start_idx = min(min_epoch_for_best - 1, L - 1)
    mask = np.arange(L) >= start_idx

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    # ---------------- Loss ----------------
    axes[0].plot(epochs, loss, label="loss")

    # Final
    axes[0].scatter(epochs[-1], loss[-1], zorder=3, label="final")

    # Best after epoch 100 (min)
    best_loss_idx = start_idx + np.argmin(loss[mask])
    axes[0].scatter(
        epochs[best_loss_idx], loss[best_loss_idx],
        marker="*", s=120, zorder=4, label="best ≥100"
    )

    axes[0].set_ylabel(loss_name)
    axes[0].set_title(f"{title_prefix} Loss")

    axes[0].annotate(
        f"final = {loss[-1]:.4g}",
        (epochs[-1], loss[-1]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
    )
    axes[0].annotate(
        f"best@≥{min_epoch_for_best} = {loss[best_loss_idx]:.4g}",
        (epochs[best_loss_idx], loss[best_loss_idx]),
        xytext=(5, -12),
        textcoords="offset points",
        fontsize=9,
    )

    axes[0].legend(loc="best", fontsize=9)

    # ---------------- R2 ----------------
    axes[1].plot(epochs, r2, label="R2")

    # Final
    axes[1].scatter(epochs[-1], r2[-1], zorder=3, label="final")

    # Best after epoch 100 (max)
    best_r2_idx = start_idx + np.argmax(r2[mask])
    axes[1].scatter(
        epochs[best_r2_idx], r2[best_r2_idx],
        marker="*", s=120, zorder=4, label="best ≥100"
    )

    axes[1].set_ylabel("R2")
    axes[1].set_title(f"{title_prefix} R2")

    axes[1].annotate(
        f"final = {r2[-1]:.4f}",
        (epochs[-1], r2[-1]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
    )
    axes[1].annotate(
        f"best@≥{min_epoch_for_best} = {r2[best_r2_idx]:.4f}",
        (epochs[best_r2_idx], r2[best_r2_idx]),
        xytext=(5, -12),
        textcoords="offset points",
        fontsize=9,
    )

    axes[1].legend(loc="best", fontsize=9)

    # ---------------- Dead features ----------------
    axes[2].plot(epochs, dead, label="dead")

    # Final
    axes[2].scatter(epochs[-1], dead[-1], zorder=3, label="final")

    # Best after epoch 100 (min)
    best_dead_idx = start_idx + np.argmin(dead[mask])
    axes[2].scatter(
        epochs[best_dead_idx], dead[best_dead_idx],
        marker="*", s=120, zorder=4, label="best ≥100"
    )

    axes[2].set_ylabel("Dead features")
    axes[2].set_xlabel("Epoch")
    axes[2].set_title(f"{title_prefix} Dead Features")

    axes[2].annotate(
        f"final = {dead[-1]}",
        (epochs[-1], dead[-1]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
    )
    axes[2].annotate(
        f"best@≥{min_epoch_for_best} = {dead[best_dead_idx]}",
        (epochs[best_dead_idx], dead[best_dead_idx]),
        xytext=(5, -12),
        textcoords="offset points",
        fontsize=9,
    )

    axes[2].legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved training curves to: {out_path}")



def train_sae_for_subset(subset: str):
    npy_paths = list_subset_files(npy_dir, subset)
    assert len(npy_paths) > 0, f"No .npy files found for subset='{subset}' in {npy_dir}"

    Activations_np = load_concat(npy_paths)
    Activations = torch.from_numpy(Activations_np)  # CPU tensor

    print(f"[{subset}] Torch activations shape:", Activations.shape)

    N_total = Activations.shape[0]
    num_layers = Activations.shape[1]

    # Select layer, flatten any extra dims
    layer_acts_list = []
    per_layer_dims = []
    for li in layer_indices:
        acts_li = Activations[:, li, ...].reshape(N_total, -1)  # [N, d_li]
        layer_acts_list.append(acts_li)
        per_layer_dims.append(acts_li.shape[1])

    layer_acts = torch.cat(layer_acts_list, dim=1)  # [N, sum(d_li)]
    d = layer_acts.shape[1]

    print(
        f"[{subset}] Training on layers {layer_indices} "
        f"with per-layer dims={per_layer_dims}, total d={d}, N={layer_acts.shape[0]}"
    )

    g = torch.Generator()
    g.manual_seed(SEED)
    dataloader = DataLoader(
        TensorDataset(layer_acts),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        generator=g,
    )

    def load_balance_loss(codes, eps=1e-8):
        # codes: [B, C], sparse with zeros for non-selected
        sel = (codes != 0).float()                 # [B, C]
        p = sel.mean(dim=0)                        # selection rate per feature
        p = p / (p.sum() + eps)                    # normalize to distribution
        # maximize entropy <=> minimize negative entropy
        entropy = -(p * (p + eps).log()).sum()
        return -entropy  # minimize negative entropy => maximize entropy

    lb_coeff = 5e-2
    def criterion(x, x_hat, pre_codes, codes, dictionary):
        mse = (x - x_hat).square().mean()
        lb = load_balance_loss(codes)

        return mse + lb_coeff * lb


    # JumpRelu
    def criterion(x, x_hat, pre_codes, codes, dictionary):
        # here we directly use the thresholds of the model to control the sparsity
        loss = (x - x_hat).square().mean()

        sparsity = (codes > 0).float().mean().detach()
        if sparsity > desired_sparsity:
            # if we are not sparse enough, increase the thresholds levels
            loss -= sae.thresholds.sum()

        return loss

    #  BatchTopKSAE
    def criterion(x, x_hat, pre_codes, codes, dictionary):
        loss = (x - x_hat).square().mean()

        # is dead of shape (k) (nb concepts) and is 1 iif
        # not a single code has fire in the batch
        is_dead = ((codes > 0).sum(dim=0) == 0).float().detach()
        # we push the pre_codes (before relu) towards the positive orthant
        reanim_loss = (pre_codes * is_dead[None, :]).mean()

        loss -= reanim_loss * 1e-3
        return loss


    for nb_concepts in NB_CONCEPTS_LIST:
        for top_k in TOPK_LIST:
            # sae = TopKSAE(d, nb_concepts=nb_concepts, top_k=top_k, device=device)
            sae = BatchTopKSAE(d, nb_concepts=nb_concepts, top_k=top_k * batch_size, device=device)
            optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

            set_seed(SEED)
            logs = train_sae(
                sae,
                dataloader,
                criterion,
                optimizer,
                nb_epochs=nb_epochs,
                device=device,
            )

            layers_tag = "-".join(map(str, layer_indices))
            ckpt_path = os.path.join(
                ckpt_dir,
                f"sae_{subset}_layer{layers_tag}_k{top_k}_c{nb_concepts}.pt"
            )


            ckpt = {
                "subset": subset,
                "layer_idx": layers_tag,
                "d": d,
                "nb_concepts": nb_concepts,
                "top_k": top_k,
                "lr": lr,
                "nb_epochs": nb_epochs,
                "model_state_dict": sae.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "logs": logs,
                "npy_paths": npy_paths,
            }
            torch.save(ckpt, ckpt_path)

            plot_path = os.path.join(
                ckpt_dir,
                f"curves_{subset}_layer{layers_tag}_k{top_k}_c{nb_concepts}.png"
            )
            plot_training_curves(
                logs,
                plot_path,
                title_prefix=f"{subset} | layer {layers_tag} | k={top_k} | c={nb_concepts}"
            )

            print(f"[{subset}] Saved checkpoint to: {ckpt_path}")

# -------------------------
# Run: train SAEs per subset
# -------------------------
for subset in LIBERO_SUBSETS:
    train_sae_for_subset(subset)