import os
import glob
import re
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")  # safe on headless SLURM
if not hasattr(np, "Inf"):
    np.Inf = np.inf
import matplotlib.pyplot as plt
from overcomplete.sae import TopKSAE, train_sae

# -------------------------
# Config
# -------------------------
SEED = 42
npy_dir = "/n/netscratch/sham_lab/Lab/chloe00/pi0_activations"
layer_indices = [ 11]
batch_size = 1024
lr = 1e-4
nb_epochs = 230
device = "cuda" if torch.cuda.is_available() else "cpu"

nb_concepts = 100
top_k = 10

# Train a separate SAE per subset:
# Put whatever LIBERO subsets 
LIBERO_SUBSETS = [
    "libero_object",
]

ckpt_dir = "./checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

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



def plot_training_curves(logs, out_path, title_prefix=""):
    """
    logs: dict-like with keys like 'avg_loss', 'r2', 'dead_features'
    Saves a 3-row plot: loss, r2, dead_features vs epoch,
    with final values annotated.
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

    loss = loss[:L]
    r2 = r2[:L]
    dead = dead[:L]

    epochs = np.arange(1, L + 1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # -------- Loss --------
    axes[0].plot(epochs, loss)
    axes[0].scatter(epochs[-1], loss[-1], zorder=3)
    axes[0].set_ylabel(loss_name)
    axes[0].set_title(f"{title_prefix} Loss")

    axes[0].annotate(
        f"final = {loss[-1]:.4g}",
        xy=(epochs[-1], loss[-1]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=10,
        ha="left",
        va="bottom",
    )

    # -------- R2 --------
    axes[1].plot(epochs, r2)
    axes[1].scatter(epochs[-1], r2[-1], zorder=3)
    axes[1].set_ylabel("R2")
    axes[1].set_title(f"{title_prefix} R2")

    axes[1].annotate(
        f"final = {r2[-1]:.4f}",
        xy=(epochs[-1], r2[-1]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=10,
        ha="left",
        va="bottom",
    )

    # -------- Dead features --------
    axes[2].plot(epochs, dead)
    axes[2].scatter(epochs[-1], dead[-1], zorder=3)
    axes[2].set_ylabel("Dead features")
    axes[2].set_xlabel("Epoch")
    axes[2].set_title(f"{title_prefix} Dead Features")

    axes[2].annotate(
        f"final = {dead[-1]}",
        xy=(epochs[-1], dead[-1]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=10,
        ha="left",
        va="bottom",
    )

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

    dataloader = DataLoader(
        TensorDataset(layer_acts),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
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

    sae = TopKSAE(d, nb_concepts=nb_concepts, top_k=top_k, device=device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

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