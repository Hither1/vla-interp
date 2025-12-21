import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from overcomplete.sae import TopKSAE, train_sae

# -------------------------
# Config
# -------------------------
npy_dir = "/n/netscratch/sham_lab/Lab/chloe00/pi0_activations"
layer_idx = 11                 # Come from earlier analysis
batch_size = 1024
lr = 5e-4
nb_epochs = 500
device = "cuda" if torch.cuda.is_available() else "cpu"

nb_concepts = 16_000
top_k = 10

ckpt_dir = "./checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(ckpt_dir, f"sae_layer{layer_idx}_k{top_k}_c{nb_concepts}.pt")

# -------------------------
# Load + concatenate .npy
# -------------------------

pattern = os.path.join(npy_dir, "*.npy")
all_paths = glob.glob(pattern)

npy_paths = sorted(
    p for p in all_paths
    if not os.path.basename(p).startswith("libero_90")
)

assert len(npy_paths) > 0, (
    f"No .npy files found in {npy_dir} after excluding 'libero_90*'"
)

all_embeds = []
for npy_path in npy_paths:
    print(f"Loading {npy_path}")
    embeds = np.load(npy_path).astype(np.float32)
    all_embeds.append(embeds)

Activations_np = np.concatenate(all_embeds, axis=0)
Activations = torch.from_numpy(Activations_np)  # CPU tensor
print("Torch activations shape:", Activations.shape)

# Your expected shape seems to be (N_total, num_layers, ?, d)
# We'll infer d from the chosen layer slice below.
N_total = Activations.shape[0]
num_layers = Activations.shape[1]
assert 0 <= layer_idx < num_layers, f"layer_idx={layer_idx} out of range (num_layers={num_layers})"

# -------------------------
# Select a single layer: (N_total, d)
# -------------------------
layer_acts = Activations[:, layer_idx, ...]      # shape: (N_total, ?, d) or (N_total, d)
layer_acts = layer_acts.reshape(layer_acts.shape[0], -1)  # flatten any extra dims -> (N_total, d)
d = layer_acts.shape[-1]
print(f"Training on layer {layer_idx} with feature dim d={d}, N={layer_acts.shape[0]}")

dataloader = DataLoader(
    TensorDataset(layer_acts),
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
)

# -------------------------
# Loss + model + optimizer
# -------------------------
def criterion(x, x_hat, pre_codes, codes, dictionary):
    return (x - x_hat).square().mean()

sae = TopKSAE(d, nb_concepts=nb_concepts, top_k=top_k, device=device)
optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

# -------------------------
# Train
# -------------------------
logs = train_sae(
    sae,
    dataloader,
    criterion,
    optimizer,
    nb_epochs=nb_epochs,
    device=device,
)

# -------------------------
# Save checkpoint
# -------------------------
ckpt = {
    "layer_idx": layer_idx,
    "d": d,
    "nb_concepts": nb_concepts,
    "top_k": top_k,
    "lr": lr,
    "nb_epochs": nb_epochs,
    "model_state_dict": sae.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "logs": logs,  # otherwise delete this line
}

torch.save(ckpt, ckpt_path)
print(f"Saved checkpoint to: {ckpt_path}")


