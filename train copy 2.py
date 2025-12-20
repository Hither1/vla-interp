import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

if not hasattr(np, "Inf"):
    np.Inf = np.inf
import matplotlib.pyplot as plt

from overcomplete.sae import TopKSAE, train_sae

npy_dir = "/n/netscratch/sham_lab/Lab/chloe00/pi0_activations"
pattern = os.path.join(npy_dir, "*.npy")
npy_paths = sorted(glob.glob(pattern))

all_embeds = [] 

for npy_path in npy_paths:
    print(f"Processing {npy_path}")
    embeds = np.load(npy_path)  
    print("  Loaded embeddings with shape:", embeds.shape)

    # ensure float32 (often safer for torch)
    embeds = embeds.astype(np.float32)
    all_embeds.append(embeds)

# ---- concatenate everything along batch dimension ----
Activations_np = np.concatenate(all_embeds, axis=0)  # shape: (N_total, num_layers, d)
Activations = torch.from_numpy(Activations_np)  # stays on CPU; DataLoader will move to GPU
print("Torch activations shape:", Activations.shape)

N_total, num_layers, _, d = Activations.shape
print("Number of layers:", num_layers)
print("Feature dimension d:", d)

# ---- training config ----
batch_size = 1024
lr = 5e-4
nb_epochs = 500
device = "cuda"



# containers for logs
all_loss_logs = {}  # layer_idx -> [loss_t]
all_r2_logs = {}    # layer_idx -> [r2_t]

# optional: store trained SAEs if you want to reuse them later
trained_saes = {}   # layer_idx -> sae

for layer_idx in range(num_layers):
    print(f"\n===== Training SAE for layer {layer_idx} =====")

    # select activations for this layer: shape (N_total, d)
    layer_acts = torch.squeeze(Activations[:, layer_idx, :])


    # dataloader
    dataloader = DataLoader(
        TensorDataset(layer_acts),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    # create SAE for this layer
    sae = TopKSAE(d, nb_concepts=16_000, top_k=10, device=device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    # train SAE
    logs = train_sae(
        sae,
        dataloader,
        criterion,
        optimizer,
        nb_epochs=nb_epochs,
        device=device,
    )

    # adapt these key names if your train_sae returns something different
    layer_loss = logs["loss"]           # iterable over training steps/epochs
    layer_r2 = logs["r2"]               # iterable over training steps/epochs

    all_loss_logs[layer_idx] = layer_loss
    all_r2_logs[layer_idx] = layer_r2
    trained_saes[layer_idx] = sae

# ---- R² (right) ----
# ax = axes[1]
cmap = plt.cm.get_cmap("viridis", num_layers) 

plt.ylim(bottom=0.6)
for layer_idx, r2_history in all_r2_logs.items():
    color = cmap() 
    plt.plot(r2_history, color=color, label=f"Layer {}")

plt.xlabel("Training step")
plt.ylabel("R²")
plt.title("SAE Training R² per Layer")
plt.legend()

plt.tight_layout()
plt.savefig('sae.png')