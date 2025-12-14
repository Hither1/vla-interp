import numpy as np
import matplotlib.pyplot as plt
if not hasattr(np, "Inf"):
    np.Inf = np.inf

def calculate_geometry_metrics(embeddings: np.ndarray):
    """
    Calculates stepwise distance and temporal curvature for a sequence of embeddings.

    Args:
        embeddings: A NumPy array of shape (N, D), where N is the number of steps
                    along some axis (time or layer) and D is the embedding dimension.
    
    Returns:
        tuple: (distances, curvatures_deg)
               - distances: NumPy array of stepwise Euclidean distances (N-1).
               - curvatures_deg: NumPy array of temporal curvatures in degrees (N-2).
    """
    if embeddings.shape[0] < 2:
        return np.array([]), np.array([])

    # 1. Displacement vectors: v_i = z_{i+1} - z_i  → shape (N-1, D)
    displacement_vectors = embeddings[1:] - embeddings[:-1]

    # 2. Stepwise distances: ||v_i||_2 → shape (N-1,)
    distances = np.linalg.norm(displacement_vectors, axis=1)

    # 3. Curvatures between consecutive displacement vectors
    if displacement_vectors.shape[0] < 2:
        return distances, np.array([])

    v_i = displacement_vectors[:-1]      # shape (N-2, D)
    v_ip1 = displacement_vectors[1:]     # shape (N-2, D)

    dot_products = np.sum(v_i * v_ip1, axis=1)                 # (N-2,)
    norms_v_i = np.linalg.norm(v_i, axis=1)                    # (N-2,)
    norms_v_ip1 = np.linalg.norm(v_ip1, axis=1)                # (N-2,)
    norm_products = norms_v_i * norms_v_ip1                    # (N-2,)

    cos_kappa = np.zeros_like(dot_products)
    non_zero = norm_products > 1e-6
    cos_kappa[non_zero] = dot_products[non_zero] / norm_products[non_zero]
    cos_kappa = np.clip(cos_kappa, -1.0, 1.0)

    curvatures_rad = np.arccos(cos_kappa)
    curvatures_deg = np.rad2deg(curvatures_rad)

    return distances, curvatures_deg


def compute_time_and_layer_metrics(embeds: np.ndarray):
    """
    Compute geometry metrics along:
      (1) time axis, for each layer
      (2) layer axis, for each time step

    Assumes `embeds` has shape (T, L, D) or (T, L, B, D).

    Returns:
        time_dists:      (T-1, L)
        time_curvs_deg:  (T-2, L)
        layer_dists:     (L-1, T)
        layer_curvs_deg: (L-2, T)
    """
    # If there is a batch dimension, average over it: (T, L, B, D) → (T, L, D)
    if embeds.ndim == 4:
        # embeds: (T, L, B, D) → mean over B
        embeds = embeds.mean(axis=2)

    assert embeds.ndim == 3, f"Expected embeds to be (T, L, D) or (T, L, B, D), got shape {embeds.shape}"

    T, L, D = embeds.shape

    # ------- (1) Along time: for each layer separately -------
    # For each layer ℓ, we look at sequence z_tℓ ∈ R^D, t = 0..T-1
    time_dists_list = []
    time_curvs_list = []
    for layer_idx in range(L):
        seq = embeds[:, layer_idx, :]  # (T, D)
        d, c = calculate_geometry_metrics(seq)
        time_dists_list.append(d)      # (T-1,)
        time_curvs_list.append(c)      # (T-2,)

    # Stack: (L, T-1) → transpose to (T-1, L) so time is first
    time_dists = np.stack(time_dists_list, axis=0).T      # (T-1, L)
    time_curvs_deg = np.stack(time_curvs_list, axis=0).T  # (T-2, L)

    # ------- (2) Along layers: for each time step separately -------
    # For each time t, we look at sequence z_tℓ ∈ R^D, ℓ = 0..L-1
    layer_dists_list = []
    layer_curvs_list = []
    for t in range(T):
        seq = embeds[t, :, :]  # (L, D)
        d, c = calculate_geometry_metrics(seq)
        layer_dists_list.append(d)     # (L-1,)
        layer_curvs_list.append(c)     # (L-2,)

    # Stack: (T, L-1) / (T, L-2), but we want layers-first for symmetry:
    # Output shapes: (L-1, T), (L-2, T)
    layer_dists = np.stack(layer_dists_list, axis=0).T      # (L-1, T)
    layer_curvs_deg = np.stack(layer_curvs_list, axis=0).T  # (L-2, T)

    return time_dists, time_curvs_deg, layer_dists, layer_curvs_deg


def plot_geometry_metrics(time_dists, time_curvs_deg, layer_dists, layer_curvs_deg, title_prefix=""):
    """
    Simple plotting helper using matplotlib.

    time_dists:      (T-1, L)
    time_curvs_deg:  (T-2, L)
    layer_dists:     (L-1, T)
    layer_curvs_deg: (L-2, T)
    """
    T_minus_1, L = time_dists.shape
    T_minus_2 = time_curvs_deg.shape[0]
    L_minus_1, T = layer_dists.shape
    L_minus_2 = layer_curvs_deg.shape[0]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{title_prefix} Geometry Metrics", fontsize=16)

    # (1) Distance over time (averaged over layers)
    mean_time_dist = time_dists.mean(axis=1)  # (T-1,)
    axs[0, 0].plot(np.arange(T_minus_1), mean_time_dist)
    axs[0, 0].set_title("Mean distance over time (avg over layers)")
    axs[0, 0].set_xlabel("time step")
    axs[0, 0].set_ylabel("distance")

    # (2) Curvature over time (averaged over layers)
    mean_time_curv = time_curvs_deg.mean(axis=1)  # (T-2,)
    axs[0, 1].plot(np.arange(T_minus_2), mean_time_curv)
    axs[0, 1].set_title("Mean curvature over time (deg, avg over layers)")
    axs[0, 1].set_xlabel("time step index (between steps)")
    axs[0, 1].set_ylabel("curvature (deg)")

    # (3) Distance over layers (averaged over time)
    mean_layer_dist = layer_dists.mean(axis=1)  # (L-1,)
    axs[1, 0].plot(np.arange(L_minus_1), mean_layer_dist)
    axs[1, 0].set_title("Mean distance over layers (avg over time)")
    axs[1, 0].set_xlabel("layer index")
    axs[1, 0].set_ylabel("distance")

    # (4) Curvature over layers (averaged over time)
    mean_layer_curv = layer_curvs_deg.mean(axis=1)  # (L-2,)
    axs[1, 1].plot(np.arange(L_minus_2), mean_layer_curv)
    axs[1, 1].set_title("Mean curvature over layers (deg, avg over time)")
    axs[1, 1].set_xlabel("layer index (between layers)")
    axs[1, 1].set_ylabel("curvature (deg)")

    plt.tight_layout()
    plt.savefig(f'{title_prefix}.png')


# ---------------- Example usage ----------------

def main():
    # Path to your saved episode file
    npy_path = "/n/netscratch/sham_lab/Lab/chloe00/pi0_activations/task0_ep0_post_ffn_last_step.npy"  # change to your path

    # Load activations
    embeds = np.load(npy_path)  # expected shape: (T, L, D) or (T, L, B, D)
    print("Loaded embeddings with shape:", embeds.shape)

    # Compute metrics
    time_dists, time_curvs_deg, layer_dists, layer_curvs_deg = compute_time_and_layer_metrics(embeds)

    print("time_dists shape:", time_dists.shape)          # (T-1, L)
    print("time_curvs_deg shape:", time_curvs_deg.shape)  # (T-2, L)
    print("layer_dists shape:", layer_dists.shape)        # (L-1, T)
    print("layer_curvs_deg shape:", layer_curvs_deg.shape)  # (L-2, T)

    # Plot
    plot_geometry_metrics(
        time_dists,
        time_curvs_deg,
        layer_dists,
        layer_curvs_deg,
        title_prefix=npy_path.split('/')[-1][:-4],
    )


if __name__ == "__main__":
    main()