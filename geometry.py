import os
import glob
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
        embeds = embeds.mean(axis=2)

    assert embeds.ndim == 3, f"Expected embeds to be (T, L, D) or (T, L, B, D), got shape {embeds.shape}"

    T, L, D = embeds.shape

    # ------- (1) Along time: for each layer separately -------
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
    layer_dists_list = []
    layer_curvs_list = []
    for t in range(T):
        seq = embeds[t, :, :]  # (L, D)
        d, c = calculate_geometry_metrics(seq)
        layer_dists_list.append(d)     # (L-1,)
        layer_curvs_list.append(c)     # (L-2,)

    # Stack: (T, L-1) / (T, L-2), but we want layers-first for symmetry:
    layer_dists = np.stack(layer_dists_list, axis=0).T      # (L-1, T)
    layer_curvs_deg = np.stack(layer_curvs_list, axis=0).T  # (L-2, T)

    return time_dists, time_curvs_deg, layer_dists, layer_curvs_deg


def plot_overlay_geometry_metrics(results, title_suffix=""):
    """
    Overlay mean geometry metric curves from multiple runs (npy files).

    `results` is a list of dicts, each with keys:
        'time_dists', 'time_curvs_deg', 'layer_dists', 'layer_curvs_deg', 'label'
    """
    if not results:
        print("No results to plot.")
        return

    # To be robust if lengths differ a bit, align by minimum length
    T_minus_1 = min(r['time_dists'].shape[0] for r in results)
    T_minus_2 = min(r['time_curvs_deg'].shape[0] for r in results)
    L_minus_1 = min(r['layer_dists'].shape[0] for r in results)
    L_minus_2 = min(r['layer_curvs_deg'].shape[0] for r in results)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Geometry Metrics {f'({title_suffix})' if title_suffix else ''}", fontsize=16)

    # (1) Distance over time (avg over layers), all files
    for r in results:
        mean_time_dist = r['time_dists'][:T_minus_1].mean(axis=1)
        axs[0, 0].plot(np.arange(T_minus_1), mean_time_dist, label=r['label'])
    axs[0, 0].set_title("Mean distance over time (avg over layers)")
    axs[0, 0].set_xlabel("time step")
    axs[0, 0].set_ylabel("distance")
    # axs[0, 0].legend()

    # (2) Curvature over time (avg over layers), all files
    for r in results:
        mean_time_curv = r['time_curvs_deg'][:T_minus_2].mean(axis=1)
        axs[0, 1].plot(np.arange(T_minus_2), mean_time_curv, label=r['label'])
    axs[0, 1].set_title("Mean curvature over time (deg, avg over layers)")
    axs[0, 1].set_xlabel("time step index (between steps)")
    axs[0, 1].set_ylabel("curvature (deg)")
    # axs[0, 1].legend()

    # (3) Distance over layers (avg over time), all files
    for r in results:
        mean_layer_dist = r['layer_dists'][:L_minus_1].mean(axis=1)
        axs[1, 0].plot(np.arange(L_minus_1), mean_layer_dist, label=r['label'])
    axs[1, 0].set_title("Mean distance over layers (avg over time)")
    axs[1, 0].set_xlabel("layer index")
    axs[1, 0].set_ylabel("distance")
    # axs[1, 0].legend()

    # (4) Curvature over layers (avg over time), all files
    for r in results:
        mean_layer_curv = r['layer_curvs_deg'][:L_minus_2].mean(axis=1)
        axs[1, 1].plot(np.arange(L_minus_2), mean_layer_curv, label=r['label'])
    axs[1, 1].set_title("Mean curvature over layers (deg, avg over time)")
    axs[1, 1].set_xlabel("layer index (between layers)")
    axs[1, 1].set_ylabel("curvature (deg)")
    # axs[1, 1].legend()

    plt.tight_layout()
    out_name = f"geometry_overlay_{title_suffix}.png" if title_suffix else "geometry_overlay.png"
    plt.savefig(out_name)
    print(f"Saved overlay plot to {out_name}")


# ---------------- Example usage ----------------

def main():
    # Directory containing your .npy files
    npy_dir = "/n/netscratch/sham_lab/Lab/chloe00/pi0_activations"
    pattern = os.path.join(npy_dir, "*.npy")
    npy_paths = sorted(glob.glob(pattern))

    results = []
    for npy_path in npy_paths:
        print(f"Processing {npy_path}")
        embeds = np.load(npy_path)  # expected shape: (T, L, D) or (T, L, B, D)
        print("  Loaded embeddings with shape:", embeds.shape)

        time_dists, time_curvs_deg, layer_dists, layer_curvs_deg = compute_time_and_layer_metrics(embeds)

        print("  time_dists shape:", time_dists.shape)
        print("  time_curvs_deg shape:", time_curvs_deg.shape)
        print("  layer_dists shape:", layer_dists.shape)
        print("  layer_curvs_deg shape:", layer_curvs_deg.shape)

        label = os.path.basename(npy_path).replace(".npy", "")
        results.append(
            dict(
                time_dists=time_dists,
                time_curvs_deg=time_curvs_deg,
                layer_dists=layer_dists,
                layer_curvs_deg=layer_curvs_deg,
                label=label,
            )
        )

    # Use directory name as a suffix for the figure title/filename
    title_suffix = os.path.basename(os.path.normpath(npy_dir))
    plot_overlay_geometry_metrics(results, title_suffix=title_suffix)


if __name__ == "__main__":
    main()