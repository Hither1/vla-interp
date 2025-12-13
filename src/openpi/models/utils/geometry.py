import numpy as np
import torch

def calculate_geometry_metrics(embeddings: np.ndarray):
    """
    Calculates stepwise distance and temporal curvature for a sequence of embeddings.

    Args:
        embeddings: A NumPy array of shape (N, D), where N is the number of frames
                    and D is the embedding dimension.
    
    Returns:
        tuple: (distances, curvatures_deg)
               - distances: NumPy array of stepwise Euclidean distances (N-1 elements).
               - curvatures_deg: NumPy array of temporal curvatures in degrees (N-2 elements).
    """
    # 1. Calculate Displacement Vectors (v_i)
    # v_i = z_{i+1} - z_i. Shape: (N-1, D)
    displacement_vectors = embeddings[1:] - embeddings[:-1]

    # 2. Calculate Stepwise Distance (d_i)
    # The L2 norm (Euclidean distance) of the displacement vectors. Shape: (N-1,)
    distances = np.linalg.norm(displacement_vectors, axis=1)

    # 3. Calculate Curvature (kappa_i)
    # We need the dot product (v_i * v_{i+1}) and the product of their norms.
    
    # Select v_i (displacement_vectors[:-1]) and v_{i+1} (displacement_vectors[1:])
    v_i = displacement_vectors[:-1]
    v_i_plus_1 = displacement_vectors[1:]

    # Dot product: v_i * v_{i+1}. Shape: (N-2,)
    dot_products = np.sum(v_i * v_i_plus_1, axis=1)

    # Norms: ||v_i||_2 and ||v_{i+1}||_2
    norms_v_i = np.linalg.norm(v_i, axis=1)
    norms_v_i_plus_1 = np.linalg.norm(v_i_plus_1, axis=1)

    # Product of norms: ||v_i||_2 * ||v_{i+1}||_2
    norm_products = norms_v_i * norms_v_i_plus_1
    
    # Avoid division by zero: if product of norms is zero, the curvature is undefined.
    # In the paper, small epsilon is often used, but for simplicity, we use a mask.
    # Curvature (cosine) is (v_i * v_{i+1}) / (||v_i||_2 * ||v_{i+1}||_2)
    cos_kappa = np.zeros_like(dot_products) # Initialize
    
    # Calculate non-zero cases
    non_zero_mask = norm_products > 1e-6 # Use a small threshold
    cos_kappa[non_zero_mask] = dot_products[non_zero_mask] / norm_products[non_zero_mask]
    
    # Clamp to [-1, 1] for arccos stability (due to floating point errors)
    cos_kappa = np.clip(cos_kappa, -1.0, 1.0)

    # Curvature in radians: kappa = arccos(cos_kappa)
    curvatures_rad = np.arccos(cos_kappa)
    
    # Convert to degrees
    curvatures_deg = np.rad2deg(curvatures_rad)

    return distances, curvatures_deg

# --- Example Usage ---
# 1. Simulate Frame Embeddings (N=5 frames, D=512 dimension)
# N_frames = 5
# D_dim = 512
# A simple, slightly curved trajectory for demonstration
# z1 = np.random.rand(D_dim)
# z2 = z1 + np.random.rand(D_dim) * 0.1
# z3 = z2 + np.random.rand(D_dim) * 0.1
# z4 = z3 + np.random.rand(D_dim) * 0.1
# z5 = z4 + np.random.rand(D_dim) * 0.1

# Stack them into the required (N, D) shape
# simulated_embeddings = np.stack([z1, z2, z3, z4, z5]) 

# 2. Calculate Metrics
# distances, curvatures_deg = calculate_geometry_metrics(simulated_embeddings)

# print(f"Number of frames (N): {simulated_embeddings.shape[0]}")
# print(f"Stepwise Distances (N-1={len(distances)}): {distances}")
# print(f"Temporal Curvatures in Degrees (N-2={len(curvatures_deg)}): {curvatures_deg}")