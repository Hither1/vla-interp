"""
Visualize attention patterns from the Pi0 VLA model on video frames.

This script extracts attention weights from the model and overlays them as heatmaps
on the input video frames to show which parts of the image the VLA is attending to.
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from typing import Dict, List, Tuple, Optional
import jax
import jax.numpy as jnp
from dataclasses import dataclass

# Add src to path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'src'))

from openpi.models import gemma
from openpi.models import pi0
from openpi.models import model as _model
from utils import get_frame_opencv, Episode


@dataclass
class AttentionVisualizationConfig:
    """Configuration for attention visualization."""
    # Which layer(s) to visualize (e.g., [0, 5, 11] for first, middle, last)
    layers_to_viz: List[int] = None
    # Which head(s) to visualize (None = average over all heads)
    heads_to_viz: Optional[List[int]] = None
    # Colormap for heatmap
    colormap: str = 'jet'
    # Alpha blending for overlay (0=transparent, 1=opaque)
    overlay_alpha: float = 0.5
    # Output resolution
    output_width: int = 1280
    output_height: int = 720

    def __post_init__(self):
        if self.layers_to_viz is None:
            # Default: visualize first, middle, and last layers
            self.layers_to_viz = [0, 8, 17]


def enable_attention_recording():
    """Enable attention weight recording in the gemma module."""
    gemma.SAVE_ATTENTION_WEIGHTS = True
    gemma.ATTENTION_WEIGHTS.clear()
    gemma._attention_layer_counter = 0  # Reset layer counter for new recording


def disable_attention_recording():
    """Disable attention weight recording."""
    gemma.SAVE_ATTENTION_WEIGHTS = False


def get_recorded_attention_weights() -> Dict[str, np.ndarray]:
    """Get the recorded attention weights and clear the buffer."""
    weights = dict(gemma.ATTENTION_WEIGHTS)
    gemma.ATTENTION_WEIGHTS.clear()
    return weights


def extract_image_attention(
    attention_weights: np.ndarray,
    image_token_start: int,
    image_token_end: int,
    query_token_idx: int,
    head_idx: Optional[int] = None
) -> np.ndarray:
    """
    Extract attention weights from a query token to image tokens.

    Args:
        attention_weights: Attention tensor of shape (B, K, G, T, S)
                          B=batch, K=num_kv_heads, G=num_heads_per_kv, T=query_len, S=key_len
        image_token_start: Start index of image tokens in the sequence
        image_token_end: End index of image tokens in the sequence
        query_token_idx: Which query token to visualize (e.g., action token)
        head_idx: Which head to use (None = average over all heads)

    Returns:
        Attention weights from query to image tokens, shape (num_image_tokens,)
    """
    # Shape: (B, K, G, T, S)
    batch_idx = 0  # Assume batch size 1 for visualization

    if head_idx is not None:
        # Select specific head
        k_idx = head_idx // attention_weights.shape[2]  # Which kv head group
        g_idx = head_idx % attention_weights.shape[2]   # Which head within group
        attn = attention_weights[batch_idx, k_idx, g_idx, query_token_idx, :]
    else:
        # Average over all heads
        # Reshape to (num_heads, T, S)
        attn = attention_weights[batch_idx].reshape(-1, attention_weights.shape[3], attention_weights.shape[4])
        attn = attn[:, query_token_idx, :].mean(axis=0)

    # Extract only image token attention
    image_attn = attn[image_token_start:image_token_end]

    return np.array(image_attn)


def create_attention_heatmap(
    attention_weights: np.ndarray,
    image_shape: Tuple[int, int],
    patch_size: int = 14
) -> np.ndarray:
    """
    Convert attention weights to a 2D heatmap matching the image spatial dimensions.

    Args:
        attention_weights: 1D array of attention weights for image tokens
        image_shape: (height, width) of the original image
        patch_size: Size of image patches (default 14 for SigLIP)

    Returns:
        2D heatmap of shape matching image spatial dimensions
    """
    height, width = image_shape

    # Calculate number of patches
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size

    # Ensure we have the right number of attention weights
    expected_num_patches = num_patches_h * num_patches_w
    if len(attention_weights) != expected_num_patches:
        print(f"Warning: Expected {expected_num_patches} patches but got {len(attention_weights)}")
        # Truncate or pad if necessary
        if len(attention_weights) > expected_num_patches:
            attention_weights = attention_weights[:expected_num_patches]
        else:
            padding = np.zeros(expected_num_patches - len(attention_weights))
            attention_weights = np.concatenate([attention_weights, padding])

    # Reshape to spatial grid
    heatmap = attention_weights.reshape(num_patches_h, num_patches_w)

    # Convert to float32 for OpenCV compatibility (handles bfloat16 from JAX)
    heatmap = np.asarray(heatmap, dtype=np.float32)

    # Upsample to match image resolution
    heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)

    # Normalize to [0, 1]
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    return heatmap


def overlay_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    colormap: str = 'jet',
    alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay an attention heatmap on an image.

    Args:
        image: RGB image array of shape (H, W, 3)
        heatmap: 2D attention heatmap of shape (H, W)
        colormap: Matplotlib colormap name
        alpha: Blending factor (0=only image, 1=only heatmap)

    Returns:
        RGB image with heatmap overlay
    """
    # Ensure image is float in [0, 1]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    # Apply colormap to heatmap
    cmap = cm.get_cmap(colormap)
    colored_heatmap = cmap(heatmap)[:, :, :3]  # Remove alpha channel

    # Blend
    blended = (1 - alpha) * image + alpha * colored_heatmap

    # Convert back to uint8
    blended = (blended * 255).astype(np.uint8)

    return blended


def visualize_attention_on_frame(
    model: pi0.Pi0,
    observation: _model.Observation,
    frame_rgb: np.ndarray,
    config: AttentionVisualizationConfig,
    output_path: Optional[str] = None,
    query_token_type: str = "action"  # "action" or "last"
) -> Dict[int, np.ndarray]:
    """
    Visualize attention patterns on a single frame.

    Args:
        model: The Pi0 model
        observation: Observation containing the frame and prompt
        frame_rgb: The original RGB frame for visualization
        config: Visualization configuration
        output_path: If provided, save visualization to this path
        query_token_type: Which token to use as query ("action" for first action token, "last" for last token)

    Returns:
        Dictionary mapping layer index to visualization image
    """
    # Enable attention recording
    enable_attention_recording()

    # Run a dummy action sampling to trigger the forward pass
    # This will record attention weights
    rng = jax.random.PRNGKey(0)
    try:
        _ = model.sample_actions(rng, observation, num_steps=1)
    except Exception as e:
        print(f"Warning: Model forward pass failed: {e}")
        print("Attempting to use compute_loss instead...")
        # Try with compute_loss if sample_actions fails
        dummy_actions = jnp.zeros((1, model.action_horizon, model.action_dim))
        _ = model.compute_loss(rng, observation, dummy_actions, train=False)

    # Get recorded attention weights
    attention_dict = get_recorded_attention_weights()
    disable_attention_recording()

    if not attention_dict:
        print("Warning: No attention weights were recorded!")
        print("You may need to modify the gemma.py Attention class to call _record_attention_weights")
        return {}

    # Get image dimensions
    image_height, image_width = frame_rgb.shape[:2]

    # Calculate image token range
    # For SigLIP with 14x14 patches on a 224x224 image, we have 16x16=256 tokens
    # Adjust based on your actual image resolution
    patch_size = 14
    num_image_tokens = (image_height // patch_size) * (image_width // patch_size)
    image_token_start = 0
    image_token_end = num_image_tokens

    # Determine which token to use as query
    if query_token_type == "action":
        # First action token (comes after image and text tokens)
        query_token_idx = image_token_end + (observation.tokenized_prompt.shape[1] if observation.tokenized_prompt is not None else 0)
    else:  # "last"
        query_token_idx = -1  # Last token

    # Visualize selected layers
    visualizations = {}

    for layer_idx in config.layers_to_viz:
        layer_key = f'layer_{layer_idx}'

        if layer_key not in attention_dict:
            print(f"Warning: Layer {layer_idx} not found in recorded attention weights")
            continue

        # Get attention weights for this layer
        # Shape: (B, K, G, T, S) where B=batch, K=num_kv_heads, G=num_heads_per_kv, T=query_len, S=key_len
        attn = attention_dict[layer_key][0]  # Get first (and likely only) batch item

        # Extract attention to image tokens
        image_attn = extract_image_attention(
            attn,
            image_token_start,
            image_token_end,
            query_token_idx,
            head_idx=config.heads_to_viz[0] if config.heads_to_viz else None
        )

        # Create heatmap
        heatmap = create_attention_heatmap(
            image_attn,
            (image_height, image_width),
            patch_size=patch_size
        )

        # Overlay on image
        vis = overlay_heatmap_on_image(
            frame_rgb,
            heatmap,
            colormap=config.colormap,
            alpha=config.overlay_alpha
        )

        visualizations[layer_idx] = vis

    # Save if output path provided
    if output_path:
        if len(visualizations) == 1:
            # Single layer - save directly
            layer_idx = list(visualizations.keys())[0]
            cv2.imwrite(output_path, cv2.cvtColor(visualizations[layer_idx], cv2.COLOR_RGB2BGR))
        else:
            # Multiple layers - create a grid
            num_layers = len(visualizations)
            fig, axes = plt.subplots(1, num_layers + 1, figsize=(5 * (num_layers + 1), 5))

            # Show original
            axes[0].imshow(frame_rgb)
            axes[0].set_title('Original')
            axes[0].axis('off')

            # Show each layer
            for idx, (layer_idx, vis) in enumerate(sorted(visualizations.items())):
                axes[idx + 1].imshow(vis)
                axes[idx + 1].set_title(f'Layer {layer_idx}')
                axes[idx + 1].axis('off')

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

    return visualizations


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Visualize VLA attention on video frames')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--frame-idx', type=int, default=0, help='Frame index to visualize')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default='pick up the bowl', help='Text prompt')
    parser.add_argument('--output', type=str, default='attention_viz.png', help='Output path')
    parser.add_argument('--layers', type=int, nargs='+', default=[0, 8, 17], help='Layers to visualize')
    parser.add_argument('--colormap', type=str, default='jet', help='Colormap for heatmap')
    parser.add_argument('--alpha', type=float, default=0.5, help='Overlay alpha')

    args = parser.parse_args()

    # Load frame
    print(f"Loading frame {args.frame_idx} from {args.video}...")
    frame_rgb = get_frame_opencv(args.video, args.frame_idx)

    # TODO: Load model from checkpoint
    # This requires initializing the Pi0 model with the checkpoint
    # You'll need to adapt this based on your model loading code
    print("Loading model...")
    print("ERROR: Model loading not implemented yet!")
    print("Please add model loading code here based on your checkpoint format")
    return

    # TODO: Create observation from frame and prompt
    # observation = create_observation_from_frame(frame_rgb, args.prompt)

    # Create config
    config = AttentionVisualizationConfig(
        layers_to_viz=args.layers,
        colormap=args.colormap,
        overlay_alpha=args.alpha
    )

    # Visualize
    # visualizations = visualize_attention_on_frame(
    #     model,
    #     observation,
    #     frame_rgb,
    #     config,
    #     output_path=args.output
    # )

    print(f"Saved visualization to {args.output}")


if __name__ == '__main__':
    main()
