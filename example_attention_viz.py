"""
Example script demonstrating how to visualize attention on video frames.

This script shows how to:
1. Load a video and extract frames
2. Run the Pi0 model on a frame
3. Extract and visualize attention patterns
"""

import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from openpi.models import gemma
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.models import model as _model
from openpi.models import tokenizer
from utils import get_frame_opencv, index_libero_dataset
from visualize_attention import (
    AttentionVisualizationConfig,
    visualize_attention_on_frame,
    enable_attention_recording,
    disable_attention_recording,
    get_recorded_attention_weights,
)
from visualize_text_attention import (
    analyze_text_attention_from_recorded,
)
from visualize_combined_attention import (
    visualize_combined_attention,
    visualize_multimodal_attention_evolution,
)


def load_model_from_checkpoint(checkpoint_path: str):
    """
    Load a Pi0 model from a checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint

    Returns:
        Loaded Pi0 model
    """
    # TODO: Implement model loading based on your checkpoint format
    # This is a placeholder - you'll need to adapt this based on how you save/load models

    # Example configuration (adjust based on your model)
    config = pi0_config.Pi0Config(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        action_dim=7,
        action_horizon=10,
        max_token_len=256,
        pi05=True,
        dtype="bfloat16"
    )

    # Initialize model
    rngs = jax.random.PRNGKey(0)
    # You would load the actual checkpoint weights here
    # For now, this creates a model with random weights
    model = pi0.Pi0(config, rngs)

    print("WARNING: Model loaded with random weights! Load your checkpoint here.")

    return model


def create_observation_from_frame(
    frame_rgb: np.ndarray,
    prompt_text: str,
    state: np.ndarray = None
) -> _model.Observation:
    """
    Create an observation object from a video frame and prompt.

    Args:
        frame_rgb: RGB image array of shape (H, W, 3)
        prompt_text: Text prompt describing the task
        state: Robot state (default: zeros)

    Returns:
        Observation object
    """
    # Resize frame to expected size (e.g., 224x224 for SigLIP)
    from PIL import Image
    pil_image = Image.fromarray(frame_rgb)
    pil_image = pil_image.resize((224, 224), Image.Resampling.LANCZOS)
    frame_resized = np.array(pil_image)

    # Convert to expected format
    # Add batch dimension and normalize to [0, 1]
    frame_batch = frame_resized[None, ...].astype(np.float32) / 255.0
    frame_jax = jnp.array(frame_batch)

    # Tokenize prompt
    tok = tokenizer.get_tokenizer()
    tokens = tok.encode(prompt_text)
    tokenized_prompt = jnp.array([tokens], dtype=jnp.int32)
    tokenized_prompt_mask = jnp.ones_like(tokenized_prompt, dtype=jnp.bool_)

    # Create state (use zeros if not provided)
    if state is None:
        state = np.zeros(7, dtype=np.float32)  # Adjust dimension as needed
    state_jax = jnp.array(state[None, ...])

    # Create observation
    observation = _model.Observation(
        images={'primary': frame_jax},
        image_masks={'primary': jnp.ones((1,), dtype=jnp.bool_)},
        state=state_jax,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask
    )

    return observation


def visualize_video_frame_attention(
    video_path: str,
    frame_idx: int,
    prompt: str,
    checkpoint_path: str,
    output_path: str = "attention_visualization.png",
    layers: list = [0, 8, 17],
    viz_type: str = "image"  # "image", "text", "combined", or "all"
):
    """
    Visualize attention on a specific video frame.

    Args:
        video_path: Path to video file
        frame_idx: Frame index to visualize
        prompt: Text prompt for the task
        checkpoint_path: Path to model checkpoint
        output_path: Where to save the visualization
        layers: Which layers to visualize
        viz_type: Type of visualization - "image", "text", "combined", or "all"
    """
    print(f"Loading frame {frame_idx} from {video_path}...")
    frame_rgb = get_frame_opencv(video_path, frame_idx)

    print("Loading model...")
    model = load_model_from_checkpoint(checkpoint_path)

    print("Creating observation...")
    observation = create_observation_from_frame(frame_rgb, prompt)

    # Enable attention recording
    enable_attention_recording()

    print("Running model and extracting attention...")

    # Run model to record attention
    rng = jax.random.PRNGKey(0)
    try:
        _ = model.sample_actions(rng, observation, num_steps=1)
    except Exception as e:
        print(f"Warning: sample_actions failed: {e}")
        dummy_actions = jnp.zeros((1, model.action_horizon, model.action_dim))
        _ = model.compute_loss(rng, observation, dummy_actions, train=False)

    # Get recorded attention
    attention_dict = get_recorded_attention_weights()
    disable_attention_recording()

    if not attention_dict:
        print("⚠ No attention weights were recorded!")
        return

    # Calculate number of image tokens
    num_image_tokens = 256  # Adjust based on your image size and patch size

    # Generate visualizations based on type
    base_path = output_path.rsplit('.', 1)[0]
    ext = output_path.rsplit('.', 1)[1] if '.' in output_path else 'png'

    if viz_type in ["image", "all"]:
        print("\n📊 Generating image attention visualization...")
        config = AttentionVisualizationConfig(
            layers_to_viz=layers,
            colormap='jet',
            overlay_alpha=0.5
        )
        visualizations = visualize_attention_on_frame(
            model,
            observation,
            frame_rgb,
            config,
            output_path=f"{base_path}_image.{ext}" if viz_type == "all" else output_path,
            query_token_type="action"
        )
        if visualizations:
            print(f"  ✓ Saved image attention visualization")

    if viz_type in ["text", "all"]:
        print("\n📝 Generating text attention visualization...")
        analyze_text_attention_from_recorded(
            attention_dict,
            observation.tokenized_prompt,
            num_image_tokens,
            prompt,
            layers_to_analyze=layers,
            output_path=f"{base_path}_text.{ext}" if viz_type == "all" else output_path
        )

    if viz_type in ["combined", "all"]:
        print("\n🔄 Generating combined attention visualization...")
        token_ids = observation.tokenized_prompt[0].tolist()
        visualize_combined_attention(
            frame_rgb,
            prompt,
            token_ids,
            attention_dict,
            num_image_tokens,
            layer_idx=layers[-1],  # Use last layer for combined viz
            output_path=f"{base_path}_combined.{ext}" if viz_type == "all" else output_path
        )

        print("\n📈 Generating attention evolution visualization...")
        visualize_multimodal_attention_evolution(
            frame_rgb,
            prompt,
            token_ids,
            attention_dict,
            num_image_tokens,
            layers_to_viz=layers,
            output_path=f"{base_path}_evolution.{ext}"
        )

    print(f"\n✓ All visualizations saved!")
    print(f"  Base path: {base_path}")
    print(f"  Visualized layers: {layers}")


def visualize_episode_attention(
    data_root: str,
    activations_root: str,
    checkpoint_path: str,
    group: str = "90",
    episode_idx: int = 0,
    frame_indices: list = [0, 10, 20, 30],
    output_dir: str = "attention_viz_output",
    viz_type: str = "all"
):
    """
    Visualize attention across multiple frames from an episode.

    Args:
        data_root: Root directory for libero data
        activations_root: Root directory for saved activations
        checkpoint_path: Path to model checkpoint
        group: Libero group name (e.g., "90", "goal", "spatial")
        episode_idx: Which episode to visualize
        frame_indices: Which frames to visualize
        output_dir: Where to save visualizations
        viz_type: Type of visualization - "image", "text", "combined", or "all"
    """
    # Index dataset
    print("Indexing dataset...")
    episodes = index_libero_dataset(
        data_root=data_root,
        activations_root=activations_root,
        groups=(group,)
    )

    if episode_idx >= len(episodes):
        print(f"Error: Episode index {episode_idx} out of range (max: {len(episodes)-1})")
        return

    episode = episodes[episode_idx]
    print(f"\nVisualizing episode {episode_idx}:")
    print(f"  Group: {episode.group}")
    print(f"  Episode ID: {episode.episode_id}")
    print(f"  Prompt: {episode.prompt}")
    print(f"  Video: {episode.video_path}")

    # Load model
    print("\nLoading model...")
    model = load_model_from_checkpoint(checkpoint_path)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Visualize each frame
    for frame_idx in frame_indices:
        print(f"\n{'='*60}")
        print(f"Processing frame {frame_idx}...")
        print(f"{'='*60}")

        output_path = os.path.join(
            output_dir,
            f"{episode.group}_{episode.episode_id}_frame{frame_idx:03d}.png"
        )

        try:
            visualize_video_frame_attention(
                video_path=episode.video_path,
                frame_idx=frame_idx,
                prompt=episode.prompt,
                checkpoint_path=checkpoint_path,
                output_path=output_path,
                layers=[0, 8, 17],  # First, middle, last layers
                viz_type=viz_type
            )
        except Exception as e:
            print(f"  ✗ Error processing frame {frame_idx}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"✓ All visualizations saved to {output_dir}/")
    print(f"{'='*60}")


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Visualize VLA attention on video frames')
    parser.add_argument('--mode', type=str, choices=['single', 'episode'], default='single',
                        help='Visualization mode: single frame or full episode')

    # Common arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')

    # Single frame mode arguments
    parser.add_argument('--video', type=str, help='Path to video file (single mode)')
    parser.add_argument('--frame-idx', type=int, default=0, help='Frame index (single mode)')
    parser.add_argument('--prompt', type=str, default='pick up the bowl',
                        help='Text prompt (single mode)')
    parser.add_argument('--output', type=str, default='attention_viz.png',
                        help='Output path (single mode)')

    # Episode mode arguments
    parser.add_argument('--data-root', type=str, default='data/libero',
                        help='Data root directory (episode mode)')
    parser.add_argument('--activations-root', type=str,
                        default='/n/netscratch/sham_lab/Lab/chloe00/pi0_activations',
                        help='Activations root directory (episode mode)')
    parser.add_argument('--group', type=str, default='90',
                        help='Libero group (episode mode)')
    parser.add_argument('--episode-idx', type=int, default=0,
                        help='Episode index (episode mode)')
    parser.add_argument('--frames', type=int, nargs='+', default=[0, 10, 20, 30],
                        help='Frame indices to visualize (episode mode)')
    parser.add_argument('--output-dir', type=str, default='attention_viz_output',
                        help='Output directory (episode mode)')

    # Visualization arguments
    parser.add_argument('--layers', type=int, nargs='+', default=[0, 8, 17],
                        help='Layers to visualize')
    parser.add_argument('--viz-type', type=str, choices=['image', 'text', 'combined', 'all'],
                        default='all',
                        help='Type of visualization: image (visual attention), text (linguistic attention), '
                             'combined (both), or all (generates all types)')

    args = parser.parse_args()

    if args.mode == 'single':
        if not args.video:
            parser.error('--video is required for single mode')

        visualize_video_frame_attention(
            video_path=args.video,
            frame_idx=args.frame_idx,
            prompt=args.prompt,
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            layers=args.layers,
            viz_type=args.viz_type
        )

    elif args.mode == 'episode':
        visualize_episode_attention(
            data_root=args.data_root,
            activations_root=args.activations_root,
            checkpoint_path=args.checkpoint,
            group=args.group,
            episode_idx=args.episode_idx,
            frame_indices=args.frames,
            output_dir=args.output_dir,
            viz_type=args.viz_type
        )


if __name__ == '__main__':
    main()
