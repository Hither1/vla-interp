"""
End-to-end script to visualize attention on video frames for OpenPI Pi0.

Fixes included:
- Loads Orbax OCDBT checkpoints from a directory (e.g. .../params with manifest.ocdbt)
- Infers action_dim (and state_dim when possible) from checkpoint params to avoid shape mismatches
- Uses PaligemmaTokenizer from openpi.models.tokenizer (NOT FSQ tokenizer)
- Properly handles Pi0 vs Pi05 tokenization (state as continuous vs discrete input)
- Pads/truncates prompt to max_token_len

Usage examples:
  python example_attention_viz.py --mode single \
    --checkpoint ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
    --video /path/to/video.mp4 --frame-idx 0 --prompt "pick up the bowl" \
    --output attention_viz.png --viz-type all

  python example_attention_viz.py --mode episode \
    --checkpoint ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
    --data-root data/libero --activations-root /path/to/pi0_activations \
    --group 90 --episode-idx 0 --frames 0 10 20 30 --output-dir attention_viz_output
"""

import os
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image
import matplotlib.pyplot as plt

# Add src to path (your repo layout)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from openpi.models import pi0_config
from openpi.models import model as _model
from openpi.models import gemma
from openpi.models import tokenizer as _tokenizer

from utils import get_frame_opencv, index_libero_dataset

from visualize_attention import (
    AttentionVisualizationConfig,
    visualize_attention_on_frame,
    enable_attention_recording,
    disable_attention_recording,
    get_recorded_attention_weights,
)
from visualize_text_attention import analyze_text_attention_from_recorded
from visualize_combined_attention import (
    visualize_combined_attention,
    visualize_multimodal_attention_evolution,
)


# -------------------------
# Checkpoint + model loading
# -------------------------

def _resolve_params_dir(checkpoint_path: str) -> Path:
    """Accepts either ckpt root or ckpt/params; returns params directory."""
    ckpt = Path(checkpoint_path).expanduser()
    if ckpt.is_dir() and (ckpt / "params").is_dir():
        params_dir = ckpt / "params"
    else:
        params_dir = ckpt
    if not params_dir.is_dir():
        raise ValueError(f"Checkpoint params path is not a directory: {params_dir}")
    # quick sanity for OCDBT
    if not (params_dir / "manifest.ocdbt").exists():
        # Not strictly required, but very common for OCDBT.
        print(f"[warn] params_dir doesn't contain manifest.ocdbt: {params_dir}")
    return params_dir


def _infer_action_dim(params) -> int:
    try:
        return int(params["action_in_proj"]["kernel"].shape[0])
    except Exception as e:
        raise ValueError(
            "Could not infer action_dim from params['action_in_proj']['kernel']. "
            f"Available top-level keys include: {list(params.keys())[:30]} ...\n"
            f"Original error: {e}"
        )


def _infer_state_dim(params, fallback: int = 7) -> int:
    """
    Try to infer state_dim from checkpoint params. If unavailable, fallback.
    Commonly there may be something like params['state_in_proj']['kernel'] of shape (state_dim, hidden).
    """
    candidates = [
        ("state_in_proj", "kernel"),
        ("state_proj", "kernel"),
        ("state_encoder", "kernel"),
    ]
    for a, b in candidates:
        try:
            if a in params and b in params[a]:
                return int(params[a][b].shape[0])
        except Exception:
            pass
    print(f"[warn] Could not infer state_dim from params; using fallback={fallback}")
    return fallback


def load_model_from_checkpoint(
    checkpoint_path: str,
    paligemma_variant: str = "gemma_2b",
    action_expert_variant: str = "gemma_300m",
    action_horizon: int = 10,
    max_token_len: int = 256,
    pi05: bool = True,
    dtype: str = "bfloat16",
):
    """
    Restore params from Orbax OCDBT checkpoint directory, infer action_dim/state_dim,
    build matching Pi0Config, and load model.
    """
    params_dir = _resolve_params_dir(checkpoint_path)

    # Restore parameter PyTree
    params = _model.restore_params(
        params_dir,
        restore_type=np.ndarray,
        dtype=jnp.bfloat16 if dtype == "bfloat16" else None,
    )

    action_dim = _infer_action_dim(params)
    state_dim = _infer_state_dim(params, fallback=7)

    # Show inferred dims
    hidden = int(params["action_in_proj"]["kernel"].shape[1])
    print(f"[ckpt] inferred action_dim={action_dim}, hidden={hidden}, inferred state_dim={state_dim}")

    config = pi0_config.Pi0Config(
        paligemma_variant=paligemma_variant,
        action_expert_variant=action_expert_variant,
        action_dim=action_dim,
        action_horizon=action_horizon,
        max_token_len=max_token_len,
        pi05=pi05,
        dtype=dtype,
        # If your Pi0Config supports state_dim explicitly, you can add it here.
        # state_dim=state_dim,
    )

    model = config.load(params)
    return model, config, state_dim


# -------------------------
# Text tokenizer (Gemma)
# -------------------------

def get_paligemma_tokenizer(max_token_len: int):
    """
    Create a PaligemmaTokenizer instance.
    """
    return _tokenizer.PaligemmaTokenizer(max_len=max_token_len)


def tokenize_prompt(text_tok, prompt_text: str, state: np.ndarray = None):
    """
    Tokenize prompt text using PaligemmaTokenizer.
    Returns (token_ids, mask) with batch dimension added.
    """
    # PaligemmaTokenizer.tokenize returns (tokens, mask) arrays already padded/truncated
    tokens, mask = text_tok.tokenize(prompt_text, state=state)

    # Add batch dimension
    tokenized_prompt = jnp.array([tokens], dtype=jnp.int32)
    tokenized_prompt_mask = jnp.array([mask], dtype=jnp.bool_)
    return tokenized_prompt, tokenized_prompt_mask


# -------------------------
# Observation construction
# -------------------------

def create_observation_from_frame(
    frame_rgb: np.ndarray,
    prompt_text: str,
    paligemma_variant: str,
    max_token_len: int,
    state_dim: int,
    state: np.ndarray = None,
    pi05: bool = False,
) -> _model.Observation:
    """
    Create an Observation from an RGB frame and prompt.
    """

    # Resize frame to expected size for vision encoder (often 224x224)
    pil_image = Image.fromarray(frame_rgb)
    pil_image = pil_image.resize((224, 224), Image.Resampling.LANCZOS)
    frame_resized = np.array(pil_image)

    # Normalize and add batch dimension
    frame_batch = frame_resized[None, ...].astype(np.float32) / 255.0
    frame_jax = jnp.array(frame_batch)

    # State
    if state is None:
        state = np.zeros(state_dim, dtype=np.float32)
    else:
        state = np.asarray(state, dtype=np.float32)
        if state.shape[-1] != state_dim:
            raise ValueError(f"state has dim {state.shape[-1]} but expected state_dim={state_dim}")

    # Text tokenizer (Gemma)
    # For Pi05, pass state to tokenizer (it will be embedded in tokens)
    # For Pi0, pass state=None to tokenizer (state will be continuous input)
    text_tok = get_paligemma_tokenizer(max_token_len)
    tokenized_prompt, tokenized_prompt_mask = tokenize_prompt(
        text_tok, prompt_text, state=state if pi05 else None
    )

    state_jax = jnp.array(state[None, ...])

    # Model expects three camera views: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
    # For single-camera visualization, use the same frame for all three views
    observation = _model.Observation(
        images={
            "base_0_rgb": frame_jax,
            "left_wrist_0_rgb": frame_jax,
            "right_wrist_0_rgb": frame_jax,
        },
        image_masks={
            "base_0_rgb": jnp.ones((1,), dtype=jnp.bool_),
            "left_wrist_0_rgb": jnp.ones((1,), dtype=jnp.bool_),
            "right_wrist_0_rgb": jnp.ones((1,), dtype=jnp.bool_),
        },
        state=state_jax,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )
    return observation


# -------------------------
# Visualization routines
# -------------------------

def visualize_video_frame_attention(
    video_path: str,
    frame_idx: int,
    prompt: str,
    checkpoint_path: str,
    output_path: str = "attention_visualization.png",
    layers: list = (0, 8, 17),
    viz_type: str = "image",  # "image", "text", "combined", "all"
    paligemma_variant: str = "gemma_2b",
    action_expert_variant: str = "gemma_300m",
    action_horizon: int = 10,
    max_token_len: int = 256,
    pi05: bool = True,
    num_image_tokens: int = 256,  # used for text/combined parsing; adjust if needed
):
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Loading frame {frame_idx} from {video_path}...")
    frame_rgb = get_frame_opencv(video_path, frame_idx)

    print("Loading model from checkpoint...")
    model, cfg, state_dim = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        paligemma_variant=paligemma_variant,
        action_expert_variant=action_expert_variant,
        action_horizon=action_horizon,
        max_token_len=max_token_len,
        pi05=pi05,
        dtype="bfloat16",
    )

    print("Creating observation...")
    observation = create_observation_from_frame(
        frame_rgb=frame_rgb,
        prompt_text=prompt,
        paligemma_variant=paligemma_variant,
        max_token_len=max_token_len,
        state_dim=state_dim,
        pi05=pi05,
    )

    # Enable attention recording
    enable_attention_recording()

    print("Running model and recording attention...")
    rng = jax.random.PRNGKey(0)

    # Try sample_actions first; fall back to compute_loss if needed
    try:
        _ = model.sample_actions(rng, observation, num_steps=1)
    except Exception as e:
        print(f"[warn] sample_actions failed: {e}\nFalling back to compute_loss(train=False).")
        dummy_actions = jnp.zeros((1, cfg.action_horizon, cfg.action_dim), dtype=jnp.float32)
        _ = model.compute_loss(rng, observation, dummy_actions, train=False)

    attention_dict = get_recorded_attention_weights()
    disable_attention_recording()

    if not attention_dict:
        print("⚠ No attention weights were recorded! Check that attention hooks are installed correctly.")
        return

    base_path = output_path.rsplit(".", 1)[0]
    ext = output_path.rsplit(".", 1)[1] if "." in output_path else "png"

    layers = list(layers)

    if viz_type in ["image", "all"]:
        print("\n📊 Generating image attention visualization...")
        config = AttentionVisualizationConfig(
            layers_to_viz=layers,
            colormap="jet",
            overlay_alpha=0.5,
        )
        _ = visualize_attention_on_frame(
            model,
            observation,
            frame_rgb,
            config,
            output_path=f"{base_path}_image.{ext}" if viz_type == "all" else output_path,
            query_token_type="action",
        )
        print("  ✓ Saved image attention visualization")

    if viz_type in ["text", "all"]:
        print("\n📝 Generating text attention visualization...")
        analyze_text_attention_from_recorded(
            attention_dict=attention_dict,
            tokenized_prompt=observation.tokenized_prompt,
            num_image_tokens=num_image_tokens,
            prompt_text=prompt,
            layers_to_analyze=layers,
            output_path=f"{base_path}_text.{ext}" if viz_type == "all" else output_path,
        )
        print("  ✓ Saved text attention visualization")

    if viz_type in ["combined", "all"]:
        print("\n🔄 Generating combined attention visualization...")
        token_ids = observation.tokenized_prompt[0].tolist()

        # Generate separate combined visualization for each layer
        for layer_idx in layers:
            layer_key = f'layer_{layer_idx}'
            if layer_key not in attention_dict:
                print(f"  Warning: Layer {layer_idx} not found in attention_dict, skipping")
                continue

            layer_output_path = f"{base_path}_combined_layer{layer_idx}.{ext}"
            fig = visualize_combined_attention(
                frame_rgb=frame_rgb,
                prompt_text=prompt,
                token_ids=token_ids,
                attention_dict=attention_dict,
                num_image_tokens=num_image_tokens,
                layer_idx=layer_idx,
                output_path=layer_output_path,
            )
            if fig is not None:
                plt.close(fig)
            print(f"  ✓ Saved combined attention for layer {layer_idx}")

        print("\n📈 Generating attention evolution visualization...")
        visualize_multimodal_attention_evolution(
            frame_rgb=frame_rgb,
            prompt_text=prompt,
            token_ids=token_ids,
            attention_dict=attention_dict,
            num_image_tokens=num_image_tokens,
            layers_to_viz=layers,
            output_path=f"{base_path}_evolution.{ext}",
        )
        # if fig is not None:
        #     plt.close(fig)
        # print("  ✓ Saved attention evolution visualization")

    print("\n✓ Done.")
    print(f"  Base path: {base_path}")
    print(f"  Layers: {layers}")
    print(f"  action_dim: {cfg.action_dim}, action_horizon: {cfg.action_horizon}, max_token_len: {cfg.max_token_len}")


def visualize_episode_attention(
    data_root: str,
    activations_root: str,
    checkpoint_path: str,
    group: str = "90",
    episode_idx: int = 0,
    frame_indices: list = (0, 10, 20, 30),
    output_dir: str = "attention_viz_output",
    viz_type: str = "all",
    paligemma_variant: str = "gemma_2b",
    action_expert_variant: str = "gemma_300m",
    action_horizon: int = 10,
    max_token_len: int = 256,
    pi05: bool = True,
    num_image_tokens: int = 256,
    layers: list = (0, 8, 17),
):
    print("Indexing dataset...")
    episodes = index_libero_dataset(
        data_root=data_root,
        activations_root=activations_root,
        groups=(group,),
    )

    if episode_idx >= len(episodes):
        print(f"Error: Episode index {episode_idx} out of range (max: {len(episodes) - 1})")
        return

    episode = episodes[episode_idx]
    print(f"\nVisualizing episode {episode_idx}:")
    print(f"  Group: {episode.group}")
    print(f"  Episode ID: {episode.episode_id}")
    print(f"  Prompt: {episode.prompt}")
    print(f"  Video: {episode.video_path}")

    os.makedirs(output_dir, exist_ok=True)

    for frame_idx in frame_indices:
        print(f"\n{'=' * 60}")
        print(f"Processing frame {frame_idx}...")
        print(f"{'=' * 60}")

        output_path = os.path.join(
            output_dir,
            f"{episode.group}_{episode.episode_id}_frame{frame_idx:03d}.png",
        )

        try:
            visualize_video_frame_attention(
                video_path=episode.video_path,
                frame_idx=frame_idx,
                prompt=episode.prompt,
                checkpoint_path=checkpoint_path,
                output_path=output_path,
                layers=layers,
                viz_type=viz_type,
                paligemma_variant=paligemma_variant,
                action_expert_variant=action_expert_variant,
                action_horizon=action_horizon,
                max_token_len=max_token_len,
                pi05=pi05,
                num_image_tokens=num_image_tokens,
            )
        except Exception as e:
            print(f"  ✗ Error processing frame {frame_idx}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"✓ All visualizations saved to {output_dir}/")
    print(f"{'=' * 60}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize VLA attention on video frames (OpenPI Pi0)")

    parser.add_argument("--mode", type=str, choices=["single", "episode"], default="single",
                        help="Visualization mode: single frame or full episode")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory (either root or root/params)")

    # Single mode
    parser.add_argument("--video", type=str, help="Path to video file (single mode)")
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame index (single mode)")
    parser.add_argument("--prompt", type=str, default="pick up the bowl", help="Text prompt (single mode)")
    parser.add_argument("--output", type=str, default="outputs_attention/attention_viz.png", help="Output path (single mode)")

    # Episode mode
    parser.add_argument("--data-root", type=str, default="data/libero", help="Data root directory (episode mode)")
    parser.add_argument("--activations-root", type=str,
                        default="/n/netscratch/sham_lab/Lab/chloe00/pi0_activations",
                        help="Activations root directory (episode mode)")
    parser.add_argument("--group", type=str, default="90", help="Libero group (episode mode)")
    parser.add_argument("--episode-idx", type=int, default=0, help="Episode index (episode mode)")
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 10, 20, 30],
                        help="Frame indices to visualize (episode mode)")
    parser.add_argument("--output-dir", type=str, default="outputs_attention",
                        help="Output directory (episode mode)")

    # Model/config knobs
    parser.add_argument("--paligemma-variant", type=str, default="gemma_2b",
                        help="paligemma_variant used for text tokenizer/model")
    parser.add_argument("--action-expert-variant", type=str, default="gemma_300m",
                        help="action_expert_variant used by Pi0")
    parser.add_argument("--action-horizon", type=int, default=10, help="Action horizon (must match checkpoint training)")
    parser.add_argument("--max-token-len", type=int, default=256, help="Max prompt token length")
    parser.add_argument("--pi05", action="store_true", help="Enable pi05 variant behavior")
    parser.add_argument("--no-pi05", action="store_true", help="Disable pi05 variant behavior")

    # Visualization knobs
    parser.add_argument("--layers", type=int, nargs="+", default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17], help="Layers to visualize")
    parser.add_argument("--viz-type", type=str, choices=["image", "text", "combined", "all"], default="all",
                        help="Type of visualization")
    parser.add_argument("--num-image-tokens", type=int, default=256,
                        help="Number of image tokens (used for text/combined parsing). Adjust if needed.")

    args = parser.parse_args()

    # resolve pi05 flag
    if args.no_pi05:
        pi05 = False
    else:
        pi05 = bool(args.pi05) if args.pi05 else True  # default True unless explicitly disabled

    if args.mode == "single":
        if not args.video:
            parser.error("--video is required for single mode")

        visualize_video_frame_attention(
            video_path=args.video,
            frame_idx=args.frame_idx,
            prompt=args.prompt,
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            layers=args.layers,
            viz_type=args.viz_type,
            paligemma_variant=args.paligemma_variant,
            action_expert_variant=args.action_expert_variant,
            action_horizon=args.action_horizon,
            max_token_len=args.max_token_len,
            pi05=pi05,
            num_image_tokens=args.num_image_tokens,
        )
    else:
        visualize_episode_attention(
            data_root=args.data_root,
            activations_root=args.activations_root,
            checkpoint_path=args.checkpoint,
            group=args.group,
            episode_idx=args.episode_idx,
            frame_indices=args.frames,
            output_dir=args.output_dir,
            viz_type=args.viz_type,
            paligemma_variant=args.paligemma_variant,
            action_expert_variant=args.action_expert_variant,
            action_horizon=args.action_horizon,
            max_token_len=args.max_token_len,
            pi05=pi05,
            num_image_tokens=args.num_image_tokens,
            layers=args.layers,
        )


if __name__ == "__main__":
    main()