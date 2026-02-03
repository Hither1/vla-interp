"""
Standalone script to run the PaliGemma VLM backbone on the same visual inputs as the VLA.

This loads the same PaliGemma 2B backbone (SigLIP vision encoder + Gemma 2B LLM) that's used
in the pi0.5 VLA model, and performs autoregressive text generation to see what the VLM
"sees" and "thinks" about the input images.

Task/Prompt Extraction:
    For videos with filenames like "rollout_<task>_trial<N>_<success/failure>.mp4",
    the task is automatically extracted from the filename if --prompt is not provided.
    Example: rollout_put_the_bowl_on_the_stove_trial16_success.mp4 -> "put the bowl on the stove"

Usage:
    # Video with auto-extracted task from filename (no --prompt needed):
    python run_paligemma_standalone.py --video rollout_put_the_bowl_on_the_stove_trial16_success.mp4

    # Video with multiple frames (task auto-extracted):
    python run_paligemma_standalone.py --video rollout_open_the_drawer_trial5_success.mp4 --frame-idx 0 10 20

    # Override auto-extracted task with explicit prompt:
    python run_paligemma_standalone.py --video rollout_task_trial1_success.mp4 --prompt "custom task"

    # Image with explicit prompt (required for images):
    python run_paligemma_standalone.py --image /path/to/image.png --prompt "pick up the bowl"

    # Multiple images with multiple prompts (runs all combinations):
    python run_paligemma_standalone.py --image img1.png img2.png \
        --prompt "Describe the scene" "What objects are visible?"

    # Extract hidden states:
    python run_paligemma_standalone.py --mode hidden_states \
        --video rollout_put_the_bowl_on_the_stove_trial16_success.mp4 --output hidden_states.npz

    # Extract attention maps (no generation):
    python run_paligemma_standalone.py --mode attention \
        --video rollout_put_the_bowl_on_the_stove_trial16_success.mp4 --output attention.npz

    # Visualize attention (simple 3-panel: image | heatmap | overlay):
    python run_paligemma_standalone.py --mode attention --visualize \
        --video rollout_put_the_bowl_on_the_stove_trial16_success.mp4

    # Visualize specific layer:
    python run_paligemma_standalone.py --mode attention --visualize \
        --layers 11 --video rollout_task_trial1_success.mp4

    # Visualize attention from last text token (instead of all text tokens):
    python run_paligemma_standalone.py --mode attention --visualize \
        --query-tokens last --video rollout_task_trial1_success.mp4

    # Show full attention matrix heatmap:
    python run_paligemma_standalone.py --mode attention --show-matrix \
        --layers 0 8 17 --video rollout_task_trial1_success.mp4

    # Save visualization to file:
    python run_paligemma_standalone.py --mode attention --visualize \
        --save-fig attention_viz.png --video rollout_task_trial1_success.mp4
"""

import os
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import gemma as _gemma
from openpi.models import tokenizer as _tokenizer
import sentencepiece
import re

from utils import get_frame_opencv


def enable_attention_saving():
    """Enable attention weight saving in the Gemma model."""
    _gemma.SAVE_ATTENTION_WEIGHTS = True
    _gemma.ATTENTION_WEIGHTS.clear()
    _gemma._attention_layer_counter = 0


def disable_attention_saving():
    """Disable attention weight saving."""
    _gemma.SAVE_ATTENTION_WEIGHTS = False


def get_attention_weights():
    """Get the captured attention weights and reset the counter."""
    weights = dict(_gemma.ATTENTION_WEIGHTS)
    _gemma.ATTENTION_WEIGHTS.clear()
    _gemma._attention_layer_counter = 0
    return weights


def visualize_attention(
    attention_results: dict,
    image_rgb: np.ndarray = None,
    layers: list[int] | None = None,
    heads: list[int] | None = None,
    query_tokens: str | list[int] | None = None,
    show_image_attention: bool = True,
    show_text_attention: bool = True,
    save_path: str | None = None,
):
    """
    Visualize attention maps from PaliGemma.

    Args:
        attention_results: Output from extract_paligemma_attention()
        image_rgb: Original image (optional, for overlay)
        layers: Which layers to visualize (None = all)
        heads: Which attention heads to visualize (None = average across heads)
        query_tokens: Which query tokens to visualize attention FROM:
            - "text": attention from text tokens to image
            - "image": attention from image tokens to text
            - list of ints: specific token indices
            - None: last text token (default)
        show_image_attention: Show attention over image patches
        show_text_attention: Show attention over text tokens
        save_path: Path to save figure (None = display)
    """
    attn_weights = attention_results["attention_weights"]
    token_info = attention_results["token_info"]
    num_img_tokens = token_info["total_image_tokens"]
    num_text_tokens = token_info["num_text_tokens"]
    total_tokens = token_info["total_tokens"]
    decoded_tokens = token_info.get("decoded_text_tokens", [])

    # Determine which layers to show
    available_layers = sorted(attn_weights.keys())
    if layers is None:
        # Show first, middle, and last layer by default
        if len(available_layers) >= 3:
            layers = [available_layers[0], available_layers[len(available_layers)//2], available_layers[-1]]
        else:
            layers = available_layers
    layers = [l for l in layers if l in available_layers]

    # Determine query token indices
    if query_tokens is None or query_tokens == "last":
        # Default: last text token
        query_indices = [total_tokens - 1]
        query_label = "last text token"
    elif query_tokens == "text":
        # All text tokens
        query_indices = list(range(num_img_tokens, total_tokens))
        query_label = "all text tokens (averaged)"
    elif query_tokens == "image":
        # All image tokens (attention FROM image TO text)
        query_indices = list(range(num_img_tokens))
        query_label = "all image tokens (averaged)"
    elif isinstance(query_tokens, list):
        query_indices = query_tokens
        query_label = f"tokens {query_tokens}"
    else:
        query_indices = [int(query_tokens)]
        query_label = f"token {query_tokens}"

    n_layers = len(layers)
    n_cols = 2 if (show_image_attention and show_text_attention) else 1

    fig = plt.figure(figsize=(6 * n_cols, 5 * n_layers))
    gs = gridspec.GridSpec(n_layers, n_cols, figure=fig, hspace=0.3, wspace=0.2)

    for row, layer_idx in enumerate(layers):
        attn = attn_weights[layer_idx]  # Shape: [B, K, G, T, S] or [K, G, T, S]

        # Handle batch dimension
        if attn.ndim == 5:
            attn = attn[0]  # Remove batch dim -> [K, G, T, S]

        # Average over heads if not specified
        if heads is None:
            # attn is [K, G, T, S], average over K and G -> [T, S]
            attn = attn.mean(axis=(0, 1))
        else:
            # Select specific heads and average
            attn = attn[:, heads, :, :].mean(axis=(0, 1))

        # Get attention FROM query tokens
        query_attn = attn[query_indices, :].mean(axis=0)  # [S]

        col = 0

        # Image attention visualization
        if show_image_attention:
            ax = fig.add_subplot(gs[row, col])

            # Get attention to image tokens
            img_attn = query_attn[:num_img_tokens]

            # Reshape to image grid (assuming 16x16 patches per view)
            patches_per_side = 16
            num_views = token_info["num_views"]

            if num_views == 1:
                img_attn_2d = img_attn.reshape(patches_per_side, patches_per_side)
            else:
                # Multiple views - show them side by side
                img_attn_2d = img_attn.reshape(num_views, patches_per_side, patches_per_side)
                img_attn_2d = img_attn_2d.mean(axis=0)  # Average across views

            # Normalize attention to [0, 1]
            img_attn_2d = img_attn_2d - img_attn_2d.min()
            if img_attn_2d.max() > 0:
                img_attn_2d = img_attn_2d / img_attn_2d.max()

            # Plot heatmap
            if image_rgb is not None:
                # Resize image to display size
                display_size = (224, 224)
                img_resized = np.array(Image.fromarray(image_rgb).resize(display_size))

                # Upsample attention to image size using bilinear interpolation
                attn_upsampled = np.array(Image.fromarray(
                    (img_attn_2d * 255).astype(np.uint8)).resize(
                    display_size, Image.BILINEAR)) / 255.0

                # Create heatmap overlay
                cmap = plt.cm.jet
                heatmap = cmap(attn_upsampled)[:, :, :3]  # Get RGB, drop alpha

                # Blend: image * (1 - alpha * attn) + heatmap * (alpha * attn)
                alpha = 0.6
                blended = img_resized / 255.0 * (1 - alpha * attn_upsampled[:, :, None]) + \
                          heatmap * (alpha * attn_upsampled[:, :, None])
                blended = np.clip(blended, 0, 1)

                ax.imshow(blended)
                # Add colorbar for reference
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
                plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label='Attention')
            else:
                im = ax.imshow(img_attn_2d, cmap='jet', aspect='equal')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax.set_title(f"Layer {layer_idx}: Image Attention\n(from {query_label})")
            ax.axis('off')
            col += 1

        # Text attention visualization
        if show_text_attention:
            ax = fig.add_subplot(gs[row, col])

            # Get attention to text tokens
            text_attn = query_attn[num_img_tokens:]

            # Bar plot for text tokens
            x = np.arange(len(text_attn))
            bars = ax.bar(x, text_attn, color='steelblue', alpha=0.8)

            # Add token labels
            if decoded_tokens and len(decoded_tokens) <= 30:
                ax.set_xticks(x)
                ax.set_xticklabels(decoded_tokens[:len(text_attn)], rotation=45, ha='right', fontsize=8)
            else:
                ax.set_xlabel("Text Token Index")

            ax.set_ylabel("Attention Weight")
            ax.set_title(f"Layer {layer_idx}: Text Attention\n(from {query_label})")

    plt.suptitle(f"Attention Visualization\nPrompt: \"{attention_results['prompt'][:50]}...\"",
                 fontsize=12, y=1.02)


    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {save_path}")
  



def visualize_image_attention(
    attention_results: dict,
    layer: int = None,
    head: int | None = None,
    query_tokens: str | list[int] | None = "text",
    save_path: str | None = None,
):
    """
    Simple visualization showing attention overlaid on the image.

    Args:
        attention_results: Output from extract_paligemma_attention()
        layer: Which layer to visualize (None = last layer)
        head: Which head (None = average)
        query_tokens: Which tokens to visualize attention FROM
        save_path: Path to save figure
    """
    attn_weights = attention_results["attention_weights"]
    token_info = attention_results["token_info"]
    image_rgb = attention_results.get("image_rgb")
    num_img_tokens = token_info["total_image_tokens"]
    total_tokens = token_info["total_tokens"]
    num_views = token_info["num_views"]
    patches_per_side = 16

    # Select layer
    available_layers = sorted(attn_weights.keys())
    if layer is None:
        layer = available_layers[-1]  # Default to last layer
    if layer not in available_layers:
        print(f"Layer {layer} not found. Available: {available_layers}")
        return

    # Get attention and process
    attn = attn_weights[layer]
    if attn.ndim == 5:
        attn = attn[0]
    if head is None:
        attn = attn.mean(axis=(0, 1))
    else:
        attn = attn[:, head, :, :].mean(axis=0)

    # Determine query indices
    if query_tokens is None or query_tokens == "last":
        query_indices = [total_tokens - 1]
        query_label = "last text token"
    elif query_tokens == "text":
        query_indices = list(range(num_img_tokens, total_tokens))
        query_label = "text tokens"
    elif query_tokens == "image":
        query_indices = list(range(num_img_tokens))
        query_label = "image tokens"
    elif isinstance(query_tokens, list):
        query_indices = query_tokens
        query_label = f"tokens {query_tokens}"
    else:
        query_indices = [int(query_tokens)]
        query_label = f"token {query_tokens}"

    # Get attention to image
    query_attn = attn[query_indices, :num_img_tokens].mean(axis=0)

    # Reshape to 2D
    if num_views == 1:
        img_attn_2d = query_attn.reshape(patches_per_side, patches_per_side)
    else:
        img_attn_2d = query_attn.reshape(num_views, patches_per_side, patches_per_side)
        img_attn_2d = img_attn_2d.mean(axis=0)

    # Normalize
    img_attn_2d = img_attn_2d - img_attn_2d.min()
    if img_attn_2d.max() > 0:
        img_attn_2d = img_attn_2d / img_attn_2d.max()

    # Create figure - just the overlay
    fig, ax = plt.subplots(figsize=(8, 8))

    if image_rgb is not None:
        # Upsample attention to image size
        h, w = image_rgb.shape[:2]
        attn_upsampled = np.array(Image.fromarray(
            (img_attn_2d * 255).astype(np.uint8)).resize(
            (w, h), Image.BILINEAR)) / 255.0

        # Create heatmap overlay
        cmap = plt.cm.jet
        heatmap = cmap(attn_upsampled)[:, :, :3]
        alpha = 0.5
        blended = image_rgb / 255.0 * (1 - alpha) + heatmap * alpha
        blended = np.clip(blended, 0, 1)

        ax.imshow(blended)
    else:
        im = ax.imshow(img_attn_2d, cmap='jet', aspect='equal')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(f"Layer {layer} | From: {query_label}\n\"{attention_results['prompt'][:50]}...\"", fontsize=10)
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_attention_matrix(
    attention_results: dict,
    layer: int,
    head: int | None = None,
    save_path: str | None = None,
):
    """
    Visualize the full attention matrix for a specific layer.

    Args:
        attention_results: Output from extract_paligemma_attention()
        layer: Which layer to visualize
        head: Which head to show (None = average)
        save_path: Path to save figure
    """
    attn_weights = attention_results["attention_weights"]
    token_info = attention_results["token_info"]
    num_img_tokens = token_info["total_image_tokens"]

    if layer not in attn_weights:
        print(f"Layer {layer} not found. Available: {list(attn_weights.keys())}")
        return

    attn = attn_weights[layer]
    if attn.ndim == 5:
        attn = attn[0]  # Remove batch

    if head is None:
        attn = attn.mean(axis=(0, 1))  # Average over K, G -> [T, S]
        head_label = "averaged"
    else:
        attn = attn[:, head, :, :].mean(axis=0)  # [T, S]
        head_label = f"head {head}"

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(attn, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label='Attention Weight')

    # Add separators for image/text boundaries
    ax.axhline(y=num_img_tokens - 0.5, color='red', linestyle='--', linewidth=2, label='Image/Text boundary')
    ax.axvline(x=num_img_tokens - 0.5, color='red', linestyle='--', linewidth=2)

    ax.set_xlabel("Key Position (attending TO)")
    ax.set_ylabel("Query Position (attending FROM)")
    ax.set_title(f"Layer {layer} Attention Matrix ({head_label})\n"
                 f"Image tokens: 0-{num_img_tokens-1}, Text tokens: {num_img_tokens}-{attn.shape[0]-1}")

    # Add region labels
    ax.text(num_img_tokens // 2, -0.02 * attn.shape[0], 'Image', ha='center', fontsize=10, color='red')
    ax.text(num_img_tokens + (attn.shape[1] - num_img_tokens) // 2, -0.02 * attn.shape[0],
            'Text', ha='center', fontsize=10, color='blue')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention matrix to: {save_path}")
    else:
        plt.show()

    plt.close()


def extract_task_from_filename(filename: str) -> str:
    """
    Extract the task/prompt from a video filename.

    Expected format: rollout_<task>_trial<N>_<success/failure>.mp4
    Example: rollout_put_the_bowl_on_the_stove_trial16_success.mp4
             -> "put the bowl on the stove"

    Args:
        filename: Video filename (with or without path)

    Returns:
        Extracted task string with underscores replaced by spaces
    """
    # Get just the filename without path and extension
    name = Path(filename).stem

    # Pattern: rollout_<task>_trial<N>_<success/failure>
    match = re.match(r'rollout_(.+)_trial\d+_(success|failure)$', name)
    if match:
        task = match.group(1)
        # Replace underscores with spaces
        return task.replace('_', ' ')

    # Fallback: try to extract anything between rollout_ and _trial
    match = re.match(r'rollout_(.+)_trial', name)
    if match:
        task = match.group(1)
        return task.replace('_', ' ')

    # If no pattern matches, return the filename as-is (without extension)
    return name.replace('_', ' ')


def load_pi0_model(
    checkpoint_path: str,
    paligemma_variant: str = "gemma_2b",
    action_expert_variant: str = "gemma_300m",
    action_dim: int = None,
    action_horizon: int = 10,
    max_token_len: int = 256,
    pi05: bool = True,
    dtype: str = "bfloat16",
):
    """
    Load the full Pi0 model from checkpoint, which includes the PaliGemma backbone.

    This uses the same loading code as the VLA to ensure identical behavior.
    """
    # Resolve checkpoint path
    ckpt = Path(checkpoint_path).expanduser()
    if ckpt.is_dir() and (ckpt / "params").is_dir():
        params_dir = ckpt / "params"
    else:
        params_dir = ckpt

    print(f"Loading checkpoint from: {params_dir}")

    # Restore parameters
    params = _model.restore_params(
        params_dir,
        restore_type=np.ndarray,
        dtype=jnp.bfloat16 if dtype == "bfloat16" else None,
    )

    # Infer action_dim from checkpoint if not provided
    if action_dim is None:
        try:
            action_dim = int(params["action_in_proj"]["kernel"].shape[0])
            print(f"  Inferred action_dim={action_dim} from checkpoint")
        except Exception:
            action_dim = 7  # Default for most robot setups
            print(f"  Using default action_dim={action_dim}")

    # Create config
    config = pi0_config.Pi0Config(
        paligemma_variant=paligemma_variant,
        action_expert_variant=action_expert_variant,
        action_dim=action_dim,
        action_horizon=action_horizon,
        max_token_len=max_token_len,
        pi05=pi05,
        dtype=dtype,
    )

    print(f"  PaliGemma variant: {paligemma_variant}")
    print(f"  Action expert variant: {action_expert_variant}")
    print(f"  Pi0.5 mode: {pi05}")

    # Load model with parameters
    model = config.load(params)
    return model, config


def load_tokenizer():
    """Load the PaliGemma SentencePiece tokenizer."""
    import openpi.shared.download as download

    path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    with path.open("rb") as f:
        tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
    return tokenizer


def preprocess_image(image_path_or_array, target_size=(224, 224)):
    """Load and preprocess an image for the vision encoder.

    Returns image normalized to [-1, 1] as expected by the model.
    """
    if isinstance(image_path_or_array, (str, Path)):
        pil_image = Image.open(image_path_or_array).convert("RGB")
    elif isinstance(image_path_or_array, np.ndarray):
        pil_image = Image.fromarray(image_path_or_array)
    else:
        pil_image = image_path_or_array

    # Resize to target size
    pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)

    # Convert to numpy and normalize to [-1, 1] (model expectation)
    image_array = np.array(pil_image).astype(np.float32) / 255.0 * 2.0 - 1.0

    # Add batch dimension
    return jnp.array(image_array[None, ...])


def create_observation_for_paligemma(
    image: jnp.ndarray,
    prompt: str,
    tokenizer,
    max_token_len: int = 256,
    state_dim: int = 7,
    pi05: bool = True,
):
    """
    Create an Observation object for the Pi0 model from an image and prompt.

    Args:
        image: Preprocessed image tensor [1, H, W, 3] in [-1, 1]
        prompt: Text prompt
        tokenizer: PaligemmaTokenizer instance
        max_token_len: Maximum token length
        state_dim: State dimension (for dummy state)
        pi05: Whether using pi0.5 mode

    Returns:
        Observation object ready for the model
    """
    # Create dummy state (zeros)
    state = np.zeros(state_dim, dtype=np.float32)

    # Tokenize prompt using the PaligemmaTokenizer
    paligemma_tok = _tokenizer.PaligemmaTokenizer(max_len=max_token_len)
    tokens, mask = paligemma_tok.tokenize(prompt, state=state if pi05 else None)

    tokenized_prompt = jnp.array([tokens], dtype=jnp.int32)
    tokenized_prompt_mask = jnp.array([mask], dtype=jnp.bool_)
    state_jax = jnp.array(state[None, ...])

    # Model expects three camera views - use the same image for all
    observation = _model.Observation(
        images={
            "base_0_rgb": image,
            "left_wrist_0_rgb": image,
            "right_wrist_0_rgb": image,
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


def get_paligemma_embeddings(
    model,
    observation: _model.Observation,
):
    """
    Extract the PaliGemma prefix embeddings (image + text tokens) from the Pi0 model.

    This uses the same code path as the VLA to ensure identical representations.

    Args:
        model: The Pi0 model
        observation: Preprocessed observation

    Returns:
        tuple: (prefix_tokens, prefix_mask, prefix_ar_mask)
            - prefix_tokens: [B, S, D] - concatenated image and text embeddings
            - prefix_mask: [B, S] - input mask
            - prefix_ar_mask: [S] - autoregressive mask
    """
    # Preprocess observation (same as VLA)
    observation = _model.preprocess_observation(None, observation, train=False)

    # Get prefix embeddings using the model's own method
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)

    return prefix_tokens, prefix_mask, prefix_ar_mask


def generate_text_from_model(
    model,
    observation: _model.Observation,
    tokenizer,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_k: int = 40,
):
    """
    Generate text using the PaliGemma backbone from the Pi0 model.

    This runs the VLM in autoregressive mode to generate text,
    showing what the model "understands" about the visual input.

    Args:
        model: The Pi0 model (contains PaliGemma)
        observation: Observation with images and prompt
        tokenizer: SentencePiece tokenizer for decoding
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)
        top_k: Top-k sampling parameter

    Returns:
        Generated text string
    """
    # Preprocess observation
    observation = _model.preprocess_observation(None, observation, train=False)

    # Get prefix embeddings (image + text)
    print("  Getting PaliGemma prefix embeddings...")
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    print(f"  Prefix tokens shape: {prefix_tokens.shape}")  # [1, num_img_tokens + num_text_tokens, 2048]

    seq_len = prefix_tokens.shape[1]

    # Build attention mask
    def make_attn_mask(input_mask, ar_mask):
        """Build attention mask (same as model's internal logic)."""
        ar_mask = jnp.broadcast_to(ar_mask, input_mask.shape)
        cumsum = jnp.cumsum(ar_mask, axis=1)
        attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
        valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
        return jnp.logical_and(attn_mask, valid_mask)

    # Forward pass through PaliGemma LLM
    print("  Running forward pass through Gemma LLM...")
    attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1

    # Run only the PaliGemma expert (index 0), not the action expert
    outputs, kv_cache = model.PaliGemma.llm(
        [prefix_tokens, None],  # Only PaliGemma input, no action expert
        mask=attn_mask,
        positions=positions,
        adarms_cond=[None, None],
    )
    hidden_states = outputs[0]  # [1, seq_len, 2048]
    print(f"  Hidden states shape: {hidden_states.shape}")

    # Get the embedding table directly from params for decoding to logits
    # In Flax NNX bridge, params are accessible via nnx.state()
    import flax.nnx as nnx
    llm_state = nnx.state(model.PaliGemma.llm)
    # Find the embedding table in the state - it should be under 'embedder/input_embedding'
    embedding_key = None
    for key in llm_state.flat_state():
        if 'embedder' in str(key) and 'input_embedding' in str(key):
            embedding_key = key
            break
    if embedding_key is None:
        raise ValueError(f"Could not find embedding table. Available keys: {list(llm_state.flat_state().keys())[:10]}")
    embedding_table = llm_state.flat_state()[embedding_key].value
    embed_dim = embedding_table.shape[1]  # Should be 2048 for Gemma 2B

    # Helper functions for encode/decode
    def embed_decode(hidden_states):
        """Convert hidden states to logits via embedding table."""
        return jnp.dot(hidden_states, embedding_table.T)

    def embed_encode(token_ids):
        """Convert token IDs to embeddings."""
        return embedding_table[token_ids] * jnp.sqrt(embed_dim).astype(embedding_table.dtype)

    # Autoregressive generation
    generated_tokens = []
    rng = jax.random.PRNGKey(42)

    print(f"\n  Generating up to {max_new_tokens} tokens...")

    current_embeddings = prefix_tokens
    current_mask = prefix_mask

    for i in range(max_new_tokens):
        # Get logits from the last position
        last_hidden = hidden_states[:, -1, :]  # [1, 2048]
        logits = embed_decode(last_hidden)  # [1, vocab_size]

        # Apply temperature
        if temperature > 0:
            logits = logits / temperature

        # Apply top-k filtering
        if top_k > 0:
            top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
            # Create mask for non-top-k tokens
            mask_topk = jnp.zeros_like(logits).at[0, top_k_indices[0]].set(1)
            logits = jnp.where(mask_topk, logits, -1e10)

        # Sample or take argmax
        if temperature > 0:
            rng, sample_rng = jax.random.split(rng)
            next_token = jax.random.categorical(sample_rng, logits, axis=-1)
        else:
            next_token = jnp.argmax(logits, axis=-1)

        next_token_id = int(next_token[0])
        generated_tokens.append(next_token_id)

        # Check for EOS token
        if next_token_id == tokenizer.eos_id() or next_token_id == 1:
            print(f"  Reached EOS token at position {i+1}")
            break

        # Prepare next input
        next_token_emb = embed_encode(next_token[:, None])  # [1, 1, 2048]

        # Update embeddings and mask
        current_embeddings = jnp.concatenate([current_embeddings, next_token_emb], axis=1)
        current_mask = jnp.concatenate([current_mask, jnp.ones((1, 1), dtype=jnp.bool_)], axis=1)

        # Build new attention mask (causal for new tokens)
        new_ar_mask = jnp.concatenate([prefix_ar_mask, jnp.ones((i + 1,), dtype=jnp.bool_)])
        attn_mask = make_attn_mask(current_mask, new_ar_mask)
        positions = jnp.cumsum(current_mask, axis=1) - 1

        # Forward pass
        outputs, _ = model.PaliGemma.llm(
            [current_embeddings, None],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, None],
        )
        hidden_states = outputs[0]

        # Progress indicator
        if (i + 1) % 10 == 0:
            partial_text = tokenizer.decode(generated_tokens)
            print(f"  [{i+1}/{max_new_tokens}] Generated so far: {partial_text[:50]}...")

    # Decode generated tokens
    generated_text = tokenizer.decode(generated_tokens)
    return generated_text


def run_paligemma_on_image(
    checkpoint_path: str,
    image_source: str | np.ndarray,
    prompt: str,
    video_frame_idx: int = 0,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    paligemma_variant: str = "gemma_2b",
    action_expert_variant: str = "gemma_300m",
    max_token_len: int = 256,
    pi05: bool = True,
):
    """
    Main function to run PaliGemma on an image and generate text.

    This uses the exact same model and code paths as your VLA to ensure
    identical visual processing.
    """
    print("=" * 60)
    print("Running PaliGemma VLM Standalone")
    print("=" * 60)

    # Load image
    print("\n1. Loading image...")
    if isinstance(image_source, str) and image_source.endswith(('.mp4', '.avi', '.mov')):
        print(f"   Extracting frame {video_frame_idx} from video: {image_source}")
        image_rgb = get_frame_opencv(image_source, video_frame_idx)
        if image_rgb is None:
            raise ValueError(f"Could not extract frame {video_frame_idx} from {image_source}")
    elif isinstance(image_source, str):
        print(f"   Loading image: {image_source}")
        image_rgb = np.array(Image.open(image_source).convert("RGB"))
    else:
        image_rgb = image_source

    print(f"   Image shape: {image_rgb.shape}")

    # Preprocess image
    image_tensor = preprocess_image(image_rgb)
    print(f"   Preprocessed tensor shape: {image_tensor.shape}")

    # Load model
    print("\n2. Loading Pi0 model (contains PaliGemma)...")
    model, config = load_pi0_model(
        checkpoint_path=checkpoint_path,
        paligemma_variant=paligemma_variant,
        action_expert_variant=action_expert_variant,
        max_token_len=max_token_len,
        pi05=pi05,
    )

    # Load tokenizer
    print("\n3. Loading tokenizer...")
    tokenizer = load_tokenizer()
    print(f"   Vocabulary size: {tokenizer.vocab_size()}")

    # Create observation
    print("\n4. Creating observation...")
    observation = create_observation_for_paligemma(
        image=image_tensor,
        prompt=prompt,
        tokenizer=tokenizer,
        max_token_len=max_token_len,
        state_dim=config.action_dim,
        pi05=pi05,
    )

    # Generate text
    print(f"\n5. Generating text...")
    print(f"   Prompt: {prompt}")
    print(f"   Max new tokens: {max_new_tokens}")
    print(f"   Temperature: {temperature}")

    generated_text = generate_text_from_model(
        model=model,
        observation=observation,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    print("\n" + "=" * 60)
    print("GENERATED OUTPUT:")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print(f"Response: {generated_text}")
    print("=" * 60)

    return generated_text


def extract_paligemma_hidden_states(
    checkpoint_path: str,
    image_source: str | np.ndarray,
    prompt: str,
    video_frame_idx: int = 0,
    paligemma_variant: str = "gemma_2b",
    action_expert_variant: str = "gemma_300m",
    max_token_len: int = 256,
    pi05: bool = True,
):
    """
    Get the hidden states from PaliGemma without generating text.

    This extracts the exact same representations that the VLA sees,
    useful for comparing with the action expert's representations.

    Returns:
        dict with:
            - prefix_tokens: Combined image + text embeddings [1, S, 2048]
            - hidden_states: LLM output hidden states [1, S, 2048]
            - num_image_tokens: Number of image tokens (256 for 224x224 with 14x14 patches)
            - num_text_tokens: Number of text tokens
            - prompt: The input prompt
            - image_shape: Original image shape
    """
    print("=" * 60)
    print("Extracting PaliGemma Hidden States")
    print("=" * 60)

    # Load image
    print("\n1. Loading image...")
    if isinstance(image_source, str) and image_source.endswith(('.mp4', '.avi', '.mov')):
        image_rgb = get_frame_opencv(image_source, video_frame_idx)
    elif isinstance(image_source, str):
        image_rgb = np.array(Image.open(image_source).convert("RGB"))
    else:
        image_rgb = image_source

    image_tensor = preprocess_image(image_rgb)

    # Load model
    print("\n2. Loading Pi0 model...")
    model, config = load_pi0_model(
        checkpoint_path=checkpoint_path,
        paligemma_variant=paligemma_variant,
        action_expert_variant=action_expert_variant,
        max_token_len=max_token_len,
        pi05=pi05,
    )

    # Load tokenizer
    tokenizer = load_tokenizer()

    # Create observation
    print("\n3. Creating observation...")
    observation = create_observation_for_paligemma(
        image=image_tensor,
        prompt=prompt,
        tokenizer=tokenizer,
        max_token_len=max_token_len,
        state_dim=config.action_dim,
        pi05=pi05,
    )

    # Get embeddings
    print("\n4. Extracting embeddings...")
    observation = _model.preprocess_observation(None, observation, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)

    # Count image vs text tokens
    # Image tokens: 3 images * 256 tokens each = 768 tokens (for 3 cameras)
    # But only base_0_rgb is typically valid, so let's check
    num_image_tokens_per_view = 256  # 16x16 patches for 224x224 image with patch size 14
    num_views = sum(1 for k in observation.images if observation.image_masks[k].any())
    num_image_tokens = num_views * num_image_tokens_per_view

    # Forward pass through LLM
    print("\n5. Running forward pass...")

    def make_attn_mask(input_mask, ar_mask):
        ar_mask = jnp.broadcast_to(ar_mask, input_mask.shape)
        cumsum = jnp.cumsum(ar_mask, axis=1)
        attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
        valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
        return jnp.logical_and(attn_mask, valid_mask)

    attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1

    outputs, kv_cache = model.PaliGemma.llm(
        [prefix_tokens, None],
        mask=attn_mask,
        positions=positions,
        adarms_cond=[None, None],
    )
    hidden_states = outputs[0]

    print(f"  Prefix tokens shape: {prefix_tokens.shape}")
    print(f"  Hidden states shape: {hidden_states.shape}")
    print(f"  Number of image tokens: {num_image_tokens}")
    print(f"  Number of text tokens: {prefix_tokens.shape[1] - num_image_tokens}")

    return {
        "prefix_tokens": np.array(prefix_tokens),
        "hidden_states": np.array(hidden_states),
        "prefix_mask": np.array(prefix_mask),
        "num_image_tokens": num_image_tokens,
        "num_text_tokens": int(prefix_tokens.shape[1] - num_image_tokens),
        "prompt": prompt,
        "image_shape": image_rgb.shape,
        "tokenized_prompt": np.array(observation.tokenized_prompt),
    }


def extract_paligemma_attention(
    checkpoint_path: str,
    image_source: str | np.ndarray,
    prompt: str,
    video_frame_idx: int = 0,
    paligemma_variant: str = "gemma_2b",
    action_expert_variant: str = "gemma_300m",
    max_token_len: int = 256,
    pi05: bool = True,
    layers: list[int] | None = None,
):
    """
    Extract attention maps from PaliGemma without generating text.

    This runs a single forward pass and captures attention weights from all
    transformer layers. Useful for visualizing what the model attends to.

    Args:
        checkpoint_path: Path to model checkpoint
        image_source: Image path, video path, or numpy array
        prompt: Text prompt
        video_frame_idx: Frame index if video
        paligemma_variant: Model variant
        action_expert_variant: Action expert variant
        max_token_len: Maximum token length
        pi05: Whether using pi0.5 mode
        layers: Which layers to return (None = all layers)

    Returns:
        dict with:
            - attention_weights: dict mapping layer_idx -> attention tensor
              Each tensor has shape [B, K, G, T, S] where:
                B = batch size
                K = num_kv_heads
                G = num_heads_per_kv_head
                T = query sequence length
                S = key/value sequence length
            - num_image_tokens: Number of image tokens (per view)
            - num_text_tokens: Number of text tokens
            - num_views: Number of image views
            - prompt: The input prompt
            - token_info: Dict with token position info
    """
    print("=" * 60)
    print("Extracting PaliGemma Attention Maps")
    print("=" * 60)

    # Load image
    print("\n1. Loading image...")
    if isinstance(image_source, str) and image_source.endswith(('.mp4', '.avi', '.mov')):
        image_rgb = get_frame_opencv(image_source, video_frame_idx)
    elif isinstance(image_source, str):
        image_rgb = np.array(Image.open(image_source).convert("RGB"))
    else:
        image_rgb = image_source

    image_tensor = preprocess_image(image_rgb)

    # Load model
    print("\n2. Loading Pi0 model...")
    model, config = load_pi0_model(
        checkpoint_path=checkpoint_path,
        paligemma_variant=paligemma_variant,
        action_expert_variant=action_expert_variant,
        max_token_len=max_token_len,
        pi05=pi05,
    )

    # Load tokenizer
    tokenizer = load_tokenizer()

    # Create observation
    print("\n3. Creating observation...")
    observation = create_observation_for_paligemma(
        image=image_tensor,
        prompt=prompt,
        tokenizer=tokenizer,
        max_token_len=max_token_len,
        state_dim=config.action_dim,
        pi05=pi05,
    )

    # Preprocess observation
    observation = _model.preprocess_observation(None, observation, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)

    # Count token types
    num_image_tokens_per_view = 256  # 16x16 patches for 224x224 image with patch size 14
    num_views = sum(1 for k in observation.images if observation.image_masks[k].any())
    total_image_tokens = num_views * num_image_tokens_per_view
    total_tokens = prefix_tokens.shape[1]
    num_text_tokens = total_tokens - total_image_tokens

    print(f"  Total tokens: {total_tokens}")
    print(f"  Image tokens: {total_image_tokens} ({num_views} views x {num_image_tokens_per_view})")
    print(f"  Text tokens: {num_text_tokens}")

    # Enable attention saving
    print("\n4. Running forward pass with attention capture...")
    enable_attention_saving()

    try:
        # Build attention mask
        def make_attn_mask(input_mask, ar_mask):
            ar_mask = jnp.broadcast_to(ar_mask, input_mask.shape)
            cumsum = jnp.cumsum(ar_mask, axis=1)
            attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
            valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
            return jnp.logical_and(attn_mask, valid_mask)

        attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1

        # Forward pass (this captures attention weights)
        outputs, kv_cache = model.PaliGemma.llm(
            [prefix_tokens, None],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, None],
        )

        # Get captured attention weights
        attention_weights = get_attention_weights()

    finally:
        disable_attention_saving()

    # Process attention weights
    processed_weights = {}
    num_layers = len(attention_weights)
    print(f"\n5. Processing attention from {num_layers} layers...")

    for layer_key, attn_list in attention_weights.items():
        layer_idx = int(layer_key.split('_')[1])

        # Filter layers if specified
        if layers is not None and layer_idx not in layers:
            continue

        # Stack attention from this layer (usually just one entry per layer)
        attn = np.stack([np.array(a) for a in attn_list], axis=0)
        if attn.shape[0] == 1:
            attn = attn[0]  # Remove extra dimension

        processed_weights[layer_idx] = attn
        print(f"  Layer {layer_idx}: shape {attn.shape}")

    # Decode tokens for reference
    tokenized = np.array(observation.tokenized_prompt[0])
    valid_tokens = tokenized[tokenized != 0]
    decoded_tokens = [tokenizer.id_to_piece(int(t)) for t in valid_tokens]

    # Create token info
    token_info = {
        "total_tokens": total_tokens,
        "num_image_tokens_per_view": num_image_tokens_per_view,
        "num_views": num_views,
        "total_image_tokens": total_image_tokens,
        "num_text_tokens": num_text_tokens,
        "image_token_range": (0, total_image_tokens),
        "text_token_range": (total_image_tokens, total_tokens),
        "decoded_text_tokens": decoded_tokens,
    }

    print(f"\n  Token breakdown:")
    print(f"    Image tokens [0:{total_image_tokens}]")
    print(f"    Text tokens [{total_image_tokens}:{total_tokens}]")
    print(f"    Decoded text: {' '.join(decoded_tokens[:20])}{'...' if len(decoded_tokens) > 20 else ''}")

    return {
        "attention_weights": processed_weights,
        "num_image_tokens": total_image_tokens,
        "num_text_tokens": num_text_tokens,
        "num_views": num_views,
        "prompt": prompt,
        "image_shape": image_rgb.shape,
        "token_info": token_info,
        "prefix_mask": np.array(prefix_mask),
        "image_rgb": image_rgb,  # Include for visualization
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run PaliGemma VLM backbone standalone on images/videos"
    )

    # Input source (supports multiple images and/or prompts)
    parser.add_argument("--image", type=str, nargs='+', help="Path(s) to input image(s)")
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--frame-idx", type=int, nargs='+', default=[0], help="Frame index(es) for video input")

    # Model
    parser.add_argument("--checkpoint", type=str,
                        default="~/.cache/openpi/openpi-assets/checkpoints/pi05_libero",
                        help="Path to checkpoint directory")
    parser.add_argument("--paligemma-variant", type=str, default="gemma_2b",
                        help="PaliGemma variant (gemma_2b)")
    parser.add_argument("--action-expert-variant", type=str, default="gemma_300m",
                        help="Action expert variant")
    parser.add_argument("--max-token-len", type=int, default=256,
                        help="Maximum token length")
    parser.add_argument("--pi05", action="store_true", default=True,
                        help="Use Pi0.5 mode (default: True)")
    parser.add_argument("--no-pi05", action="store_true",
                        help="Disable Pi0.5 mode")

    # Generation
    parser.add_argument("--prompt", type=str, nargs='+', default=None,
                        help="Text prompt(s)/task(s) for the model. If not provided, extracts from video filename.")
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (0 for greedy)")

    # Mode
    parser.add_argument("--mode", type=str, choices=["generate", "hidden_states", "attention"],
                        default="generate",
                        help="Mode: 'generate' for text generation, 'hidden_states' for extracting representations, 'attention' for attention maps")
    parser.add_argument("--output", type=str, help="Output path for hidden states or attention (numpy format)")
    parser.add_argument("--layers", type=int, nargs='+', default=None,
                        help="Which layers to extract attention from (default: all)")

    # Visualization options (for attention mode)
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize attention maps (for --mode attention)")
    parser.add_argument("--show-matrix", action="store_true",
                        help="Show full attention matrix heatmap")
    parser.add_argument("--simple-viz", action="store_true",
                        help="Show simple 3-panel view: image | attention | overlay")
    parser.add_argument("--query-tokens", type=str, default="text",
                        help="Which tokens to visualize attention FROM: 'last', 'text', 'image', or comma-separated indices")
    parser.add_argument("--save-fig", type=str, default=None,
                        help="Save visualization to file instead of displaying")

    args = parser.parse_args()

    # Resolve pi05 flag
    pi05 = not args.no_pi05

    # Determine image sources and prompts
    image_sources = []
    auto_prompts = []  # Prompts extracted from filenames

    if args.image:
        image_sources = args.image  # Already a list due to nargs='+'
        # For images, we can't auto-extract prompts, so use provided or default
        if args.prompt is None:
            auto_prompts = ["Describe what you see in this image."]
        else:
            auto_prompts = args.prompt
    elif args.video:
        # For video, create (video_path, frame_idx) tuples
        for frame_idx in args.frame_idx:
            image_sources.append((args.video, frame_idx))
        # Auto-extract task from video filename if no prompt provided
        if args.prompt is None:
            extracted_task = extract_task_from_filename(args.video)
            auto_prompts = [extracted_task]
            print(f"Auto-extracted task from filename: \"{extracted_task}\"")
        else:
            auto_prompts = args.prompt
    else:
        parser.error("Either --image or --video must be provided")

    # Get prompts list
    prompts = auto_prompts

    # Run for all combinations of images and prompts
    all_results = []
    for img_idx, image_source in enumerate(image_sources):
        # Handle video frame tuples
        if isinstance(image_source, tuple):
            video_path, frame_idx = image_source
            img_name = f"{Path(video_path).stem}_frame{frame_idx}"
        else:
            video_path = None
            frame_idx = args.frame_idx[0] if args.frame_idx else 0
            img_name = Path(image_source).stem if isinstance(image_source, str) else f"image_{img_idx}"

        for prompt_idx, prompt in enumerate(prompts):
            print(f"\n{'=' * 60}")
            print(f"Processing: Image {img_idx + 1}/{len(image_sources)}, Prompt {prompt_idx + 1}/{len(prompts)}")
            print(f"  Image: {img_name}")
            print(f"  Prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
            print(f"{'=' * 60}")

            if args.mode == "generate":
                generated_text = run_paligemma_on_image(
                    checkpoint_path=args.checkpoint,
                    image_source=video_path if video_path else image_source,
                    prompt=prompt,
                    video_frame_idx=frame_idx,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    paligemma_variant=args.paligemma_variant,
                    action_expert_variant=args.action_expert_variant,
                    max_token_len=args.max_token_len,
                    pi05=pi05,
                )
                all_results.append({
                    "image": img_name,
                    "prompt": prompt,
                    "generated_text": generated_text,
                })
            elif args.mode == "attention":
                results = extract_paligemma_attention(
                    checkpoint_path=args.checkpoint,
                    image_source=video_path if video_path else image_source,
                    prompt=prompt,
                    video_frame_idx=frame_idx,
                    paligemma_variant=args.paligemma_variant,
                    action_expert_variant=args.action_expert_variant,
                    max_token_len=args.max_token_len,
                    pi05=pi05,
                    layers=args.layers,
                )

                print(f"\n  ATTENTION SUMMARY:")
                print(f"    Number of layers: {len(results['attention_weights'])}")
                print(f"    Number of image tokens: {results['num_image_tokens']}")
                print(f"    Number of text tokens: {results['num_text_tokens']}")
                if results['attention_weights']:
                    sample_layer = list(results['attention_weights'].keys())[0]
                    print(f"    Attention shape (layer {sample_layer}): {results['attention_weights'][sample_layer].shape}")

                # Visualize if requested
                if args.visualize or args.show_matrix or args.simple_viz:
                    # Parse query tokens
                    if args.query_tokens in ["last", "text", "image"]:
                        query_tokens = args.query_tokens
                    else:
                        try:
                            query_tokens = [int(x) for x in args.query_tokens.split(",")]
                        except ValueError:
                            query_tokens = "text"

                    # Simple 3-panel visualization (default if --simple-viz or just --visualize)
                    if args.simple_viz or (args.visualize and not args.show_matrix):
                        viz_layers = args.layers if args.layers else [sorted(results["attention_weights"].keys())[-1]]
                        for layer in viz_layers:
                            save_path = args.save_fig
                            if save_path:
                                base, ext = os.path.splitext(save_path)
                                if len(image_sources) > 1:
                                    save_path = f"{base}_{img_idx}_layer{layer}{ext}"
                                elif len(viz_layers) > 1:
                                    save_path = f"{base}_layer{layer}{ext}"

                            visualize_image_attention(
                                results,
                                layer=layer,
                                query_tokens=query_tokens,
                                save_path=save_path,
                            )

                    # Full multi-layer visualization
                    elif args.visualize:
                        save_path = args.save_fig
                        if save_path and len(image_sources) > 1:
                            base, ext = os.path.splitext(save_path)
                            save_path = f"{base}_{img_idx}{ext}"

                        visualize_attention(
                            results,
                            image_rgb=results["image_rgb"],
                            layers=args.layers,
                            query_tokens=query_tokens,
                            save_path=save_path,
                        )

                    if args.show_matrix:
                        # Show attention matrix for each requested layer
                        viz_layers = args.layers if args.layers else [0, 8, 17]
                        for layer in viz_layers:
                            if layer in results["attention_weights"]:
                                matrix_save_path = None
                                if args.save_fig:
                                    base, ext = os.path.splitext(args.save_fig)
                                    matrix_save_path = f"{base}_matrix_layer{layer}{ext}"
                                visualize_attention_matrix(results, layer=layer, save_path=matrix_save_path)

                results["image_name"] = img_name
                all_results.append(results)
            else:  # hidden_states
                results = extract_paligemma_hidden_states(
                    checkpoint_path=args.checkpoint,
                    image_source=video_path if video_path else image_source,
                    prompt=prompt,
                    video_frame_idx=frame_idx,
                    paligemma_variant=args.paligemma_variant,
                    action_expert_variant=args.action_expert_variant,
                    max_token_len=args.max_token_len,
                    pi05=pi05,
                )

                print(f"\n  HIDDEN STATE SUMMARY:")
                print(f"    Prefix tokens shape: {results['prefix_tokens'].shape}")
                print(f"    Hidden states shape: {results['hidden_states'].shape}")
                print(f"    Number of image tokens: {results['num_image_tokens']}")
                print(f"    Number of text tokens: {results['num_text_tokens']}")

                results["image_name"] = img_name
                all_results.append(results)

    # Summary and output
    if args.mode == "generate":
        print(f"\n{'=' * 60}")
        print("GENERATION SUMMARY:")
        print(f"{'=' * 60}")
        for r in all_results:
            print(f"\n[Image: {r['image']}]")
            print(f"[Prompt: {r['prompt']}]")
            print(f"Generated: {r['generated_text']}")
    elif args.mode == "attention":
        print(f"\n{'=' * 60}")
        print("ATTENTION EXTRACTION COMPLETE")
        print(f"{'=' * 60}")
        for r in all_results:
            print(f"\n[Image: {r['image_name']}]")
            print(f"[Prompt: {r['prompt']}]")
            print(f"Layers captured: {sorted(r['attention_weights'].keys())}")
            print(f"Token info: {r['num_image_tokens']} image + {r['num_text_tokens']} text tokens")

        # Only save npz if --output is provided (and not just visualizing)
        if args.output and not (args.visualize or args.show_matrix):
            output_path = Path(args.output)
            if len(all_results) == 1:
                # For attention, we need to handle the nested dict structure
                results = all_results[0]
                save_dict = {
                    "prompt": results["prompt"],
                    "num_image_tokens": results["num_image_tokens"],
                    "num_text_tokens": results["num_text_tokens"],
                    "num_views": results["num_views"],
                    "image_shape": results["image_shape"],
                    "prefix_mask": results["prefix_mask"],
                }
                # Add attention weights with layer prefix
                for layer_idx, attn in results["attention_weights"].items():
                    save_dict[f"attn_layer_{layer_idx}"] = attn
                # Add token info
                for k, v in results["token_info"].items():
                    if isinstance(v, (list, np.ndarray)):
                        save_dict[f"token_info_{k}"] = np.array(v) if isinstance(v, list) else v
                    else:
                        save_dict[f"token_info_{k}"] = v

                np.savez(args.output, **save_dict)
                print(f"\nSaved attention maps to: {args.output}")
            else:
                for i, results in enumerate(all_results):
                    out_file = output_path.parent / f"{output_path.stem}_{i}{output_path.suffix}"
                    save_dict = {
                        "prompt": results["prompt"],
                        "num_image_tokens": results["num_image_tokens"],
                        "num_text_tokens": results["num_text_tokens"],
                    }
                    for layer_idx, attn in results["attention_weights"].items():
                        save_dict[f"attn_layer_{layer_idx}"] = attn
                    np.savez(str(out_file), **save_dict)
                    print(f"Saved: {out_file}")
    else:  # hidden_states
        if args.output:
            # Save all results
            output_path = Path(args.output)
            if len(all_results) == 1:
                np.savez(args.output, **all_results[0])
                print(f"\nSaved hidden states to: {args.output}")
            else:
                # Save each result separately
                for i, results in enumerate(all_results):
                    out_file = output_path.parent / f"{output_path.stem}_{i}{output_path.suffix}"
                    np.savez(str(out_file), **results)
                    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
