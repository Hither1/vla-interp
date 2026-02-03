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
"""

import os
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import gemma as _gemma
from openpi.models import tokenizer as _tokenizer
import sentencepiece
import re

from utils import get_frame_opencv


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
    parser.add_argument("--mode", type=str, choices=["generate", "hidden_states"],
                        default="generate",
                        help="Mode: 'generate' for text generation, 'hidden_states' for extracting representations")
    parser.add_argument("--output", type=str, help="Output path for hidden states (numpy format)")

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
            else:
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
    else:
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
