"""
Visualize attention to text tokens in the Pi0 VLA model.

This script shows which words/tokens in the text prompt the model is attending to
when making action predictions.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, LinearSegmentedColormap
from typing import Dict, List, Tuple, Optional
import jax
import jax.numpy as jnp

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from openpi.models import gemma
from openpi.models import tokenizer as tok_module


def extract_text_attention(
    attention_weights: np.ndarray,
    text_token_start: int,
    text_token_end: int,
    query_token_idx: int,
    head_idx: Optional[int] = None
) -> np.ndarray:
    """
    Extract attention weights from a query token to text tokens.

    Args:
        attention_weights: Attention tensor of shape (B, K, G, T, S)
                          B=batch, K=num_kv_heads, G=num_heads_per_kv, T=query_len, S=key_len
        text_token_start: Start index of text tokens in the sequence
        text_token_end: End index of text tokens in the sequence
        query_token_idx: Which query token to visualize (e.g., action token)
        head_idx: Which head to use (None = average over all heads)

    Returns:
        Attention weights from query to text tokens, shape (num_text_tokens,)
    """
    batch_idx = 0  # Assume batch size 1 for visualization

    if head_idx is not None:
        # Select specific head
        k_idx = head_idx // attention_weights.shape[2]
        g_idx = head_idx % attention_weights.shape[2]
        attn = attention_weights[batch_idx, k_idx, g_idx, query_token_idx, :]
    else:
        # Average over all heads
        attn = attention_weights[batch_idx].reshape(-1, attention_weights.shape[3], attention_weights.shape[4])
        attn = attn[:, query_token_idx, :].mean(axis=0)

    # Extract only text token attention
    text_attn = attn[text_token_start:text_token_end]

    return np.array(text_attn)


def decode_tokens(token_ids: List[int]) -> List[str]:
    """
    Decode token IDs to words/subwords.

    Args:
        token_ids: List of token IDs

    Returns:
        List of decoded tokens (strings)
    """
    tokenizer = tok_module.get_tokenizer()

    # Decode each token individually to see subword boundaries
    tokens = []
    for tid in token_ids:
        try:
            # Decode single token
            decoded = tokenizer.decode([tid])
            tokens.append(decoded)
        except:
            tokens.append(f"<{tid}>")

    return tokens


def visualize_text_attention_bars(
    token_ids: List[int],
    attention_weights: np.ndarray,
    title: str = "Text Attention",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Create a bar chart showing attention to each token.

    Args:
        token_ids: List of token IDs
        attention_weights: Attention weight for each token
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Decode tokens
    tokens = decode_tokens(token_ids)

    # Normalize attention weights
    if attention_weights.max() > 0:
        norm_weights = attention_weights / attention_weights.max()
    else:
        norm_weights = attention_weights

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create bar chart
    x = np.arange(len(tokens))
    colors = plt.cm.RdYlBu_r(norm_weights)
    bars = ax.bar(x, attention_weights, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

    # Customize
    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_ylabel('Attention Weight', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=Normalize(vmin=0, vmax=attention_weights.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Attention Weight', fontsize=10)

    plt.tight_layout()
    return fig


def visualize_text_attention_heatmap(
    token_ids: List[int],
    attention_weights_per_layer: Dict[int, np.ndarray],
    title: str = "Text Attention Across Layers",
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Create a heatmap showing attention to each token across multiple layers.

    Args:
        token_ids: List of token IDs
        attention_weights_per_layer: Dict mapping layer_idx -> attention weights
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Decode tokens
    tokens = decode_tokens(token_ids)

    # Stack attention weights from all layers
    layers = sorted(attention_weights_per_layer.keys())
    attn_matrix = np.stack([attention_weights_per_layer[l] for l in layers])

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(attn_matrix, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')

    # Customize
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(layers)))
    ax.set_yticklabels([f'Layer {l}' for l in layers])
    ax.set_xlabel('Tokens', fontsize=12)
    ax.set_ylabel('Layers', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Attention Weight', fontsize=10)

    # Add grid
    ax.set_xticks(np.arange(len(tokens)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(layers)) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_text_attention_highlighted(
    text: str,
    token_ids: List[int],
    attention_weights: np.ndarray,
    title: str = "Highlighted Text Attention",
    figsize: Tuple[int, int] = (14, 3),
    colormap: str = 'YlOrRd'
) -> plt.Figure:
    """
    Visualize attention by highlighting text with background colors.

    Args:
        text: Original text prompt
        token_ids: List of token IDs
        attention_weights: Attention weight for each token
        title: Plot title
        figsize: Figure size
        colormap: Matplotlib colormap name

    Returns:
        Matplotlib figure
    """
    # Decode tokens
    tokens = decode_tokens(token_ids)

    # Normalize attention weights to [0, 1]
    if attention_weights.max() > attention_weights.min():
        norm_weights = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min())
    else:
        norm_weights = np.ones_like(attention_weights) * 0.5

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    # Get colormap
    cmap = plt.cm.get_cmap(colormap)

    # Position for text
    x_pos = 0.05
    y_pos = 0.5

    # Draw each token with background color
    for i, (token, weight) in enumerate(zip(tokens, norm_weights)):
        color = cmap(weight)

        # Add token with colored background
        bbox = dict(boxstyle='round,pad=0.3', facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.8)
        ax.text(x_pos, y_pos, token, fontsize=14, bbox=bbox,
                verticalalignment='center', family='monospace')

        # Update position (rough estimate, adjust as needed)
        x_pos += len(token) * 0.015 + 0.02

        # Wrap to next line if needed
        if x_pos > 0.95:
            x_pos = 0.05
            y_pos -= 0.2

    # Add title
    ax.text(0.5, 0.95, title, fontsize=14, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=Normalize(vmin=attention_weights.min(), vmax=attention_weights.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1, shrink=0.6)
    cbar.set_label('Attention Weight', fontsize=10)

    plt.tight_layout()
    return fig


def visualize_text_attention_summary(
    text: str,
    token_ids: List[int],
    attention_weights_per_layer: Dict[int, np.ndarray],
    output_path: Optional[str] = None,
    query_name: str = "Action Token"
) -> None:
    """
    Create a comprehensive visualization with multiple views of text attention.

    Args:
        text: Original text prompt
        token_ids: List of token IDs
        attention_weights_per_layer: Dict mapping layer_idx -> attention weights
        output_path: Where to save the visualization
        query_name: Name of the query token being visualized
    """
    # Select a representative layer (middle or last)
    representative_layer = max(attention_weights_per_layer.keys())
    rep_weights = attention_weights_per_layer[representative_layer]

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Bar chart for representative layer
    ax1 = fig.add_subplot(gs[0, :])
    tokens = decode_tokens(token_ids)
    x = np.arange(len(tokens))
    if rep_weights.max() > 0:
        norm_weights = rep_weights / rep_weights.max()
    else:
        norm_weights = rep_weights
    colors = plt.cm.RdYlBu_r(norm_weights)
    ax1.bar(x, rep_weights, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tokens, rotation=45, ha='right')
    ax1.set_ylabel('Attention Weight', fontsize=12)
    ax1.set_title(f'Text Attention - Layer {representative_layer} ({query_name})', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # 2. Heatmap across all layers
    ax2 = fig.add_subplot(gs[1, :])
    layers = sorted(attention_weights_per_layer.keys())
    attn_matrix = np.stack([attention_weights_per_layer[l] for l in layers])
    im = ax2.imshow(attn_matrix, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
    ax2.set_xticks(np.arange(len(tokens)))
    ax2.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
    ax2.set_yticks(np.arange(len(layers)))
    ax2.set_yticklabels([f'L{l}' for l in layers])
    ax2.set_xlabel('Tokens', fontsize=12)
    ax2.set_ylabel('Layers', fontsize=12)
    ax2.set_title('Attention Across All Layers', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax2, pad=0.02, label='Attention Weight')

    # 3. Top attended tokens
    ax3 = fig.add_subplot(gs[2, 0])
    top_k = min(5, len(tokens))
    top_indices = np.argsort(rep_weights)[-top_k:][::-1]
    top_tokens = [tokens[i] for i in top_indices]
    top_weights = [rep_weights[i] for i in top_indices]
    ax3.barh(range(top_k), top_weights, color='coral', edgecolor='black')
    ax3.set_yticks(range(top_k))
    ax3.set_yticklabels(top_tokens)
    ax3.set_xlabel('Attention Weight', fontsize=11)
    ax3.set_title(f'Top {top_k} Attended Tokens', fontsize=12, fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(axis='x', alpha=0.3, linestyle='--')

    # 4. Statistics
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')

    # Calculate statistics
    stats_text = f"""
    Text Attention Statistics
    {'='*40}

    Prompt: "{text}"

    Number of tokens: {len(tokens)}

    Layer {representative_layer} Statistics:
    - Max attention: {rep_weights.max():.4f}
    - Min attention: {rep_weights.min():.4f}
    - Mean attention: {rep_weights.mean():.4f}
    - Std attention: {rep_weights.std():.4f}

    Most attended token:
    - "{tokens[np.argmax(rep_weights)]}" ({rep_weights.max():.4f})

    Least attended token:
    - "{tokens[np.argmin(rep_weights)]}" ({rep_weights.min():.4f})

    Attention distribution:
    - Top 3 tokens account for {(np.sort(rep_weights)[-3:].sum() / rep_weights.sum() * 100):.1f}% of total
    """

    ax4.text(0.1, 0.95, stats_text, fontsize=10, verticalalignment='top',
             family='monospace', transform=ax4.transAxes)

    plt.suptitle(f'Text Attention Analysis: {query_name}', fontsize=16, fontweight='bold', y=0.98)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved text attention visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def analyze_text_attention_from_recorded(
    attention_dict: Dict[str, np.ndarray],
    tokenized_prompt: np.ndarray,
    num_image_tokens: int,
    prompt_text: str,
    layers_to_analyze: List[int] = [0, 8, 17],
    query_token_idx: Optional[int] = None,
    output_path: Optional[str] = None
) -> Dict[int, np.ndarray]:
    """
    Analyze text attention from recorded attention weights.

    Args:
        attention_dict: Dictionary of recorded attention weights
        tokenized_prompt: The tokenized prompt tensor
        num_image_tokens: Number of image tokens before text tokens
        prompt_text: Original prompt text
        layers_to_analyze: Which layers to analyze
        query_token_idx: Which token to use as query (None = first action token)
        output_path: Where to save visualization

    Returns:
        Dictionary mapping layer_idx -> text attention weights
    """
    # Get token IDs
    token_ids = tokenized_prompt[0].tolist()  # Remove batch dim

    # Calculate text token range
    text_token_start = num_image_tokens
    text_token_end = num_image_tokens + len(token_ids)

    # If query token not specified, use first action token (after image and text)
    if query_token_idx is None:
        query_token_idx = text_token_end

    # Extract text attention for each layer
    text_attention_per_layer = {}

    for layer_idx in layers_to_analyze:
        layer_key = f'layer_{layer_idx}'

        if layer_key not in attention_dict:
            print(f"Warning: Layer {layer_idx} not found in attention_dict")
            continue

        attn = attention_dict[layer_key][0]  # Get first batch item

        # Extract text attention
        text_attn = extract_text_attention(
            attn,
            text_token_start,
            text_token_end,
            query_token_idx,
            head_idx=None  # Average over all heads
        )

        text_attention_per_layer[layer_idx] = text_attn

    # Create visualization
    if text_attention_per_layer:
        visualize_text_attention_summary(
            prompt_text,
            token_ids,
            text_attention_per_layer,
            output_path=output_path,
            query_name="First Action Token"
        )

    return text_attention_per_layer


if __name__ == '__main__':
    # Example usage
    print("Text attention visualization module loaded.")
    print("\nTo use this module, import it and call analyze_text_attention_from_recorded()")
    print("after running your model with attention recording enabled.")
