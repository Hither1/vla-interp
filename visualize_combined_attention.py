"""
Combined visualization of image and text attention.

Shows both:
1. Which parts of the image the model attends to (visual attention)
2. Which words in the prompt the model attends to (linguistic attention)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Tuple
import cv2

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from visualize_attention import (
    extract_image_attention,
    create_attention_heatmap,
    overlay_heatmap_on_image,
)
from visualize_text_attention import (
    extract_text_attention,
    decode_tokens,
)


def visualize_combined_attention(
    frame_rgb: np.ndarray,
    prompt_text: str,
    token_ids: List[int],
    attention_dict: Dict[str, np.ndarray],
    num_image_tokens: int,
    layer_idx: int = 17,
    query_token_idx: Optional[int] = None,
    output_path: Optional[str] = None,
    patch_size: int = 14,
) -> plt.Figure:
    """
    Create a combined visualization showing both image and text attention.

    Args:
        frame_rgb: Original RGB frame
        prompt_text: Original text prompt
        token_ids: Tokenized prompt
        attention_dict: Recorded attention weights
        num_image_tokens: Number of image tokens
        layer_idx: Which layer to visualize
        query_token_idx: Which token to use as query (None = first action token)
        output_path: Where to save the figure
        patch_size: Size of image patches

    Returns:
        Matplotlib figure
    """
    layer_key = f'layer_{layer_idx}'

    if layer_key not in attention_dict:
        print(f"Error: Layer {layer_idx} not found in attention_dict")
        return None

    attn = attention_dict[layer_key][0]  # Get first batch item

    # Calculate token ranges
    text_token_start = num_image_tokens
    text_token_end = num_image_tokens + len(token_ids)
    image_token_start = 0
    image_token_end = num_image_tokens

    if query_token_idx is None:
        query_token_idx = text_token_end  # First action token

    # Extract image attention
    image_attn = extract_image_attention(
        attn,
        image_token_start,
        image_token_end,
        query_token_idx,
        head_idx=None
    )

    # Extract text attention
    text_attn = extract_text_attention(
        attn,
        text_token_start,
        text_token_end,
        query_token_idx,
        head_idx=None
    )

    # Create heatmap for image
    height, width = frame_rgb.shape[:2]
    heatmap = create_attention_heatmap(image_attn, (height, width), patch_size=patch_size)
    frame_with_attn = overlay_heatmap_on_image(frame_rgb, heatmap, colormap='jet', alpha=0.5)

    # Decode tokens
    tokens = decode_tokens(token_ids)

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3, height_ratios=[2, 1, 1])

    # 1. Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(frame_rgb)
    ax1.set_title('Original Frame', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # 2. Image with attention heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(frame_with_attn)
    ax2.set_title(f'Visual Attention (Layer {layer_idx})', fontsize=14, fontweight='bold')
    ax2.axis('off')

    # 3. Text attention bar chart
    ax3 = fig.add_subplot(gs[1, :])
    x = np.arange(len(tokens))
    if text_attn.max() > 0:
        norm_weights = text_attn / text_attn.max()
    else:
        norm_weights = text_attn
    colors = plt.cm.RdYlBu_r(norm_weights)
    ax3.bar(x, text_attn, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax3.set_xticks(x)
    ax3.set_xticklabels(tokens, rotation=45, ha='right', fontsize=10)
    ax3.set_ylabel('Attention Weight', fontsize=12)
    ax3.set_title(f'Linguistic Attention (Layer {layer_idx})', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    # 4. Combined statistics
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    # Calculate statistics
    total_image_attn = image_attn.sum()
    total_text_attn = text_attn.sum()
    total_attn = total_image_attn + total_text_attn

    # Get full attention distribution
    full_attn = attn.reshape(-1, attn.shape[-2], attn.shape[-1]).mean(axis=0)[query_token_idx]

    stats_text = f"""
    Combined Attention Analysis - Layer {layer_idx}
    {'='*80}

    Prompt: "{prompt_text}"

    Attention Distribution:
    ├─ Visual Attention:     {total_image_attn:.4f} ({total_image_attn/total_attn*100:.1f}% of total)
    ├─ Linguistic Attention: {total_text_attn:.4f} ({total_text_attn/total_attn*100:.1f}% of total)
    └─ Other (state/action): {full_attn.sum() - total_attn:.4f} ({(full_attn.sum() - total_attn)/full_attn.sum()*100:.1f}% of total)

    Top Visual Regions:
    - Highest attention patch: {image_attn.max():.4f} (position {np.argmax(image_attn)})

    Top Linguistic Tokens:
    - Most attended: "{tokens[np.argmax(text_attn)]}" ({text_attn.max():.4f})
    - 2nd most:      "{tokens[np.argsort(text_attn)[-2]]}" ({text_attn[np.argsort(text_attn)[-2]]:.4f})
    - 3rd most:      "{tokens[np.argsort(text_attn)[-3]]}" ({text_attn[np.argsort(text_attn)[-3]]:.4f})

    Interpretation:
    {"- Model focuses MORE on visual input" if total_image_attn > total_text_attn else "- Model focuses MORE on linguistic input"}
    - The model is {"highly" if text_attn.max() > 2*text_attn.mean() else "moderately"} selective in text attention
    - The model is {"highly" if image_attn.max() > 2*image_attn.mean() else "moderately"} selective in visual attention
    """

    ax4.text(0.05, 0.95, stats_text, fontsize=11, verticalalignment='top',
             family='monospace', transform=ax4.transAxes)

    plt.suptitle('Combined Visual and Linguistic Attention Analysis',
                 fontsize=16, fontweight='bold', y=0.98)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved combined attention visualization to {output_path}")

    return fig


def visualize_multimodal_attention_evolution(
    frame_rgb: np.ndarray,
    prompt_text: str,
    token_ids: List[int],
    attention_dict: Dict[str, np.ndarray],
    num_image_tokens: int,
    layers_to_viz: List[int] = [0, 5, 11, 17],
    query_token_idx: Optional[int] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Show how the balance between visual and linguistic attention evolves across layers.

    Args:
        frame_rgb: Original RGB frame
        prompt_text: Original text prompt
        token_ids: Tokenized prompt
        attention_dict: Recorded attention weights
        num_image_tokens: Number of image tokens
        layers_to_viz: Which layers to visualize
        query_token_idx: Which token to use as query
        output_path: Where to save the figure

    Returns:
        Matplotlib figure
    """
    # Calculate token ranges
    text_token_start = num_image_tokens
    text_token_end = num_image_tokens + len(token_ids)
    image_token_start = 0
    image_token_end = num_image_tokens

    if query_token_idx is None:
        query_token_idx = text_token_end

    # Collect attention stats for each layer
    visual_attn_per_layer = []
    linguistic_attn_per_layer = []
    layer_indices = []

    for layer_idx in layers_to_viz:
        layer_key = f'layer_{layer_idx}'
        if layer_key not in attention_dict:
            continue

        attn = attention_dict[layer_key][0]

        # Extract image and text attention
        image_attn = extract_image_attention(
            attn, image_token_start, image_token_end, query_token_idx, head_idx=None
        )
        text_attn = extract_text_attention(
            attn, text_token_start, text_token_end, query_token_idx, head_idx=None
        )

        visual_attn_per_layer.append(image_attn.sum())
        linguistic_attn_per_layer.append(text_attn.sum())
        layer_indices.append(layer_idx)

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 1. Stacked area plot
    ax1 = axes[0]
    visual_arr = np.array(visual_attn_per_layer)
    linguistic_arr = np.array(linguistic_attn_per_layer)

    ax1.fill_between(layer_indices, 0, visual_arr, alpha=0.6, color='steelblue', label='Visual Attention')
    ax1.fill_between(layer_indices, visual_arr, visual_arr + linguistic_arr,
                     alpha=0.6, color='coral', label='Linguistic Attention')
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Total Attention Weight', fontsize=12)
    ax1.set_title('Visual vs. Linguistic Attention Evolution', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_xticks(layer_indices)

    # 2. Percentage distribution
    ax2 = axes[1]
    total_attn = visual_arr + linguistic_arr
    visual_pct = (visual_arr / total_attn) * 100
    linguistic_pct = (linguistic_arr / total_attn) * 100

    width = 0.8
    x = np.arange(len(layer_indices))
    ax2.bar(x, visual_pct, width, label='Visual %', color='steelblue', alpha=0.7)
    ax2.bar(x, linguistic_pct, width, bottom=visual_pct, label='Linguistic %', color='coral', alpha=0.7)

    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Attention Percentage', fontsize=12)
    ax2.set_title('Relative Attention Distribution', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'L{l}' for l in layer_indices])
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim([0, 100])

    plt.suptitle(f'Multimodal Attention Evolution\nPrompt: "{prompt_text}"',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention evolution visualization to {output_path}")

    return fig


if __name__ == '__main__':
    print("Combined attention visualization module loaded.")
    print("\nThis module provides functions to visualize both image and text attention together.")
