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
    # Convert to float32 for matplotlib and formatting compatibility
    image_attn = np.asarray(image_attn, dtype=np.float32)

    # Extract text attention
    text_attn = extract_text_attention(
        attn,
        text_token_start,
        text_token_end,
        query_token_idx,
        head_idx=None
    )
    # Convert to float32 for matplotlib and formatting compatibility
    text_attn = np.asarray(text_attn, dtype=np.float32)

    # Create heatmap for image
    height, width = frame_rgb.shape[:2]
    heatmap = create_attention_heatmap(image_attn, (height, width), patch_size=patch_size)
    frame_with_attn = overlay_heatmap_on_image(frame_rgb, heatmap, colormap='jet', alpha=0.5)

    # Decode tokens and filter out padding tokens (token_id == 0)
    tokens = decode_tokens(token_ids)
    non_pad_mask = np.array([tid != 0 and tid is not False for tid in token_ids])
    filtered_tokens = [t for t, m in zip(tokens, non_pad_mask) if m]
    filtered_text_attn = text_attn[non_pad_mask]

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

    # 3. Text attention bar chart (excluding padding tokens)
    ax3 = fig.add_subplot(gs[1, :])
    x = np.arange(len(filtered_tokens))
    if len(filtered_text_attn) > 0 and filtered_text_attn.max() > 0:
        norm_weights = filtered_text_attn / filtered_text_attn.max()
    else:
        norm_weights = filtered_text_attn if len(filtered_text_attn) > 0 else np.array([])
    colors = plt.cm.RdYlBu_r(norm_weights) if len(norm_weights) > 0 else []
    ax3.bar(x, filtered_text_attn, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax3.set_xticks(x)
    ax3.set_xticklabels(filtered_tokens, rotation=45, ha='right', fontsize=10)
    ax3.set_ylabel('Attention Weight', fontsize=12)
    ax3.set_title(f'Linguistic Attention (Layer {layer_idx})', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    # 4. Combined statistics
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    # Calculate statistics (using filtered text attention, excluding padding)
    total_image_attn = float(image_attn.sum())
    total_text_attn = float(filtered_text_attn.sum()) if len(filtered_text_attn) > 0 else 0.0
    total_attn = total_image_attn + total_text_attn

    # Get full attention distribution
    full_attn = attn.reshape(-1, attn.shape[-2], attn.shape[-1]).mean(axis=0)[query_token_idx]
    full_attn = np.asarray(full_attn, dtype=np.float32)

    # Get top linguistic tokens (from filtered, non-padding tokens)
    if len(filtered_tokens) >= 3:
        top_indices = np.argsort(filtered_text_attn)[-3:][::-1]
        top1_token, top1_weight = filtered_tokens[top_indices[0]], filtered_text_attn[top_indices[0]]
        top2_token, top2_weight = filtered_tokens[top_indices[1]], filtered_text_attn[top_indices[1]]
        top3_token, top3_weight = filtered_tokens[top_indices[2]], filtered_text_attn[top_indices[2]]
    elif len(filtered_tokens) > 0:
        top_indices = np.argsort(filtered_text_attn)[::-1]
        top1_token, top1_weight = filtered_tokens[top_indices[0]], filtered_text_attn[top_indices[0]]
        top2_token = top3_token = "N/A"
        top2_weight = top3_weight = 0.0
    else:
        top1_token = top2_token = top3_token = "N/A"
        top1_weight = top2_weight = top3_weight = 0.0

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

    Top Linguistic Tokens (excluding padding):
    - Most attended: "{top1_token}" ({top1_weight:.4f})
    - 2nd most:      "{top2_token}" ({top2_weight:.4f})
    - 3rd most:      "{top3_token}" ({top3_weight:.4f})

    Interpretation:
    {"- Model focuses MORE on visual input" if total_image_attn > total_text_attn else "- Model focuses MORE on linguistic input"}
    - The model is {"highly" if len(filtered_text_attn) > 0 and filtered_text_attn.max() > 2*filtered_text_attn.mean() else "moderately"} selective in text attention
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

    # Create mask to filter out padding tokens (token_id == 0)
    non_pad_mask = np.array([tid != 0 and tid is not False for tid in token_ids])

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

        # Convert to float32
        image_attn = np.asarray(image_attn, dtype=np.float32)
        text_attn = np.asarray(text_attn, dtype=np.float32)

        # Filter out padding tokens from text attention before summing
        filtered_text_attn = text_attn[non_pad_mask] if len(text_attn) == len(non_pad_mask) else text_attn

        # Sum attention (excluding padding for text)
        visual_attn_per_layer.append(float(image_attn.sum()))
        linguistic_attn_per_layer.append(float(filtered_text_attn.sum()))
        layer_indices.append(layer_idx)

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 1. Stacked area plot
    ax1 = axes[0]
    visual_arr = np.array(visual_attn_per_layer, dtype=np.float32)
    linguistic_arr = np.array(linguistic_attn_per_layer, dtype=np.float32)

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


def compute_frame_attention_stats(
    attention_dict: Dict[str, np.ndarray],
    token_ids: List[int],
    num_image_tokens: int,
    layers_to_analyze: List[int],
    query_token_idx: Optional[int] = None,
) -> Dict[str, Dict[int, float]]:
    """
    Compute visual and linguistic attention statistics for a single frame across layers.

    Args:
        attention_dict: Recorded attention weights for this frame
        token_ids: Tokenized prompt
        num_image_tokens: Number of image tokens
        layers_to_analyze: Which layers to analyze
        query_token_idx: Which token to use as query

    Returns:
        Dictionary with 'visual' and 'linguistic' keys, each mapping layer_idx to attention sum
    """
    text_token_start = num_image_tokens
    text_token_end = num_image_tokens + len(token_ids)
    image_token_start = 0
    image_token_end = num_image_tokens

    if query_token_idx is None:
        query_token_idx = text_token_end

    # Create mask to filter out padding tokens (token_id == 0)
    non_pad_mask = np.array([tid != 0 and tid is not False for tid in token_ids])

    visual_by_layer = {}
    linguistic_by_layer = {}

    for layer_idx in layers_to_analyze:
        layer_key = f'layer_{layer_idx}'
        if layer_key not in attention_dict:
            continue

        attn = attention_dict[layer_key][0]

        image_attn = extract_image_attention(
            attn, image_token_start, image_token_end, query_token_idx, head_idx=None
        )
        text_attn = extract_text_attention(
            attn, text_token_start, text_token_end, query_token_idx, head_idx=None
        )

        # Convert to float32
        image_attn = np.asarray(image_attn, dtype=np.float32)
        text_attn = np.asarray(text_attn, dtype=np.float32)

        # Filter out padding tokens from text attention before summing
        filtered_text_attn = text_attn[non_pad_mask] if len(text_attn) == len(non_pad_mask) else text_attn

        visual_by_layer[layer_idx] = float(image_attn.sum())
        linguistic_by_layer[layer_idx] = float(filtered_text_attn.sum())

    return {'visual': visual_by_layer, 'linguistic': linguistic_by_layer}


def visualize_episode_attention_evolution(
    frame_attention_stats: List[Dict[str, Dict[int, float]]],
    frame_indices: List[int],
    prompt_text: str,
    layers_to_viz: List[int],
    output_path: Optional[str] = None,
    mode: str = "across_layers",  # "across_layers" or "across_frames"
) -> plt.Figure:
    """
    Visualize how visual/linguistic attention evolves across an episode.

    Args:
        frame_attention_stats: List of attention stats per frame (from compute_frame_attention_stats)
        frame_indices: Frame indices corresponding to each stats entry
        prompt_text: The prompt text
        layers_to_viz: Layers to visualize
        output_path: Where to save the figure
        mode: "across_layers" shows layer evolution averaged over frames,
              "across_frames" shows frame evolution for selected layers

    Returns:
        Matplotlib figure
    """
    if mode == "across_frames":
        # Show how attention evolves across frames for each layer (stacked visual + linguistic)
        num_layers = len(layers_to_viz)
        fig, axes = plt.subplots(num_layers, 1, figsize=(14, 4 * num_layers))
        if num_layers == 1:
            axes = [axes]

        for ax_idx, layer_idx in enumerate(layers_to_viz):
            visual_per_frame = []
            linguistic_per_frame = []
            valid_frame_indices = []

            for i, stats in enumerate(frame_attention_stats):
                if layer_idx in stats['visual']:
                    visual_per_frame.append(stats['visual'][layer_idx])
                    linguistic_per_frame.append(stats['linguistic'][layer_idx])
                    valid_frame_indices.append(frame_indices[i])

            if visual_per_frame:
                visual_arr = np.array(visual_per_frame, dtype=np.float32)
                linguistic_arr = np.array(linguistic_per_frame, dtype=np.float32)

                # Stacked area plot: visual on bottom, linguistic on top
                axes[ax_idx].fill_between(valid_frame_indices, 0, visual_arr,
                                         alpha=0.6, color='steelblue', label='Visual Attention')
                axes[ax_idx].fill_between(valid_frame_indices, visual_arr, visual_arr + linguistic_arr,
                                         alpha=0.6, color='coral', label='Linguistic Attention')
                axes[ax_idx].plot(valid_frame_indices, visual_arr + linguistic_arr,
                                 color='darkred', linewidth=1, alpha=0.5)

            axes[ax_idx].set_xlabel('Frame Index', fontsize=12)
            axes[ax_idx].set_ylabel('Total Attention Weight', fontsize=12)
            axes[ax_idx].set_title(f'Layer {layer_idx}: Visual vs. Linguistic Attention', fontsize=13, fontweight='bold')
            axes[ax_idx].legend(loc='upper right', fontsize=10)
            axes[ax_idx].grid(alpha=0.3, linestyle='--')

        plt.suptitle(f'Attention Evolution Across Episode (Stacked)\nPrompt: "{prompt_text}"',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

    else:  # across_layers - show layer evolution with stacked visual + linguistic
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Compute mean across all frames for each layer
        all_layers = sorted(set(l for stats in frame_attention_stats for l in stats['visual'].keys()))

        visual_mean = []
        linguistic_mean = []

        for layer in all_layers:
            v_vals = [s['visual'].get(layer, np.nan) for s in frame_attention_stats]
            l_vals = [s['linguistic'].get(layer, np.nan) for s in frame_attention_stats]
            v_vals = [v for v in v_vals if not np.isnan(v)]
            l_vals = [v for v in l_vals if not np.isnan(v)]

            visual_mean.append(np.mean(v_vals) if v_vals else 0)
            linguistic_mean.append(np.mean(l_vals) if l_vals else 0)

        visual_mean = np.array(visual_mean, dtype=np.float32)
        linguistic_mean = np.array(linguistic_mean, dtype=np.float32)

        # 1. Stacked area plot
        ax1 = axes[0]
        ax1.fill_between(all_layers, 0, visual_mean, alpha=0.6, color='steelblue', label='Visual Attention')
        ax1.fill_between(all_layers, visual_mean, visual_mean + linguistic_mean,
                        alpha=0.6, color='coral', label='Linguistic Attention')
        ax1.set_xlabel('Layer', fontsize=12)
        ax1.set_ylabel('Total Attention Weight', fontsize=12)
        ax1.set_title('Visual vs. Linguistic Attention Evolution', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=11)
        ax1.grid(alpha=0.3, linestyle='--')
        ax1.set_xticks(all_layers)

        # 2. Percentage distribution (stacked bar)
        ax2 = axes[1]
        total_attn = visual_mean + linguistic_mean
        # Avoid division by zero
        total_attn_safe = np.where(total_attn == 0, 1, total_attn)
        visual_pct = (visual_mean / total_attn_safe) * 100
        linguistic_pct = (linguistic_mean / total_attn_safe) * 100

        width = 0.8
        x = np.arange(len(all_layers))
        ax2.bar(x, visual_pct, width, label='Visual %', color='steelblue', alpha=0.7)
        ax2.bar(x, linguistic_pct, width, bottom=visual_pct, label='Linguistic %', color='coral', alpha=0.7)

        ax2.set_xlabel('Layer', fontsize=12)
        ax2.set_ylabel('Attention Percentage', fontsize=12)
        ax2.set_title('Relative Attention Distribution', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'L{l}' for l in all_layers])
        ax2.legend(loc='upper right', fontsize=11)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.set_ylim([0, 100])

        plt.suptitle(f'Attention Evolution Analysis (Episode Mean)\nPrompt: "{prompt_text}" ({len(frame_attention_stats)} frames)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved episode attention evolution to {output_path}")

    return fig


if __name__ == '__main__':
    print("Combined attention visualization module loaded.")
    print("\nThis module provides functions to visualize both image and text attention together.")
