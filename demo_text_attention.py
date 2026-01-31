"""
Simple demo showing how to extract and visualize text attention.

This is a minimal example showing the core functionality without all the bells and whistles.
"""

import numpy as np
import matplotlib.pyplot as plt

# Mock data for demonstration
def create_mock_attention_data():
    """
    Create mock attention data for demonstration.
    In real usage, this would come from gemma.ATTENTION_WEIGHTS
    """
    # Simulate attention from action token to text tokens
    # Format: (batch=1, kv_heads=1, heads_per_kv=8, query_len, key_len)

    # Example: 10 text tokens, 1 query token (action token)
    num_text_tokens = 10
    num_heads = 8

    # Create realistic-looking attention weights
    # Higher attention to action words (index 1, 5) and object words (index 3, 7)
    base_attention = np.random.rand(num_heads, num_text_tokens) * 0.2

    # Boost attention to specific tokens
    base_attention[:, 1] += 0.5  # "pick" (action verb)
    base_attention[:, 3] += 0.4  # "bowl" (object)
    base_attention[:, 5] += 0.3  # "place" (action verb)
    base_attention[:, 7] += 0.4  # "plate" (target object)

    # Add some head diversity
    for i in range(num_heads):
        if i % 2 == 0:
            base_attention[i, 3] *= 1.5  # Some heads focus more on "bowl"
        else:
            base_attention[i, 7] *= 1.5  # Other heads focus more on "plate"

    # Normalize
    base_attention = base_attention / base_attention.sum(axis=1, keepdims=True)

    return base_attention


def demo_basic_text_attention():
    """
    Basic example: Show attention to each token with a bar chart.
    """
    # Mock tokens and attention
    tokens = ["pick", "up", "the", "bowl", "and", "place", "it", "on", "the", "plate"]
    attention = create_mock_attention_data()

    # Average over heads
    avg_attention = attention.mean(axis=0)

    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 5))

    # Bar chart
    x = np.arange(len(tokens))
    colors = plt.cm.RdYlBu_r(avg_attention / avg_attention.max())
    bars = ax.bar(x, avg_attention, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Customize
    ax.set_xticks(x)
    ax.set_xticklabels(tokens, fontsize=12, fontweight='bold')
    ax.set_ylabel('Attention Weight', fontsize=13)
    ax.set_title('Text Attention: Which Words Does the Model Focus On?', fontsize=15, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on top of bars
    for i, (bar, val) in enumerate(zip(bars, avg_attention)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Add interpretation
    top_3_indices = np.argsort(avg_attention)[-3:][::-1]
    interpretation = f"Top 3 attended tokens: {', '.join([f'{tokens[i]} ({avg_attention[i]:.3f})' for i in top_3_indices])}"
    ax.text(0.5, -0.15, interpretation, transform=ax.transAxes,
            ha='center', fontsize=11, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig('demo_text_attention_basic.png', dpi=150, bbox_inches='tight')
    print("✓ Saved demo_text_attention_basic.png")
    plt.close()


def demo_multihead_text_attention():
    """
    Advanced example: Show how different attention heads focus on different words.
    """
    tokens = ["pick", "up", "the", "bowl", "and", "place", "it", "on", "the", "plate"]
    attention = create_mock_attention_data()  # Shape: (num_heads, num_tokens)

    num_heads = attention.shape[0]

    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 8))

    # Heatmap
    im = ax.imshow(attention, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')

    # Customize
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, fontsize=11, fontweight='bold')
    ax.set_yticks(np.arange(num_heads))
    ax.set_yticklabels([f'Head {i}' for i in range(num_heads)], fontsize=10)
    ax.set_xlabel('Tokens', fontsize=13, fontweight='bold')
    ax.set_ylabel('Attention Heads', fontsize=13, fontweight='bold')
    ax.set_title('Multi-Head Text Attention: Different Heads Focus on Different Words',
                 fontsize=15, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Attention Weight', fontsize=12)

    # Add grid
    ax.set_xticks(np.arange(len(tokens)) - 0.5, minor=True)
    ax.set_yticks(np.arange(num_heads) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    # Add interpretation
    interpretation = (
        "Each row shows one attention head's focus.\n"
        "Different heads specialize in different aspects:\n"
        "• Some focus on action words (pick, place)\n"
        "• Others focus on objects (bowl, plate)\n"
        "• This specialization helps the model understand complex tasks"
    )
    ax.text(1.25, 0.5, interpretation, transform=ax.transAxes,
            va='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()
    plt.savefig('demo_text_attention_multihead.png', dpi=150, bbox_inches='tight')
    print("✓ Saved demo_text_attention_multihead.png")
    plt.close()


def demo_text_attention_evolution():
    """
    Show how text attention changes across layers.
    """
    tokens = ["pick", "up", "the", "bowl", "and", "place", "it", "on", "the", "plate"]

    # Simulate attention across 4 layers
    layers = [0, 5, 11, 17]
    attention_per_layer = []

    for layer_idx in layers:
        # Simulate different attention patterns at different layers
        layer_attn = create_mock_attention_data().mean(axis=0)

        # Early layers: more uniform
        if layer_idx < 6:
            layer_attn = layer_attn * 0.5 + 0.05  # Flatten distribution

        # Late layers: more focused
        elif layer_idx > 12:
            # Emphasize task-critical words even more
            layer_attn[1] *= 1.5  # "pick"
            layer_attn[3] *= 1.8  # "bowl"
            layer_attn[5] *= 1.5  # "place"
            layer_attn[7] *= 1.8  # "plate"
            layer_attn = layer_attn / layer_attn.sum()  # Re-normalize

        attention_per_layer.append(layer_attn)

    attention_matrix = np.stack(attention_per_layer)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Heatmap
    im = ax1.imshow(attention_matrix, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
    ax1.set_xticks(np.arange(len(tokens)))
    ax1.set_xticklabels(tokens, fontsize=11, fontweight='bold')
    ax1.set_yticks(np.arange(len(layers)))
    ax1.set_yticklabels([f'Layer {l}' for l in layers], fontsize=10)
    ax1.set_xlabel('Tokens', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Layers', fontsize=12, fontweight='bold')
    ax1.set_title('Text Attention Evolution Across Layers', fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax1, pad=0.02)
    cbar.set_label('Attention Weight', fontsize=11)

    # Line plot showing evolution of specific tokens
    ax2.plot(layers, attention_matrix[:, 1], 'o-', label='pick (action)', linewidth=2, markersize=8)
    ax2.plot(layers, attention_matrix[:, 3], 's-', label='bowl (object)', linewidth=2, markersize=8)
    ax2.plot(layers, attention_matrix[:, 5], '^-', label='place (action)', linewidth=2, markersize=8)
    ax2.plot(layers, attention_matrix[:, 7], 'd-', label='plate (target)', linewidth=2, markersize=8)
    ax2.plot(layers, attention_matrix[:, 2], 'x--', label='the (article)', linewidth=1.5, markersize=8, alpha=0.5)

    ax2.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Attention Weight', fontsize=12, fontweight='bold')
    ax2.set_title('How Attention to Key Words Changes Through Layers', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.set_xticks(layers)

    # Add interpretation
    interpretation = (
        "Observations:\n"
        "• Early layers (0-5): Attention is more uniform\n"
        "• Later layers (11-17): Attention becomes more focused on task-critical words\n"
        "• Action words (pick, place) and object words (bowl, plate) get more attention\n"
        "• Function words (the, it) get less attention over time"
    )
    fig.text(0.5, -0.02, interpretation, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig('demo_text_attention_evolution.png', dpi=150, bbox_inches='tight')
    print("✓ Saved demo_text_attention_evolution.png")
    plt.close()


if __name__ == '__main__':
    print("Generating text attention demo visualizations...\n")

    print("1. Basic text attention (bar chart)")
    demo_basic_text_attention()

    print("\n2. Multi-head text attention (heatmap)")
    demo_multihead_text_attention()

    print("\n3. Text attention evolution across layers")
    demo_text_attention_evolution()

    print("\n" + "="*60)
    print("✓ All demos generated successfully!")
    print("="*60)
    print("\nThese are mock examples to show what the visualizations look like.")
    print("With real model data, you would:")
    print("  1. Enable attention recording: gemma.SAVE_ATTENTION_WEIGHTS = True")
    print("  2. Run your model on an observation")
    print("  3. Extract attention from gemma.ATTENTION_WEIGHTS")
    print("  4. Use visualize_text_attention.py functions to create similar plots")
