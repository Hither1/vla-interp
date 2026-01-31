# VLA Attention Visualization

This directory contains tools for visualizing attention patterns in the Pi0 Vision-Language-Action (VLA) model. These tools help you understand which parts of the input (images, text, previous actions) the model is paying attention to when making predictions.

## Overview

The attention visualization system consists of:

1. **Modified Attention Recording** ([gemma.py](src/openpi/models/gemma.py))
   - Added `SAVE_ATTENTION_WEIGHTS` flag to enable/disable recording
   - Added `ATTENTION_WEIGHTS` dictionary to store attention patterns
   - Added `_record_attention_weights()` function to capture attention during forward pass

2. **Visual Attention Library** ([visualize_attention.py](visualize_attention.py))
   - Functions to extract attention weights from the model
   - Convert attention to 2D heatmaps matching image dimensions
   - Overlay heatmaps on video frames

3. **Linguistic Attention Library** ([visualize_text_attention.py](visualize_text_attention.py))
   - Extract attention to text tokens (words/subwords)
   - Visualize with bar charts, heatmaps, and highlighted text
   - Analyze which words the model focuses on

4. **Combined Multimodal Visualization** ([visualize_combined_attention.py](visualize_combined_attention.py))
   - Show both visual and linguistic attention together
   - Compare attention distribution across modalities
   - Track attention evolution through layers

5. **Example Scripts** ([example_attention_viz.py](example_attention_viz.py))
   - Complete examples for visualizing single frames or full episodes
   - Support for image-only, text-only, combined, or all visualization types
   - Integration with existing data loading utilities

## Quick Start

### Visualize Visual Attention (Image Only)

```bash
python example_attention_viz.py \
    --mode single \
    --video data/libero/90/videos/rollout_task0_ep0.mp4 \
    --frame-idx 10 \
    --prompt "pick up the black bowl and place it on the plate" \
    --checkpoint checkpoints/pi0_model.ckpt \
    --output attention_frame10.png \
    --layers 0 8 17 \
    --viz-type image
```

### Visualize Linguistic Attention (Text Only)

See which words in the prompt the model focuses on:

```bash
python example_attention_viz.py \
    --mode single \
    --video data/libero/90/videos/rollout_task0_ep0.mp4 \
    --frame-idx 10 \
    --prompt "pick up the black bowl and place it on the plate" \
    --checkpoint checkpoints/pi0_model.ckpt \
    --output text_attention.png \
    --layers 0 8 17 \
    --viz-type text
```

### Visualize Combined (Visual + Linguistic)

Show both image and text attention together:

```bash
python example_attention_viz.py \
    --mode single \
    --video data/libero/90/videos/rollout_task0_ep0.mp4 \
    --frame-idx 10 \
    --prompt "pick up the black bowl and place it on the plate" \
    --checkpoint checkpoints/pi0_model.ckpt \
    --output combined_attention.png \
    --layers 0 8 17 \
    --viz-type combined
```

### Generate All Visualizations

Create all visualization types at once:

```bash
python example_attention_viz.py \
    --mode single \
    --video data/libero/90/videos/rollout_task0_ep0.mp4 \
    --frame-idx 10 \
    --prompt "pick up the black bowl and place it on the plate" \
    --checkpoint checkpoints/pi0_model.ckpt \
    --output attention.png \
    --layers 0 8 17 \
    --viz-type all
```

This will generate:
- `attention_image.png` - Visual attention heatmaps
- `attention_text.png` - Linguistic attention analysis
- `attention_combined.png` - Combined multimodal visualization
- `attention_evolution.png` - Attention distribution across layers

### Visualize Multiple Frames from an Episode

```bash
python example_attention_viz.py \
    --mode episode \
    --data-root data/libero \
    --group 90 \
    --episode-idx 0 \
    --frames 0 10 20 30 40 \
    --checkpoint checkpoints/pi0_model.ckpt \
    --output-dir attention_viz_output/ \
    --viz-type all
```

## How It Works

### 1. Attention Recording

The model's `Attention` class in [gemma.py:222-313](src/openpi/models/gemma.py#L222-L313) computes attention probabilities:

```python
probs = jax.nn.softmax(masked_logits, axis=-1)  # Shape: (B, K, G, T, S)
```

Where:
- `B` = batch size
- `K` = number of key/value heads
- `G` = number of query heads per key/value head
- `T` = query sequence length
- `S` = key sequence length

When `SAVE_ATTENTION_WEIGHTS = True`, these probabilities are saved for each layer.

### 2. Token Layout

For a typical forward pass, the token sequence looks like:

```
[IMAGE_TOKENS] [TEXT_TOKENS] [STATE_TOKEN?] [ACTION_TOKENS]
|--- 256 ---| |---- N -----| |------ 1 -----| |--- 10 ----|
```

- **Image tokens**: Typically 256 tokens (16x16 patches from SigLIP for a 224x224 image)
- **Text tokens**: Variable length (N) depending on the prompt
- **State token**: Optional, only in non-pi0.5 models
- **Action tokens**: Action horizon (default 10)

### 3. Visualization Process

1. **Extract attention weights** for a specific query token (e.g., first action token)
2. **Focus on image tokens** - extract the attention weights from the query to image positions
3. **Reshape to spatial grid** - convert 1D attention (256,) to 2D grid (16, 16)
4. **Upsample to image resolution** - resize to match original frame size
5. **Apply colormap and overlay** - create heatmap and blend with original image

## Interpreting the Visualizations

### Visual Attention (Image)

**Color Coding:**
- **Red/Yellow**: High attention (model focuses here)
- **Blue/Purple**: Low attention (model ignores these areas)

**What to Look For:**
1. **Object-centric attention**: Does the model focus on relevant objects mentioned in the prompt?
2. **Spatial reasoning**: Does attention shift based on spatial relationships?
3. **Temporal patterns**: How does attention change across frames during an episode?
4. **Layer differences**:
   - Early layers (0-5): Often focus on low-level visual features
   - Middle layers (6-12): Task-relevant objects and relationships
   - Late layers (13-17): Fine-grained task-specific details

### Linguistic Attention (Text)

**What the Visualization Shows:**
- Bar charts showing attention weight for each word/token
- Heatmaps across layers showing how attention evolves
- Highlighted text with color intensity indicating attention strength

**What to Look For:**
1. **Action verbs**: Does the model focus on action words like "pick", "place", "open"?
2. **Object nouns**: Does it attend to object names like "bowl", "drawer", "cabinet"?
3. **Spatial relations**: Does it focus on spatial words like "top", "bottom", "on", "in"?
4. **Modifiers**: Which adjectives/modifiers get attention? ("black", "left", "right")
5. **Task-irrelevant words**: Are common words ("the", "a", "and") properly ignored?

**Common Patterns:**
- **Early layers**: Often attend more uniformly to all tokens
- **Middle layers**: Start focusing on task-relevant words (objects, actions)
- **Late layers**: Highly selective attention to the most critical task words

### Combined Attention (Multimodal)

**What the Visualization Shows:**
- Side-by-side comparison of visual and linguistic attention
- Statistics showing attention distribution across modalities
- Evolution of attention balance through layers

**Key Metrics:**
1. **Visual vs. Linguistic Ratio**: What percentage of attention goes to each modality?
   - High visual attention: Model relies more on visual grounding
   - High linguistic attention: Model relies more on language understanding
   - Balanced: Model integrates both modalities equally

2. **Cross-modal Consistency**: Do attended words match attended visual regions?
   - E.g., if "bowl" has high text attention, does the bowl region have high visual attention?

3. **Layer-wise Evolution**:
   - Early layers: Often more visual-dominant
   - Late layers: May shift more toward linguistic or become more balanced

## Advanced Usage

### Customize Visualization

```python
from visualize_attention import AttentionVisualizationConfig

config = AttentionVisualizationConfig(
    layers_to_viz=[0, 5, 11, 17],  # Visualize 4 layers
    heads_to_viz=None,              # Average over all heads (or specify [0, 1, 2])
    colormap='viridis',             # Try 'hot', 'plasma', 'viridis', etc.
    overlay_alpha=0.6,              # More opaque heatmap
    output_width=1920,
    output_height=1080
)
```

### Visualize Different Query Tokens

By default, we visualize what the **first action token** attends to. You can change this:

```python
visualizations = visualize_attention_on_frame(
    model, observation, frame_rgb, config,
    query_token_type="last"  # Use last token instead of first action token
)
```

### Programmatic Access

```python
from visualize_attention import enable_attention_recording, get_recorded_attention_weights

# Enable recording
enable_attention_recording()

# Run model
model.sample_actions(rng, observation, num_steps=10)

# Get attention weights
attn_dict = get_recorded_attention_weights()

# attn_dict = {
#     'layer_0': [array(B, K, G, T, S)],
#     'layer_1': [array(B, K, G, T, S)],
#     ...
# }
```

## Troubleshooting

### "No attention weights were recorded"

This means the attention recording hook isn't being called. Make sure:
1. `SAVE_ATTENTION_WEIGHTS = True` before running the model
2. The model actually runs a forward pass (not cached)
3. The scope path is being parsed correctly to determine layer index

### Heatmap doesn't align with image

This can happen if:
1. Image resolution doesn't match expected size (224x224 for SigLIP)
2. Number of image tokens is incorrect (should be 256 for 14x14 patches)
3. Token indices are wrong (check `image_token_start` and `image_token_end`)

### Attention looks uniform/random

This could indicate:
1. Model hasn't been trained properly (using random weights)
2. Wrong query token selected
3. Layer you're visualizing doesn't have meaningful attention patterns yet

## Examples

### Example 1: Comparing Visual Attention Across Layers

See how visual attention evolves through the network:

```bash
python example_attention_viz.py \
    --mode single \
    --video data/libero/90/videos/rollout_task0_ep0.mp4 \
    --frame-idx 15 \
    --prompt "open the bottom drawer of the cabinet" \
    --checkpoint checkpoints/pi0_model.ckpt \
    --layers 0 3 6 9 12 15 17 \
    --viz-type image \
    --output attention_all_layers.png
```

This creates a grid showing all 7 layers side-by-side.

### Example 2: Analyzing Which Words Matter

Compare text attention for different prompts to see which words the model focuses on:

```bash
# Spatial task
python example_attention_viz.py \
    --mode single \
    --video data/libero/90/videos/rollout_task0_ep0.mp4 \
    --frame-idx 10 \
    --prompt "pick up the black bowl on the left and place it on the right" \
    --checkpoint checkpoints/pi0_model.ckpt \
    --viz-type text \
    --output text_attn_spatial.png

# Object manipulation task
python example_attention_viz.py \
    --mode single \
    --video data/libero/90/videos/rollout_task0_ep0.mp4 \
    --frame-idx 10 \
    --prompt "open the top drawer of the cabinet" \
    --checkpoint checkpoints/pi0_model.ckpt \
    --viz-type text \
    --output text_attn_open.png
```

Compare the outputs to see if the model attends to spatial words ("left", "right") vs. action words ("open") vs. object words ("drawer", "cabinet").

### Example 3: Multimodal Analysis

Understand the balance between vision and language:

```bash
python example_attention_viz.py \
    --mode single \
    --video data/libero/90/videos/rollout_task0_ep0.mp4 \
    --frame-idx 10 \
    --prompt "pick up the black bowl and place it on the plate" \
    --checkpoint checkpoints/pi0_model.ckpt \
    --viz-type combined \
    --layers 0 5 11 17 \
    --output multimodal_analysis.png
```

This generates:
- Combined visualization showing visual and linguistic attention together
- Evolution plot showing how the vision/language balance changes across layers
- Statistics about attention distribution

### Example 4: Attention Over Time

Visualize how attention changes during task execution:

```bash
python example_attention_viz.py \
    --mode episode \
    --group 90 \
    --episode-idx 5 \
    --frames 0 5 10 15 20 25 30 35 40 \
    --checkpoint checkpoints/pi0_model.ckpt \
    --viz-type all \
    --output-dir attention_temporal/
```

Then create a video from the frames:
```bash
# Visual attention video
ffmpeg -framerate 5 -pattern_type glob -i 'attention_temporal/*_image.png' \
    -c:v libx264 -pix_fmt yuv420p visual_attention_over_time.mp4

# Text attention video
ffmpeg -framerate 5 -pattern_type glob -i 'attention_temporal/*_text.png' \
    -c:v libx264 -pix_fmt yuv420p linguistic_attention_over_time.mp4
```

### Example 5: Debugging Task Failures

Compare attention between success and failure cases:

```bash
# Success case
python example_attention_viz.py \
    --mode single \
    --video data/libero/90/correct_videos/rollout_put_the_black_bowl_on_top_of_the_cabinet_trial11_success.mp4 \
    --frame-idx 20 \
    --prompt "put the black bowl on top of the cabinet" \
    --checkpoint checkpoints/pi0_model.ckpt \
    --viz-type all \
    --output success_attention.png

# Failure case
python example_attention_viz.py \
    --mode single \
    --video data/libero/90/correct_videos/rollout_put_the_black_bowl_on_top_of_the_cabinet_trial5_failure.mp4 \
    --frame-idx 20 \
    --prompt "put the black bowl on top of the cabinet" \
    --checkpoint checkpoints/pi0_model.ckpt \
    --viz-type all \
    --output failure_attention.png
```

Compare the outputs to see:
- Does the failure case attend to the wrong objects?
- Is there a difference in linguistic attention?
- Does the visual/linguistic balance differ?

## Integration with Existing Analysis

You can combine attention visualization with other interpretability tools:

1. **SAE (Sparse Autoencoder) features**: Visualize which image regions activate specific SAE features
2. **Linear probes**: See if high-attention regions correlate with probe predictions
3. **Activation patterns**: Compare attention with activations from `ACTION_EXPERT_ACTS`

## Future Enhancements

Potential improvements to this visualization system:

- [ ] Support for attention between action tokens (action-to-action attention)
- [ ] Text token highlighting (which words does the model attend to?)
- [ ] Multi-head comparison (visualize all heads separately)
- [ ] Interactive visualization with sliders for different layers/heads
- [ ] Attention flow visualization (how attention propagates through layers)
- [ ] Comparison mode (side-by-side for success vs. failure cases)

## Citation

If you use this visualization tool in your research, please cite the Pi0 paper and this repository.

## Contact

For questions or issues with the attention visualization system, please open an issue on the repository.
