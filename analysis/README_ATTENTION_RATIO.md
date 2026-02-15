# Visual/Linguistic Attention Ratio Analysis

This suite of tools systematically analyzes the ratio of visual to linguistic attention in VLA models across LIBERO task suites. It computes how much attention action tokens place on visual features vs. text instructions at every replan step during task execution.

## Overview

The attention ratio analysis helps answer:
- **Does the model rely more on visual features or linguistic instructions?**
- **How does the attention distribution change across different tasks?**
- **Are there specific layers that show stronger visual or linguistic bias?**

### Key Metrics

- **Visual/Linguistic Ratio**: `visual_mass / linguistic_mass`
  - Ratio > 1.0: Visual-dominant (relies more on images)
  - Ratio < 1.0: Linguistic-dominant (relies more on text)
  - Ratio ≈ 1.0: Balanced attention

- **Attention Masses**: Raw attention weights summed over token types
  - Visual mass: Sum of attention to image tokens
  - Linguistic mass: Sum of attention to text tokens
  - Action mass: Sum of attention to action tokens

- **Attention Fractions**: Normalized proportions (sum to 1.0)

## Scripts

### 1. Pi0/Pi0.5 Model (Gemma-based)

**Main Script**: `evaluate_attention_ratio.py`

```bash
python analysis/evaluate_attention_ratio.py \
  --checkpoint ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  --task-suite libero_10 \
  --num-episodes 5 \
  --layers 0 8 17 25 26 27 \
  --output-dir results/attention_ratio
```

**SLURM Script**: `scripts/run_attention_ratio.sh`

```bash
# Run with defaults
sbatch scripts/run_attention_ratio.sh

# Customize via environment variables
TASK_SUITE=libero_spatial NUM_EPISODES=10 LAYERS="17 25 26 27" \
  sbatch scripts/run_attention_ratio.sh

# Run specific task with visualization
TASK_ID=5 SAVE_VIZ=1 sbatch scripts/run_attention_ratio.sh
```

### 2. Cosmos Policy Model (DiT-based)

**Main Script**: `evaluate_attention_ratio_cosmos.py`

```bash
python analysis/evaluate_attention_ratio_cosmos.py \
  --ckpt-path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
  --task-suite libero_10 \
  --num-episodes 5 \
  --layers 25 26 27 \
  --query-frame action \
  --visual-frame curr_last \
  --text-frame frame0 \
  --output-dir results/attention_ratio_cosmos
```

**SLURM Script**: `scripts/run_attention_ratio_cosmos.sh`

```bash
# Run with defaults
sbatch scripts/run_attention_ratio_cosmos.sh

# Customize frame selection
QUERY_FRAME=action VISUAL_FRAME=curr_last TEXT_FRAME=frame0 \
  sbatch scripts/run_attention_ratio_cosmos.sh
```

#### Cosmos Frame Selection

Cosmos uses multi-frame token sequences. You can control which frames to analyze:

- `--query-frame`: Which frame's tokens to use as queries (default: `action`)
  - `action`: Action prediction frame
  - `curr_last`: Last observation frame
  - `frame0`, `frame1`, etc.: Specific frame indices

- `--visual-frame`: Which frame contains visual tokens to measure (default: `curr_last`)
- `--text-frame`: Which frame contains text embeddings (default: `frame0`)

### 3. Results Analysis

**Parser Script**: `parse_attention_ratio_results.py`

```bash
# Print summary statistics
python analysis/parse_attention_ratio_results.py \
  --results results/attention_ratio/attention_ratio_results_libero_10.json

# Generate all plots
python analysis/parse_attention_ratio_results.py \
  --results results/attention_ratio/attention_ratio_results_libero_10.json \
  --output summary.txt \
  --output-dir results/attention_ratio_analysis \
  --plot-all

# Individual plots
python analysis/parse_attention_ratio_results.py \
  --results results.json \
  --plot-distribution    # Histogram + boxplot
  --plot-per-task       # Comparison across tasks
  --plot-fractions      # Visual vs linguistic fractions
```

## Output Structure

### Results JSON

```json
{
  "task_id": 0,
  "episode_idx": 0,
  "task_description": "pick up the bowl",
  "success": true,
  "num_steps": 150,
  "summary": {
    "layer_25": {
      "visual_linguistic_ratio": {
        "mean": 2.345,
        "std": 0.456,
        "median": 2.301,
        "min": 1.892,
        "max": 3.102
      },
      "visual_fraction": {"mean": 0.65, "std": 0.08},
      "linguistic_fraction": {"mean": 0.28, "std": 0.07}
    },
    "layers_avg": { ... }
  },
  "per_step_ratios": {
    "layer_25": [
      {
        "step": 10,
        "visual_linguistic_ratio": 2.34,
        "visual_mass": 0.234,
        "linguistic_mass": 0.100,
        "visual_fraction": 0.65,
        "linguistic_fraction": 0.28
      },
      ...
    ]
  }
}
```

### Visualizations (with `--save-viz`)

Per episode:
- `{prefix}_layer{N}_ratio_evolution.png`: Ratio evolution over time
  - Plot 1: Visual/linguistic ratio over steps
  - Plot 2: Attention masses (visual, linguistic, action)
  - Plot 3: Attention fractions (stacked area)
  - Plot 4: Summary statistics

Aggregated (from parser):
- `ratio_distribution.png`: Histogram and boxplot of ratios
- `ratio_per_task.png`: Mean ratio per task with error bars
- `attention_fractions.png`: Average visual vs linguistic fractions

## Command-Line Arguments

### Common Arguments

```bash
# Model
--checkpoint PATH              # Model checkpoint path
--paligemma-variant STR       # Gemma variant (default: gemma_2b)
--action-expert-variant STR   # Action expert variant (default: gemma_300m)

# Evaluation
--task-suite {libero_spatial,libero_object,libero_goal,libero_10,libero_90}
--task-id INT                 # Specific task (default: all tasks)
--num-episodes INT            # Episodes per task (default: 5)
--seed INT                    # Random seed (default: 7)

# Attention
--layers INT [INT ...]        # Layers to analyze (default: [25, 26, 27])
--num-image-tokens INT        # Number of image tokens (default: 256)

# Output
--output-dir PATH             # Output directory
--save-viz                    # Save per-step visualizations
```

### Cosmos-Specific Arguments

```bash
--ckpt-path PATH              # Cosmos checkpoint path
--config-name STR             # Cosmos config name
--query-frame STR             # Query frame (default: action)
--visual-frame STR            # Visual frame (default: curr_last)
--text-frame STR              # Text frame (default: frame0)
--prefer-S INT                # Prefer calls with specific S
--prefer-call {last,first}    # Which call to use per layer
```

## Example Workflows

### 1. Quick Single-Task Analysis

```bash
# Evaluate one task with visualization
python analysis/evaluate_attention_ratio.py \
  --checkpoint ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  --task-suite libero_10 \
  --task-id 0 \
  --num-episodes 3 \
  --save-viz \
  --output-dir results/attention_ratio_test
```

### 2. Full Suite Evaluation

```bash
# All tasks, multiple episodes
sbatch scripts/run_attention_ratio.sh

# Wait for completion, then analyze
python analysis/parse_attention_ratio_results.py \
  --results results/attention_ratio_libero_10/attention_ratio_results_libero_10.json \
  --plot-all \
  --output-dir results/attention_ratio_analysis
```

### 3. Cosmos Multi-Suite Comparison

```bash
# Run multiple suites
for suite in libero_spatial libero_object libero_goal; do
  TASK_SUITE=$suite OUTPUT_DIR=results/attention_ratio_cosmos_${suite} \
    sbatch scripts/run_attention_ratio_cosmos.sh
done

# Compare results
python analysis/parse_attention_ratio_results.py \
  --results results/attention_ratio_cosmos_*/attention_ratio_results_*.json \
  --plot-comparison \
  --output-dir results/cosmos_comparison
```

### 4. Layer-by-Layer Analysis

```bash
# Analyze different layer groups
LAYERS="0 1 2 3 4 5" sbatch scripts/run_attention_ratio.sh  # Early layers
LAYERS="13 14 15 16 17" sbatch scripts/run_attention_ratio.sh  # Middle layers
LAYERS="25 26 27" sbatch scripts/run_attention_ratio.sh  # Late layers
```

## Interpretation Guide

### Visual-Dominant Model (Ratio > 1.5)
- Relies heavily on visual features
- May struggle with tasks requiring precise linguistic understanding
- Good for visually grounded manipulation

### Linguistic-Dominant Model (Ratio < 0.67)
- Relies heavily on text instructions
- May struggle with tasks requiring fine visual perception
- Good for instruction following

### Balanced Model (0.67 ≤ Ratio ≤ 1.5)
- Uses both modalities effectively
- Likely more robust across diverse tasks
- Optimal for general VLA tasks

### Per-Layer Patterns

- **Early layers**: Often show balanced attention (feature extraction)
- **Middle layers**: May show modality-specific patterns
- **Late layers**: Often visual-dominant (action prediction benefits from spatial info)

## Comparison with IoU Analysis

| Aspect | IoU Analysis | Ratio Analysis |
|--------|-------------|----------------|
| **What** | Attention-object overlap | Visual vs linguistic attention |
| **Measures** | Spatial grounding quality | Modality preference |
| **Output** | IoU scores per object | Ratio + attention masses |
| **Use Case** | Object-centric grounding | Modality balance understanding |
| **Requires** | Segmentation masks | No extra data needed |

Both analyses are complementary:
- **IoU**: "Is the model looking at the right objects?"
- **Ratio**: "Is the model using visual or linguistic information more?"

## Troubleshooting

### No Attention Weights Recorded
- Check that attention hooks are properly installed
- Verify model architecture has expected attention modules
- For Cosmos: ensure DiT blocks are found

### Inf/NaN Ratios
- Happens when linguistic_mass ≈ 0
- Filtered automatically in visualizations
- Check `total_mass` to verify attention is being captured

### Mismatched Token Counts
- For Pi0: Adjust `--num-image-tokens` (default 256)
- For Cosmos: Verify frame layout matches model config
- Check `num_visual_tokens` and `num_text_tokens` in results

## Citation

If you use this analysis in your research, please cite:

```bibtex
@misc{vla-attention-ratio,
  title={Visual-Linguistic Attention Ratio Analysis for Vision-Language-Action Models},
  author={Your Name},
  year={2026},
  note={Analysis tools for VLA attention distribution}
}
```

## See Also

- `evaluate_attention_iou.py` - Attention-object IoU analysis
- `visualize_attention.py` - Attention heatmap visualization
- `example_attention_viz.py` - Attention visualization utilities
