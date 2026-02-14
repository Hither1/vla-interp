# Action Entropy Analysis for LIBERO Tasks

This directory contains scripts to calculate and analyze action entropy across LIBERO task suites, separated by success/failure status.

## Overview

The analysis calculates **action entropy** using Kernel Density Estimation (KDE) to measure how random or deterministic the robot's actions are during task execution. By comparing entropy between successful and failed trials, we can understand whether the policy is confident, uncertain, or systematically wrong.

## Scripts

### 1. `calculate_entropy_by_suite.py`

Main script that processes action JSON files and calculates entropy statistics.

**Usage:**
```bash
python calculate_entropy_by_suite.py \
    --action_data_dir third_party/cosmos-policy/cosmos_policy/experiments/robot/libero/logs/action_data \
    --output entropy_results.json
```

**Arguments:**
- `--action_data_dir`: Directory containing `actions_*.json` files (default: path shown above)
- `--output`: Output JSON file for aggregated results (default: `entropy_results.json`)
- `--use_simple_entropy`: Use simple binning instead of KDE (optional)

**Outputs:**
- `entropy_results.json`: Aggregated statistics by suite and success/failure
- `entropy_results_raw.json`: Raw per-trial entropy data

**What it does:**
1. Scans all `actions_*.json` files in the specified directory
2. Extracts action sequences from each trial
3. Calculates action entropy using KDE
4. Groups results by:
   - LIBERO suite (10, 90, spatial, object, goal, long)
   - Task description
   - Success/failure status
5. Computes average entropy for each suite × status combination

### 2. `summarize_entropy_results.py`

Generates a human-readable summary report from the entropy results.

**Usage:**
```bash
python summarize_entropy_results.py \
    --results entropy_results.json \
    --output entropy_analysis_report.md
```

**Arguments:**
- `--results`: Path to entropy results JSON (default: `entropy_results.json`)
- `--output`: Output markdown report path (default: `entropy_analysis_report.md`)

**Outputs:**
- `entropy_analysis_report.md`: Comprehensive markdown report with:
  - Suite-level statistics
  - Task-level breakdowns
  - Interpretations and recommendations
  - Comparison tables

### 3. `visualize_entropy_results.py`

Creates visualizations of entropy results (requires matplotlib).

**Usage:**
```bash
python visualize_entropy_results.py \
    --results entropy_results.json \
    --results_raw entropy_results_raw.json \
    --output_dir entropy_plots
```

**Arguments:**
- `--results`: Path to aggregated results JSON
- `--results_raw`: Path to raw results JSON
- `--output_dir`: Directory for output plots (default: `entropy_plots`)

**Outputs (if matplotlib available):**
- `entropy_by_suite_comparison.png`: Bar chart comparing success vs failure
- `entropy_difference.png`: Difference visualization
- `task_level_*.png`: Per-task comparisons
- `entropy_distribution_violin.png`: Distribution plots

## Understanding the Results

### Entropy Values

**Action Entropy** is calculated as: `-E[log p(x)]` where `p(x)` is estimated using KDE.

- **More negative** (e.g., -10.0) = **Lower entropy** = More deterministic/consistent
- **Less negative** (e.g., -6.0) = **Higher entropy** = More random/exploratory

### Interpreting Success vs Failure Differences

1. **Failure has HIGHER entropy** (positive difference):
   - Policy is uncertain during failures
   - Actions are more random/exploratory
   - Suggests the policy knows it's struggling

2. **Failure has LOWER entropy** (negative difference):
   - Policy is confident but wrong
   - Actions are deterministic but incorrect
   - Suggests learned incorrect behavior patterns

3. **Similar entropy**:
   - Randomness is not the distinguishing factor
   - Success/failure may depend on other factors

## Example Results

From our analysis of the COSMOS policy on LIBERO tasks:

### LIBERO-10 (Easy Tasks)
- **Success rate**: 97.4%
- **Success entropy**: -9.03 ± 1.87
- **Failure entropy**: -8.60 ± 1.23
- **Difference**: +0.43 (failures are more random)
- **Interpretation**: Policy explores more during rare failures

### LIBERO-90 (Challenging Tasks)
- **Success rate**: 18.6%
- **Failure entropy**: -9.66 ± 0.98
- **Difference**: -0.66 (failures are more deterministic)
- **Interpretation**: Policy has learned incorrect patterns

## File Format

### Input: `actions_*.json`

Expected format for action JSON files:
```json
{
  "task_id": 5,
  "trial_id": 3,
  "seed": 195,
  "task_description": "put the bowl on the plate",
  "success": false,
  "actions": [
    {
      "t": 10,
      "action": [0.407, 0.356, -0.007, ...],
      "reward": 0.0,
      "done": false
    },
    ...
  ]
}
```

### Output: `entropy_results.json`

Structure:
```json
{
  "LIBERO-90": {
    "num_tasks": 14,
    "success": {
      "num_trials": 52,
      "entropy_mean": -8.996,
      "entropy_std": 0.953,
      "entropy_min": -10.794,
      "entropy_max": -7.301
    },
    "failure": {
      "num_trials": 228,
      "entropy_mean": -9.661,
      "entropy_std": 0.979,
      "entropy_min": -13.167,
      "entropy_max": -7.312
    },
    "task_details": {
      "put the red mug on the right plate": {
        "num_success_trials": 0,
        "num_failure_trials": 20,
        "success_entropy_mean": null,
        "failure_entropy_mean": -10.093
      },
      ...
    }
  }
}
```

## Quick Start

Run the complete analysis pipeline:

```bash
# 1. Calculate entropy from action files
python calculate_entropy_by_suite.py

# 2. Generate summary report
python summarize_entropy_results.py

# 3. (Optional) Create visualizations if matplotlib is available
python visualize_entropy_results.py
```

Results will be saved in:
- `entropy_results.json` - Aggregated statistics
- `entropy_results_raw.json` - Raw per-trial data
- `entropy_analysis_report.md` - Human-readable report

## Dependencies

**Required:**
- Python 3.7+
- numpy
- scipy (for KDE calculation)

**Optional:**
- matplotlib (for visualizations)
- seaborn (for enhanced plots)
- pandas (for data manipulation)

Install with:
```bash
pip install numpy scipy matplotlib seaborn pandas
```

## Suite Mapping

The scripts automatically map suite identifiers to names:

| ID | Suite Name |
|----|------------|
| 10 | LIBERO-10 (generic/mixed) |
| 90 | LIBERO-90 |
| spatial | LIBERO-Spatial |
| object | LIBERO-Object |
| goal | LIBERO-Goal |
| long | LIBERO-Long |

Files with "libero_XX" in the name use that suite ID. Other files are grouped based on task_id patterns.

## Troubleshooting

**Issue: "Could not extract suite_id"**
- The script tries to infer suite from filename or task_id
- Files without "libero_" prefix are grouped by task_id:
  - task_id ≥ 10 → LIBERO-90
  - task_id < 10 → LIBERO-10

**Issue: "KDE calculation failed"**
- Occurs when there are too few action samples
- Try using `--use_simple_entropy` flag for binning-based entropy

**Issue: "No module named matplotlib"**
- Visualization script requires matplotlib
- Skip visualization or install: `pip install matplotlib seaborn`

## Citation

If you use this analysis in your research, please cite the relevant papers:
- LIBERO benchmark
- COSMOS policy (if applicable)
- Your specific work

## License

[Add your license information here]
