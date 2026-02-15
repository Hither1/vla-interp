#!/bin/bash
#SBATCH -J attn_ratio          # Job name
#SBATCH -p gpu                 # Partition
#SBATCH -t 12:00:00            # Time limit
#SBATCH --gres=gpu:1           # GPU request
#SBATCH --mem=32G              # Memory
#SBATCH -o logs/attn_ratio_%j.log
#SBATCH -e logs/attn_ratio_%j.log

# Visual/Linguistic Attention Ratio Evaluation Script
# Systematically evaluates the ratio of visual to linguistic attention
# across LIBERO task suites.

set -e


source ~/.bashrc
conda deactivate
conda activate vla

module load gcc/12.2.0-fasrc01
module load cuda/12.4.1-fasrc01

export HF_HOME=/n/netscratch/sham_lab/Lab/chloe00/huggingface
export TRANSFORMERS_CACHE=/n/netscratch/sham_lab/Lab/chloe00/huggingface

export PYTHONPATH="${PYTHONPATH:-}:/n/netscratch/sham_lab/Lab/chloe00/libero"
export LIBERO_CONFIG_PATH=/n/netscratch/sham_lab/Lab/chloe00/libero


# Configuration
CHECKPOINT="${CHECKPOINT:-$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero}"
TASK_SUITE="${TASK_SUITE:-libero_10}"
TASK_ID="${TASK_ID:-}"  # Empty means all tasks
NUM_EPISODES="${NUM_EPISODES:-5}"
LAYERS="${LAYERS:-0 8 17 25 26 27}"
OUTPUT_DIR="${OUTPUT_DIR:-results/attention_ratio_${TASK_SUITE}}"
SAVE_VIZ="${SAVE_VIZ:-}"

# Script directory (handle SLURM execution)
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
fi

echo "========================================="
echo "Visual/Linguistic Attention Ratio Eval"
echo "========================================="
echo "Checkpoint:   $CHECKPOINT"
echo "Task Suite:   $TASK_SUITE"
echo "Task ID:      ${TASK_ID:-all}"
echo "Episodes:     $NUM_EPISODES"
echo "Layers:       $LAYERS"
echo "Output:       $OUTPUT_DIR"
echo "Save Viz:     ${SAVE_VIZ:-no}"
echo "========================================="
echo

# Build command
CMD="python $PROJECT_ROOT/analysis/evaluate_attention_ratio.py \
  --checkpoint $CHECKPOINT \
  --task-suite $TASK_SUITE \
  --num-episodes $NUM_EPISODES \
  --layers $LAYERS \
  --output-dir $OUTPUT_DIR"

# Add task ID if specified
if [ -n "$TASK_ID" ]; then
  CMD="$CMD --task-id $TASK_ID"
fi

# Add save-viz flag if requested
if [ -n "$SAVE_VIZ" ]; then
  CMD="$CMD --save-viz"
fi

echo "Running: $CMD"
echo

# Run evaluation
$CMD

echo
echo "========================================="
echo "Evaluation Complete!"
echo "========================================="
echo "Results saved to: $OUTPUT_DIR"
echo

# Parse results if they exist
RESULTS_FILE="$OUTPUT_DIR/attention_ratio_results_${TASK_SUITE}.json"
if [ -f "$RESULTS_FILE" ]; then
  echo "Generating analysis..."
  python "$PROJECT_ROOT/analysis/parse_attention_ratio_results.py" \
    --results "$RESULTS_FILE" \
    --output "$OUTPUT_DIR/summary.txt" \
    --output-dir "$OUTPUT_DIR/analysis" \
    --plot-all

  echo
  echo "Analysis saved to: $OUTPUT_DIR/analysis/"
  echo "Summary saved to: $OUTPUT_DIR/summary.txt"
  echo

  # Display summary
  if [ -f "$OUTPUT_DIR/summary.txt" ]; then
    echo "========================================="
    echo "SUMMARY"
    echo "========================================="
    cat "$OUTPUT_DIR/summary.txt"
    echo "========================================="
  fi
fi

echo
echo "Done!"
