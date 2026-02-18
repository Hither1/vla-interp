#!/bin/bash
#SBATCH -J attn_ratio_cosmos   # Job name
#SBATCH -p gpu                 # Partition
#SBATCH -t 12:00:00            # Time limit
#SBATCH --gres=gpu:1           # GPU request
#SBATCH --mem=64G              # Memory (Cosmos needs more)
#SBATCH -o logs/cosmos_attn_ratio_%j.log
#SBATCH -e logs/cosmos_attn_ratio_%j.log

# Visual/Linguistic Attention Ratio Evaluation Script for Cosmos Policy
# Systematically evaluates the ratio of visual to linguistic attention
# across LIBERO task suites using Cosmos DiT architecture.

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
CKPT_PATH="${CKPT_PATH:-nvidia/Cosmos-Policy-LIBERO-Predict2-2B}"
CONFIG_NAME="${CONFIG_NAME:-cosmos_predict2_2b_480p_libero__inference_only}"
TASK_SUITE="${TASK_SUITE:-libero_10}"
TASK_ID="${TASK_ID:-}"  # Empty means all tasks
NUM_EPISODES="${NUM_EPISODES:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-results/attention_ratio_cosmos_${TASK_SUITE}}"

# Frame selection for Cosmos (important!)
QUERY_FRAME="${QUERY_FRAME:-action}"     # Which frame provides queries
VISUAL_FRAME="${VISUAL_FRAME:-curr_last}" # Which frame has visual tokens
TEXT_FRAME="${TEXT_FRAME:-frame0}"       # Which frame has text tokens

# Script directory (handle SLURM execution)
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
fi

echo "========================================="
echo "Cosmos Visual/Linguistic Attention Ratio"
echo "========================================="
echo "Checkpoint:    $CKPT_PATH"
echo "Config:        $CONFIG_NAME"
echo "Task Suite:    $TASK_SUITE"
echo "Task ID:       ${TASK_ID:-all}"
echo "Episodes:      $NUM_EPISODES"
echo "Query Frame:   $QUERY_FRAME"
echo "Visual Frame:  $VISUAL_FRAME"
echo "Text Frame:    $TEXT_FRAME"
echo "Output:        $OUTPUT_DIR"
echo "========================================="
echo

# Build command
CMD="python $PROJECT_ROOT/analysis/attention/evaluate_attention_ratio_cosmos.py \
  --ckpt-path $CKPT_PATH \
  --config-name $CONFIG_NAME \
  --task-suite $TASK_SUITE \
  --num-episodes $NUM_EPISODES \
  --output-dir $OUTPUT_DIR"

# Add task ID if specified
if [ -n "$TASK_ID" ]; then
  CMD="$CMD --task-id $TASK_ID"
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
  python "$PROJECT_ROOT/analysis/attention/parse_attention_ratio_results.py" \
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
