#!/bin/bash
#SBATCH -J attn_ratio_openvla   # Job name
#SBATCH -p gpu                  # Partition
#SBATCH -t 12:00:00             # Time limit
#SBATCH --gres=gpu:1            # GPU request
#SBATCH --mem=64G               # Memory
#SBATCH -o logs/openvla_attn_ratio_%j.log
#SBATCH -e logs/openvla_attn_ratio_%j.log

# Visual/Linguistic Attention Ratio Evaluation Script for OpenVLA
# Systematically evaluates the ratio of visual to linguistic attention
# across LIBERO task suites using OpenVLA (Prismatic VLM + LLaMA-2 decoder).
#
# Attention is captured during the LLM prefill phase of each predict_action call.
# The last text-token's attention to image patches vs. text tokens gives the ratio.
#
# Usage examples:
#   sbatch run_attention_ratio_openvla.sh
#   TASK_SUITE=libero_object TASK_ID=0 NUM_EPISODES=10 bash run_attention_ratio_openvla.sh

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
CHECKPOINT="${CHECKPOINT:-}"          # Required: path to OpenVLA checkpoint
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
TASK_ID="${TASK_ID:-}"               # Empty means all tasks
NUM_EPISODES="${NUM_EPISODES:-5}"
LAYERS="${LAYERS:-20 21 22}"         # LLM decoder layer indices to analyze
OUTPUT_DIR="${OUTPUT_DIR:-results/attention_ratio_openvla_${TASK_SUITE}}"

if [ -z "$CHECKPOINT" ]; then
  echo "ERROR: CHECKPOINT is required. Set it via: CHECKPOINT=/path/to/ckpt bash $0"
  exit 1
fi

# Script directory (handle SLURM execution)
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
fi

echo "========================================="
echo "OpenVLA Visual/Linguistic Attention Ratio"
echo "========================================="
echo "Checkpoint:    $CHECKPOINT"
echo "Task Suite:    $TASK_SUITE"
echo "Task ID:       ${TASK_ID:-all}"
echo "Episodes:      $NUM_EPISODES"
echo "LLM Layers:    $LAYERS"
echo "Output:        $OUTPUT_DIR"
echo "========================================="
echo

mkdir -p logs

# Build command
CMD="python $PROJECT_ROOT/openvla/experiments/robot/libero/evaluate_attention_ratio_openvla.py \
  --checkpoint $CHECKPOINT \
  --task-suite $TASK_SUITE \
  --num-episodes $NUM_EPISODES \
  --layers $LAYERS \
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

echo "Done!"
