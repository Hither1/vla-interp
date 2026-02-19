#!/bin/bash
#SBATCH -J attn_iou_openvla     # Job name
#SBATCH -p gpu                  # Partition
#SBATCH -t 12:00:00             # Time limit
#SBATCH --gres=gpu:1            # GPU request
#SBATCH --mem=64G               # Memory
#SBATCH -o logs/openvla_attn_iou_%j.log
#SBATCH -e logs/openvla_attn_iou_%j.log

# Attention-Segmentation IoU Evaluation Script for OpenVLA
# Evaluates how well attention heatmaps overlap with ground-truth
# object segmentation masks across LIBERO task suites.
#
# Attention heatmaps are built from the last-text-token's attention to image
# patch positions (256 patches → 16×16 spatial grid → upsampled to 256×256).
# IoU is computed against SegmentationRenderEnv instance masks.
#
# Usage examples:
#   sbatch run_attention_iou_openvla.sh
#   TASK_SUITE=libero_object TASK_ID=0 NUM_EPISODES=5 bash run_attention_iou_openvla.sh

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
TASK_ID="${TASK_ID:-}"
NUM_EPISODES="${NUM_EPISODES:-5}"
LAYERS="${LAYERS:-20 21 22}"
OUTPUT_DIR="${OUTPUT_DIR:-results/iou_openvla_${TASK_SUITE}}"
SAVE_VIZ="${SAVE_VIZ:-}"             # Set to "--save-viz" to save visualizations

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
echo "OpenVLA Attention-Segmentation IoU"
echo "========================================="
echo "Checkpoint:    $CHECKPOINT"
echo "Task Suite:    $TASK_SUITE"
echo "Task ID:       ${TASK_ID:-all}"
echo "Episodes:      $NUM_EPISODES"
echo "LLM Layers:    $LAYERS"
echo "Save Viz:      ${SAVE_VIZ:-no}"
echo "Output:        $OUTPUT_DIR"
echo "========================================="
echo

mkdir -p logs

CMD="python $PROJECT_ROOT/openvla/experiments/robot/libero/evaluate_attention_iou_openvla.py \
  --checkpoint $CHECKPOINT \
  --task-suite $TASK_SUITE \
  --num-episodes $NUM_EPISODES \
  --layers $LAYERS \
  --output-dir $OUTPUT_DIR \
  ${SAVE_VIZ}"

if [ -n "$TASK_ID" ]; then
  CMD="$CMD --task-id $TASK_ID"
fi

echo "Running: $CMD"
echo

$CMD

echo
echo "========================================="
echo "Evaluation Complete!"
echo "========================================="
echo "Results saved to: $OUTPUT_DIR"
echo

echo "Done!"
