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
SEED="${SEED:-7}"
LAYERS="${LAYERS:-20 21 22}"         # LLM decoder layer indices to analyze

# Visual perturbation
# mode: none | rotate | translate | rotate_translate
VISUAL_PERTURB_MODE="${VISUAL_PERTURB_MODE:-none}"
ROTATION_DEGREES="${ROTATION_DEGREES:-0.0}"
TRANSLATE_X_FRAC="${TRANSLATE_X_FRAC:-0.0}"
TRANSLATE_Y_FRAC="${TRANSLATE_Y_FRAC:-0.0}"

# Policy perturbation
# mode: none | random_action | object_shift
POLICY_PERTURB_MODE="${POLICY_PERTURB_MODE:-none}"
RANDOM_ACTION_PROB="${RANDOM_ACTION_PROB:-0.0}"
RANDOM_ACTION_SCALE="${RANDOM_ACTION_SCALE:-1.0}"
OBJECT_SHIFT_X_STD="${OBJECT_SHIFT_X_STD:-0.0}"
OBJECT_SHIFT_Y_STD="${OBJECT_SHIFT_Y_STD:-0.0}"

if [ -z "$CHECKPOINT" ]; then
  echo "ERROR: CHECKPOINT is required. Set it via: CHECKPOINT=/path/to/ckpt bash $0"
  exit 1
fi

# ── Derived perturbation tag ────────────────────────────────────────────
if [[ "$VISUAL_PERTURB_MODE" != "none" ]]; then
    if [[ "$VISUAL_PERTURB_MODE" == "rotate" ]]; then
        VIS_TAG="vis_rotate_${ROTATION_DEGREES}deg"
    elif [[ "$VISUAL_PERTURB_MODE" == "translate" ]]; then
        VIS_TAG="vis_translate_x${TRANSLATE_X_FRAC}_y${TRANSLATE_Y_FRAC}"
    else
        VIS_TAG="vis_rotate_${ROTATION_DEGREES}deg_translate_x${TRANSLATE_X_FRAC}_y${TRANSLATE_Y_FRAC}"
    fi
else
    VIS_TAG=""
fi

if [[ "$POLICY_PERTURB_MODE" != "none" ]]; then
    if [[ "$POLICY_PERTURB_MODE" == "random_action" ]]; then
        POL_TAG="pol_random_action_p${RANDOM_ACTION_PROB}_s${RANDOM_ACTION_SCALE}"
    elif [[ "$POLICY_PERTURB_MODE" == "object_shift" ]]; then
        POL_TAG="pol_object_shift_x${OBJECT_SHIFT_X_STD}_y${OBJECT_SHIFT_Y_STD}"
    else
        POL_TAG="pol_${POLICY_PERTURB_MODE}"
    fi
else
    POL_TAG=""
fi

if [[ -n "${VIS_TAG}" && -n "${POL_TAG}" ]]; then
    PERTURB_TAG="${VIS_TAG}__${POL_TAG}"
elif [[ -n "${VIS_TAG}" ]]; then
    PERTURB_TAG="${VIS_TAG}"
elif [[ -n "${POL_TAG}" ]]; then
    PERTURB_TAG="${POL_TAG}"
else
    PERTURB_TAG="none"
fi

OUTPUT_DIR="${OUTPUT_DIR:-results/attention_ratio_openvla/${TASK_SUITE}_seed${SEED}_perturb_${PERTURB_TAG}}"

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
echo "Seed:          $SEED"
echo "LLM Layers:    $LAYERS"
echo "Visual perturbation: ${VISUAL_PERTURB_MODE} (rotation=${ROTATION_DEGREES} tx=${TRANSLATE_X_FRAC} ty=${TRANSLATE_Y_FRAC})"
echo "Policy perturbation: ${POLICY_PERTURB_MODE} (prob=${RANDOM_ACTION_PROB} scale=${RANDOM_ACTION_SCALE} ox=${OBJECT_SHIFT_X_STD} oy=${OBJECT_SHIFT_Y_STD})"
echo "Perturbation tag:    ${PERTURB_TAG}"
echo "Output:        $OUTPUT_DIR"
echo "========================================="
echo

mkdir -p logs

# Build command
CMD="python $PROJECT_ROOT/openvla/experiments/robot/libero/evaluate_attention_ratio_openvla.py \
  --checkpoint $CHECKPOINT \
  --task-suite $TASK_SUITE \
  --num-episodes $NUM_EPISODES \
  --seed $SEED \
  --layers $LAYERS \
  --visual-perturb-mode ${VISUAL_PERTURB_MODE} \
  --rotation-degrees ${ROTATION_DEGREES} \
  --translate-x-frac ${TRANSLATE_X_FRAC} \
  --translate-y-frac ${TRANSLATE_Y_FRAC} \
  --policy-perturb-mode ${POLICY_PERTURB_MODE} \
  --random-action-prob ${RANDOM_ACTION_PROB} \
  --random-action-scale ${RANDOM_ACTION_SCALE} \
  --object-shift-x-std ${OBJECT_SHIFT_X_STD} \
  --object-shift-y-std ${OBJECT_SHIFT_Y_STD} \
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
