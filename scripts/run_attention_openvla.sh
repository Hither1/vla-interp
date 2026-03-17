#!/bin/bash
#SBATCH -J attn_openvla             # Job name
#SBATCH -p gpu                      # Partition
#SBATCH -t 12:00:00                 # Time limit
#SBATCH --gres=gpu:1                # GPU request
#SBATCH --mem=64G                   # Memory
#SBATCH -o logs/openvla_attn_%j.log
#SBATCH -e logs/openvla_attn_%j.log

# Combined Attention Analysis Script for OpenVLA
# Runs both visual/linguistic ratio and attention-segmentation IoU analyses
# in a single forward pass per timestep (evaluate_attention_openvla.py).
#
# Usage examples:
#   sbatch run_attention_openvla.sh
#   TASK_SUITE=libero_object TASK_ID=0 NUM_EPISODES=5 bash run_attention_openvla.sh

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

# Expand TASK_SUITE into array of suites to run
if [ "${TASK_SUITE}" = "all" ]; then
    SUITES=(libero_spatial libero_object libero_goal libero_10)
elif [ "${TASK_SUITE}" = "90_all" ]; then
    SUITES=(libero_90_obj libero_90_spa libero_90_act libero_90_com)
else
    SUITES=("${TASK_SUITE}")
fi
NUM_EPISODES="${NUM_EPISODES:-5}"
SEED="${SEED:-7}"
LAYERS="${LAYERS:-20 21 22}"         # LLM decoder layer indices to analyze
SAVE_VIZ="${SAVE_VIZ:-0}"            # Set to 1 to save IoU visualizations

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
        VIS_TAG_VALUE="rotate_${ROTATION_DEGREES}deg"
    elif [[ "$VISUAL_PERTURB_MODE" == "translate" ]]; then
        VIS_TAG_VALUE="translate_x${TRANSLATE_X_FRAC}_y${TRANSLATE_Y_FRAC}"
    else
        VIS_TAG_VALUE="rotate_${ROTATION_DEGREES}deg_translate_x${TRANSLATE_X_FRAC}_y${TRANSLATE_Y_FRAC}"
    fi
else
    VIS_TAG_VALUE="none"
fi

if [[ "$POLICY_PERTURB_MODE" != "none" ]]; then
    if [[ "$POLICY_PERTURB_MODE" == "random_action" ]]; then
        POL_TAG_VALUE="random_action_p${RANDOM_ACTION_PROB}_s${RANDOM_ACTION_SCALE}"
    elif [[ "$POLICY_PERTURB_MODE" == "object_shift" ]]; then
        POL_TAG_VALUE="object_shift_x${OBJECT_SHIFT_X_STD}_y${OBJECT_SHIFT_Y_STD}"
    else
        POL_TAG_VALUE="${POLICY_PERTURB_MODE}"
    fi
else
    POL_TAG_VALUE="none"
fi

# Script directory (handle SLURM execution)
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
fi
WORKDIR="$PROJECT_ROOT"

mkdir -p logs

VIZ_FLAG=""
if [[ "$SAVE_VIZ" == "1" ]]; then
    VIZ_FLAG="--save-viz"
fi

# Shared args used by both Python scripts
SHARED_ARGS=(
    --checkpoint "$CHECKPOINT"
    --num-episodes "$NUM_EPISODES"
    --seed "$SEED"
    --layers $LAYERS
    --visual-perturb-mode "${VISUAL_PERTURB_MODE}"
    --rotation-degrees "${ROTATION_DEGREES}"
    --translate-x-frac "${TRANSLATE_X_FRAC}"
    --translate-y-frac "${TRANSLATE_Y_FRAC}"
    --policy-perturb-mode "${POLICY_PERTURB_MODE}"
    --random-action-prob "${RANDOM_ACTION_PROB}"
    --random-action-scale "${RANDOM_ACTION_SCALE}"
    --object-shift-x-std "${OBJECT_SHIFT_X_STD}"
    --object-shift-y-std "${OBJECT_SHIFT_Y_STD}"
)
if [ -n "$TASK_ID" ]; then
    SHARED_ARGS+=(--task-id "$TASK_ID")
fi

for SUITE in "${SUITES[@]}"; do
    if [[ "$VISUAL_PERTURB_MODE" != "none" || "$POLICY_PERTURB_MODE" != "none" ]]; then
        PERTURB_TAG="vis_${VIS_TAG_VALUE}__pol_${POL_TAG_VALUE}"
    else
        PERTURB_TAG="none"
    fi
    OUTPUT_DIR="${WORKDIR}/results/attention/combined/openvla/perturb/${PERTURB_TAG}/${SUITE}_seed${SEED}"

    echo "========================================="
    echo "OpenVLA Attention Analysis (ratio + IoU)"
    echo "========================================="
    echo "Checkpoint:          $CHECKPOINT"
    echo "Task Suite:          $SUITE"
    echo "Task ID:             ${TASK_ID:-all}"
    echo "Episodes:            $NUM_EPISODES"
    echo "Seed:                $SEED"
    echo "LLM Layers:          $LAYERS"
    echo "Save Viz:            $SAVE_VIZ"
    echo "Visual perturbation: ${VISUAL_PERTURB_MODE} (rotation=${ROTATION_DEGREES} tx=${TRANSLATE_X_FRAC} ty=${TRANSLATE_Y_FRAC})"
    echo "Policy perturbation: ${POLICY_PERTURB_MODE} (prob=${RANDOM_ACTION_PROB} scale=${RANDOM_ACTION_SCALE} ox=${OBJECT_SHIFT_X_STD} oy=${OBJECT_SHIFT_Y_STD})"
    echo "Perturbation tag:    ${PERTURB_TAG}"
    echo "Output:              $OUTPUT_DIR"
    echo "========================================="
    echo

    python "$PROJECT_ROOT/openvla/experiments/robot/libero/evaluate_attention_openvla.py" \
        --task-suite "$SUITE" \
        --output-dir "$OUTPUT_DIR" \
        "${SHARED_ARGS[@]}" \
        ${VIZ_FLAG}

    echo
    echo "========================================="
    echo "Suite ${SUITE} complete! Results in: $OUTPUT_DIR"
    echo "========================================="
    echo
done

echo "Done!"
