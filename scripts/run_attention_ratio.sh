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

# Expand TASK_SUITE into array of suites to run
if [ "${TASK_SUITE}" = "all" ]; then
    SUITES=(libero_spatial libero_object libero_goal libero_10)
elif [ "${TASK_SUITE}" = "90_all" ]; then
    SUITES=(libero_90_obj libero_90_spa libero_90_act libero_90_com)
else
    SUITES=("${TASK_SUITE}")
fi
NUM_EPISODES="${NUM_EPISODES:-5}"
LAYERS="${LAYERS:-0 8 17 25 26 27}"
SAVE_VIZ="${SAVE_VIZ:-}"

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

# Derived perturbation tags
if [[ "$VISUAL_PERTURB_MODE" == "none" ]]; then
    VIS_TAG="none"
elif [[ "$VISUAL_PERTURB_MODE" == "rotate" ]]; then
    VIS_TAG="rotate_${ROTATION_DEGREES}deg"
elif [[ "$VISUAL_PERTURB_MODE" == "translate" ]]; then
    VIS_TAG="translate_x${TRANSLATE_X_FRAC}_y${TRANSLATE_Y_FRAC}"
else
    VIS_TAG="rotate_${ROTATION_DEGREES}deg_translate_x${TRANSLATE_X_FRAC}_y${TRANSLATE_Y_FRAC}"
fi

if [[ "$POLICY_PERTURB_MODE" == "none" ]]; then
    POL_TAG="none"
elif [[ "$POLICY_PERTURB_MODE" == "random_action" ]]; then
    POL_TAG="random_action_p${RANDOM_ACTION_PROB}_s${RANDOM_ACTION_SCALE}"
elif [[ "$POLICY_PERTURB_MODE" == "object_shift" ]]; then
    POL_TAG="object_shift_x${OBJECT_SHIFT_X_STD}_y${OBJECT_SHIFT_Y_STD}"
else
    POL_TAG="${POLICY_PERTURB_MODE}"
fi

# Script directory (handle SLURM execution)
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
fi

for SUITE in "${SUITES[@]}"; do
    OUTPUT_DIR="results/attention/ratio_pi05/${SUITE}/vis_${VIS_TAG}__pol_${POL_TAG}"

    echo "========================================="
    echo "Visual/Linguistic Attention Ratio Eval"
    echo "========================================="
    echo "Checkpoint:   $CHECKPOINT"
    echo "Task Suite:   $SUITE"
    echo "Task ID:      ${TASK_ID:-all}"
    echo "Episodes:     $NUM_EPISODES"
    echo "Layers:       $LAYERS"
    echo "Output:       $OUTPUT_DIR"
    echo "Save Viz:     ${SAVE_VIZ:-no}"
    echo "Visual perturb: $VISUAL_PERTURB_MODE ($VIS_TAG)"
    echo "  rotation_degrees:  $ROTATION_DEGREES"
    echo "  translate_x_frac:  $TRANSLATE_X_FRAC"
    echo "  translate_y_frac:  $TRANSLATE_Y_FRAC"
    echo "Policy perturb: $POLICY_PERTURB_MODE ($POL_TAG)"
    echo "  random_action_prob:   $RANDOM_ACTION_PROB"
    echo "  random_action_scale:  $RANDOM_ACTION_SCALE"
    echo "  object_shift_x_std:   $OBJECT_SHIFT_X_STD"
    echo "  object_shift_y_std:   $OBJECT_SHIFT_Y_STD"
    echo "========================================="
    echo

    # Build command
    CMD="python $PROJECT_ROOT/analysis/attention/evaluate_attention_ratio.py \
      --checkpoint $CHECKPOINT \
      --task-suite $SUITE \
      --num-episodes $NUM_EPISODES \
      --layers $LAYERS \
      --visual-perturb-mode $VISUAL_PERTURB_MODE \
      --rotation-degrees $ROTATION_DEGREES \
      --translate-x-frac $TRANSLATE_X_FRAC \
      --translate-y-frac $TRANSLATE_Y_FRAC \
      --policy-perturb-mode $POLICY_PERTURB_MODE \
      --random-action-prob $RANDOM_ACTION_PROB \
      --random-action-scale $RANDOM_ACTION_SCALE \
      --object-shift-x-std $OBJECT_SHIFT_X_STD \
      --object-shift-y-std $OBJECT_SHIFT_Y_STD \
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
    RESULTS_FILE="$OUTPUT_DIR/attention_ratio_results_${SUITE}.json"
    if [ -f "$RESULTS_FILE" ]; then
      echo "Generating analysis..."
      python "$PROJECT_ROOT/analysis/attention/parse_attention_ratio_results.py" \
        --results "$RESULTS_FILE" \
        --output "$OUTPUT_DIR/summary.txt" \
        --output-dir "$OUTPUT_DIR/analysis/attention" \
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
done

echo
echo "Done!"
