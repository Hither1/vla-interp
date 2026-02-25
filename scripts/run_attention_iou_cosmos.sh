#!/usr/bin/env bash
#SBATCH --job-name=cosmos-attn-iou
#SBATCH --output=/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/logs/cosmos_attn_iou_%j.log
#SBATCH -p gpu                 # Partition
#SBATCH -t 12:00:00            # Time limit
#SBATCH --gres=gpu:1           # GPU request
#SBATCH --mem=64G              # Memory (Cosmos needs more)
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END
#SBATCH --exclusive

# ── Cosmos Policy — Attention IoU Evaluation (with optional perturbations) ───
#
# Quick-start examples
# --------------------
# Baseline (no perturbation):
#   sbatch scripts/run_attention_iou_cosmos.sh
#
# 30° visual rotation:
#   VISUAL_PERTURB_MODE=rotate ROTATION_DEGREES=30 \
#       sbatch scripts/run_attention_iou_cosmos.sh
#
# 20% rightward translation:
#   VISUAL_PERTURB_MODE=translate TRANSLATE_X_FRAC=0.2 sbatch scripts/run_attention_iou_cosmos.sh
#
# Rotate then translate:
#   VISUAL_PERTURB_MODE=rotate_translate ROTATION_DEGREES=15 TRANSLATE_X_FRAC=0.1 \
#       sbatch scripts/run_attention_iou_cosmos.sh
#
# 25% random action replacement:
#   POLICY_PERTURB_MODE=random_action RANDOM_ACTION_PROB=0.25 \
#       sbatch scripts/run_attention_iou_cosmos.sh
#
# Random object shift (x-axis, std=5cm):
#   POLICY_PERTURB_MODE=object_shift OBJECT_SHIFT_X_STD=0.05 \
#       sbatch scripts/run_attention_iou_cosmos.sh
# ─────────────────────────────────────────────────────────────────────────────

# ── Environment ──────────────────────────────────────────────────────────────
source ~/.bashrc
conda deactivate
conda activate vla

set -euo pipefail

module load gcc/12.2.0-fasrc01
module load cuda/12.4.1-fasrc01

export HF_HOME=/n/netscratch/sham_lab/Lab/chloe00/huggingface
export TRANSFORMERS_CACHE=/n/netscratch/sham_lab/Lab/chloe00/huggingface

WORKDIR="/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp"
export PYTHONPATH="${WORKDIR}/third_party/cosmos-policy:${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH="${WORKDIR}:${PYTHONPATH}"
export PYTHONPATH="${PYTHONPATH}:/n/netscratch/sham_lab/Lab/chloe00/libero"
export LIBERO_CONFIG_PATH=/n/netscratch/sham_lab/Lab/chloe00/libero

# Offscreen rendering for MuJoCo (no display on compute nodes)
export MUJOCO_GL=egl

# ── Configuration (override via environment or edit here) ────────────────────
TASK_SUITE="${TASK_SUITE:-libero_10}"
NUM_EPISODES="${NUM_EPISODES:-5}"
SEED="${SEED:-7}"
SAVE_VIZ="${SAVE_VIZ:-0}"

# Cosmos model paths
CKPT_PATH="${CKPT_PATH:-nvidia/Cosmos-Policy-LIBERO-Predict2-2B}"
CONFIG_NAME="${CONFIG_NAME:-cosmos_predict2_2b_480p_libero__inference_only}"
CONFIG_FILE="${CONFIG_FILE:-cosmos_policy/config/config.py}"
DATASET_STATS="${DATASET_STATS:-nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json}"
T5_EMBEDDINGS="${T5_EMBEDDINGS:-nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl}"

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

# ── Derived output tag ────────────────────────────────────────────────────────
if [[ "$VISUAL_PERTURB_MODE" == "none" && "$POLICY_PERTURB_MODE" == "none" ]]; then
    PERTURB_TAG="none"
elif [[ "$VISUAL_PERTURB_MODE" != "none" && "$POLICY_PERTURB_MODE" == "none" ]]; then
    if [[ "$VISUAL_PERTURB_MODE" == "rotate" ]]; then
        PERTURB_TAG="vis_rotate_${ROTATION_DEGREES}deg"
    elif [[ "$VISUAL_PERTURB_MODE" == "translate" ]]; then
        PERTURB_TAG="vis_translate_x${TRANSLATE_X_FRAC}_y${TRANSLATE_Y_FRAC}"
    else
        PERTURB_TAG="vis_rotate_${ROTATION_DEGREES}deg_translate_x${TRANSLATE_X_FRAC}_y${TRANSLATE_Y_FRAC}"
    fi
elif [[ "$VISUAL_PERTURB_MODE" == "none" && "$POLICY_PERTURB_MODE" != "none" ]]; then
    if [[ "$POLICY_PERTURB_MODE" == "random_action" ]]; then
        PERTURB_TAG="pol_random_action_p${RANDOM_ACTION_PROB}_s${RANDOM_ACTION_SCALE}"
    elif [[ "$POLICY_PERTURB_MODE" == "object_shift" ]]; then
        PERTURB_TAG="pol_object_shift_x${OBJECT_SHIFT_X_STD}_y${OBJECT_SHIFT_Y_STD}"
    else
        PERTURB_TAG="pol_${POLICY_PERTURB_MODE}"
    fi
else
    PERTURB_TAG="vis_${VISUAL_PERTURB_MODE}_pol_${POLICY_PERTURB_MODE}"
fi

OUTPUT_DIR="${WORKDIR}/results/attention/outputs_attn_iou_cosmos/${TASK_SUITE}_seed${SEED}_perturb_${PERTURB_TAG}"

# ── Run ──────────────────────────────────────────────────────────────────────
mkdir -p logs "$OUTPUT_DIR"

echo "============================================================"
echo "Job ID:              ${SLURM_JOB_ID:-local}"
echo "Task suite:          $TASK_SUITE"
echo "Num episodes:        $NUM_EPISODES"
echo "Seed:                $SEED"
echo "Checkpoint:          $CKPT_PATH"
echo "Visual perturbation: $VISUAL_PERTURB_MODE"
echo "  rotation_degrees:    $ROTATION_DEGREES"
echo "  translate_x_frac:    $TRANSLATE_X_FRAC"
echo "  translate_y_frac:    $TRANSLATE_Y_FRAC"
echo "Policy perturbation: $POLICY_PERTURB_MODE"
echo "  random_action_prob:  $RANDOM_ACTION_PROB"
echo "  random_action_scale: $RANDOM_ACTION_SCALE"
echo "  object_shift_x_std:  $OBJECT_SHIFT_X_STD"
echo "  object_shift_y_std:  $OBJECT_SHIFT_Y_STD"
echo "Output dir:          $OUTPUT_DIR"
echo "Save viz:            $SAVE_VIZ"
echo "============================================================"

VIZ_FLAG=""
if [[ "$SAVE_VIZ" == "1" ]]; then
    VIZ_FLAG="--save-viz"
fi

python analysis/attention/evaluate_attention_iou_cosmos.py \
    --ckpt-path "$CKPT_PATH" \
    --config-name "$CONFIG_NAME" \
    --config-file "$CONFIG_FILE" \
    --dataset-stats-path "$DATASET_STATS" \
    --t5-text-embeddings-path "$T5_EMBEDDINGS" \
    --task-suite "$TASK_SUITE" \
    --num-episodes "$NUM_EPISODES" \
    --seed "$SEED" \
    --metric attention_ratio \
    --visual-perturb-mode "$VISUAL_PERTURB_MODE" \
    --rotation-degrees "$ROTATION_DEGREES" \
    --translate-x-frac "$TRANSLATE_X_FRAC" \
    --translate-y-frac "$TRANSLATE_Y_FRAC" \
    --policy-perturb-mode "$POLICY_PERTURB_MODE" \
    --random-action-prob "$RANDOM_ACTION_PROB" \
    --random-action-scale "$RANDOM_ACTION_SCALE" \
    --object-shift-x-std "$OBJECT_SHIFT_X_STD" \
    --object-shift-y-std "$OBJECT_SHIFT_Y_STD" \
    --output-dir "$OUTPUT_DIR" \
    $VIZ_FLAG

echo "Done. Results saved to $OUTPUT_DIR/iou_results_${TASK_SUITE}.json"
