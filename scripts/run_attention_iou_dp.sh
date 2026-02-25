#!/usr/bin/env bash
#SBATCH --job-name=attn-iou-dp
#SBATCH --output=/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/logs/dp_attn_iou_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner_h100
#SBATCH --time=08:00:00
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END

# GradCAM saliency IoU for Diffusion Policy.
# Hooks multiple ResNet-18 layers, averages GradCAM heatmaps at each replan
# step, and measures IoU against LIBERO instance segmentation masks.
#
# Override variables at submission time, e.g.:
#   sbatch --export=ALL,TASK_SUITE=libero_goal,NUM_EPISODES=10,SAVE_VIZ=1 run_attention_iou_dp.sh
#
# Visual perturbation examples:
#   VISUAL_PERTURB_MODE=rotate ROTATION_DEGREES=30 sbatch scripts/run_attention_iou_dp.sh
#   VISUAL_PERTURB_MODE=translate TRANSLATE_X_FRAC=0.2 sbatch scripts/run_attention_iou_dp.sh
#
# Policy perturbation examples:
#   POLICY_PERTURB_MODE=random_action RANDOM_ACTION_PROB=0.25 sbatch scripts/run_attention_iou_dp.sh
#   POLICY_PERTURB_MODE=object_shift OBJECT_SHIFT_X_STD=0.05 sbatch scripts/run_attention_iou_dp.sh

source ~/.bashrc
conda deactivate
conda activate vla

module load gcc/12.2.0-fasrc01
module load cuda/12.4.1-fasrc01

# Offscreen rendering (no display on compute nodes)
export MUJOCO_GL=egl

WORKDIR="/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp"

export PYTHONPATH="${WORKDIR}:${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH="${PYTHONPATH}:/n/netscratch/sham_lab/Lab/chloe00/libero"
export LIBERO_CONFIG_PATH=/n/netscratch/sham_lab/Lab/chloe00/libero

export HF_HOME=/n/netscratch/sham_lab/Lab/chloe00/huggingface
export TRANSFORMERS_CACHE=/n/netscratch/sham_lab/Lab/chloe00/huggingface

# ── Configuration ─────────────────────────────────────────────
CKPT="${CKPT:-dp_scratch/ckpt_300.pt}"
TASK_SUITE="${TASK_SUITE:-libero_10}"
TASK_ID="${TASK_ID:-}"           # Leave empty to run all tasks
NUM_EPISODES="${NUM_EPISODES:-5}"
SEED="${SEED:-7}"
REPLAN_STEPS="${REPLAN_STEPS:-16}"
T_FRAC="${T_FRAC:-0.5}"         # Diffusion timestep fraction for GradCAM
LAYERS="${LAYERS:-layer2 layer3 layer4}"  # ResNet-18 layers for GradCAM (space-separated)
SAVE_VIZ="${SAVE_VIZ:-1}"       # Set to 1 to save GradCAM overlay images

# Visual perturbation
# mode: none | rotate | translate | rotate_translate
VISUAL_PERTURB_MODE="${VISUAL_PERTURB_MODE:-none}"
ROTATION_DEGREES="${ROTATION_DEGREES:-0.0}"
TRANSLATE_X_FRAC="${TRANSLATE_X_FRAC:-0.0}"
TRANSLATE_Y_FRAC="${TRANSLATE_Y_FRAC:-0.0}"

# Policy perturbation
# mode: none | random_action | object_shift
POLICY_PERTURB_MODE="${POLICY_PERTURB_MODE:-none}"
RANDOM_ACTION_PROB="${RANDOM_ACTION_PROB:-0.25}"
RANDOM_ACTION_SCALE="${RANDOM_ACTION_SCALE:-1.0}"
OBJECT_SHIFT_X_STD="${OBJECT_SHIFT_X_STD:-0.0}"
OBJECT_SHIFT_Y_STD="${OBJECT_SHIFT_Y_STD:-0.0}"

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

# Build combined tag (empty = no perturbation)
if [[ -n "${VIS_TAG}" && -n "${POL_TAG}" ]]; then
    PERTURB_TAG="${VIS_TAG}__${POL_TAG}"
elif [[ -n "${VIS_TAG}" ]]; then
    PERTURB_TAG="${VIS_TAG}"
elif [[ -n "${POL_TAG}" ]]; then
    PERTURB_TAG="${POL_TAG}"
else
    PERTURB_TAG="none"
fi

OUTPUT_DIR="${OUTPUT_DIR:-${WORKDIR}/results/attention/iou_dp/${TASK_SUITE}_seed${SEED}_perturb_${PERTURB_TAG}}"

# ── Run ────────────────────────────────────────────────────────────────────────────
mkdir -p logs "${OUTPUT_DIR}"

echo "============================================================"
echo "Diffusion Policy: GradCAM Saliency IoU"
echo "============================================================"
echo "Job ID:              ${SLURM_JOB_ID}"
echo "Checkpoint:          ${CKPT}"
echo "Task suite:          ${TASK_SUITE}"
echo "Task ID:             ${TASK_ID:-all}"
echo "Num episodes:        ${NUM_EPISODES}"
echo "Replan steps:        ${REPLAN_STEPS}"
echo "t_frac:              ${T_FRAC}"
echo "Layers:              ${LAYERS}"
echo "Seed:                ${SEED}"
echo "Save viz:            ${SAVE_VIZ}"
echo "Visual perturbation: ${VISUAL_PERTURB_MODE} (rotation=${ROTATION_DEGREES} tx=${TRANSLATE_X_FRAC} ty=${TRANSLATE_Y_FRAC})"
echo "Policy perturbation: ${POLICY_PERTURB_MODE} (prob=${RANDOM_ACTION_PROB} scale=${RANDOM_ACTION_SCALE} ox=${OBJECT_SHIFT_X_STD} oy=${OBJECT_SHIFT_Y_STD})"
echo "Perturbation tag:    ${PERTURB_TAG}"
echo "Output dir:          ${OUTPUT_DIR}"
echo "============================================================"

TASK_ID_FLAG=""
if [ -n "${TASK_ID}" ]; then
    TASK_ID_FLAG="--task-id ${TASK_ID}"
fi

VIZ_FLAG=""
if [[ "${SAVE_VIZ}" == "1" ]]; then
    VIZ_FLAG="--save-viz"
fi

python "${WORKDIR}/analysis/attention/evaluate_attention_iou_dp.py" \
    --ckpt "${CKPT}" \
    --task-suite "${TASK_SUITE}" \
    --num-episodes "${NUM_EPISODES}" \
    --seed "${SEED}" \
    --replan-steps "${REPLAN_STEPS}" \
    --t-frac "${T_FRAC}" \
    --layers ${LAYERS} \
    --visual-perturb-mode "${VISUAL_PERTURB_MODE}" \
    --rotation-degrees "${ROTATION_DEGREES}" \
    --translate-x-frac "${TRANSLATE_X_FRAC}" \
    --translate-y-frac "${TRANSLATE_Y_FRAC}" \
    --policy-perturb-mode "${POLICY_PERTURB_MODE}" \
    --random-action-prob "${RANDOM_ACTION_PROB}" \
    --random-action-scale "${RANDOM_ACTION_SCALE}" \
    --object-shift-x-std "${OBJECT_SHIFT_X_STD}" \
    --object-shift-y-std "${OBJECT_SHIFT_Y_STD}" \
    --output-dir "${OUTPUT_DIR}" \
    ${TASK_ID_FLAG} \
    ${VIZ_FLAG}

echo
echo "Done. Results saved to ${OUTPUT_DIR}/iou_results_${TASK_SUITE}.json"
