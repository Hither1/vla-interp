#!/usr/bin/env bash
#SBATCH --job-name=openvla-perturb
#SBATCH --output=/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/logs/openvla_perturb_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner_h100
#SBATCH --time=16:00:00
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END
#SBATCH --exclusive

# ── OpenVLA — Perturbation Evaluation ────────────────────────────────────────
#
# Evaluates OpenVLA under any combination of language, visual, and policy
# perturbations.  All three types can be set independently; the output path
# encodes every active perturbation.
#
# Prompt modes (PROMPT_MODE)
# --------------------------
#   original  (default) use ground-truth task description
#   empty               pass an empty string
#   shuffle             randomly shuffle instruction words
#   random              replace with a random other task's instruction
#   synonym             substitute key verbs with synonyms
#   opposite            substitute key phrases with antonyms
#   custom              use CUSTOM_PROMPT verbatim
#
# Visual modes (VISUAL_PERTURB_MODE)
# ------------------------------------
#   none          (default) no perturbation
#   rotate        CCW rotation by ROTATION_DEGREES
#   translate     shift by TRANSLATE_{X,Y}_FRAC * image size
#   rotate_translate  rotation then translation
#
# Policy modes (POLICY_PERTURB_MODE)
# ------------------------------------
#   none          (default) no perturbation
#   random_action replace action with probability RANDOM_ACTION_PROB
#   object_shift  displace objects at episode start by OBJECT_SHIFT_{X,Y}_STD
#
# 
# --------------------
# Baseline:
#   CHECKPOINT=/path/to/openvla-7b sbatch scripts/run_openvla_perturb.sh
#
# Prompt — empty:
#   CHECKPOINT=/path/to/openvla-7b PROMPT_MODE=empty TASK_SUITE=all \
#       sbatch scripts/run_openvla_perturb.sh
#
# Prompt — shuffle, all suites:
#   CHECKPOINT=/path/to/openvla-7b PROMPT_MODE=shuffle TASK_SUITE=all \
#       sbatch scripts/run_openvla_perturb.sh
#
# Prompt — random, all suites:
#   CHECKPOINT=/path/to/openvla-7b PROMPT_MODE=random TASK_SUITE=all \
#       sbatch scripts/run_openvla_perturb.sh
#
# Prompt — synonym, all suites:
#   CHECKPOINT=/path/to/openvla-7b PROMPT_MODE=synonym TASK_SUITE=all \
#       sbatch scripts/run_openvla_perturb.sh
#
# Prompt — opposite, all suites:
#   CHECKPOINT=/path/to/openvla-7b PROMPT_MODE=opposite TASK_SUITE=all \
#       sbatch scripts/run_openvla_perturb.sh
# 
# Visual — 30° rotation:
#   CHECKPOINT=/path/to/openvla-7b \
#   VISUAL_PERTURB_MODE=rotate ROTATION_DEGREES=30 TASK_SUITE=all \
#       sbatch scripts/run_openvla_perturb.sh
#
# Visual — 20% rightward translation:
#   CHECKPOINT=/path/to/openvla-7b \
#   VISUAL_PERTURB_MODE=translate TRANSLATE_X_FRAC=0.2 TASK_SUITE=all \
#       sbatch scripts/run_openvla_perturb.sh
#
# Visual — 20% rightward translation:
#   CHECKPOINT=/path/to/openvla-7b \
#   VISUAL_PERTURB_MODE=translate TRANSLATE_X_FRAC=0.2 TASK_SUITE=all \
#       sbatch scripts/run_openvla_perturb.sh
# 
# VISUAL_PERTURB_MODE=rotate_translate ROTATION_DEGREES=15 TRANSLATE_X_FRAC=0.1 \
#     CHECKPOINT=/path/to/openvla-7b \
#     TASK_SUITE=all sbatch scripts/run_openvla_perturb.sh
#
# Policy — 25% random action:
#   CHECKPOINT=/path/to/openvla-7b \
#   POLICY_PERTURB_MODE=random_action RANDOM_ACTION_PROB=0.25 TASK_SUITE=all \
#       sbatch scripts/run_openvla_perturb.sh
#
# Policy — object shift (x-axis, std=5cm):
#   CHECKPOINT=/path/to/openvla-7b \
#   POLICY_PERTURB_MODE=object_shift OBJECT_SHIFT_X_STD=0.05 TASK_SUITE=all \
#       sbatch scripts/run_openvla_perturb.sh
#
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

source ~/.bashrc
conda deactivate
conda activate vla

module load gcc/12.2.0-fasrc01
module load cuda/12.4.1-fasrc01

export MUJOCO_GL=egl

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKDIR="/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp"
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH="${WORKDIR}/openvla:${PYTHONPATH}"
export PYTHONPATH="${PYTHONPATH}:/n/netscratch/sham_lab/Lab/chloe00/libero"
export LIBERO_CONFIG_PATH=/n/netscratch/sham_lab/Lab/chloe00/libero
export HF_HOME=/n/netscratch/sham_lab/Lab/chloe00/huggingface
export TRANSFORMERS_CACHE=/n/netscratch/sham_lab/Lab/chloe00/huggingface

# ── Model ─────────────────────────────────────────────────────────────────────
CHECKPOINT="${CHECKPOINT:-}"
if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: CHECKPOINT is required.  Example:"
    echo "  CHECKPOINT=/path/to/openvla-7b sbatch $0"
    exit 1
fi
UNNORM_KEY="${UNNORM_KEY:-}"

# ── Prompt perturbation ───────────────────────────────────────────────────────
PROMPT_MODE="${PROMPT_MODE:-original}"
CUSTOM_PROMPT="${CUSTOM_PROMPT:-}"

# ── Visual perturbation ───────────────────────────────────────────────────────
VISUAL_PERTURB_MODE="${VISUAL_PERTURB_MODE:-none}"
ROTATION_DEGREES="${ROTATION_DEGREES:-0.0}"
TRANSLATE_X_FRAC="${TRANSLATE_X_FRAC:-0.0}"
TRANSLATE_Y_FRAC="${TRANSLATE_Y_FRAC:-0.0}"

# ── Policy perturbation ───────────────────────────────────────────────────────
POLICY_PERTURB_MODE="${POLICY_PERTURB_MODE:-none}"
RANDOM_ACTION_PROB="${RANDOM_ACTION_PROB:-0.0}"
RANDOM_ACTION_SCALE="${RANDOM_ACTION_SCALE:-1.0}"
OBJECT_SHIFT_X_STD="${OBJECT_SHIFT_X_STD:-0.0}"
OBJECT_SHIFT_Y_STD="${OBJECT_SHIFT_Y_STD:-0.0}"

# ── LIBERO settings ───────────────────────────────────────────────────────────
TASK_SUITE="${TASK_SUITE:-libero_10}"
NUM_TRIALS="${NUM_TRIALS:-20}"
SEED="${SEED:-7}"

# ── Derived output tag (one segment per active perturbation) ──────────────────
PROMPT_TAG=""
if [[ "$PROMPT_MODE" != "original" ]]; then
    if [[ "$PROMPT_MODE" == "custom" && -n "$CUSTOM_PROMPT" ]]; then
        SLUG="${CUSTOM_PROMPT:0:30}"
        SLUG="${SLUG// /_}"
        PROMPT_TAG="prompt_custom_${SLUG}"
    else
        PROMPT_TAG="prompt_${PROMPT_MODE}"
    fi
fi

VIS_TAG=""
if [[ "$VISUAL_PERTURB_MODE" != "none" ]]; then
    if [[ "$VISUAL_PERTURB_MODE" == "rotate" ]]; then
        VIS_TAG="vis_rotate_${ROTATION_DEGREES}deg"
    elif [[ "$VISUAL_PERTURB_MODE" == "translate" ]]; then
        VIS_TAG="vis_translate_x${TRANSLATE_X_FRAC}_y${TRANSLATE_Y_FRAC}"
    else
        VIS_TAG="vis_rotate_${ROTATION_DEGREES}deg_translate_x${TRANSLATE_X_FRAC}_y${TRANSLATE_Y_FRAC}"
    fi
fi

POL_TAG=""
if [[ "$POLICY_PERTURB_MODE" != "none" ]]; then
    if [[ "$POLICY_PERTURB_MODE" == "random_action" ]]; then
        POL_TAG="pol_random_p${RANDOM_ACTION_PROB}_s${RANDOM_ACTION_SCALE}"
    elif [[ "$POLICY_PERTURB_MODE" == "object_shift" ]]; then
        POL_TAG="pol_objshift_x${OBJECT_SHIFT_X_STD}_y${OBJECT_SHIFT_Y_STD}"
    else
        POL_TAG="pol_${POLICY_PERTURB_MODE}"
    fi
fi

# Join non-empty tags with double underscore
PERTURB_TAG=""
for _tag in "$PROMPT_TAG" "$VIS_TAG" "$POL_TAG"; do
    if [[ -n "$_tag" ]]; then
        PERTURB_TAG="${PERTURB_TAG:+${PERTURB_TAG}__}${_tag}"
    fi
done
PERTURB_TAG="${PERTURB_TAG:-none}"

# ── Task suite list ───────────────────────────────────────────────────────────
if [[ "$TASK_SUITE" == "all" ]]; then
    SUITES=(libero_spatial libero_object libero_goal libero_10)
elif [[ "$TASK_SUITE" == "90_all" ]]; then
    SUITES=(libero_90_obj libero_90_spa libero_90_act libero_90_com)
else
    SUITES=("${TASK_SUITE}")
fi

mkdir -p "${WORKDIR}/logs"

echo "============================================================"
echo "Job ID:              ${SLURM_JOB_ID:-local}"
echo "Model:               OpenVLA"
echo "Checkpoint:          ${CHECKPOINT}"
echo "Prompt perturbation: ${PROMPT_MODE}"
[[ "$PROMPT_MODE" == "custom" ]] && echo "  custom_prompt:       '${CUSTOM_PROMPT}'"
echo "Visual perturbation: ${VISUAL_PERTURB_MODE}"
[[ "$VISUAL_PERTURB_MODE" != "none" ]] && echo "  rotation_degrees:    ${ROTATION_DEGREES}"
[[ "$VISUAL_PERTURB_MODE" != "none" ]] && echo "  translate_x_frac:    ${TRANSLATE_X_FRAC}"
[[ "$VISUAL_PERTURB_MODE" != "none" ]] && echo "  translate_y_frac:    ${TRANSLATE_Y_FRAC}"
echo "Policy perturbation: ${POLICY_PERTURB_MODE}"
[[ "$POLICY_PERTURB_MODE" == "random_action" ]] && echo "  random_action_prob:  ${RANDOM_ACTION_PROB}"
[[ "$POLICY_PERTURB_MODE" == "random_action" ]] && echo "  random_action_scale: ${RANDOM_ACTION_SCALE}"
[[ "$POLICY_PERTURB_MODE" == "object_shift"  ]] && echo "  object_shift_x_std:  ${OBJECT_SHIFT_X_STD}"
[[ "$POLICY_PERTURB_MODE" == "object_shift"  ]] && echo "  object_shift_y_std:  ${OBJECT_SHIFT_Y_STD}"
echo "Perturbation tag:    ${PERTURB_TAG}"
echo "Task suites:         ${SUITES[*]}"
echo "Trials per task:     ${NUM_TRIALS}"
echo "Seed:                ${SEED}"
echo "============================================================"

for SUITE in "${SUITES[@]}"; do
    echo ""
    echo "── Suite: ${SUITE} ──────────────────────────────────────────"

    VIDEO_OUT="${WORKDIR}/data/libero/openvla/perturb/${PERTURB_TAG}/${SUITE}"
    mkdir -p "${VIDEO_OUT}"

    UNNORM_KEY_ARG=""
    [[ -n "${UNNORM_KEY}" ]] && UNNORM_KEY_ARG="--unnorm-key ${UNNORM_KEY}"

    python "${WORKDIR}/examples/libero/openvla_eval.py" \
        --checkpoint "${CHECKPOINT}" \
        ${UNNORM_KEY_ARG} \
        --task-suite-name "${SUITE}" \
        --num-trials-per-task "${NUM_TRIALS}" \
        --seed "${SEED}" \
        --prompt-mode "${PROMPT_MODE}" \
        --custom-prompt "${CUSTOM_PROMPT}" \
        --visual-perturb-mode "${VISUAL_PERTURB_MODE}" \
        --rotation-degrees "${ROTATION_DEGREES}" \
        --translate-x-frac "${TRANSLATE_X_FRAC}" \
        --translate-y-frac "${TRANSLATE_Y_FRAC}" \
        --policy-perturb-mode "${POLICY_PERTURB_MODE}" \
        --random-action-prob "${RANDOM_ACTION_PROB}" \
        --random-action-scale "${RANDOM_ACTION_SCALE}" \
        --object-shift-x-std "${OBJECT_SHIFT_X_STD}" \
        --object-shift-y-std "${OBJECT_SHIFT_Y_STD}" \
        --video-out-path "${VIDEO_OUT}"

    echo "Finished: ${SUITE}"
done

echo ""
echo "============================================================"
echo "All suites complete.  Perturbation tag: ${PERTURB_TAG}"
echo "============================================================"
