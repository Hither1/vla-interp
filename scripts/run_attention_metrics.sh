#!/usr/bin/env bash
#SBATCH --job-name=attn-metrics
#SBATCH --output=logs/attn_metrics_%j.log
#SBATCH -p gpu
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=240G

set -euo pipefail

source ~/.bashrc
conda deactivate
conda activate vla

module load gcc/12.2.0-fasrc01
module load cuda/12.4.1-fasrc01

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
fi
WORKDIR="$PROJECT_ROOT"

export HF_HOME="${HF_HOME:-/n/netscratch/sham_lab/Lab/chloe00/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/n/netscratch/sham_lab/Lab/chloe00/huggingface}"
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH="${PYTHONPATH}:/n/netscratch/sham_lab/Lab/chloe00/libero"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/n/netscratch/sham_lab/Lab/chloe00/libero}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${TMPDIR:-/tmp}/numba_cache}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

mkdir -p "$NUMBA_CACHE_DIR" logs

# Shared configuration
CHECKPOINT="${CHECKPOINT:-$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero}"
TASK_SUITE="${TASK_SUITE:-libero_10}"
TASK_ID="${TASK_ID:-}"
NUM_EPISODES="${NUM_EPISODES:-5}"
SEED="${SEED:-7}"
LAYERS="${LAYERS:-0 8 17 25 26 27}"
SAVE_VIZ="${SAVE_VIZ:-0}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"

# IoU-specific configuration
IOU_METRIC="${IOU_METRIC:-iou}"
THRESHOLD_METHODS="${THRESHOLD_METHODS:-percentile_90 percentile_75 otsu_0}"

# Expand TASK_SUITE into array of suites to run
if [ "${TASK_SUITE}" = "all" ]; then
    SUITES=(libero_spatial libero_object libero_goal libero_10)
elif [ "${TASK_SUITE}" = "90_all" ]; then
    SUITES=(libero_90_obj libero_90_spa libero_90_act libero_90_com)
else
    SUITES=("${TASK_SUITE}")
fi

# Visual perturbation
VISUAL_PERTURB_MODE="${VISUAL_PERTURB_MODE:-none}"
ROTATION_DEGREES="${ROTATION_DEGREES:-0.0}"
TRANSLATE_X_FRAC="${TRANSLATE_X_FRAC:-0.0}"
TRANSLATE_Y_FRAC="${TRANSLATE_Y_FRAC:-0.0}"

# Policy perturbation
POLICY_PERTURB_MODE="${POLICY_PERTURB_MODE:-none}"
RANDOM_ACTION_PROB="${RANDOM_ACTION_PROB:-0.0}"
RANDOM_ACTION_SCALE="${RANDOM_ACTION_SCALE:-1.0}"
OBJECT_SHIFT_X_STD="${OBJECT_SHIFT_X_STD:-0.0}"
OBJECT_SHIFT_Y_STD="${OBJECT_SHIFT_Y_STD:-0.0}"

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

if [[ "$VISUAL_PERTURB_MODE" != "none" || "$POLICY_PERTURB_MODE" != "none" ]]; then
    PERTURB_TAG="vis_${VIS_TAG}__pol_${POL_TAG}"
else
    PERTURB_TAG="none"
fi

TASK_ID_FLAG=()
if [ -n "$TASK_ID" ]; then
    TASK_ID_FLAG+=(--task-id "$TASK_ID")
fi

SAVE_VIZ_FLAG=()
if [[ "$SAVE_VIZ" == "1" || "$SAVE_VIZ" == "true" ]]; then
    SAVE_VIZ_FLAG+=(--save-viz)
fi

read -r -a LAYERS_ARR <<< "$LAYERS"
read -r -a THRESHOLD_METHODS_ARR <<< "$THRESHOLD_METHODS"

for SUITE in "${SUITES[@]}"; do
    RATIO_OUTPUT_DIR="${WORKDIR}/results/attention/ratio/pi05/perturb/${PERTURB_TAG}/${SUITE}_seed${SEED}"
    IOU_OUTPUT_DIR="${WORKDIR}/results/attention/iou/pi05/perturb/${PERTURB_TAG}/${SUITE}_seed${SEED}"
    mkdir -p "${RATIO_OUTPUT_DIR}" "${IOU_OUTPUT_DIR}"

    echo "============================================================"
    echo "Attention Metrics Eval"
    echo "============================================================"
    echo "Job ID:        ${SLURM_JOB_ID:-local}"
    echo "Task suite:    ${SUITE}"
    echo "Task ID:       ${TASK_ID:-all}"
    echo "Num episodes:  ${NUM_EPISODES}"
    echo "Checkpoint:    ${CHECKPOINT}"
    echo "Seed:          ${SEED}"
    echo "Layers:        ${LAYERS}"
    echo "Save viz:      ${SAVE_VIZ}"
    echo "IoU metric:    ${IOU_METRIC}"
    echo "Thresholds:    ${THRESHOLD_METHODS}"
    echo "Visual perturbation: ${VISUAL_PERTURB_MODE} (rotation=${ROTATION_DEGREES} tx=${TRANSLATE_X_FRAC} ty=${TRANSLATE_Y_FRAC})"
    echo "Policy perturbation: ${POLICY_PERTURB_MODE} (prob=${RANDOM_ACTION_PROB} scale=${RANDOM_ACTION_SCALE} ox=${OBJECT_SHIFT_X_STD} oy=${OBJECT_SHIFT_Y_STD})"
    echo "Perturbation tag: ${PERTURB_TAG}"
    echo "Ratio output:  ${RATIO_OUTPUT_DIR}"
    echo "IoU output:    ${IOU_OUTPUT_DIR}"
    echo "============================================================"

    python "${WORKDIR}/analysis/attention/evaluate_attention_ratio.py" \
        --checkpoint "${CHECKPOINT}" \
        --task-suite "${SUITE}" \
        --num-episodes "${NUM_EPISODES}" \
        --seed "${SEED}" \
        --replan-steps "${REPLAN_STEPS}" \
        --layers "${LAYERS_ARR[@]}" \
        --visual-perturb-mode "${VISUAL_PERTURB_MODE}" \
        --rotation-degrees "${ROTATION_DEGREES}" \
        --translate-x-frac "${TRANSLATE_X_FRAC}" \
        --translate-y-frac "${TRANSLATE_Y_FRAC}" \
        --policy-perturb-mode "${POLICY_PERTURB_MODE}" \
        --random-action-prob "${RANDOM_ACTION_PROB}" \
        --random-action-scale "${RANDOM_ACTION_SCALE}" \
        --object-shift-x-std "${OBJECT_SHIFT_X_STD}" \
        --object-shift-y-std "${OBJECT_SHIFT_Y_STD}" \
        --output-dir "${RATIO_OUTPUT_DIR}" \
        "${TASK_ID_FLAG[@]}" \
        "${SAVE_VIZ_FLAG[@]}"

    RATIO_RESULTS_FILE="${RATIO_OUTPUT_DIR}/attention_ratio_results_${SUITE}.json"
    if [ -f "${RATIO_RESULTS_FILE}" ]; then
        python "${WORKDIR}/analysis/attention/parse_attention_ratio_results.py" \
            --results "${RATIO_RESULTS_FILE}" \
            --output "${RATIO_OUTPUT_DIR}/summary.txt" \
            --output-dir "${RATIO_OUTPUT_DIR}/analysis/attention" \
            --plot-all
    fi

    python "${WORKDIR}/analysis/attention/evaluate_attention_iou.py" \
        --checkpoint "${CHECKPOINT}" \
        --task-suite "${SUITE}" \
        --num-episodes "${NUM_EPISODES}" \
        --seed "${SEED}" \
        --replan-steps "${REPLAN_STEPS}" \
        --layers "${LAYERS_ARR[@]}" \
        --metric "${IOU_METRIC}" \
        --threshold-methods "${THRESHOLD_METHODS_ARR[@]}" \
        --visual-perturb-mode "${VISUAL_PERTURB_MODE}" \
        --rotation-degrees "${ROTATION_DEGREES}" \
        --translate-x-frac "${TRANSLATE_X_FRAC}" \
        --translate-y-frac "${TRANSLATE_Y_FRAC}" \
        --policy-perturb-mode "${POLICY_PERTURB_MODE}" \
        --random-action-prob "${RANDOM_ACTION_PROB}" \
        --random-action-scale "${RANDOM_ACTION_SCALE}" \
        --object-shift-x-std "${OBJECT_SHIFT_X_STD}" \
        --object-shift-y-std "${OBJECT_SHIFT_Y_STD}" \
        --output-dir "${IOU_OUTPUT_DIR}" \
        "${TASK_ID_FLAG[@]}" \
        "${SAVE_VIZ_FLAG[@]}"

    echo
    echo "Finished suite ${SUITE}"
    echo "  Ratio results: ${RATIO_OUTPUT_DIR}/attention_ratio_results_${SUITE}.json"
    echo "  IoU results:   ${IOU_OUTPUT_DIR}/iou_results_${SUITE}.json"
    if [ -f "${RATIO_OUTPUT_DIR}/summary.txt" ]; then
        echo "  Ratio summary: ${RATIO_OUTPUT_DIR}/summary.txt"
    fi
    echo
done

echo "All attention metric runs completed."
