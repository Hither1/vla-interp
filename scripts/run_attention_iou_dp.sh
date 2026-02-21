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

# ── Configuration ─────────────────────────────────────────────────────────────
CKPT="${CKPT:-dp_scratch/ckpt_300.pt}"
TASK_SUITE="${TASK_SUITE:-libero_10}"
TASK_ID="${TASK_ID:-}"           # Leave empty to run all tasks
NUM_EPISODES="${NUM_EPISODES:-5}"
SEED="${SEED:-7}"
REPLAN_STEPS="${REPLAN_STEPS:-16}"
T_FRAC="${T_FRAC:-0.5}"         # Diffusion timestep fraction for GradCAM
LAYERS="${LAYERS:-layer2 layer3 layer4}"  # ResNet-18 layers for GradCAM (space-separated)
SAVE_VIZ="${SAVE_VIZ:-1}"       # Set to 1 to save GradCAM overlay images

OUTPUT_DIR="${OUTPUT_DIR:-${WORKDIR}/results/attention/iou_dp/${TASK_SUITE}_seed${SEED}}"

# ── Run ───────────────────────────────────────────────────────────────────────
mkdir -p logs "${OUTPUT_DIR}"

echo "============================================================"
echo "Diffusion Policy: GradCAM Saliency IoU"
echo "============================================================"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Checkpoint:    ${CKPT}"
echo "Task suite:    ${TASK_SUITE}"
echo "Task ID:       ${TASK_ID:-all}"
echo "Num episodes:  ${NUM_EPISODES}"
echo "Replan steps:  ${REPLAN_STEPS}"
echo "t_frac:        ${T_FRAC}"
echo "Layers:        ${LAYERS}"
echo "Seed:          ${SEED}"
echo "Save viz:      ${SAVE_VIZ}"
echo "Output dir:    ${OUTPUT_DIR}"
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
    --output-dir "${OUTPUT_DIR}" \
    ${TASK_ID_FLAG} \
    ${VIZ_FLAG}

echo
echo "Done. Results saved to ${OUTPUT_DIR}/iou_results_${TASK_SUITE}.json"
