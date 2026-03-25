#!/usr/bin/env bash
#SBATCH --job-name=pi0fast-robocasa-eval
#SBATCH --output=/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/logs/pi0fast_robocasa_eval_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner
#SBATCH --time=24:00:00
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END,FAIL

# ── Pi0-FAST RoboCasa Evaluation ─────────────────────────────────────────────
#
# Loads the policy in-process (no separate server needed) and runs eval on all
# 16 Composite-Seen tasks.
#
# Usage:
#   sbatch scripts/eval_pi0fast_robocasa.sh
#
# Override checkpoint dir and other settings via env vars:
#   CKPT_DIR=/path/to/checkpoint/49999 \
#   NUM_TRIALS=20 \
#   sbatch scripts/eval_pi0fast_robocasa.sh
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
export PYTHONPATH="${WORKDIR}/src:${WORKDIR}/robosuite:${WORKDIR}/robocasa:${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME=/n/netscratch/sham_lab/Lab/chloe00/huggingface
export TRANSFORMERS_CACHE=/n/netscratch/sham_lab/Lab/chloe00/huggingface
export HF_HUB_CACHE=/n/netscratch/sham_lab/Lab/chloe00

# ── Config ────────────────────────────────────────────────────────────────────
POLICY_CONFIG="${POLICY_CONFIG:-pi0_fast_robocasa_target_composite_seen_lora}"
CKPT_DIR="${CKPT_DIR:-/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/checkpoints/pi0_fast_robocasa_target_composite_seen_lora/composite_seen_pi0fast_v1_lora/49999}"
NUM_TRIALS="${NUM_TRIALS:-20}"
SEED="${SEED:-0}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
VIDEO_OUT="${VIDEO_OUT:-${WORKDIR}/data/robocasa/pi0fast_eval/${SLURM_JOB_ID:-local}}"

cd "${WORKDIR}"

mkdir -p "${WORKDIR}/logs"
mkdir -p "${VIDEO_OUT}"

echo "============================================================"
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Policy config: ${POLICY_CONFIG}"
echo "Checkpoint:    ${CKPT_DIR}"
echo "Trials/task:   ${NUM_TRIALS}"
echo "Seed:          ${SEED}"
echo "Video out:     ${VIDEO_OUT}"
echo "============================================================"

echo ""
echo "Running evaluation on all 16 Composite-Seen tasks..."

python "${WORKDIR}/examples/robocasa/main.py" \
    --policy-config "${POLICY_CONFIG}" \
    --checkpoint-dir "${CKPT_DIR}" \
    --num-trials-per-task "${NUM_TRIALS}" \
    --seed "${SEED}" \
    --replan-steps "${REPLAN_STEPS}" \
    --video-out-path "${VIDEO_OUT}"

echo ""
echo "============================================================"
echo "Evaluation complete. Results: ${VIDEO_OUT}/summary.json"
echo "============================================================"
