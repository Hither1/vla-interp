#!/usr/bin/env bash
#SBATCH --job-name=attn-iou
#SBATCH --output=/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/logs/attn_iou_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=10:30:00
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner_h100
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END
#SBATCH --exclusive

# ── Environment ──────────────────────────────────────────────────────────────
source ~/.bashrc
conda deactivate
conda activate vla

set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:/n/netscratch/sham_lab/Lab/chloe00/libero"
export LIBERO_CONFIG_PATH=/n/netscratch/sham_lab/Lab/chloe00/libero

export NUMBA_CACHE_DIR="$TMPDIR/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

# Offscreen rendering for MuJoCo (no display on compute nodes)
export MUJOCO_GL=egl

# ── Configuration (override via environment or edit here) ────────────────────
TASK_SUITE="${TASK_SUITE:-libero_10}"
NUM_EPISODES="${NUM_EPISODES:-5}"
LAYERS="${LAYERS:-17}"
CHECKPOINT="${CHECKPOINT:-$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero}"
SAVE_VIZ="${SAVE_VIZ:-1}"   # set to 1 for per-step visualizations (slow, disk-heavy)
SEED="${SEED:-7}"

OUTPUT_DIR="outputs_iou/${TASK_SUITE}_seed${SEED}"

# ── Run ──────────────────────────────────────────────────────────────────────
mkdir -p logs "$OUTPUT_DIR"

echo "============================================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Task suite:    $TASK_SUITE"
echo "Num episodes:  $NUM_EPISODES"
echo "Layers:        $LAYERS"
echo "Checkpoint:    $CHECKPOINT"
echo "Output dir:    $OUTPUT_DIR"
echo "Save viz:      $SAVE_VIZ"
echo "============================================================"

VIZ_FLAG=""
if [[ "$SAVE_VIZ" == "1" ]]; then
    VIZ_FLAG="--save-viz"
fi

python evaluate_attention_iou.py \
    --checkpoint "$CHECKPOINT" \
    --task-suite "$TASK_SUITE" \
    --num-episodes "$NUM_EPISODES" \
    --layers $LAYERS \
    --seed "$SEED" \
    --output-dir "$OUTPUT_DIR" \
    $VIZ_FLAG

echo "Done. Results saved to $OUTPUT_DIR/iou_results.json"
