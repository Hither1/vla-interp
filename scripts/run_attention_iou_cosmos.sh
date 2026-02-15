#!/usr/bin/env bash
#SBATCH --job-name=cosmos-attn-iou
#SBATCH --output=/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/logs/cosmos_attn_iou_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=11:30:00
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

module load gcc/12.2.0-fasrc01
module load cuda/12.4.1-fasrc01

export HF_HOME=/n/netscratch/sham_lab/Lab/chloe00/huggingface
export TRANSFORMERS_CACHE=/n/netscratch/sham_lab/Lab/chloe00/huggingface

export PYTHONPATH="${PYTHONPATH:-}:/n/netscratch/sham_lab/Lab/chloe00/libero"
export LIBERO_CONFIG_PATH=/n/netscratch/sham_lab/Lab/chloe00/libero

# Offscreen rendering for MuJoCo (no display on compute nodes)
export MUJOCO_GL=egl

# ── Configuration (override via environment or edit here) ────────────────────
TASK_SUITE="${TASK_SUITE:-libero_10}"
NUM_EPISODES="${NUM_EPISODES:-5}"
# LAYERS="${LAYERS:-27}"
SEED="${SEED:-7}"
SAVE_VIZ="${SAVE_VIZ:-0}"

# Cosmos model paths
CKPT_PATH="${CKPT_PATH:-nvidia/Cosmos-Policy-LIBERO-Predict2-2B}"
CONFIG_NAME="${CONFIG_NAME:-cosmos_predict2_2b_480p_libero__inference_only}"
CONFIG_FILE="${CONFIG_FILE:-cosmos_policy/config/config.py}"
DATASET_STATS="${DATASET_STATS:-nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json}"
T5_EMBEDDINGS="${T5_EMBEDDINGS:-nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl}"

OUTPUT_DIR="outputs_iou_cosmos/${TASK_SUITE}_seed${SEED}"

# ── Run ──────────────────────────────────────────────────────────────────────
mkdir -p logs "$OUTPUT_DIR"

echo "============================================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Task suite:    $TASK_SUITE"
echo "Num episodes:  $NUM_EPISODES"
# echo "Layers:        $LAYERS"
echo "Checkpoint:    $CKPT_PATH"
echo "Output dir:    $OUTPUT_DIR"
echo "Save viz:      $SAVE_VIZ"
echo "============================================================"

VIZ_FLAG=""
if [[ "$SAVE_VIZ" == "1" ]]; then
    VIZ_FLAG="--save-viz"
fi

python analysis/evaluate_attention_iou_cosmos.py \
    --ckpt-path "$CKPT_PATH" \
    --config-name "$CONFIG_NAME" \
    --config-file "$CONFIG_FILE" \
    --dataset-stats-path "$DATASET_STATS" \
    --t5-text-embeddings-path "$T5_EMBEDDINGS" \
    --task-suite "$TASK_SUITE" \
    --num-episodes "$NUM_EPISODES" \
    # --layers $LAYERS \
    --seed "$SEED" \
    --output-dir "$OUTPUT_DIR" \
    $VIZ_FLAG

echo "Done. Results saved to $OUTPUT_DIR/iou_results.json"
