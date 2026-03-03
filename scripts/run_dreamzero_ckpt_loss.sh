#!/bin/bash
#SBATCH --job-name=dz_ckpt_loss
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_sham_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --output=logs/dreamzero_ckpt_loss_%j.out
#SBATCH --error=logs/dreamzero_ckpt_loss_%j.err
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END,FAIL

# Evaluate training loss of a DreamZero finetuned checkpoint.
#
# Usage:
#   sbatch scripts/run_dreamzero_ckpt_loss.sh
#
# Override defaults with env vars:
#   NUM_BATCHES=200 sbatch scripts/run_dreamzero_ckpt_loss.sh

module load gcc/12.2.0-fasrc01
module load cuda/12.4.1-fasrc01

source ~/.bashrc
conda deactivate
conda activate vla

TORCHRUN=/n/home13/chloe00/miniforge3/envs/vla/bin/torchrun

# ── config ───────────────────────────────────────────────────────────────────
CKPT_PATH="${CKPT_PATH:-/n/netscratch/sham_lab/Lab/chloe00/libero/dreamzero_libero_all_lora/DreamZero_libero/checkpoint-3400}"
NUM_BATCHES="${NUM_BATCHES:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_GPUS="${NUM_GPUS:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-analysis/loss_results}"
# ─────────────────────────────────────────────────────────────────────────────

# Resolve project root (works both from sbatch and direct bash execution)
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
fi

mkdir -p "$PROJECT_ROOT/logs"

echo "============================================"
echo "DreamZero checkpoint loss evaluation"
echo "============================================"
echo "Checkpoint : $CKPT_PATH"
echo "Batches    : $NUM_BATCHES"
echo "Batch size : $BATCH_SIZE"
echo "GPUs       : $NUM_GPUS"
echo "Output     : $OUTPUT_DIR"
echo "============================================"
echo

$TORCHRUN \
    --standalone \
    --nproc_per_node "$NUM_GPUS" \
    "$PROJECT_ROOT/analysis/dreamzero_ckpt_loss.py" \
    --ckpt-path "$CKPT_PATH" \
    --num-batches "$NUM_BATCHES" \
    --batch-size "$BATCH_SIZE" \
    --output-dir "$PROJECT_ROOT/$OUTPUT_DIR"

echo
echo "Done. Results in $PROJECT_ROOT/$OUTPUT_DIR"
