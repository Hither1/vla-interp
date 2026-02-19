#!/usr/bin/env bash
#SBATCH --job-name=action-entropy
#SBATCH --output=/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/logs/action_entropy_%j.log
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

export PYTHONPATH="${PYTHONPATH:-}:$PWD/third_party/libero"
export LIBERO_CONFIG_PATH=/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/third_party/libero

# Offscreen rendering for MuJoCo (no display on compute nodes)
export MUJOCO_GL=egl

# ── Configuration (override via environment or edit here) ────────────────────
# Which task suites to evaluate. Space-separated list, or "all" for all five.
TASK_SUITES="${TASK_SUITES:-all}"
NUM_TRIALS="${NUM_TRIALS:-20}"
SEED="${SEED:-7}"
PORT="${PORT:-8000}"
ENV_NAME="LIBERO"
REPLAN_STEPS="${REPLAN_STEPS:-5}"

# ── Resolve task suite list ──────────────────────────────────────────────────
if [[ "$TASK_SUITES" == "all" ]]; then
    SUITES=(libero_spatial libero_object libero_goal libero_10 libero_90)
else
    read -ra SUITES <<< "$TASK_SUITES"
fi

# ── Utils ────────────────────────────────────────────────────────────────────
wait_for_server() {
    echo "Waiting for policy server on port $PORT..."
    for i in {1..60}; do
        if nc -z localhost "$PORT" 2>/dev/null; then
            echo "Server is up."
            return
        fi
        sleep 2
    done
    echo "ERROR: Server failed to start within 120s"
    exit 1
}

kill_server() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo "Stopping server (PID=$SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        unset SERVER_PID
    fi
}

trap kill_server EXIT

# ── Run ──────────────────────────────────────────────────────────────────────
mkdir -p logs

echo "============================================================"
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Task suites:   ${SUITES[*]}"
echo "Num trials:    $NUM_TRIALS"
echo "Seed:          $SEED"
echo "Port:          $PORT"
echo "============================================================"

# Start policy server (shared across all suites)
echo "Starting policy server..."
python scripts/serve_policy.py --env "$ENV_NAME" \
    > "logs/action_entropy_server_${SLURM_JOB_ID:-0}.log" 2>&1 &
SERVER_PID=$!
wait_for_server

# Run evaluation for each task suite
for SUITE in "${SUITES[@]}"; do
    echo ""
    echo "============================================================"
    echo "Evaluating: $SUITE"
    echo "============================================================"

    # video_out_path default is computed at class-definition time, so we
    # must pass it explicitly when overriding task_suite_name.
    SUITE_SHORT="${SUITE#libero_}"   # e.g. libero_spatial -> spatial
    VIDEO_OUT="data/libero/${SUITE_SHORT}/videos"
    mkdir -p "$VIDEO_OUT"

    python examples/libero/main.py \
        --task-suite-name "$SUITE" \
        --num-trials-per-task "$NUM_TRIALS" \
        --seed "$SEED" \
        --port "$PORT" \
        --replan-steps "$REPLAN_STEPS" \
        --video-out-path "$VIDEO_OUT" \
        2>&1 | tee "logs/action_entropy_${SUITE}_${SLURM_JOB_ID:-0}.log"

    echo "Finished: $SUITE"
done

# Stop server
kill_server

# Post-hoc: aggregate entropy across all suites
echo ""
echo "============================================================"
echo "Aggregating action entropy across suites..."
echo "============================================================"

python compute_action_entropy.py \
    --auto-discover \
    --output "data/libero/action_entropy_results.json"

echo "Done. Results saved to data/libero/action_entropy_results.json"
