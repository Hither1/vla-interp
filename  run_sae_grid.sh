#!/usr/bin/env bash
#SBATCH --job-name=train-token-adapter

#SBATCH --output=/n/holylfs06/LABS/sham_lab/Users/chloe00/adaptive-decoding/logs/%A_%a.log
#SBATCH --nodes=1

#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4    
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=71:30:00

#SBATCH --account=kempner_grads
#SBATCH --partition=kempner_h100
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END
#SBATCH --exclusive


# Custom environment
source ~/.bashrc
conda deactivate
conda activate vla

set -euo pipefail

export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export LIBERO_CONFIG_PATH=/n/holylfs06/LABS/sham_lab/Users/chloe00/vla-interp/third_party/libero

############################################
# CONFIG
############################################

ENV_NAME="LIBERO"
PORT=8000

LAYERS=(10 11 12 13)
FEATURES=(3 7 12 19 42)
STRENGTHS=(-2.0 -1.0 1.0 2.0)

RESULTS_DIR="results"
SERVER_LOG="server.log"
EVAL_LOG="eval.log"

############################################
# UTILS
############################################

wait_for_server () {
  echo "Waiting for policy server..."
  for i in {1..30}; do
    if nc -z localhost $PORT; then
      echo "Server is up."
      return
    fi
    sleep 1
  done
  echo "Server failed to start"
  exit 1
}

kill_server () {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill $SERVER_PID || true
    wait $SERVER_PID 2>/dev/null || true
  fi
}

trap kill_server EXIT

############################################
# BASELINE (no intervention)
############################################

echo "Running baseline..."

unset SAE_ENABLED
mkdir -p $RESULTS_DIR/baseline

uv run scripts/serve_policy.py --env $ENV_NAME \
  > $RESULTS_DIR/baseline/$SERVER_LOG 2>&1 &
SERVER_PID=$!

wait_for_server

python examples/libero/main.py \
  > $RESULTS_DIR/baseline/$EVAL_LOG 2>&1

kill_server

############################################
# ABLATION GRID
############################################

for LAYER in "${LAYERS[@]}"; do
  for FEATURE in "${FEATURES[@]}"; do
    echo "Ablation: layer=$LAYER feature=$FEATURE"

    OUTDIR="$RESULTS_DIR/ablate/layer_$LAYER/feature_$FEATURE"
    mkdir -p "$OUTDIR"

    export SAE_ENABLED=1
    export SAE_MODE="ablate"
    export SAE_LAYER=$LAYER
    export SAE_FEATURE=$FEATURE

    uv run scripts/serve_policy.py --env $ENV_NAME \
      > "$OUTDIR/$SERVER_LOG" 2>&1 &
    SERVER_PID=$!

    wait_for_server

    python examples/libero/main.py \
      > "$OUTDIR/$EVAL_LOG" 2>&1

    kill_server
  done
done

############################################
# STEERING GRID
############################################

for LAYER in "${LAYERS[@]}"; do
  for FEATURE in "${FEATURES[@]}"; do
    for STRENGTH in "${STRENGTHS[@]}"; do
      echo "Steering: layer=$LAYER feature=$FEATURE strength=$STRENGTH"

      OUTDIR="$RESULTS_DIR/steer/layer_$LAYER/feature_$FEATURE/strength_$STRENGTH"
      mkdir -p "$OUTDIR"

      export SAE_ENABLED=1
      export SAE_MODE="steer"
      export SAE_LAYER=$LAYER
      export SAE_FEATURE=$FEATURE
      export SAE_STRENGTH=$STRENGTH

      python scripts/serve_policy.py --env $ENV_NAME \
        > "$OUTDIR/$SERVER_LOG" 2>&1 &
      SERVER_PID=$!

      wait_for_server

      python examples/libero/main.py --args.port 8000 \
        > "$OUTDIR/$EVAL_LOG" 2>&1

      kill_server
    done
  done
done

echo "Grid search complete."