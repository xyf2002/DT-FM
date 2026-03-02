#!/usr/bin/env bash
# Launch baselines distributed training on a single node via mp.spawn.
# Usage:
#   bash baselines/scripts/launch_single_node.sh [NUM_GPUS] [STRATEGY] [CONFIG]
#
# Examples:
#   bash baselines/scripts/launch_single_node.sh 4 dtfm
#   bash baselines/scripts/launch_single_node.sh 8 asteroid baselines/configs/asteroid_default.yaml
#   bash baselines/scripts/launch_single_node.sh 2 dtfm baselines/configs/dtfm_default.yaml --dry-run
#   bash baselines/scripts/launch_single_node.sh 4 dtfm baselines/configs/dtfm_default.yaml --enable-mps

set -euo pipefail

NUM_GPUS="${1:-4}"
STRATEGY="${2:-dtfm}"
CONFIG="${3:-baselines/configs/${STRATEGY}_default.yaml}"
shift 3 2>/dev/null || true  # consume positional args, ignore if fewer

echo "============================================"
echo "  Baselines Single-Node Distributed Training"
echo "  GPUs:     ${NUM_GPUS}"
echo "  Strategy: ${STRATEGY}"
echo "  Config:   ${CONFIG}"
echo "============================================"

python -m baselines.train \
    --spawn \
    --num-gpus "${NUM_GPUS}" \
    --strategy "${STRATEGY}" \
    --config "${CONFIG}" \
    --dist-url "tcp://127.0.0.1:29500" \
    "$@"
