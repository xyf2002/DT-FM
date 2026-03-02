#!/usr/bin/env bash
# Launch baselines distributed training via torchrun (elastic launcher).
# Each rank is a separate process — no mp.spawn.
#
# Single-node usage:
#   bash baselines/scripts/launch_torchrun.sh [NPROC] [STRATEGY] [CONFIG]
#
# Multi-node usage (run on each node):
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 \
#     bash baselines/scripts/launch_torchrun.sh 4 dtfm
#
# Examples:
#   bash baselines/scripts/launch_torchrun.sh 4 dtfm
#   bash baselines/scripts/launch_torchrun.sh 8 asteroid
#   bash baselines/scripts/launch_torchrun.sh 4 dtfm --enable-mps

set -euo pipefail

NPROC_PER_NODE="${1:-4}"
STRATEGY="${2:-dtfm}"
CONFIG="${3:-baselines/configs/${STRATEGY}_default.yaml}"
shift 3 2>/dev/null || true

# Multi-node defaults (override with env vars)
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

echo "============================================"
echo "  Baselines torchrun Distributed Training"
echo "  Nodes:      ${NNODES}"
echo "  Node rank:  ${NODE_RANK}"
echo "  Procs/node: ${NPROC_PER_NODE}"
echo "  World size: ${WORLD_SIZE}"
echo "  Strategy:   ${STRATEGY}"
echo "  Config:     ${CONFIG}"
echo "  Master:     ${MASTER_ADDR}:${MASTER_PORT}"
echo "============================================"

torchrun \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m baselines.train \
    --strategy "${STRATEGY}" \
    --config "${CONFIG}" \
    --world-size "${WORLD_SIZE}" \
    --dist-url "tcp://${MASTER_ADDR}:${MASTER_PORT}" \
    "$@"
