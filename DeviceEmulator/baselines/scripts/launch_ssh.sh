#!/usr/bin/env bash
# Launch baselines training across multiple nodes via SSH.
# Mirrors the DT-FM multi-node SSH-per-rank launch pattern.
#
# Setup:
#   1. Edit IP_LIST and GPUS_PER_NODE below
#   2. Ensure passwordless SSH to all nodes
#   3. Ensure baselines/ is at the same path on all nodes
#
# Usage:
#   bash baselines/scripts/launch_ssh.sh [STRATEGY] [CONFIG]

set -euo pipefail

# ── Configuration ────────────────────────────────────────
# Edit these for your cluster:
IP_LIST=("10.0.0.1" "10.0.0.2")  # Node IPs
GPUS_PER_NODE=4                    # GPUs per node
STRATEGY="${1:-dtfm}"
CONFIG="${2:-baselines/configs/${STRATEGY}_default.yaml}"
shift 2 2>/dev/null || true

MASTER_IP="${IP_LIST[0]}"
MASTER_PORT="${MASTER_PORT:-29500}"
NUM_NODES=${#IP_LIST[@]}
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))

echo "============================================"
echo "  Baselines SSH Multi-Node Training"
echo "  Nodes:      ${NUM_NODES} (${IP_LIST[*]})"
echo "  GPUs/node:  ${GPUS_PER_NODE}"
echo "  World size: ${WORLD_SIZE}"
echo "  Master:     ${MASTER_IP}:${MASTER_PORT}"
echo "  Strategy:   ${STRATEGY}"
echo "============================================"

RANK=0
PIDS=()

for node_idx in "${!IP_LIST[@]}"; do
    NODE_IP="${IP_LIST[$node_idx]}"
    for ((gpu=0; gpu<GPUS_PER_NODE; gpu++)); do
        echo "Launching rank ${RANK} on ${NODE_IP} GPU ${gpu}"
        ssh "${NODE_IP}" \
            "cd $(pwd) && python -m baselines.train \
                --rank ${RANK} \
                --world-size ${WORLD_SIZE} \
                --cuda-id ${gpu} \
                --strategy ${STRATEGY} \
                --config ${CONFIG} \
                --dist-url tcp://${MASTER_IP}:${MASTER_PORT} \
                $*" &
        PIDS+=($!)
        RANK=$((RANK + 1))
    done
done

echo "Waiting for ${#PIDS[@]} processes..."
FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=$((FAIL + 1))
done

if [ "$FAIL" -gt 0 ]; then
    echo "ERROR: ${FAIL} rank(s) failed."
    exit 1
fi
echo "All ranks finished successfully."
