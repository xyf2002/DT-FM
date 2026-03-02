#!/usr/bin/env bash
# Launch baselines distributed training inside a Docker container.
# Requires nvidia-docker / NVIDIA Container Toolkit.
#
# Usage:
#   bash baselines/scripts/launch_docker.sh [NUM_GPUS] [STRATEGY] [CONFIG]
#
# Build image first:
#   docker build -t baselines:latest -f baselines/Dockerfile .
#
# Examples:
#   bash baselines/scripts/launch_docker.sh 4 dtfm
#   bash baselines/scripts/launch_docker.sh 8 asteroid

set -euo pipefail

NUM_GPUS="${1:-4}"
STRATEGY="${2:-dtfm}"
CONFIG="${3:-baselines/configs/${STRATEGY}_default.yaml}"
shift 3 2>/dev/null || true

IMAGE="${BASELINES_IMAGE:-baselines:latest}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"

mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "  Baselines Docker Distributed Training"
echo "  Image:    ${IMAGE}"
echo "  GPUs:     ${NUM_GPUS}"
echo "  Strategy: ${STRATEGY}"
echo "  Config:   ${CONFIG}"
echo "  Output:   ${OUTPUT_DIR}"
echo "============================================"

docker run --rm \
    --gpus "\"device=0,1,2,3,4,5,6,7\"" \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$(pwd)/${OUTPUT_DIR}:/workspace/output" \
    "${IMAGE}" \
    python -m baselines.train \
        --spawn \
        --num-gpus "${NUM_GPUS}" \
        --strategy "${STRATEGY}" \
        --config "${CONFIG}" \
        --output-dir /workspace/output \
        "$@"
