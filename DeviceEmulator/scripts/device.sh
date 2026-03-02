#!/bin/bash

# Device launch script with device ID argument
# Usage: ./device.sh <device_id>
#
# Examples:
#   ./device.sh 0
#   DEVICE_FLOPS=2T DEVICE_MEMORY=16G ./device.sh 1
#   DEVICE_BACKEND=proxy DEVICE_CFG=./config/proxy/cli.ini \
#     DEVICE_PROXY_HOST=127.0.0.1:39000 ./device.sh 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEVICE_ID=${1:-${DEVICE_ID:-0}}  # Default to device 0 if not provided
DEVICE_FLOPS=${DEVICE_FLOPS:-1T}
DEVICE_MEMORY=${DEVICE_MEMORY:-8G}
DEVICE_UL_BW=${DEVICE_UL_BW:-100M}
DEVICE_DL_BW=${DEVICE_DL_BW:-100M}
DEVICE_UL_LAT=${DEVICE_UL_LAT:-10}
DEVICE_DL_LAT=${DEVICE_DL_LAT:-10}
DEVICE_BACKEND=${DEVICE_BACKEND:-proxy}
DEVICE_CFG=${DEVICE_CFG:-"${ROOT_DIR}/config/proxy/cli.ini"}
DEVICE_PROXY_HOST=${DEVICE_PROXY_HOST:-127.0.0.1:39000}

CMD=(
  morphling_device
  --id "$DEVICE_ID"
  --flops "$DEVICE_FLOPS"
  --memory "$DEVICE_MEMORY"
  --ul_bw "$DEVICE_UL_BW"
  --dl_bw "$DEVICE_DL_BW"
  --ul_lat "$DEVICE_UL_LAT"
  --dl_lat "$DEVICE_DL_LAT"
  --backend "$DEVICE_BACKEND"
  --cfg "$DEVICE_CFG"
)

if [ -n "$DEVICE_PROXY_HOST" ]; then
  CMD+=(--proxy_host "$DEVICE_PROXY_HOST")
fi

"${CMD[@]}"
