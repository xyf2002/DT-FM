#!/usr/bin/env bash

# Launch a single device with explicit arguments.
# Usage: ./run_device.sh <id> <flops> <memory> <ul_bw> <dl_bw> <ul_lat> <dl_lat> <backend> [proxy_host]
#
# Examples:
#   ./run_device.sh 0 1T 8G 100M 100M 10 10 proxy
#   ./run_device.sh 1 2T 16G 200M 200M 5 5 proxy 127.0.0.1:39000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ID=$1
FLOPS=$2
MEMORY=$3
UL_BW=$4
DL_BW=$5
UL_LAT=$6
DL_LAT=$7
BACKEND=$8

# Redis host (host:port)
# REDIS_HOST=${9:-127.0.0.1:6379}

# Proxy host (host:port) - optional, will override config file
PROXY_HOST=${9:-}

CFG_PATH=${CFG_PATH:-"${ROOT_DIR}/config/proxy/cli.ini"}

# Build the command
CMD="morphling_device --id $ID --flops $FLOPS --memory $MEMORY --ul_bw $UL_BW --dl_bw $DL_BW --ul_lat $UL_LAT --dl_lat $DL_LAT --backend $BACKEND --cfg $CFG_PATH"

# Add proxy_host parameter if provided
if [ -n "$PROXY_HOST" ]; then
    CMD="$CMD --proxy_host $PROXY_HOST"
fi

# Execute the command in background
eval $CMD &
