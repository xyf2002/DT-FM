#!/bin/bash

# Batch device launch script
# Usage: ./start_multiple_devices.sh <num_devices> [interval_seconds] [batch_size]
#
# Examples:
#   ./start_multiple_devices.sh 4           # Start 4 devices with default 5s interval
#   ./start_multiple_devices.sh 4 3         # Start 4 devices with 3s interval
#   ./start_multiple_devices.sh 8 5 2       # Start 8 devices in batches of 2, with 5s interval between batches
#   DEVICE_BACKEND=proxy DEVICE_PROXY_HOST=127.0.0.1:39000 ./start_multiple_devices.sh 6 2 3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

NUM_DEVICES=${1:-${NUM_DEVICES:-4}}           # Default: 4 devices
INTERVAL=${2:-${INTERVAL:-5}}                 # Default: 5 seconds between launches
BATCH_SIZE=${3:-${BATCH_SIZE:-1}}             # Default: 1 device per batch (original sequential behavior)

DEVICE_FLOPS=${DEVICE_FLOPS:-1T}
DEVICE_MEMORY=${DEVICE_MEMORY:-8G}
DEVICE_UL_BW=${DEVICE_UL_BW:-100M}
DEVICE_DL_BW=${DEVICE_DL_BW:-100M}
DEVICE_UL_LAT=${DEVICE_UL_LAT:-10}
DEVICE_DL_LAT=${DEVICE_DL_LAT:-10}
DEVICE_BACKEND=${DEVICE_BACKEND:-proxy}
DEVICE_CFG=${DEVICE_CFG:-"${ROOT_DIR}/config/proxy/cli.ini"}
DEVICE_PROXY_HOST=${DEVICE_PROXY_HOST:-127.0.0.1:39000}
LOG_DIR=${LOG_DIR:-"${ROOT_DIR}/logs"}

# Validate input
if ! [[ "$NUM_DEVICES" =~ ^[0-9]+$ ]] || [ "$NUM_DEVICES" -lt 1 ]; then
    echo "Error: NUM_DEVICES must be a positive integer"
    exit 1
fi

if ! [[ "$INTERVAL" =~ ^[0-9]+$ ]]; then
    echo "Error: INTERVAL must be a non-negative integer"
    exit 1
fi

if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || [ "$BATCH_SIZE" -lt 1 ]; then
    echo "Error: BATCH_SIZE must be a positive integer"
    exit 1
fi

echo "Starting $NUM_DEVICES device(s)..."
echo "Interval: ${INTERVAL}s"
echo "Batch size: $BATCH_SIZE"
echo "Backend: ${DEVICE_BACKEND}"
echo "Config: ${DEVICE_CFG}"
echo "Log dir: ${LOG_DIR}"
echo ""

mkdir -p "${LOG_DIR}"

# Array to store PIDs of launched devices
pids=()

# Launch devices in batches
batch_num=0
for i in $(seq 0 $((NUM_DEVICES - 1))); do
    DEVICE_ID=$i
    current_batch=$((i / BATCH_SIZE))

    # Print batch info at the start of each batch
    if [ $((i % BATCH_SIZE)) -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] Launching batch $((current_batch + 1)) (devices $DEVICE_ID to $((i + BATCH_SIZE - 1)))..."
    fi

    echo "[$(date '+%H:%M:%S')] Launching device $DEVICE_ID..."

    morphling_device \
      --id "$DEVICE_ID" \
      --flops "$DEVICE_FLOPS" \
      --memory "$DEVICE_MEMORY" \
      --ul_bw "$DEVICE_UL_BW" \
      --dl_bw "$DEVICE_DL_BW" \
      --ul_lat "$DEVICE_UL_LAT" \
      --dl_lat "$DEVICE_DL_LAT" \
      --backend "$DEVICE_BACKEND" \
      --cfg "$DEVICE_CFG" \
      --proxy_host "$DEVICE_PROXY_HOST" | tee "${LOG_DIR}/device_${DEVICE_ID}.log" &
    
    # Save the PID
    pids+=($!)

    # Wait before launching the next batch (not after individual devices)
    if [ $((((i + 1) % BATCH_SIZE))) -eq 0 ] && [ $((i + 1)) -lt $NUM_DEVICES ]; then
        echo "Batch complete. Waiting ${INTERVAL}s before launching next batch..."
        sleep "$INTERVAL"
    fi
done

echo ""
echo "All $NUM_DEVICES device(s) launched!"
echo "Device PIDs: ${pids[@]}"
echo ""
echo "To stop all devices, run:"
echo "  kill ${pids[@]}"
echo ""
echo "To monitor devices:"
echo "  ps aux | grep morphling_device"

# Optional: wait for all background processes
# Uncomment the line below if you want the script to wait until all devices exit
# wait "${pids[@]}"
