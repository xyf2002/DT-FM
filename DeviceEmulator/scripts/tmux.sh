#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME=${SESSION_NAME:-device_emulator}
ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}

# Usage: ./scripts/tmux.sh
#
# Examples:
#   ./scripts/tmux.sh
#   SESSION_NAME=my_session START_DEVICES_CMD="./scripts/start_multiple_devices.sh 8 0" ./scripts/tmux.sh

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' already exists."
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  exit 0
fi

SERVER_CMD=${SERVER_CMD:-"./scripts/server.sh"}
START_DEVICES_CMD=${START_DEVICES_CMD:-"./scripts/start_multiple_devices.sh 100 0"}
START_DEVICES_DELAY=${START_DEVICES_DELAY:-20s}

tmux new-session -d -s "${SESSION_NAME}" -n server "cd ${ROOT_DIR} && ${SERVER_CMD}"
tmux new-window -t "${SESSION_NAME}" -n devices "cd ${ROOT_DIR} && sleep ${START_DEVICES_DELAY} & ${START_DEVICES_CMD}"

tmux select-window -t "${SESSION_NAME}:server"
tmux attach -t "${SESSION_NAME}"
