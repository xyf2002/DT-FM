#!/bin/bash

# Server launch script
# Usage: ./server.sh [backend] [model_name]
#
# Examples:
#   ./server.sh
#   ./server.sh proxy facebook/opt-125m
#   SERVER_BACKEND=proxy SERVER_MODEL=facebook/opt-125m ./server.sh

SERVER_BACKEND=${1:-${SERVER_BACKEND:-proxy}}
SERVER_MODEL=${2:-${SERVER_MODEL:-facebook/opt-125m}}
SERVER_CFG=${SERVER_CFG:-}
SERVER_ENABLE_HOOKS=${SERVER_ENABLE_HOOKS:-1}
SERVER_ENABLE_VERIFICATION=${SERVER_ENABLE_VERIFICATION:-1}

CMD=(python3 scripts/run_server.py --backend "${SERVER_BACKEND}" --model_name "${SERVER_MODEL}")

if [ -n "${SERVER_CFG}" ]; then
  CMD+=(--cfg "${SERVER_CFG}")
fi

if [ "${SERVER_ENABLE_HOOKS}" -eq 1 ]; then
  CMD+=(--enable-hooks)
fi

if [ "${SERVER_ENABLE_VERIFICATION}" -eq 1 ]; then
  CMD+=(--enable-verification)
fi

"${CMD[@]}"
