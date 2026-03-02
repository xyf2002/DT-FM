#!/usr/bin/env bash

# Record perf profile for a target binary.
# Usage: ./scripts/profile.sh [target_binary]
#
# Examples:
#   ./scripts/profile.sh
#   ./scripts/profile.sh ./build/temp.linux-x86_64-cpython-39/bin/tests/echo_server_test

TARGET_BINARY=${1:-./build/temp.linux-x86_64-cpython-39/bin/tests/echo_server_test}

sudo perf record -e cycles -e sched:sched_switch --switch-events \
    --sample-cpu -g \
    -m 8M --aio \
    --call-graph dwarf \
    "${TARGET_BINARY}"
