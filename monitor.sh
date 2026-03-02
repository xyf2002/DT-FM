#!/usr/bin/env bash
# ============================================================================
# Asteroid Training Monitor
# ============================================================================
# Real-time dashboard for monitoring distributed training progress.
#
# Usage:
#   ./monitor.sh                 # Full dashboard (auto-refresh)
#   ./monitor.sh --once          # Single snapshot
#   ./monitor.sh --logs          # Stream raw logs from all pods
#   ./monitor.sh --rank 0        # Follow logs for a specific rank
#   ./monitor.sh --watch 10      # Refresh every 10 seconds (default: 5)
# ============================================================================

set -euo pipefail

export KUBECONFIG="${KUBECONFIG:-${HOME}/.kube/config}"

# Defaults
MODE="dashboard"
WATCH_INTERVAL=5
TARGET_RANK=""
NAMESPACE="default"
LABEL="app=asteroid"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ============================================================================
# Argument Parsing
# ============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --once)     MODE="once"; shift ;;
        --logs)     MODE="logs"; shift ;;
        --rank)     MODE="rank"; TARGET_RANK="$2"; shift 2 ;;
        --watch)    WATCH_INTERVAL="$2"; shift 2 ;;
        --tensorboard|--tb)  MODE="tensorboard"; shift ;;
        -h|--help)
            echo "Usage: $(basename "$0") [--once|--logs|--rank N|--watch SECS|--tensorboard]"
            echo ""
            echo "Modes:"
            echo "  (default)    Auto-refreshing dashboard"
            echo "  --once       Print single status snapshot"
            echo "  --logs       Stream raw logs from all training pods"
            echo "  --rank N     Follow logs for rank N only"
            echo "  --watch N    Set refresh interval in seconds (default: 5)"
            echo "  --tensorboard  Launch persistent TensorBoard dashboard"
            exit 0
            ;;
        *)          echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Functions
# ============================================================================

get_pods() {
    kubectl get pods -l "${LABEL}" -n "${NAMESPACE}" --no-headers 2>/dev/null
}

get_pod_for_rank() {
    local rank="$1"
    kubectl get pods -l "${LABEL},rank=${rank}" -n "${NAMESPACE}" --no-headers -o custom-columns=NAME:.metadata.name 2>/dev/null | head -1
}

extract_training_metrics() {
    # Parse training log lines like: iter   30 | loss=0.6866 | lr=1.86e-04 | 5,436 tok/s | dt=6028.4ms
    local pod="$1"
    kubectl logs "${pod}" --tail=100 -n "${NAMESPACE}" 2>/dev/null | grep -E "^\s*iter\s+" | tail -1
}

extract_all_training_metrics() {
    local pod="$1"
    kubectl logs "${pod}" --tail=500 -n "${NAMESPACE}" 2>/dev/null | grep -E "^\s*iter\s+"
}

render_dashboard() {
    clear
    local now
    now=$(date '+%Y-%m-%d %H:%M:%S')

    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${BOLD}              ASTEROID TRAINING MONITOR                              ${NC}${CYAN}║${NC}"
    echo -e "${CYAN}║${DIM}              ${now}                              ${NC}${CYAN}║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # ---- Cluster Status ----
    echo -e "${BOLD}  CLUSTER${NC}"
    echo -e "  ─────────────────────────────────────────────────────"
    local nodes
    nodes=$(kubectl get nodes --no-headers 2>/dev/null)
    local total_nodes ready_nodes
    total_nodes=$(echo "$nodes" | wc -l)
    ready_nodes=$(echo "$nodes" | grep -c " Ready" || true)
    
    if [[ "$ready_nodes" -eq "$total_nodes" ]]; then
        echo -e "  Nodes:  ${GREEN}${ready_nodes}/${total_nodes} Ready${NC}"
    else
        echo -e "  Nodes:  ${YELLOW}${ready_nodes}/${total_nodes} Ready${NC}"
    fi

    local gpu_info
    gpu_info=$(kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}={.status.allocatable.nvidia\.com/gpu}{" "}{end}' 2>/dev/null || true)
    echo -e "  GPUs:   ${gpu_info}"
    echo ""

    # ---- Pod Status ----
    echo -e "${BOLD}  TRAINING PODS${NC}"
    echo -e "  ─────────────────────────────────────────────────────"
    
    local pods
    pods=$(get_pods)
    
    if [[ -z "$pods" ]]; then
        echo -e "  ${YELLOW}No training pods found${NC}"
        echo ""
        echo -e "  Start training: ./deploy.sh --redeploy"
        return
    fi

    while IFS= read -r line; do
        local pod_name status restarts age
        pod_name=$(echo "$line" | awk '{print $1}')
        status=$(echo "$line" | awk '{print $3}')
        restarts=$(echo "$line" | awk '{print $4}')
        age=$(echo "$line" | awk '{print $5}')

        local status_color="${GREEN}"
        [[ "$status" != "Running" ]] && status_color="${YELLOW}"
        [[ "$status" == "Error" || "$status" == "CrashLoopBackOff" ]] && status_color="${RED}"

        local restart_color="${NC}"
        [[ "$restarts" -gt 0 ]] && restart_color="${YELLOW}"
        [[ "$restarts" -gt 3 ]] && restart_color="${RED}"

        printf "  %-35s ${status_color}%-12s${NC} restarts: ${restart_color}%s${NC}  age: %s\n" \
            "$pod_name" "$status" "$restarts" "$age"
    done <<< "$pods"
    echo ""

    # ---- Training Progress ----
    echo -e "${BOLD}  TRAINING PROGRESS${NC}"
    echo -e "  ─────────────────────────────────────────────────────"

    # Find the last-rank pod (it typically prints the training metrics)
    local last_pod
    last_pod=$(echo "$pods" | sort | tail -1 | awk '{print $1}')

    local latest_metric
    latest_metric=$(extract_training_metrics "$last_pod")

    if [[ -n "$latest_metric" ]]; then
        # Parse the metric line
        local iter loss lr toks dt
        iter=$(echo "$latest_metric" | grep -oP 'iter\s+\K\d+' || echo "?")
        loss=$(echo "$latest_metric" | grep -oP 'loss=\K[0-9.]+' || echo "?")
        lr=$(echo "$latest_metric" | grep -oP 'lr=\K[0-9.e+-]+' || echo "?")
        toks=$(echo "$latest_metric" | grep -oP '[\d,]+ tok/s' || echo "?")
        dt=$(echo "$latest_metric" | grep -oP 'dt=\K[0-9.]+ms' || echo "?")

        echo -e "  Iteration:   ${BOLD}${iter}${NC}"
        echo -e "  Loss:        ${BOLD}${loss}${NC}"
        echo -e "  Learning Rate: ${lr}"
        echo -e "  Throughput:  ${toks}"
        echo -e "  Step Time:   ${dt}"

        # Show recent progress (last 5 iterations)
        echo ""
        echo -e "  ${DIM}Recent iterations:${NC}"
        extract_all_training_metrics "$last_pod" | tail -5 | while IFS= read -r line; do
            echo -e "  ${DIM}  ${line}${NC}"
        done
    else
        # Check initialization status
        local init_status
        init_status=$(kubectl logs "$last_pod" --tail=10 -n "${NAMESPACE}" 2>/dev/null | tail -3)
        echo -e "  ${YELLOW}Training not producing metrics yet${NC}"
        echo -e "  ${DIM}Latest log output:${NC}"
        echo "$init_status" | while IFS= read -r line; do
            echo -e "    ${DIM}${line}${NC}"
        done
    fi
    echo ""

    # ---- Per-Rank Status ----
    echo -e "${BOLD}  PER-RANK STATUS${NC}"
    echo -e "  ─────────────────────────────────────────────────────"
    while IFS= read -r line; do
        local pod_name status
        pod_name=$(echo "$line" | awk '{print $1}')
        status=$(echo "$line" | awk '{print $3}')

        # Extract rank from pod name
        local rank
        rank=$(echo "$pod_name" | grep -oP 'rank-\K\d+' || echo "?")

        # Get last meaningful log line
        local last_log
        last_log=$(kubectl logs "$pod_name" --tail=5 -n "${NAMESPACE}" 2>/dev/null | grep -E "RANK|iter|NCCL|Ready|Error|distributed" | tail -1 || echo "(no output)")
        
        printf "  Rank %s: %s\n" "$rank" "$last_log"
    done <<< "$pods"
    echo ""

    echo -e "${DIM}  Refreshing every ${WATCH_INTERVAL}s | Ctrl+C to exit | ./monitor.sh --logs for raw output${NC}"
}

mode_once() {
    render_dashboard
}

mode_dashboard() {
    trap 'echo ""; echo "Monitor stopped."; exit 0' INT
    while true; do
        render_dashboard
        sleep "${WATCH_INTERVAL}"
    done
}

mode_logs() {
    echo -e "${CYAN}Streaming logs from all training pods...${NC}"
    echo -e "${DIM}Press Ctrl+C to stop${NC}"
    echo ""

    # Use kubectl logs with --follow for all pods
    local pods
    pods=$(get_pods | awk '{print $1}')

    if [[ -z "$pods" ]]; then
        echo "No training pods found"
        exit 1
    fi

    # Stream logs from all pods with prefix
    kubectl logs -l "${LABEL}" -n "${NAMESPACE}" --follow --prefix --tail=20 2>/dev/null || {
        # Fallback: stream from last pod only
        local last_pod
        last_pod=$(echo "$pods" | tail -1)
        echo "Following logs from ${last_pod}..."
        kubectl logs -f "${last_pod}" -n "${NAMESPACE}" --tail=50
    }
}

mode_rank() {
    local pod
    pod=$(get_pod_for_rank "${TARGET_RANK}")

    if [[ -z "$pod" ]]; then
        echo "No pod found for rank ${TARGET_RANK}"
        echo "Available pods:"
        get_pods
        exit 1
    fi

    echo -e "${CYAN}Following logs for Rank ${TARGET_RANK} (${pod})...${NC}"
    echo ""
    kubectl logs -f "${pod}" -n "${NAMESPACE}" --tail=50
}

mode_tensorboard() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    echo -e "${CYAN}Launching TensorBoard persistent dashboard...${NC}"
    echo ""

    # Delegate to deploy.sh phase
    "${script_dir}/deploy.sh" --phase tensorboard
}

# ============================================================================
# Main
# ============================================================================
case "$MODE" in
    dashboard)    mode_dashboard ;;
    once)         mode_once ;;
    logs)         mode_logs ;;
    rank)         mode_rank ;;
    tensorboard)  mode_tensorboard ;;
esac
