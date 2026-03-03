#!/usr/bin/env bash
# ============================================================================
# Asteroid Training Monitor — Persistent Streaming Dashboard
# ============================================================================
# Outputs append-only — no screen clearing, no flickering.
# Each poll prints only NEW training iterations since the last check.
# A full status banner is printed once at startup and again on state changes.
#
# Usage:
#   ./monitor.sh                 # Persistent streaming dashboard
#   ./monitor.sh --once          # Single snapshot (same look, exits)
#   ./monitor.sh --logs          # Stream raw logs from all pods
#   ./monitor.sh --rank 0        # Follow logs for a specific rank
#   ./monitor.sh --interval 10   # Poll every 10s (default: 5)
#   ./monitor.sh --tensorboard   # Launch TensorBoard
# ============================================================================

set -euo pipefail

export KUBECONFIG="${KUBECONFIG:-${HOME}/.kube/config}"

# ── Defaults ────────────────────────────────────────────────────────────────
MODE="dashboard"
POLL_INTERVAL=5
TARGET_RANK=""
NAMESPACE="default"
LABEL="app=asteroid"

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ── State tracking (for persistent mode) ────────────────────────────────────
_LAST_SEEN_ITER=-1        # highest iter we already printed
_PREV_POD_STATUS=""       # serialised pod statuses from last poll
_HEADER_PRINTED=false     # whether the startup banner was shown
_STARTED_AT=""            # monitor start time

# ============================================================================
# Argument Parsing
# ============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --once)       MODE="once"; shift ;;
        --logs)       MODE="logs"; shift ;;
        --rank)       MODE="rank"; TARGET_RANK="$2"; shift 2 ;;
        --interval)   POLL_INTERVAL="$2"; shift 2 ;;
        --watch)      POLL_INTERVAL="$2"; shift 2 ;;   # compat alias
        --tensorboard|--tb)  MODE="tensorboard"; shift ;;
        -h|--help)
            echo "Usage: $(basename "$0") [OPTIONS]"
            echo ""
            echo "Modes:"
            echo "  (default)       Persistent streaming dashboard (append-only)"
            echo "  --once          Print single status snapshot and exit"
            echo "  --logs          Stream raw kubectl logs from all pods"
            echo "  --rank N        Follow logs for rank N only"
            echo "  --tensorboard   Launch TensorBoard"
            echo ""
            echo "Options:"
            echo "  --interval N    Poll every N seconds (default: 5)"
            exit 0
            ;;
        *)  echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Helpers
# ============================================================================

get_pods() {
    kubectl get pods -l "${LABEL}" -n "${NAMESPACE}" --no-headers 2>/dev/null
}

get_pod_for_rank() {
    kubectl get pods -l "${LABEL},rank=${1}" -n "${NAMESPACE}" \
        --no-headers -o custom-columns=NAME:.metadata.name 2>/dev/null | head -1
}

# Return the last-rank pod name (the one that prints loss)
get_last_rank_pod() {
    get_pods | sort | tail -1 | awk '{print $1}'
}

# ── Separator lines ────────────────────────────────────────────────────────
sep()     { echo -e "${CYAN}  ──────────────────────────────────────────────────────────────────${NC}"; }
thin_sep(){ echo -e "${DIM}  ··································································${NC}"; }

# ============================================================================
# Banner — printed once at start and when pod state changes
# ============================================================================
print_banner() {
    local now
    now=$(date '+%Y-%m-%d %H:%M:%S')

    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${BOLD}               ASTEROID TRAINING MONITOR                             ${NC}${CYAN}║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
    echo -e "${DIM}  Started ${now}   •   polling every ${POLL_INTERVAL}s   •   Ctrl-C to stop${NC}"
    echo ""

    # ── Cluster ──────────────────────────────────────────────────────────
    echo -e "${BOLD}  CLUSTER${NC}"
    sep
    local nodes total_nodes ready_nodes
    nodes=$(kubectl get nodes --no-headers 2>/dev/null || true)
    total_nodes=$(echo "$nodes" | grep -c . || echo 0)
    ready_nodes=$(echo "$nodes" | grep -c " Ready" || echo 0)

    if [[ "$ready_nodes" -eq "$total_nodes" ]]; then
        echo -e "  Nodes:  ${GREEN}${ready_nodes}/${total_nodes} Ready${NC}"
    else
        echo -e "  Nodes:  ${YELLOW}${ready_nodes}/${total_nodes} Ready${NC}"
    fi

    # GPU column
    local gpu_total=0
    while IFS= read -r n; do
        local nname ngpu
        nname=$(echo "$n" | awk '{print $1}')
        ngpu=$(kubectl get node "$nname" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}' 2>/dev/null || echo 0)
        [[ -z "$ngpu" ]] && ngpu=0
        echo -e "  GPU:    ${nname}  ×${ngpu}"
        gpu_total=$((gpu_total + ngpu))
    done <<< "$nodes"
    echo -e "  Total:  ${BOLD}${gpu_total} GPU(s)${NC}"
    echo ""

    # ── Pod table ────────────────────────────────────────────────────────
    echo -e "${BOLD}  TRAINING PODS${NC}"
    sep

    local pods
    pods=$(get_pods)

    if [[ -z "$pods" ]]; then
        echo -e "  ${YELLOW}No training pods found.${NC}  Run: ${CYAN}./deploy.sh --redeploy${NC}"
        echo ""
        return 1
    fi

    printf "  ${DIM}%-32s %-10s %-10s %-8s %-20s${NC}\n" "POD" "STATUS" "RESTARTS" "AGE" "NODE"
    thin_sep
    while IFS= read -r line; do
        local pname pstatus prestarts page pnode
        pname=$(echo "$line" | awk '{print $1}')
        pstatus=$(echo "$line" | awk '{print $3}')
        prestarts=$(echo "$line" | awk '{print $4}')
        page=$(echo "$line" | awk '{print $5}')
        pnode=$(kubectl get pod "$pname" -n "${NAMESPACE}" -o jsonpath='{.spec.nodeName}' 2>/dev/null || echo "—")

        local sc="${GREEN}"
        [[ "$pstatus" != "Running" ]] && sc="${YELLOW}"
        [[ "$pstatus" == "Error" || "$pstatus" == "CrashLoopBackOff" ]] && sc="${RED}"

        local rc="${NC}"
        [[ "$prestarts" -gt 0 ]] && rc="${YELLOW}"
        [[ "$prestarts" -gt 3 ]] && rc="${RED}"

        printf "  %-32s ${sc}%-10s${NC} ${rc}%-10s${NC} %-8s %-20s\n" \
            "$pname" "$pstatus" "$prestarts" "$page" "$pnode"
    done <<< "$pods"
    echo ""

    # ── Per-rank init status ─────────────────────────────────────────────
    echo -e "${BOLD}  PER-RANK STATUS${NC}"
    sep
    while IFS= read -r line; do
        local pod_name
        pod_name=$(echo "$line" | awk '{print $1}')
        local rank
        rank=$(echo "$pod_name" | grep -oP 'rank-\K\d+' || echo "?")
        local last_log
        last_log=$(kubectl logs "$pod_name" --tail=5 -n "${NAMESPACE}" 2>/dev/null \
            | grep -E "RANK|Ready|NCCL|distributed|Error|iter" | tail -1 || echo "(no output)")
        echo -e "  Rank ${BOLD}${rank}${NC}: ${DIM}${last_log}${NC}"
    done <<< "$pods"
    echo ""

    _HEADER_PRINTED=true
}

# ============================================================================
# Streaming iteration rows — only NEW iterations since last poll
# ============================================================================
print_training_header() {
    echo -e "${BOLD}  TRAINING STREAM${NC}"
    sep
    printf "  ${DIM}%-8s  %-10s  %-12s  %-14s  %-12s  %-20s${NC}\n" \
        "TIME" "ITER" "LOSS" "LR" "TOK/S" "STEP TIME"
    thin_sep
}

# Fetch iter lines from the last-rank pod that are newer than _LAST_SEEN_ITER.
# On first call, shows only the last HISTORY_ROWS rows to avoid dumping
# hundreds of historical lines. After that, streams only new rows.
_FIRST_POLL=true
HISTORY_ROWS=5

poll_new_iters() {
    local pod="$1"
    [[ -z "$pod" ]] && return

    # Grab enough tail to cover several poll intervals
    local raw
    raw=$(kubectl logs "$pod" --tail=200 -n "${NAMESPACE}" 2>/dev/null \
        | grep -E '^\s*iter\s+' || true)
    [[ -z "$raw" ]] && return

    # On first poll, fast-forward _LAST_SEEN_ITER so we only show the
    # tail HISTORY_ROWS of previously-logged iterations.
    if [[ "$_FIRST_POLL" == "true" ]]; then
        _FIRST_POLL=false
        local total_lines skip_lines
        total_lines=$(echo "$raw" | wc -l | tr -d '[:space:]')
        skip_lines=$((total_lines - HISTORY_ROWS))
        if (( skip_lines > 0 )); then
            local cutoff_line
            cutoff_line=$(echo "$raw" | head -n "$skip_lines" | tail -1)
            local cutoff_iter
            cutoff_iter=$(echo "$cutoff_line" | grep -oP 'iter\s+\K\d+' || true)
            if [[ -n "$cutoff_iter" ]]; then
                _LAST_SEEN_ITER=$cutoff_iter
            fi
        fi
    fi

    local now
    now=$(date '+%H:%M:%S')

    while IFS= read -r line; do
        local it
        it=$(echo "$line" | grep -oP 'iter\s+\K\d+' || true)
        [[ -z "$it" ]] && continue
        (( it <= _LAST_SEEN_ITER )) && continue

        # Parse fields
        local loss lr toks dt
        loss=$(echo "$line" | grep -oP 'loss=\K[0-9.]+' || echo "---")
        lr=$(echo "$line"   | grep -oP 'lr=\K[0-9.e+-]+' || echo "---")
        toks=$(echo "$line" | grep -oP '[0-9,]+ tok/s' || echo "---")
        dt=$(echo "$line"   | grep -oP 'dt=\K[0-9.]+ms' || echo "---")

        # Color loss: green < 0.69, yellow < 0.72, red >= 0.72
        local lc="${YELLOW}"
        if [[ "$loss" != "---" ]]; then
            lc="${RED}"
            if (( $(echo "$loss < 0.69" | bc -l 2>/dev/null || echo 0) )); then
                lc="${GREEN}"
            elif (( $(echo "$loss < 0.72" | bc -l 2>/dev/null || echo 0) )); then
                lc="${YELLOW}"
            fi
        fi

        printf "  ${DIM}%-8s${NC}  ${BOLD}%-10s${NC}  ${lc}%-12s${NC}  %-14s  %-14s  %-12s\n" \
            "$now" "$it" "$loss" "$lr" "$toks" "$dt"

        _LAST_SEEN_ITER=$it
    done <<< "$raw"
}

# ============================================================================
# Detect pod state changes (status/restarts/count) and reprint banner
# ============================================================================
check_pod_state_change() {
    local current
    current=$(get_pods | awk '{print $1,$3,$4}' | sort | tr '\n' '|')
    if [[ "$current" != "$_PREV_POD_STATUS" ]]; then
        _PREV_POD_STATUS="$current"
        return 0   # changed
    fi
    return 1       # no change
}

# ============================================================================
# Completion detection
# ============================================================================
check_training_complete() {
    local jobs_output
    jobs_output=$(kubectl get jobs -l "${LABEL}" -n "${NAMESPACE}" --no-headers 2>/dev/null || true)
    [[ -z "$jobs_output" ]] && return 1

    local jobs_done jobs_total
    jobs_done=$(echo "$jobs_output"  | awk '{print $2}' | grep -c "1/1" 2>/dev/null || true)
    jobs_total=$(echo "$jobs_output" | wc -l 2>/dev/null || true)
    jobs_done=$(echo "$jobs_done"  | tr -d '[:space:]')
    jobs_total=$(echo "$jobs_total" | tr -d '[:space:]')
    [[ -z "$jobs_done" ]]  && jobs_done=0
    [[ -z "$jobs_total" ]] && jobs_total=0

    if [[ "$jobs_total" -gt 0 ]] && [[ "$jobs_done" -eq "$jobs_total" ]]; then
        return 0  # all complete
    fi
    return 1
}

# ============================================================================
# Mode: persistent dashboard (default)
# ============================================================================
mode_dashboard() {
    trap 'echo ""; echo -e "${DIM}  Monitor stopped.${NC}"; exit 0' INT
    _STARTED_AT=$(date '+%Y-%m-%d %H:%M:%S')

    # Initial banner
    print_banner || { echo "  Waiting for pods..."; }

    # Training stream header
    print_training_header
    _PREV_POD_STATUS=$(get_pods | awk '{print $1,$3,$4}' | sort | tr '\n' '|')

    while true; do
        # If pod states changed, print a state-change notice (not a full clear)
        if check_pod_state_change; then
            echo ""
            echo -e "  ${YELLOW}⚡ Pod state change detected${NC}  $(date '+%H:%M:%S')"
            sep
            local pods
            pods=$(get_pods)
            if [[ -n "$pods" ]]; then
                while IFS= read -r line; do
                    local pn ps pr
                    pn=$(echo "$line" | awk '{print $1}')
                    ps=$(echo "$line" | awk '{print $3}')
                    pr=$(echo "$line" | awk '{print $4}')
                    local sc="${GREEN}"
                    [[ "$ps" != "Running" ]] && sc="${YELLOW}"
                    [[ "$ps" == "Error" || "$ps" == "CrashLoopBackOff" ]] && sc="${RED}"
                    echo -e "  ${pn}  ${sc}${ps}${NC}  restarts: ${pr}"
                done <<< "$pods"
            else
                echo -e "  ${RED}No pods found${NC}"
            fi
            thin_sep
        fi

        # Stream new iterations
        local last_pod
        last_pod=$(get_last_rank_pod)
        poll_new_iters "$last_pod"

        # Check completion
        if check_training_complete; then
            echo ""
            echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
            echo -e "${GREEN}║${BOLD}               TRAINING COMPLETE                                    ${NC}${GREEN}║${NC}"
            echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════╝${NC}"

            # Print final summary
            echo ""
            echo -e "${BOLD}  FINAL RESULTS${NC}"
            sep
            local final_line
            final_line=$(kubectl logs "$last_pod" --tail=200 -n "${NAMESPACE}" 2>/dev/null \
                | grep -E '^\s*iter\s+' | tail -1 || true)
            if [[ -n "$final_line" ]]; then
                local f_iter f_loss f_toks
                f_iter=$(echo "$final_line" | grep -oP 'iter\s+\K\d+' || echo "?")
                f_loss=$(echo "$final_line" | grep -oP 'loss=\K[0-9.]+' || echo "?")
                f_toks=$(echo "$final_line" | grep -oP '[0-9,]+ tok/s' || echo "?")
                echo -e "  Final iteration:  ${BOLD}${f_iter}${NC}"
                echo -e "  Final loss:       ${BOLD}${f_loss}${NC}"
                echo -e "  Throughput:       ${f_toks}"
            fi
            echo -e "  Started:  ${_STARTED_AT}"
            echo -e "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
            echo ""
            exit 0
        fi

        sleep "${POLL_INTERVAL}"
    done
}

# ============================================================================
# Mode: single snapshot (--once)
# ============================================================================
mode_once() {
    print_banner || true

    echo -e "${BOLD}  TRAINING PROGRESS${NC}"
    sep

    local last_pod
    last_pod=$(get_last_rank_pod)
    if [[ -z "$last_pod" ]]; then
        echo -e "  ${YELLOW}No training pods found${NC}"
        return
    fi

    local metrics
    metrics=$(kubectl logs "$last_pod" --tail=500 -n "${NAMESPACE}" 2>/dev/null \
        | grep -E '^\s*iter\s+' || true)

    if [[ -z "$metrics" ]]; then
        echo -e "  ${YELLOW}No training iterations yet${NC}"
        local init_log
        init_log=$(kubectl logs "$last_pod" --tail=5 -n "${NAMESPACE}" 2>/dev/null || true)
        echo -e "  ${DIM}${init_log}${NC}"
    else
        local latest
        latest=$(echo "$metrics" | tail -1)
        local iter loss lr toks dt
        iter=$(echo "$latest" | grep -oP 'iter\s+\K\d+' || echo "?")
        loss=$(echo "$latest" | grep -oP 'loss=\K[0-9.]+' || echo "?")
        lr=$(echo "$latest"   | grep -oP 'lr=\K[0-9.e+-]+' || echo "?")
        toks=$(echo "$latest" | grep -oP '[0-9,]+ tok/s' || echo "?")
        dt=$(echo "$latest"   | grep -oP 'dt=\K[0-9.]+ms' || echo "?")

        echo -e "  Iteration:     ${BOLD}${iter}${NC}"
        echo -e "  Loss:          ${BOLD}${loss}${NC}"
        echo -e "  Learning Rate: ${lr}"
        echo -e "  Throughput:    ${toks}"
        echo -e "  Step Time:     ${dt}"
        echo ""
        echo -e "  ${DIM}Recent:${NC}"
        echo "$metrics" | tail -10 | while IFS= read -r line; do
            echo -e "  ${DIM}  ${line}${NC}"
        done
    fi
    echo ""
}

# ============================================================================
# Mode: stream raw logs (--logs)
# ============================================================================
mode_logs() {
    echo -e "${CYAN}Streaming logs from all training pods...${NC}"
    echo -e "${DIM}Press Ctrl+C to stop${NC}"
    echo ""

    local pods
    pods=$(get_pods | awk '{print $1}')

    if [[ -z "$pods" ]]; then
        echo "No training pods found"
        exit 1
    fi

    kubectl logs -l "${LABEL}" -n "${NAMESPACE}" --follow --prefix --tail=20 2>/dev/null || {
        local last_pod
        last_pod=$(echo "$pods" | tail -1)
        echo "Following logs from ${last_pod}..."
        kubectl logs -f "${last_pod}" -n "${NAMESPACE}" --tail=50
    }
}

# ============================================================================
# Mode: follow single rank (--rank N)
# ============================================================================
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

# ============================================================================
# Mode: TensorBoard (--tensorboard)
# ============================================================================
mode_tensorboard() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    echo -e "${CYAN}Launching TensorBoard persistent dashboard...${NC}"
    echo ""
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
