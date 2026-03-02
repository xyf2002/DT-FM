#!/usr/bin/env bash
# ============================================================================
# Asteroid - Full Deployment Script
# ============================================================================
# Single entry-point to deploy Asteroid distributed training on a K3s cluster.
#
# Usage:
#   ./deploy.sh                  # Full deployment (all phases)
#   ./deploy.sh --phase gpu      # Run only GPU setup phase
#   ./deploy.sh --skip-build     # Skip Docker image rebuild
#   ./deploy.sh --skip-profile   # Use existing profiles
#   ./deploy.sh --redeploy       # Just redeploy K8s jobs (fastest)
#   ./deploy.sh --monitor        # Deploy + open live monitor
#
# Prerequisites:
#   1. deploy/inventory.ini configured with cluster nodes
#   2. deploy/secrets.yml encrypted with ansible-vault (see README)
#   3. ~/.asteroid_vault_pass contains vault password
#   4. Python venv at .venv/ with project dependencies
#
# Environment Variables:
#   IMAGE_TAG     - Docker image tag (default: latest)
#   NUM_STAGES    - Number of pipeline stages (default: auto from planner)
#   KUBECONFIG    - Path to kubeconfig (default: ~/.kube/config)
# ============================================================================

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="${SCRIPT_DIR}/deploy"
VENV_DIR="${SCRIPT_DIR}/.venv"
PROFILES_DIR="${SCRIPT_DIR}/profiles"
GENERATED_DIR="${DEPLOY_DIR}/generated"
PLAN_FILE="${SCRIPT_DIR}/hpp_plan.json"

IMAGE_NAME="asteroid"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_FULL="${IMAGE_NAME}:${IMAGE_TAG}"

export KUBECONFIG="${KUBECONFIG:-${HOME}/.kube/config}"
export ANSIBLE_CONFIG="${DEPLOY_DIR}/ansible.cfg"
ANSIBLE_INVENTORY="${DEPLOY_DIR}/inventory.ini"
ANSIBLE_SECRETS="${DEPLOY_DIR}/secrets.yml"

# Ansible flags that use vault-encrypted secrets (no manual -k/-K needed)
ANSIBLE_COMMON_FLAGS="-i ${ANSIBLE_INVENTORY} -e @${ANSIBLE_SECRETS}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Phase control
SKIP_K3S=false
SKIP_GPU=false
SKIP_PROFILE=false
SKIP_BUILD=false
SKIP_PLAN=false
SKIP_MANIFESTS=false
SKIP_DEPLOY=false
REDEPLOY_ONLY=false
RUN_MONITOR=false
SINGLE_PHASE=""

# ============================================================================
# Helpers
# ============================================================================
log()    { echo -e "${GREEN}[✓]${NC} $*"; }
info()   { echo -e "${BLUE}[ℹ]${NC} $*"; }
warn()   { echo -e "${YELLOW}[!]${NC} $*"; }
err()    { echo -e "${RED}[✗]${NC} $*" >&2; }
header() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN} $*${NC}"
    echo -e "${CYAN}========================================${NC}"
}

check_prereqs() {
    local ok=true

    # Python venv
    if [[ ! -f "${VENV_DIR}/bin/python" ]]; then
        err "Python venv not found at ${VENV_DIR}"
        err "Run: python3 -m venv .venv && source .venv/bin/activate && pip install -e ."
        ok=false
    fi

    # Ansible
    if ! command -v ansible-playbook &>/dev/null && ! "${VENV_DIR}/bin/ansible-playbook" --version &>/dev/null 2>&1; then
        err "ansible-playbook not found. Install: pip install ansible"
        ok=false
    fi

    # Docker
    if ! command -v docker &>/dev/null; then
        err "docker not found. Install Docker first."
        ok=false
    fi

    # kubectl
    if ! command -v kubectl &>/dev/null; then
        warn "kubectl not found locally — will rely on K3s nodes"
    fi

    # Vault password file
    if [[ ! -f "${HOME}/.asteroid_vault_pass" ]]; then
        err "Vault password file not found: ~/.asteroid_vault_pass"
        err "Create it: echo 'your-vault-password' > ~/.asteroid_vault_pass && chmod 600 ~/.asteroid_vault_pass"
        ok=false
    fi

    # Secrets file
    if [[ ! -f "${ANSIBLE_SECRETS}" ]]; then
        err "Secrets file not found: ${ANSIBLE_SECRETS}"
        err "Create it: cd deploy && cp secrets.yml.template secrets.yml && ansible-vault encrypt secrets.yml"
        ok=false
    fi

    # Inventory
    if [[ ! -f "${ANSIBLE_INVENTORY}" ]]; then
        err "Inventory file not found: ${ANSIBLE_INVENTORY}"
        ok=false
    fi

    if [[ "$ok" == false ]]; then
        err "Prerequisites check failed. See errors above."
        exit 1
    fi

    log "Prerequisites OK"
}

run_ansible_playbook() {
    local playbook="$1"
    shift
    local extra_args=("$@")

    info "Running playbook: $(basename ${playbook})"

    # Use venv ansible if available, otherwise system
    local ansible_cmd="ansible-playbook"
    if [[ -f "${VENV_DIR}/bin/ansible-playbook" ]]; then
        ansible_cmd="${VENV_DIR}/bin/ansible-playbook"
    fi

    ${ansible_cmd} ${ANSIBLE_COMMON_FLAGS} "${playbook}" "${extra_args[@]}" -v
}

run_ansible_adhoc() {
    # Run ad-hoc ansible command using vault secrets (no -k/-K)
    local hosts="$1"
    shift

    local ansible_cmd="ansible"
    if [[ -f "${VENV_DIR}/bin/ansible" ]]; then
        ansible_cmd="${VENV_DIR}/bin/ansible"
    fi

    ${ansible_cmd} ${ANSIBLE_COMMON_FLAGS} "${hosts}" "$@"
}

# ============================================================================
# Phase Functions
# ============================================================================

phase_k3s() {
    header "Phase 1: K3s Cluster Setup"

    # Check if K3s is already running
    if kubectl get nodes &>/dev/null 2>&1; then
        local node_count
        node_count=$(kubectl get nodes --no-headers 2>/dev/null | wc -l)
        log "K3s cluster already running with ${node_count} node(s)"
        kubectl get nodes
        return 0
    fi

    info "Installing K3s cluster..."
    run_ansible_playbook "${DEPLOY_DIR}/setup_k3s.yaml"
    log "K3s cluster setup complete"

    # Verify
    sleep 5
    kubectl get nodes
}

phase_gpu() {
    header "Phase 2: GPU & NVIDIA Runtime Setup"

    # Check if NVIDIA device plugin is already running
    if kubectl get pods -n kube-system -l name=nvidia-device-plugin-ds --no-headers 2>/dev/null | grep -q Running; then
        local gpu_count
        gpu_count=$(kubectl get nodes -o jsonpath='{range .items[*]}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' 2>/dev/null | grep -c "1" || true)
        if [[ "$gpu_count" -gt 0 ]]; then
            log "NVIDIA device plugin running, ${gpu_count} GPU(s) available"
            return 0
        fi
    fi

    # Step 1: Install nvidia-container-toolkit on all nodes
    info "Installing nvidia-container-toolkit..."
    run_ansible_adhoc "cluster" -m shell \
        -a "apt-get update -qq && apt-get install -y -qq nvidia-container-toolkit 2>/dev/null || echo 'toolkit already installed or repo missing'" \
        --become

    # Step 2: Deploy containerd config.toml.tmpl (version 3 format, nvidia as default)
    info "Deploying containerd nvidia config template..."
    run_ansible_adhoc "cluster" -m copy \
        -a "src=${DEPLOY_DIR}/containerd-nvidia.toml.tmpl dest=/var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl mode=0644" \
        --become

    # Step 3: Restart K3s services
    info "Restarting K3s services..."
    run_ansible_adhoc "master" -m systemd -a "name=k3s state=restarted" --become
    run_ansible_adhoc "workers" -m systemd -a "name=k3s-agent state=restarted" --become

    info "Waiting for nodes to recover..."
    sleep 15

    # Step 4: Wait for all nodes to be Ready
    local retries=12
    for i in $(seq 1 $retries); do
        if ! kubectl get nodes 2>/dev/null | grep -q "NotReady"; then
            log "All nodes Ready"
            break
        fi
        if [[ $i -eq $retries ]]; then
            err "Nodes did not become Ready after restart"
            kubectl get nodes
            exit 1
        fi
        info "Waiting for nodes... (${i}/${retries})"
        sleep 10
    done

    # Step 5: Deploy NVIDIA device plugin
    info "Deploying NVIDIA device plugin..."
    kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml

    info "Waiting for device plugin pods..."
    kubectl wait --for=condition=Ready pod -l name=nvidia-device-plugin-ds -n kube-system --timeout=120s || true
    sleep 10

    # Step 6: Verify GPUs
    info "Checking GPU resources..."
    kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable."nvidia\.com/gpu"

    log "GPU setup complete"
}

phase_profile() {
    header "Phase 3: Cluster Profiling"

    mkdir -p "${PROFILES_DIR}"

    local profile_count
    profile_count=$(find "${PROFILES_DIR}" -name "profile_*.json" 2>/dev/null | wc -l)

    if [[ "$profile_count" -gt 0 ]]; then
        log "Found ${profile_count} existing profile(s), skipping profiling"
        ls -la "${PROFILES_DIR}"/profile_*.json
        return 0
    fi

    info "Running cluster profiling..."
    run_ansible_playbook "${DEPLOY_DIR}/profile_and_gather.yaml"

    profile_count=$(find "${PROFILES_DIR}" -name "profile_*.json" 2>/dev/null | wc -l)
    log "Profiling complete: ${profile_count} profile(s) collected"
}

phase_plan() {
    header "Phase 4: HPP Planning"

    if [[ -f "${PLAN_FILE}" ]]; then
        log "Existing plan found: ${PLAN_FILE}"
        "${VENV_DIR}/bin/python" -c "
import json
with open('${PLAN_FILE}') as f:
    p = json.load(f)
print(f'  Stages:    {p[\"num_stages\"]}')
print(f'  World:     {p[\"world_size\"]}')
print(f'  Partition: {p[\"partition_points\"]}')
print(f'  Latency:   {p[\"estimated_latency_ms\"]:.1f} ms')
"
        return 0
    fi

    info "Running HPP planner..."
    "${VENV_DIR}/bin/python" run_planner.py \
        --profiles-dir "${PROFILES_DIR}" \
        --cluster-conf "${SCRIPT_DIR}/cluster.conf" \
        --output "${PLAN_FILE}"

    log "Plan generated: ${PLAN_FILE}"
}

phase_build() {
    header "Phase 5: Build & Distribute Docker Image"

    # Step 1: Build the Docker image locally
    info "Building Docker image: ${IMAGE_FULL}"
    docker build -t "${IMAGE_FULL}" -f "${SCRIPT_DIR}/Dockerfile" "${SCRIPT_DIR}"
    log "Image built: ${IMAGE_FULL}"

    # Step 2: Export to tarball
    local tarball="/tmp/asteroid_image.tar"
    info "Exporting image to ${tarball}..."
    docker save "${IMAGE_FULL}" -o "${tarball}"
    local size_mb
    size_mb=$(du -m "${tarball}" | cut -f1)
    log "Tarball exported: ${size_mb} MB"

    # Step 3: Remove old images from K3s containerd on all nodes (parallel, fire-and-forget)
    info "Removing old images from K3s containerd (parallel)..."
    run_ansible_adhoc "cluster" -m shell \
        -a "k3s ctr images rm docker.io/library/${IMAGE_FULL} 2>/dev/null || true" \
        --become \
        -B 300 -P 0
    sleep 5  # brief wait for removal to finish

    # Step 4: Copy tarball to all nodes (parallel via Ansible forks — all 3 nodes concurrently)
    info "Distributing image to all nodes in parallel (${size_mb} MB each)..."
    run_ansible_adhoc "cluster" -m copy \
        -a "src=${tarball} dest=/tmp/asteroid_image.tar" \
        --become -f 10

    # Step 5: Import into K3s containerd on all nodes (parallel with async polling)
    info "Importing image into K3s containerd on all nodes (parallel)..."
    run_ansible_adhoc "cluster" -m shell \
        -a "k3s ctr images import /tmp/asteroid_image.tar && rm -f /tmp/asteroid_image.tar" \
        --become \
        -B 600 -P 5

    # Step 6: Verify import
    info "Verifying image on all nodes..."
    run_ansible_adhoc "cluster" -m shell \
        -a "k3s ctr images ls | grep asteroid | head -3" \
        --become

    rm -f "${tarball}"

    log "Image distributed to all nodes"
}

phase_manifests() {
    header "Phase 6: Generate K8s Manifests"

    mkdir -p "${GENERATED_DIR}"

    info "Generating manifests from HPP plan..."
    "${VENV_DIR}/bin/python" "${DEPLOY_DIR}/generate_manifests.py" \
        --plan "${PLAN_FILE}" \
        --image "${IMAGE_FULL}" \
        --namespace default \
        --master-port 29500 \
        --output-dir "${GENERATED_DIR}"

    # The generator uses host NIC names from the plan, but K8s pods use eth0.
    # The fix is already in generate_manifests.py (nccl_ifname hardcoded to eth0),
    # but let's double-check the generated manifests:
    info "Verifying NCCL_SOCKET_IFNAME is set to eth0 in manifests..."
    local bad_ifname
    bad_ifname=$(grep -l 'value: "ens' "${GENERATED_DIR}"/02-job-rank-*.yaml 2>/dev/null || true)
    if [[ -n "$bad_ifname" ]]; then
        warn "Found host NIC name in manifests, fixing to eth0..."
        sed -i 's/value: "ens[0-9]*"/value: "eth0"/g' "${GENERATED_DIR}"/02-job-rank-*.yaml
    fi

    log "Manifests generated in ${GENERATED_DIR}/"
    ls -la "${GENERATED_DIR}"/*.yaml
}

phase_deploy() {
    header "Phase 7: Deploy to K8s"

    # Delete existing jobs
    info "Cleaning up old deployments..."
    kubectl delete jobs -l app=asteroid --ignore-not-found=true 2>/dev/null || true
    sleep 3

    # Apply all manifests
    info "Applying K8s manifests..."
    kubectl apply -f "${GENERATED_DIR}/"
    log "Manifests applied"

    # Wait for pods
    info "Waiting for pods to start..."
    sleep 10

    local retries=30
    for i in $(seq 1 $retries); do
        local running
        running=$(kubectl get pods -l app=asteroid --no-headers 2>/dev/null | grep -c "Running" || true)
        local total
        total=$(kubectl get pods -l app=asteroid --no-headers 2>/dev/null | wc -l)

        if [[ "$running" -eq "$total" && "$total" -gt 0 ]]; then
            log "All ${total} pods running"
            break
        fi

        if [[ $i -eq $retries ]]; then
            err "Pods did not all reach Running state"
            kubectl get pods -l app=asteroid -o wide
            echo ""
            err "Check logs: kubectl logs -l app=asteroid --tail=20"
            exit 1
        fi

        info "Pods: ${running}/${total} running (waiting ${i}/${retries})..."
        sleep 5
    done

    echo ""
    kubectl get pods -l app=asteroid -o wide
    echo ""

    # Quick check for training output
    info "Checking for training output (waiting 30s)..."
    sleep 30

    local last_rank_pod
    last_rank_pod=$(kubectl get pods -l app=asteroid --sort-by=.metadata.name --no-headers | tail -1 | awk '{print $1}')
    local training_line
    training_line=$(kubectl logs "${last_rank_pod}" --tail=5 2>/dev/null | grep "iter" | tail -1 || true)

    if [[ -n "$training_line" ]]; then
        log "Training is running!"
        echo "  ${training_line}"
    else
        warn "No training output yet — pods may still be initializing"
        info "Check manually: kubectl logs ${last_rank_pod} --tail=20"
    fi
}

phase_monitor() {
    header "Training Monitor"
    "${SCRIPT_DIR}/monitor.sh"
}

phase_tensorboard() {
    header "TensorBoard Dashboard"

    # Verify training pods exist
    local pod_count
    pod_count=$(kubectl get pods -l app=asteroid --no-headers 2>/dev/null | wc -l)
    if [[ "$pod_count" -eq 0 ]]; then
        err "No training pods found. Deploy first: ./deploy.sh"
        exit 1
    fi

    local local_tb_dir="${SCRIPT_DIR}/tb_logs"
    mkdir -p "${local_tb_dir}"

    # Sync TB logs from ALL nodes (each rank writes to its own subdir)
    info "Syncing TensorBoard logs from all ${pod_count} pods..."

    local pods
    pods=$(kubectl get pods -l app=asteroid --no-headers -o custom-columns=NAME:.metadata.name,NODE:.spec.nodeName 2>/dev/null)

    while IFS= read -r line; do
        local pod_name node_name
        pod_name=$(echo "$line" | awk '{print $1}')
        node_name=$(echo "$line" | awk '{print $2}')
        local rank
        rank=$(echo "$pod_name" | grep -oP 'rank-\K\d+' || echo "?")

        info "  Syncing rank-${rank} from ${node_name} (${pod_name})..."
        mkdir -p "${local_tb_dir}/rank-${rank}"
        kubectl cp "${pod_name}:/tensorboard/rank-${rank}/" "${local_tb_dir}/rank-${rank}/" 2>/dev/null || {
            # If the rank subdir doesn't exist yet, try the base dir
            kubectl cp "${pod_name}:/tensorboard/" "${local_tb_dir}/rank-${rank}/" 2>/dev/null || true
        }
    done <<< "$pods"

    # Check if we got any TB data
    local event_files
    event_files=$(find "${local_tb_dir}" -name "events.out.tfevents.*" 2>/dev/null | wc -l)
    if [[ "$event_files" -eq 0 ]]; then
        warn "No TensorBoard event files found yet."
        info "The Docker image must be rebuilt to include TensorBoard logging."
        info "Run: ./deploy.sh --skip-k3s --skip-gpu --skip-profile --skip-plan"
        info "Then try again: ./deploy.sh --phase tensorboard"
        return 0
    fi

    log "Found ${event_files} event file(s) across $(find "${local_tb_dir}" -mindepth 1 -maxdepth 1 -type d | wc -l) rank(s)"
    find "${local_tb_dir}" -mindepth 1 -maxdepth 1 -type d | sort | while read -r d; do
        local count
        count=$(find "$d" -name "events.out.tfevents.*" | wc -l)
        echo "  $(basename "$d"): ${count} event file(s)"
    done

    # Launch TensorBoard
    local tb_cmd="tensorboard"
    if [[ -f "${VENV_DIR}/bin/tensorboard" ]]; then
        tb_cmd="${VENV_DIR}/bin/tensorboard"
    fi

    info "Starting TensorBoard on http://localhost:6006"
    info "Each rank appears as a separate run in TensorBoard"
    info "Press Ctrl+C to stop"
    echo ""
    ${tb_cmd} --logdir "${local_tb_dir}" --host 0.0.0.0 --port 6006
}

# ============================================================================
# Argument Parsing
# ============================================================================
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run the full Asteroid deployment pipeline or individual phases.

Options:
  --phase PHASE     Run a single phase: k3s, gpu, profile, plan, build, manifests, deploy, monitor, tensorboard
  --skip-k3s        Skip K3s installation (already installed)
  --skip-gpu        Skip GPU/NVIDIA setup (already configured)
  --skip-profile    Skip profiling (use existing profiles)
  --skip-build      Skip Docker image build (use existing image)
  --skip-plan       Skip HPP planning (use existing plan)
  --redeploy        Only regenerate manifests and redeploy jobs
  --monitor         Launch training monitor after deployment
  --status          Show current cluster and training status
  -h, --help        Show this help message

Phases (run in order):
  1. k3s        Install K3s cluster
  2. gpu        Setup NVIDIA runtime & device plugin
  3. profile    Profile cluster hardware
  4. plan       Run HPP optimizer
  5. build      Build & distribute Docker image
  6. manifests  Generate K8s job manifests
  7. deploy     Apply manifests & start training
  8. monitor    Live training dashboard
  9. tensorboard  Persistent TensorBoard dashboard

Examples:
  ./deploy.sh                           # Full deployment
  ./deploy.sh --skip-k3s --skip-gpu     # Skip infra (already set up)
  ./deploy.sh --redeploy                # Quick redeploy after code change
  ./deploy.sh --phase monitor           # Just monitor training
  ./deploy.sh --phase tensorboard       # Launch TensorBoard dashboard
  ./deploy.sh --status                  # Check cluster status
EOF
}

show_status() {
    header "Cluster Status"
    echo ""
    echo "Nodes:"
    kubectl get nodes -o wide 2>/dev/null || echo "  kubectl not configured"
    echo ""
    echo "GPU Resources:"
    kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable."nvidia\.com/gpu" 2>/dev/null || true
    echo ""
    echo "Training Pods:"
    kubectl get pods -l app=asteroid -o wide 2>/dev/null || echo "  No training pods"
    echo ""
    echo "Training Jobs:"
    kubectl get jobs -l app=asteroid 2>/dev/null || echo "  No training jobs"
    echo ""

    # Show last training log line from last-rank pod
    local last_pod
    last_pod=$(kubectl get pods -l app=asteroid --no-headers --sort-by=.metadata.name 2>/dev/null | tail -1 | awk '{print $1}')
    if [[ -n "$last_pod" ]]; then
        echo "Latest Training Output (${last_pod}):"
        kubectl logs "${last_pod}" --tail=5 2>/dev/null | grep -E "iter|loss|epoch" || echo "  (no training output yet)"
    fi
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)
            SINGLE_PHASE="$2"
            shift 2
            ;;
        --skip-k3s)
            SKIP_K3S=true
            shift
            ;;
        --skip-gpu)
            SKIP_GPU=true
            shift
            ;;
        --skip-profile)
            SKIP_PROFILE=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-plan)
            SKIP_PLAN=true
            shift
            ;;
        --redeploy)
            REDEPLOY_ONLY=true
            shift
            ;;
        --monitor)
            RUN_MONITOR=true
            shift
            ;;
        --status)
            show_status
            exit 0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            err "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# ============================================================================
# Main Execution
# ============================================================================
main() {
    header "Asteroid Deployment Pipeline"
    echo "  Image:   ${IMAGE_FULL}"
    echo "  Config:  ${ANSIBLE_INVENTORY}"
    echo "  Secrets: ${ANSIBLE_SECRETS}"
    echo ""

    check_prereqs

    # Single phase mode
    if [[ -n "$SINGLE_PHASE" ]]; then
        case "$SINGLE_PHASE" in
            k3s)       phase_k3s ;;
            gpu)       phase_gpu ;;
            profile)   phase_profile ;;
            plan)      phase_plan ;;
            build)     phase_build ;;
            manifests) phase_manifests ;;
            deploy)    phase_deploy ;;
            monitor)   phase_monitor ;;
            tensorboard) phase_tensorboard ;;
            *)
                err "Unknown phase: ${SINGLE_PHASE}"
                usage
                exit 1
                ;;
        esac
        exit 0
    fi

    # Redeploy mode (fastest path for code changes)
    if [[ "$REDEPLOY_ONLY" == true ]]; then
        phase_manifests
        phase_deploy
        if [[ "$RUN_MONITOR" == true ]]; then
            phase_monitor
        fi
        exit 0
    fi

    # Full pipeline
    local start_time
    start_time=$(date +%s)

    [[ "$SKIP_K3S"     == false ]] && phase_k3s
    [[ "$SKIP_GPU"     == false ]] && phase_gpu
    [[ "$SKIP_PROFILE" == false ]] && phase_profile
    [[ "$SKIP_PLAN"    == false ]] && phase_plan
    [[ "$SKIP_BUILD"   == false ]] && phase_build

    phase_manifests
    phase_deploy

    local end_time duration_min
    end_time=$(date +%s)
    duration_min=$(( (end_time - start_time) / 60 ))

    header "Deployment Complete!"
    echo ""
    echo "  Duration:    ${duration_min} minutes"
    echo "  Image:       ${IMAGE_FULL}"
    echo "  Plan:        ${PLAN_FILE}"
    echo "  Manifests:   ${GENERATED_DIR}/"
    echo ""
    echo "  Useful commands:"
    echo "    ./deploy.sh --status              # Check cluster status"
    echo "    ./deploy.sh --phase monitor       # Live training dashboard"
    echo "    ./monitor.sh                      # Training monitor"
    echo "    kubectl logs -f -l app=asteroid   # Stream all logs"
    echo "    kubectl delete jobs -l app=asteroid  # Stop training"
    echo ""

    if [[ "$RUN_MONITOR" == true ]]; then
        phase_monitor
    fi
}

main
