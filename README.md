# Asteroid

Distributed training framework for heterogeneous edge GPU clusters using **Heterogeneous Pipeline Parallelism (HPP)**.

Asteroid automatically profiles GPU nodes, computes an optimal model partition using dynamic programming, generates Kubernetes manifests, and deploys distributed training jobs — all from a single script.

## Features

- **Heterogeneous Pipeline Parallelism (HPP)**: Partition models across nodes with different GPU capabilities, optimizing for end-to-end latency using DP-based planning (Section 3.3 of the HPP paper)
- **Dynamic Profiling**: Per-node compute, memory, and network bandwidth measurement
- **Automated K8s Deployment**: Generate and apply K8s Job manifests with GPU scheduling, NCCL configuration, and DNS-based rendezvous
- **Live Monitoring**: Real-time training dashboard showing loss, throughput, and per-rank status
- **TensorBoard Integration**: Persistent per-rank metrics — forward/backward timing, barrier sync, GPU memory, loss curves — viewable in browser
- **Fault Tolerance**: Built-in checkpointing and recovery via `AsteroidFaultTolerance`
- **Single-Command Deployment**: `./deploy.sh` handles the full pipeline from K3s setup to training launch
- **Parallel Image Distribution**: Docker images are distributed to all nodes concurrently via Ansible async

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Control Node (master)                    │
│   deploy.sh → Ansible → K3s setup → Profiling → Planning   │
│                         │                                    │
│                ┌────────┴────────┐                           │
│                ▼                 ▼                            │
│          profile_node.py    run_planner.py                   │
│          (per-GPU stats)    (DP optimizer)                    │
│                │                 │                            │
│                └────────┬────────┘                            │
│                         ▼                                    │
│               generate_manifests.py                          │
│               (K8s Jobs + ConfigMap)                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────  K3s Cluster  ─────────────────────┐
│                                                              │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐            │
│  │  Rank 0  │────▶│  Rank 1  │────▶│  Rank 2  │            │
│  │ Stage 0  │     │ Stage 1  │     │ Stage 2  │            │
│  │ Layers   │     │ Layers   │     │ Layers   │            │
│  │  0..3    │     │  4..7    │     │  8..11   │            │
│  │  (GPU)   │     │  (GPU)   │     │  (GPU)   │            │
│  └──────────┘     └──────────┘     └──────────┘            │
│       │                                    │                │
│       └── NCCL over eth0 (pod network) ────┘                │
│           Rendezvous: asteroid-master:29500                  │
└─────────────────────────────────────────────────────────────┘
```

## Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| Python | >= 3.8 | 3.12 tested |
| PyTorch | >= 2.0 | **2.4.0 required in Docker image** |
| CUDA | >= 12.0 | 12.4 tested |
| K3s | >= 1.30 | 1.34.4+k3s1 tested |
| Docker | >= 20.10 | For building images |
| Ansible | >= 2.14 | With `ansible-vault` |
| NVIDIA Container Toolkit | >= 1.14 | 1.18.2 tested |
| NVIDIA GPUs | Any CUDA-capable | L40S tested |
| OS | Ubuntu 22.04 / 24.04 | On all cluster nodes |

## Quick Start

### 1-Command Deployment

```bash
# Full deployment: K3s install → GPU setup → profile → plan → build → deploy
./deploy.sh
```

That's it. The script handles all 7 phases automatically.

### Selective / Incremental Usage

```bash
# Re-deploy only (fastest — rebuilds K8s jobs without rebuilding image)
./deploy.sh --redeploy

# Run a single phase
./deploy.sh --phase gpu

# Skip slow steps
./deploy.sh --skip-profile --skip-build

# Deploy and start live monitoring
./deploy.sh --monitor

# Check current status
./deploy.sh --status

# Launch TensorBoard dashboard
./deploy.sh --phase tensorboard
```

## Setup Guide

### Step 1: Clone and Create Virtual Environment

```bash
git clone <repo-url> asteroid_project
cd asteroid_project
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Step 2: Configure Cluster Nodes

**`cluster.conf`** — Define your cluster topology (one line per node):

```
# IP          NIC    RANK  GPU_ID
10.203.54.10  ens33  0     0
10.203.53.10  ens33  1     0
10.203.56.10  ens33  2     0
```

> **Note**: The NIC field (`ens33`) is the **host** network interface. Inside K8s pods, NCCL will use `eth0` automatically (this is handled by the deploy scripts).

**`deploy/inventory.ini`** — Ansible inventory matching your cluster:

```ini
[master]
master_node ansible_host=10.203.54.10 rank=0

[workers]
worker1 ansible_host=10.203.53.10 rank=1
worker3 ansible_host=10.203.56.10 rank=2

[cluster:children]
master
workers

[cluster:vars]
ansible_user=ubuntu
```

### Step 3: Configure Ansible Secrets

Secrets (SSH and sudo passwords) are managed via Ansible Vault to avoid interactive password prompts.

```bash
# Create vault password file
echo "your-vault-password" > ~/.asteroid_vault_pass
chmod 600 ~/.asteroid_vault_pass

# Create encrypted secrets file
cd deploy
ansible-vault create secrets.yml
```

Add the following to `secrets.yml`:

```yaml
ansible_ssh_pass: "ubuntu"
ansible_become_pass: "ubuntu"
```

The vault password file path is configured in `deploy/ansible.cfg`:

```ini
[defaults]
vault_password_file = ~/.asteroid_vault_pass
```

> **Important**: All Ansible commands in `deploy.sh` pass `-e @secrets.yml` automatically. You should never need to use `-k` or `-K` flags manually.

### Step 4: Deploy

```bash
./deploy.sh
```

## Deployment Phases

`deploy.sh` runs these phases in order:

| Phase | What It Does | Can Skip? |
|-------|-------------|-----------|
| **1. K3s** | Install K3s on master, join workers, install NVIDIA device plugin | `--skip-k3s` |
| **2. GPU** | Configure containerd with NVIDIA runtime on all nodes | Part of K3s |
| **3. Profile** | Run `profile_node.py` on each node to measure GPU performance | `--skip-profile` |
| **4. Plan** | Run `run_planner.py` to compute optimal HPP partition | `--skip-plan` |
| **5. Build** | Build Docker image, export, distribute to all nodes in parallel via K3s containerd | `--skip-build` |
| **6. Manifests** | Generate K8s ConfigMap, Service, and per-rank Job YAMLs | Auto |
| **7. Deploy** | Apply manifests to K3s cluster, start training | Auto |
| **8. TensorBoard** | Sync per-rank event files and launch TensorBoard at localhost:6006 | `--phase tensorboard` |

## Monitoring Training

### Live Dashboard

```bash
# Auto-refreshing dashboard (updates every 5s)
./monitor.sh

# Single snapshot
./monitor.sh --once

# Custom refresh interval
./monitor.sh --watch 10
```

The dashboard shows:
- Cluster node status and GPU availability
- Pod status (Running / CrashLoopBackOff / etc.)
- Training metrics: iteration, loss, learning rate, throughput (tok/s), step time
- Recent iteration history
- Per-rank log status

### TensorBoard (Persistent Monitoring)

For persistent, browser-based monitoring with per-node detail:

```bash
# Launch TensorBoard (syncs logs from all pods, opens at http://localhost:6006)
./deploy.sh --phase tensorboard

# Or via monitor.sh
./monitor.sh --tensorboard
```

Each rank appears as a separate run in TensorBoard. Metrics logged per rank:

| Metric | Description | What to Check |
|--------|-------------|---------------|
| `timing/forward_ms` | Forward pass duration | Compare across ranks — should reflect stage compute load |
| `timing/backward_ms` | Backward pass duration | Larger stages = longer backward |
| `timing/barrier_after_fwd_ms` | Wait time at forward barrier | High = this rank finished early (pipeline imbalance) |
| `timing/barrier_after_step_ms` | Wait time at step barrier | All ranks should converge; high = straggler |
| `timing/total_step_ms` | Full iteration wall time | Should be similar across ranks |
| `gpu/peak_memory_allocated_MB` | Peak GPU memory used | Monitor for OOM risk |
| `gpu/peak_memory_reserved_MB` | Peak GPU memory reserved | CUDA memory pool size |
| `gpu/memory_utilization_pct` | Memory utilization % | Higher = better GPU usage |
| `train/loss` | Training loss (last stage only) | Should decrease over time |
| `train/learning_rate` | Learning rate schedule | Verify warmup + decay |
| `train/throughput_tok_s` | Tokens per second (last stage only) | Overall training speed |

**Checking synchronization**: Overlay `timing/barrier_after_fwd_ms` for all ranks. If one rank consistently waits longer at the barrier, it finishes its forward pass faster than others — indicating pipeline imbalance. Well-balanced stages show similar barrier times.

### Log Streaming

```bash
# Stream logs from all training pods
./monitor.sh --logs

# Follow a specific rank
./monitor.sh --rank 0
./monitor.sh --rank 2   # last stage reports training metrics
```

### Manual Inspection

```bash
# Pod status
kubectl get pods -l app=asteroid

# Logs for a specific pod
kubectl logs -f asteroid-rank-0-xxxxx

# GPU utilization on a node
nvidia-smi
```

## Troubleshooting

### Common Issues and Fixes

#### 1. `Bootstrap: no socket interface found` (NCCL error)

**Cause**: `NCCL_SOCKET_IFNAME` is set to the host NIC name (e.g., `ens33`) but K8s pods use `eth0`.

**Fix**: All generated manifests must use `eth0`:
```yaml
- name: NCCL_SOCKET_IFNAME
  value: "eth0"
```

This is handled automatically by `generate_manifests.py` and verified by `deploy.sh`.

#### 2. `GPT2Model requires PyTorch library but it was not found`

**Cause**: K3s containerd has a stale/old Docker image cached. Even if `docker images` shows the correct version, K3s uses its own image store.

**Fix**: Clear and re-import the image:
```bash
# On each node:
sudo k3s ctr images rm docker.io/library/asteroid:latest
sudo k3s ctr images import /tmp/asteroid_image.tar
sudo k3s ctr images tag docker.io/library/asteroid:v2 docker.io/library/asteroid:latest
```

`deploy.sh` handles this automatically during the build phase.

#### 3. Ansible still prompting for passwords

**Cause**: Missing `vars_files: - secrets.yml` in a playbook, or using `-k`/`-K` flags.

**Fix**: 
- Ensure `deploy/secrets.yml` exists (vault-encrypted with `ansible_ssh_pass` and `ansible_become_pass`)
- Ensure `~/.asteroid_vault_pass` exists with the vault password
- Use `-e @secrets.yml` for ad-hoc commands:
  ```bash
  ansible -i deploy/inventory.ini cluster -m ping -e @deploy/secrets.yml
  ```

#### 4. Pod stuck in `Pending` — no GPU available

**Cause**: NVIDIA device plugin not running or containerd not configured with NVIDIA runtime.

**Fix**:
```bash
# Check device plugin
kubectl get pods -n kube-system | grep nvidia

# Check GPU allocatable on nodes
kubectl get nodes -o json | jq '.items[].status.allocatable["nvidia.com/gpu"]'

# If missing, re-run GPU setup
./deploy.sh --phase gpu
```

#### 5. Containerd/CNI crash after config change

**Cause**: Using a minimal or version-2 format `config.toml.tmpl` breaks K3s CNI networking.

**Fix**: The containerd config must be a **complete version 3** file using `'io.containerd.cri.v1.runtime'` (not `"io.containerd.grpc.v1.cri"`). The correct template is at `deploy/containerd-nvidia.toml.tmpl`. Never write a minimal config — always use the full template.

#### 6. Image import with `ctr` doesn't work on K3s

**Cause**: K3s wraps containerd; using raw `ctr -n k8s.io` commands may not work correctly.

**Fix**: Always use `k3s ctr images import` instead of `ctr -n k8s.io images import`:
```bash
sudo k3s ctr images import /tmp/asteroid_image.tar    # ✓ correct
sudo ctr -n k8s.io images import /tmp/asteroid_image.tar  # ✗ unreliable
```

#### 7. Nodes show wrong hostname in K8s

**Cause**: Multiple nodes have the same OS hostname (e.g., all cloned from the same template).

**Fix**: Set unique hostnames before installing K3s:
```bash
sudo hostnamectl set-hostname worker1    # on each node
```

#### 8. `ANSIBLE_CONFIG` not found when running `deploy.sh`

**Cause**: `deploy.sh` runs from the project root, but `ansible.cfg` is in `deploy/`. Ansible can't find the vault password file.

**Fix**: `deploy.sh` exports `ANSIBLE_CONFIG="${DEPLOY_DIR}/ansible.cfg"`. If you see vault errors, verify this line exists near the top of the script.

#### 9. TensorBoard: `No module named 'pkg_resources'`

**Cause**: `setuptools >= 82` removed `pkg_resources` into a separate package, but TensorBoard 2.x still imports it.

**Fix**:
```bash
pip install "setuptools<81"
```

### Verifying the Deployment

```bash
# All nodes ready
kubectl get nodes

# All pods running
kubectl get pods -l app=asteroid

# Check PyTorch version inside a pod
kubectl exec -it <pod-name> -- python3 -c "import torch; print(torch.__version__)"

# Check NCCL initialization in logs
kubectl logs <pod-name> | grep -i "distributed\|nccl\|initialized"

# Check training metrics (last-stage pod)
kubectl logs <last-stage-pod> | grep "iter"
```

## Project Structure

```
asteroid_project/
├── deploy.sh                 # Single-command deployment script
├── monitor.sh                # Live training monitor / dashboard
├── cluster.conf              # Cluster topology (IP / NIC / RANK / GPU)
├── Dockerfile                # Training image (PyTorch 2.4.0 + CUDA 12.4)
├── worker.py                 # Main training loop (runs inside pods)
├── run_planner.py            # HPP planner CLI
├── profile_node.py           # Per-node GPU profiler
├── hpp_plan.json             # Generated partition plan
├── pyproject.toml            # Python package metadata
├── requirements.txt          # Locked Python dependencies
├── tb_logs/                  # TensorBoard event files (synced from pods)
│
├── asteroid/                 # Core library
│   ├── core/                 # Config, state management
│   ├── model/                # GPT-2 blocks, heads, pipeline stages
│   ├── planner/              # DP-based HPP planner, profiler
│   ├── comm/                 # NCCL utilities
│   ├── optim/                # Optimizer utilities
│   ├── ft/                   # Fault tolerance (checkpointing)
│   └── utils/                # Data loading, logging
│
└── deploy/                   # Deployment configs
    ├── inventory.ini         # Ansible inventory
    ├── ansible.cfg           # Ansible settings (vault, SSH)
    ├── secrets.yml           # Encrypted passwords (ansible-vault)
    ├── setup_k3s.yaml        # K3s installation playbook
    ├── setup_gpu.yaml        # NVIDIA runtime playbook
    ├── distribute_image.yaml # Image distribution playbook
    ├── containerd-nvidia.toml.tmpl  # K3s containerd config (v3 format)
    ├── generate_manifests.py # K8s manifest generator
    ├── templates/            # Jinja2 templates for K8s manifests
    │   ├── stage_job.yaml.j2
    │   ├── configmap.yaml.j2
    │   └── headless_service.yaml.j2
    └── generated/            # Output manifests
        ├── 00-configmap.yaml
        ├── 01-headless-service.yaml
        └── 02-job-rank-*.yaml
```

## Key Design Decisions & Fixes

These are lessons learned during deployment that are baked into `deploy.sh`:

1. **PyTorch 2.4.0 base image** — Earlier versions (2.1.0) lack compatibility with `transformers >= 5.x`. The Dockerfile uses `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`.

2. **Containerd config format** — K3s uses `config.toml.tmpl` (version 3, Golang template). Using the old `config.toml` (version 2) with `"io.containerd.grpc.v1.cri"` breaks pod networking. The correct plugin key is `'io.containerd.cri.v1.runtime'`.

3. **NCCL socket interface** — The host NIC (e.g., `ens33`) does not exist inside K8s pods. NCCL must use `eth0` (the pod's virtual interface). `generate_manifests.py` hardcodes this.

4. **K3s image management** — `docker save`/`docker load` only affects Docker's image store. K3s containerd has its own store. Images must be imported with `k3s ctr images import`, not `ctr -n k8s.io images import`.

5. **Ansible vault for secrets** — All playbooks include `vars_files: - secrets.yml`. The deploy script passes `-e @secrets.yml` for ad-hoc commands. This eliminates interactive password prompts (`-k`/`-K`).

6. **Hostname uniqueness** — Nodes cloned from the same VM template share hostnames, causing K8s scheduling confusion. Each node must have a unique hostname before K3s installation.

7. **ANSIBLE_CONFIG export** — `deploy.sh` runs from the project root, but `ansible.cfg` lives in `deploy/`. Without `export ANSIBLE_CONFIG`, Ansible can't find the vault password file and fails with "no vault secrets found".

8. **Parallel image distribution** — Docker image distribution uses Ansible's `-f 10` (forks) for parallel copy to all nodes and `-B`/`-P` (async) for parallel import into K3s containerd, reducing distribution time from ~3x to ~1x.

9. **setuptools < 81** — TensorBoard 2.x requires `pkg_resources` which was removed from `setuptools >= 82`. Pin to `setuptools<81` in the local venv.

10. **Per-rank TensorBoard** — Every rank writes timing/GPU metrics to TensorBoard via `hostPath` volumes at `/var/lib/asteroid/tensorboard`. Logs persist across pod restarts and are synced to local `tb_logs/` via `kubectl cp`.

## License

Apache 2.0
