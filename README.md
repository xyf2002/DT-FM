# Distributed LLM Training

<p align="center">
  <b>Heterogeneous Pipeline Parallelism (HPP) for training large language models on clusters of edge GPUs</b>
</p>

This is a distributed training framework designed for **heterogeneous edge clusters** — environments where GPU memory, compute capacity, and network bandwidth vary across nodes. It implements research from DT-FM, Confident, and ASTEROID systems to enable efficient LLM training on commodity hardware.

---

## Table of Contents

1. [Features](#features)
2. [Architecture Overview](#architecture-overview)
3. [Requirements](#requirements)
4. [Quick Start](#quick-start)
5. [Cluster Setup](#cluster-setup)
   - [Hardware Requirements](#hardware-requirements)
   - [Network Configuration](#network-configuration)
   - [K3s Installation](#k3s-installation)
   - [GPU Setup](#gpu-setup)
6. [Configuration Reference](#configuration-reference)
   - [asteroid.yaml](#asteroidyaml)
   - [Cluster Nodes](#cluster-nodes)
   - [Model Configuration](#model-configuration)
   - [Training Parameters](#training-parameters)
   - [Parallelism Settings](#parallelism-settings)
   - [Fault Tolerance](#fault-tolerance)
7. [Deployment Guide](#deployment-guide)
   - [Prerequisites](#prerequisites)
   - [Ansible Setup](#ansible-setup)
   - [SSH Key Distribution](#ssh-key-distribution)
   - [Full Deployment](#full-deployment)
   - [Command Reference](#command-reference)
   - [Deployment Phases](#deployment-phases)
   - [Auto-Generated Inventory](#auto-generated-inventory)
8. [Pipeline Schedules](#pipeline-schedules)
   - [GPipe](#gpipe)
   - [1F1B (One Forward One Backward)](#1f1b-one-forward-one-backward)
9. [Supported Models](#supported-models)
   - [GPT-2](#gpt-2)
   - [LLaMA](#llama)
   - [Custom Models](#custom-models)
   - [HuggingFace Adapter](#huggingface-adapter)
10. [Monitoring & Debugging](#monitoring--debugging)
11. [Checkpointing](#checkpointing)
12. [Troubleshooting](#troubleshooting)
13. [API Reference](#api-reference)
14. [Contributing](#contributing)

---

## Features

- **Heterogeneous Pipeline Parallelism (HPP)** — Dynamic programming-based layer partitioning optimized for heterogeneous GPU memory and compute
- **Multiple Pipeline Schedules** — GPipe and 1F1B (interleaved) scheduling with automatic fallback
- **Fault Tolerance** — Active heartbeat detection, passive backward-timeout detection, weight replication (topology/local/global), automatic recovery
- **Multiple Model Architectures** — GPT-2, LLaMA-2/3 (with RoPE, GQA, SwiGLU), and custom models via registry
- **HuggingFace Integration** — Load pretrained models directly from HuggingFace Hub
- **Automated Deployment** — Single-command deployment to K3s clusters via Ansible with auto-generated inventory from `asteroid.yaml`
- **Hardware Profiling** — Per-node profiling with automatic bandwidth measurement
- **TensorBoard Integration** — Per-rank metrics visualization
- **Async Checkpointing** — MegaScale-inspired non-blocking checkpoint saves
- **Unified Configuration** — Single `asteroid.yaml` file for model, training, cluster, and deployment settings

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Asteroid Architecture                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Node 0    │    │   Node 1    │    │   Node 2    │    │   Node 3    │  │
│  │  (Stage 0)  │───▶│  (Stage 1)  │───▶│  (Stage 2)  │───▶│  (Stage 3)  │  │
│  │ Layers 0-7  │    │ Layers 8-15 │    │ Layers 16-23│    │ Layers 24-31│  │
│  │  Embedding  │    │   Blocks    │    │   Blocks    │    │  LM Head    │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                  │                  │                  │          │
│         └──────────────────┴──────────────────┴──────────────────┘          │
│                          NCCL / torch.distributed                           │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Components:                                                                 │
│  • HPP Planner: Dynamic programming for optimal layer partitioning          │
│  • Pipeline Schedule: GPipe or 1F1B interleaved execution                   │
│  • Fault Tolerance: Heartbeat + timeout detection + replication             │
│  • Checkpoint: Async CPU-clone saves with background I/O                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
asteroid_project/
├── asteroid/                    # Core Python package
│   ├── core/                    # Config, state, interfaces
│   │   ├── config.py            # AsteroidConfig, HPPPlanConfig, DeviceSpec
│   │   ├── state.py             # AsteroidStateManager
│   │   └── interfaces.py        # Abstract base classes
│   ├── model/                   # Model architectures
│   │   ├── blocks.py            # GPT-2, Encoder blocks
│   │   ├── llama.py             # LLaMA (RoPE, GQA, SwiGLU)
│   │   ├── heads.py             # Classification, LM heads
│   │   ├── stage.py             # AsteroidStage, MODEL_REGISTRY
│   │   └── hf_adapter.py        # HuggingFace model adapter
│   ├── pipeline/                # Pipeline scheduling
│   │   └── schedule.py          # GPipe, 1F1B schedules
│   ├── planner/                 # HPP optimization
│   │   ├── dp_planner.py        # Dynamic programming planner
│   │   └── profiler.py          # Hardware profiler
│   ├── ft/                      # Fault tolerance
│   │   ├── fault_tolerance.py   # Main FT class
│   │   ├── heartbeat.py         # Heartbeat detection
│   │   ├── replication.py       # Weight replication
│   │   └── checkpoint.py        # Async/sync checkpointing
│   ├── comm/                    # Communication backends
│   │   ├── backends.py          # TorchDist, NCCL, Gloo
│   │   └── nccl_utils.py        # CuPy NCCL utilities
│   ├── optim/                   # Optimizer utilities
│   │   └── optim_utils.py       # Flatten params, LR schedule
│   └── utils/                   # Utilities
│       ├── data_utils.py        # Dataset preparation
│       ├── logger.py            # Event logger, Chrome trace
│       └── seed.py              # Reproducibility helpers
├── deploy/                      # Deployment automation
│   ├── inventory.ini            # Ansible inventory
│   ├── secrets.yml              # Encrypted credentials
│   ├── ansible.cfg              # Ansible configuration
│   ├── setup_k3s.yaml           # K3s installation playbook
│   ├── setup_gpu.yaml           # GPU/NVIDIA setup playbook
│   ├── profile_and_gather.yaml  # Profiling playbook
│   ├── generate_manifests.py    # K8s manifest generator
│   └── generated/               # Generated K8s YAML files
├── worker.py                    # Main training worker
├── run_planner.py               # HPP planning script
├── profile_node.py              # Node profiling script
├── deploy.sh                    # Master deployment script
├── monitor.sh                   # Training monitor
├── asteroid.yaml                # Unified configuration
├── Dockerfile                   # Container image
└── requirements.txt             # Python dependencies
```

---

## Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPUs | 2+ NVIDIA GPUs | 4+ NVIDIA GPUs (L40S, A100, RTX 3090/4090) |
| VRAM per GPU | 8 GB | 24+ GB |
| RAM per Node | 16 GB | 64+ GB |
| Network | 1 Gbps | 10+ Gbps, low latency |
| Storage | 50 GB | 200+ GB NVMe |

### Software

| Component | Version |
|-----------|---------|
| Ubuntu | 20.04 / 22.04 / 24.04 |
| Python | 3.10+ |
| PyTorch | 2.0+ |
| CUDA | 11.8+ / 12.x |
| NVIDIA Driver | 525+ |
| K3s | 1.28+ |
| Docker | 20.10+ |
| Ansible | 2.12+ |

---

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/your-org/asteroid.git
cd asteroid_project

# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
pip install ansible
```

### 2. Configure Cluster

Edit `asteroid.yaml` with your cluster nodes:

```yaml
cluster:
  nodes:
    - ip: 192.168.1.10
      hostname: node0
      nic: eth0
      gpu_id: 0
      memory_mb: 24576
      rank: 0
      role: master

    - ip: 192.168.1.11
      hostname: node1
      nic: eth0
      gpu_id: 0
      memory_mb: 24576
      rank: 1
```

### 3. Setup Ansible Secrets

```bash
# Create vault password
echo "your-secure-password" > ~/.asteroid_vault_pass
chmod 600 ~/.asteroid_vault_pass

# Create encrypted secrets
cd deploy
cp secrets.yml.template secrets.yml
# Edit secrets.yml with your SSH passwords
ansible-vault encrypt secrets.yml
```

### 4. Update Inventory

Edit `deploy/inventory.ini`:

```ini
[master]
node0 ansible_host=192.168.1.10 ansible_user=ubuntu rank=0 gpu_id=0

[workers]
node1 ansible_host=192.168.1.11 ansible_user=ubuntu rank=1 gpu_id=0

[cluster:children]
master
workers
```

### 5. Deploy

```bash
# Full deployment (first time)
./deploy.sh

# Skip K3s/GPU setup (already configured)
./deploy.sh --skip-k3s --skip-gpu

# Quick redeploy after code changes
./deploy.sh --redeploy
```

### 6. Monitor Training

```bash
# Live dashboard
./deploy.sh --phase monitor

# Check status
./deploy.sh --status

# View logs
kubectl logs -f -l app=asteroid
```

---

## Cluster Setup

### Hardware Requirements

Asteroid is designed for **heterogeneous** clusters. Nodes can have different:
- GPU models (mixing A100, L40S, RTX 4090 is supported)
- GPU memory (the planner accounts for this)
- Network bandwidth (measured during profiling)

**Important**: All nodes must have:
- NVIDIA GPU with CUDA support
- Direct network connectivity to all other nodes
- Synchronized clocks (NTP recommended)

### Network Configuration

#### Required Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| 6443 | TCP | K3s API server |
| 10250 | TCP | Kubelet metrics |
| 29500 | TCP | PyTorch distributed master port |
| 29501-29600 | TCP | PyTorch NCCL communication |
| 5201 | TCP | iperf3 bandwidth measurement |

#### Firewall Rules

```bash
# On all nodes (Ubuntu)
sudo ufw allow 6443/tcp
sudo ufw allow 10250/tcp
sudo ufw allow 29500:29600/tcp
sudo ufw allow 5201/tcp
sudo ufw reload
```

#### Network Interface

Identify your network interface:

```bash
ip addr show | grep -E "^[0-9]+"
# Example output: 2: ens33: <BROADCAST,MULTICAST,UP,LOWER_UP>
```

Use this interface name (e.g., `ens33`) in `asteroid.yaml` under `cluster.nodes[].nic`.

> **Note**: Inside Kubernetes pods, the interface is always `eth0`. The deploy script automatically handles this translation.

### K3s Installation

K3s is the lightweight Kubernetes distribution used by Asteroid.

#### Automatic Installation (Recommended)

```bash
./deploy.sh --phase k3s
```

#### Manual Installation

**On the master node:**

```bash
# Install K3s server
curl -sfL https://get.k3s.io | sh -s - server \
  --write-kubeconfig-mode 644 \
  --disable traefik \
  --disable servicelb

# Get the node token
sudo cat /var/lib/rancher/k3s/server/node-token
```

**On worker nodes:**

```bash
# Replace TOKEN and MASTER_IP with your values
curl -sfL https://get.k3s.io | K3S_URL=https://MASTER_IP:6443 K3S_TOKEN=TOKEN sh -
```

**Verify cluster:**

```bash
kubectl get nodes
# All nodes should show "Ready"
```

### GPU Setup

#### NVIDIA Driver Installation

```bash
# Ubuntu 22.04/24.04
sudo apt update
sudo apt install -y nvidia-driver-535

# Reboot
sudo reboot

# Verify
nvidia-smi
```

#### NVIDIA Container Toolkit

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install toolkit
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Configure containerd for K3s
sudo nvidia-ctk runtime configure --runtime=containerd --config=/var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl

# Restart K3s
sudo systemctl restart k3s  # master
sudo systemctl restart k3s-agent  # workers
```

#### containerd Configuration

If GPUs aren't detected, you may need to manually configure containerd. Create `/var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl`:

```toml
version = 3

[plugins]
  [plugins."io.containerd.grpc.v1.cri"]
    [plugins."io.containerd.grpc.v1.cri".containerd]
      default_runtime_name = "nvidia"
      
      [plugins."io.containerd.grpc.v1.cri".containerd.runtimes]
        [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia]
          runtime_type = "io.containerd.runc.v2"
          [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia.options]
            BinaryName = "/usr/bin/nvidia-container-runtime"
            
    [plugins."io.containerd.grpc.v1.cri".cni]
      bin_dir = "/opt/cni/bin:/var/lib/rancher/k3s/data/current/bin"
      conf_dir = "/var/lib/rancher/k3s/agent/etc/cni/net.d"
```

Then restart K3s and deploy the NVIDIA device plugin:

```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml
```

Verify GPUs:

```bash
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable."nvidia\.com/gpu"
```

---

## Configuration Reference

### asteroid.yaml

The unified configuration file controls all aspects of Asteroid. Below is a complete reference:

```yaml
# =============================================================================
# Asteroid — Unified Configuration
# =============================================================================

# ── Model ────────────────────────────────────────────────────────────────────
model:
  model_name: llama2-7b              # Descriptive name for logging
  model_type: llama                  # gpt2 | llama | encoder
  task_type: lm                      # classification | lm
  num_layers: 32                     # Number of transformer layers
  embedding_dim: 4096                # Hidden dimension (d_model)
  num_heads: 32                      # Attention heads
  n_kv_heads: 32                     # GQA key/value heads (0 = MHA)
  d_ff: 11008                        # FFN intermediate dimension
  max_seq_len: 512                   # Maximum sequence length
  vocab_size: 50257                  # Vocabulary size
  num_classes: 2                     # Classification classes (if task_type=classification)
  dropout: 0.0                       # Dropout rate
  use_flash_attention: true          # Use Flash Attention v2
  hf_model_name: ""                  # HuggingFace model ID (e.g., "meta-llama/Llama-2-7b")

# ── Training ─────────────────────────────────────────────────────────────────
training:
  global_batch_size: 64              # Total batch size across all GPUs
  micro_batch_size: 2                # Batch size per GPU per microbatch
  lr: 3.0e-4                         # Peak learning rate
  min_lr: 1.0e-5                     # Minimum learning rate (for cosine decay)
  weight_decay: 0.01                 # AdamW weight decay
  max_iters: 500                     # Total training iterations
  warmup_iters: 50                   # Linear warmup iterations
  grad_clip: 1.0                     # Gradient clipping norm
  eval_interval: 100                 # Validation interval
  log_interval: 10                   # Logging interval
  seed: 42                           # Random seed
  dataset: sst2                      # Dataset name

# ── Parallelism ──────────────────────────────────────────────────────────────
parallelism:
  strategy: asteroid                 # asteroid | confident | dtfm
  schedule_type: "1f1b"              # gpipe | 1f1b (empty = auto from strategy)
  num_stages: 4                      # Number of pipeline stages
  world_size: 4                      # Total number of workers
  comm_backend: torch_dist           # torch_dist | nccl | gloo
  dist_url: "tcp://127.0.0.1:29600"  # Distributed init URL (local mode)
  d2d_bandwidth_mbps: 100.0          # Default inter-node bandwidth

# ── Fault Tolerance ──────────────────────────────────────────────────────────
fault_tolerance:
  heartbeat_interval_s: 2.0          # Heartbeat ping interval
  heartbeat_timeout_s: 8.0           # Heartbeat timeout threshold
  backward_timeout_ms: 30000.0       # Backward pass timeout (Confident-style)
  replication_mode: all              # all | topology | local | global | none
  replication_interval: 25           # Steps between weight replication
  ft_check_interval: 5               # Steps between FT checks
  checkpoint_strategy: async         # async | sync
  checkpoint_dir: ./checkpoints      # Checkpoint output directory
  checkpoint_interval: 100           # Steps between checkpoints (0 = disabled)

# ── Cluster ──────────────────────────────────────────────────────────────────
cluster:
  nodes:
    - ip: 10.203.54.11               # Node IP address
      hostname: host                 # Hostname for identification
      nic: ens33                     # Network interface
      gpu_id: 0                      # GPU index on this node
      memory_mb: 46068               # GPU memory in MB
      rank: 0                        # Global rank
      role: master                   # master | worker

    - ip: 10.203.54.10
      hostname: master
      nic: ens33
      gpu_id: 0
      memory_mb: 46068
      rank: 1

    - ip: 10.203.53.10
      hostname: worker1
      nic: ens33
      gpu_id: 0
      memory_mb: 46068
      rank: 2

    - ip: 10.203.56.10
      hostname: worker3
      nic: ens33
      gpu_id: 0
      memory_mb: 46068
      rank: 3

# ── MPS (NVIDIA Multi-Process Service) ───────────────────────────────────────
mps:
  enabled: false                     # Enable GPU sharing via MPS
  active_thread_percentage: 50       # Percentage of GPU threads per process
  pipe_directory: ""                 # MPS pipe directory
  log_directory: ""                  # MPS log directory
  pinned_device_mem_limit: ""        # Memory limit per process

# ── Deploy ───────────────────────────────────────────────────────────────────
deploy:
  image_name: asteroid               # Docker image name
  image_tag: latest                  # Docker image tag
  kubeconfig: ~/.kube/config         # Kubeconfig path
  namespace: default                 # Kubernetes namespace
  master_port: 29500                 # PyTorch master port
  nccl_debug: WARN                   # NCCL debug level (WARN | INFO | TRACE)
  nccl_ib_disable: "1"               # Disable InfiniBand (set "0" to enable)
  nccl_p2p_disable: "1"              # Disable P2P (set "0" for NVLink)
  shm_size: 4Gi                      # Shared memory size for containers
  output_dir: ./asteroid_output      # Output directory
```

### Cluster Nodes

Each node in `cluster.nodes` requires:

| Field | Description | Example |
|-------|-------------|---------|
| `ip` | Node's IP address (must be reachable from all nodes) | `192.168.1.10` |
| `hostname` | Human-readable name | `gpu-node-1` |
| `nic` | Network interface for NCCL | `ens33`, `eth0` |
| `gpu_id` | GPU index on this node | `0` |
| `memory_mb` | GPU memory in MB (from `nvidia-smi`) | `24576` |
| `rank` | Global rank (0 = master) | `0`, `1`, `2`, ... |
| `role` | `master` (one required) or `worker` | `worker` |

**Finding GPU Memory:**

```bash
nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits
# Example: 46068
```

### Model Configuration

#### model_type Options

| Type | Description | Architecture |
|------|-------------|--------------|
| `gpt2` | GPT-2 style decoder | Pre-norm, GELU, causal attention |
| `llama` | LLaMA-2/3 decoder | RMSNorm, RoPE, SwiGLU, GQA |
| `encoder` | BERT-style encoder | Bidirectional attention |

#### task_type Options

| Type | Description | Head |
|------|-------------|------|
| `lm` | Language modeling | `LMHead` (next-token prediction) |
| `classification` | Sequence classification | `ClassificationHead` |

#### Key Architecture Parameters

| Parameter | GPT-2 Small | GPT-2 XL | LLaMA-2 7B | LLaMA-2 13B |
|-----------|-------------|----------|------------|-------------|
| `num_layers` | 12 | 48 | 32 | 40 |
| `embedding_dim` | 768 | 1600 | 4096 | 5120 |
| `num_heads` | 12 | 25 | 32 | 40 |
| `n_kv_heads` | 12 | 25 | 32 | 40 |
| `d_ff` | 3072 | 6400 | 11008 | 13824 |
| `vocab_size` | 50257 | 50257 | 32000 | 32000 |

### Training Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `global_batch_size` | Total batch size across all workers | 32-128 |
| `micro_batch_size` | Per-GPU microbatch size | 1-4 (depends on GPU memory) |
| `lr` | Learning rate | 1e-4 to 3e-4 |
| `warmup_iters` | Linear warmup steps | 5-10% of max_iters |
| `grad_clip` | Gradient clipping | 1.0 |
| `max_iters` | Total iterations | Depends on dataset |

**Micro-batch sizing:**

```
num_microbatches = global_batch_size / (micro_batch_size * world_size)
```

For pipeline parallelism, `num_microbatches >= num_stages` is recommended for 1F1B efficiency.

### Parallelism Settings

#### Strategy Options

This framework implements three parallelism strategies from recent research:

---

##### **ASTEROID Strategy**

**Paper**: ASTEROID — Heterogeneous Pipeline Parallelism for edge clusters

**Planning Algorithm**: Dynamic Programming (DP) with memory-aware micro-batch allocation

**How it works**:
1. **Layer Partitioning**: Uses DP to find optimal layer boundaries across heterogeneous devices
2. **Micro-batch Allocation**: Memory-aware algorithm (Algorithm 1 in paper) that:
   - Allocates micro-batches proportional to compute capacity
   - Respects memory budget: `Mem_p(β) = Mem_MOD + Mem_OPT + K_p × Mem_ACT(β)`
   - Performs straggler offloading to balance execution times
3. **DP Objective**: Minimize `max(stage_time)` subject to memory constraints

**DP Formulation**:
```
dp[l][s] = minimum bottleneck time to assign layers 0..l to stages 0..s
dp[l][s] = min over cut points c: max(dp[c][s-1], time(layers c+1..l on stage s) + comm_cost)
```

**Schedule**: 1F1B (one-forward-one-backward) for lower pipeline bubble

---

##### **Confident Strategy**

**Paper**: Confident — Fault-tolerant distributed training

**Planning Algorithm**: Bottleneck-minimizing DP partition

**How it works**:
1. **Stage Partition**: DP finds partition points that minimize the slowest stage
2. **Prefix-sum Optimization**: Precomputes cumulative layer times per stage for O(1) range queries
3. **Communication-aware**: Includes inter-stage communication cost in the DP transition

**DP Formulation**:
```
dp[end][stage] = min bottleneck to cover layers 0..end using stages 0..stage
Transition: dp[end][s] = min over cuts: max(dp[cut][s-1], range_cost(s, cut+1, end) + comm)
```

**Fault Tolerance**: Passive backward-timeout detection — if backward pass exceeds threshold, upstream failure is assumed

**Schedule**: 1F1B

---

##### **DT-FM Strategy**

**Paper**: DT-FM — Distributed Training for Foundation Models

**Planning Algorithm**: GCMA (Genetic/Crossover-Mutation Algorithm) + DP

**How it works**:
1. **GCMA Topology Search**: Evolutionary algorithm to find optimal device-to-stage assignment
   - Population of device orderings
   - Crossover: Combine two orderings at random cut point
   - Mutation: Swap adjacent devices
   - Fitness: Inter-stage bandwidth + intra-stage communication cost
2. **DP Layer Partition**: After device assignment, uses DP to partition layers

**GCMA Parameters**:
- `population_size`: Number of candidate orderings (default: 100)
- `gcma_trails`: Number of evolution iterations (default: 4900)

**Schedule**: GPipe (all-forward then all-backward) — simpler but larger pipeline bubble

---

#### Profiling Parameters

The profiler (`AsteroidProfiler`) measures the following parameters per device:

| Parameter | Description | How Measured |
|-----------|-------------|---------------|
| `exec_times[device][layer][batch_size]` | (forward_ms, backward_ms) per layer | CUDA events timing over 20 iterations, median of last 17 |
| `activation_sizes[layer]` | Activation memory per sample (bytes) | `seq_len × d_model × 4` (float32) |
| `weight_sizes[layer]` | Parameter memory per layer (bytes) | Sum of `param.numel() × element_size()` |
| `bandwidths[(src, dst)]` | Device-to-device bandwidth (MB/s) | 50MB tensor transfer, median of 7 trials |
| `peak_memory` | Peak GPU memory during forward+backward | `torch.cuda.max_memory_allocated()` |

**Batch Size Profiling**: Non-linear scaling — profiles at multiple batch sizes because smaller batches underutilize GPU (captured by `bs^0.85` scaling factor).

**Memory Model** (ASTEROID paper Eq. 3):
```
Mem_p(β) = Mem_MOD + Mem_OPT + K_p × Mem_ACT(β)

Where:
- Mem_MOD = model weights for stage p
- Mem_OPT = optimizer states (2× weights for Adam)
- Mem_ACT(β) = activations for micro-batch size β
- K_p = 2(P - p) - 1 = number of in-flight activations at stage p
```

---

#### Strategy Summary

| Strategy | Planning Algorithm | Schedule | Best For |
|----------|-------------------|----------|----------|
| `asteroid` | DP + Memory-aware micro-batch allocation | 1F1B | Heterogeneous clusters, memory-constrained |
| `confident` | Bottleneck-minimizing DP | 1F1B | Fault-tolerant training |
| `dtfm` | GCMA evolutionary search + DP | GPipe | Large homogeneous clusters |

#### schedule_type Options

| Schedule | Description | When to Use |
|----------|-------------|-------------|
| `gpipe` | All-forward then all-backward | Simple debugging, baseline |
| `1f1b` | Interleaved forward/backward | Production (lower bubble) |

#### num_stages vs world_size

- `world_size`: Total number of GPU workers
- `num_stages`: Number of pipeline stages

With pure pipeline parallelism: `num_stages == world_size`

With hybrid DP+PP: `world_size = num_stages × data_parallel_size`

### Fault Tolerance

#### replication_mode Options

| Mode | Description | Overhead |
|------|-------------|----------|
| `none` | No replication | None |
| `local` | CPU clone on same node | Low |
| `topology` | Ring backup to neighbor stage | Medium |
| `global` | Full copies on all nodes | High |
| `all` | Combines local + topology + global | Highest |

#### Checkpoint Strategy

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `async` | CPU clone + background thread save | Production |
| `sync` | Blocking torch.save | Debugging |

---

## Deployment Guide

### Prerequisites

1. **Python Virtual Environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install ansible
```

2. **Ansible Configuration**

The `deploy/ansible.cfg` configures Ansible:

```ini
[defaults]
inventory = inventory.ini
vault_password_file = ~/.asteroid_vault_pass
host_key_checking = False
remote_user = ubuntu
timeout = 30

[privilege_escalation]
become = True
become_method = sudo

[ssh_connection]
pipelining = True
ssh_args = -o ControlMaster=auto -o ControlPersist=60s -o StrictHostKeyChecking=no
```

### Ansible Setup

#### 1. Create Vault Password File

```bash
echo "your-secure-password" > ~/.asteroid_vault_pass
chmod 600 ~/.asteroid_vault_pass
```

#### 2. Create Encrypted Secrets

```bash
cd deploy

# Option A: Create from template
cp secrets.yml.template secrets.yml
# Edit secrets.yml with your SSH and sudo passwords:
#   ansible_ssh_pass: "your-ssh-password"
#   ansible_become_pass: "your-sudo-password"
ansible-vault encrypt secrets.yml

# Option B: Create interactively
ansible-vault create secrets.yml
# Enter:
# ---
# ansible_ssh_pass: "your-ssh-password"
# ansible_become_pass: "your-sudo-password"
```

#### 3. Update Inventory

Edit `deploy/inventory.ini`:

```ini
[master]
# Master node (rank 0) - runs the PyTorch rendezvous
master_node ansible_host=10.203.54.11 ansible_user=ubuntu rank=0 gpu_id=0 nic=ens33 hostname=master

[workers]
# Worker nodes (rank 1+)
worker1 ansible_host=10.203.54.10 ansible_user=ubuntu rank=1 gpu_id=0 nic=ens33 hostname=worker1
worker2 ansible_host=10.203.53.10 ansible_user=ubuntu rank=2 gpu_id=0 nic=ens33 hostname=worker2
worker3 ansible_host=10.203.56.10 ansible_user=ubuntu rank=3 gpu_id=0 nic=ens33 hostname=worker3

[all:vars]
ansible_python_interpreter=/usr/bin/python3

[cluster:children]
master
workers
```

### SSH Key Distribution

For passwordless SSH (optional but recommended):

```bash
# Generate SSH key on deployment machine
ssh-keygen -t ed25519 -f ~/.ssh/asteroid_key -N ""

# Distribute to all nodes
for node in 10.203.54.11 10.203.54.10 10.203.53.10 10.203.56.10; do
  ssh-copy-id -i ~/.ssh/asteroid_key.pub ubuntu@$node
done

# Update Ansible to use key
# Add to inventory.ini under [all:vars]:
# ansible_ssh_private_key_file=~/.ssh/asteroid_key
```

### Full Deployment

The `deploy.sh` script is the single entry-point for all deployment operations. It auto-generates `inventory.ini` from the `cluster.nodes` section in `asteroid.yaml`, ensuring configuration stays in sync.

#### Command Reference

```bash
Usage: deploy.sh [OPTIONS]

Options:
  --phase PHASE       Run a single phase (see Phases below)
  --strategy NAME     Override parallelism strategy: asteroid | confident | dtfm
  --config PATH       Path to asteroid.yaml (default: ./asteroid.yaml)
  --skip-k3s          Skip K3s installation (already installed)
  --skip-gpu          Skip GPU/NVIDIA setup (already configured)
  --skip-profile      Skip profiling (use existing profiles)
  --skip-build        Skip Docker image build (use existing image)
  --skip-plan         Skip HPP planning (use existing plan)
  --redeploy          Only regenerate manifests and redeploy jobs
  --monitor           Launch training monitor after deployment
  --status            Show current cluster and training status
  --status --watch    Continuously refresh status (every 5s)
  --checkpoints [DIR] Collect saved checkpoints from all nodes
  --merge-checkpoints [DIR] [ITER]  Merge rank checkpoints into single model file
  --stop              Stop all training and clean up resources
  -h, --help          Show help message
```

#### Common Usage Patterns

```bash
# First-time full deployment
./deploy.sh

# Skip infrastructure (K3s and GPU already set up)
./deploy.sh --skip-k3s --skip-gpu

# Quick redeploy after code changes (fastest)
./deploy.sh --redeploy

# Use cached profiles and plan
./deploy.sh --skip-profile --skip-plan

# Deploy with a different strategy
./deploy.sh --strategy confident

# Monitor training progress
./deploy.sh --phase monitor

# Check cluster status
./deploy.sh --status
./deploy.sh --status --watch    # Auto-refresh

# Collect model checkpoints from all ranks
./deploy.sh --checkpoints ./my_checkpoints

# Merge distributed checkpoints into single model file
./deploy.sh --merge-checkpoints ./my_checkpoints 500  # Merge iteration 500

# Stop everything and clean up
./deploy.sh --stop
```

### Deployment Phases

The deployment runs these phases in order:

| Phase | Description | Skip Flag |
|-------|-------------|-----------|
| 1. k3s | Install K3s cluster | `--skip-k3s` |
| 2. gpu | Setup NVIDIA runtime + device plugin | `--skip-gpu` |
| 3. profile | Run hardware profiling on all nodes | `--skip-profile` |
| 4. plan | Run HPP dynamic programming planner | `--skip-plan` |
| 5. build | Build Docker image + distribute to nodes | `--skip-build` |
| 6. manifests | Generate K8s Job manifests | - |
| 7. deploy | Apply manifests + start training | - |
| 8. monitor | Live training dashboard | - |
| 9. tensorboard | Persistent TensorBoard dashboard | - |
| 10. mps | Start NVIDIA MPS on all nodes (opt-in) | - |

Run individual phases:

```bash
./deploy.sh --phase k3s
./deploy.sh --phase gpu
./deploy.sh --phase profile
./deploy.sh --phase plan
./deploy.sh --phase build
./deploy.sh --phase manifests
./deploy.sh --phase deploy
./deploy.sh --phase monitor
./deploy.sh --phase tensorboard
./deploy.sh --phase mps
```

#### Auto-Generated Inventory

The `deploy.sh` script automatically generates `deploy/inventory.ini` from the `cluster.nodes` section in `asteroid.yaml`. You only need to maintain one configuration file:

```yaml
# asteroid.yaml
cluster:
  nodes:
    - ip: 10.203.54.11
      hostname: master
      nic: ens33
      gpu_id: 0
      memory_mb: 46068
      rank: 0
      role: master
    - ip: 10.203.54.10
      hostname: worker1
      ...
```

This generates:
```ini
# deploy/inventory.ini (auto-generated - DO NOT EDIT)
[master]
master_node ansible_host=10.203.54.11 ansible_user='ubuntu' rank=0 ...

[workers]
worker1_node ansible_host=10.203.54.10 ansible_user='ubuntu' rank=1 ...
```

#### Checkpoint Management

Distributed training saves checkpoints per-rank (each GPU saves its pipeline stage). Use these commands to collect and merge them into a single model file.

**1. Collect checkpoints from all nodes:**
```bash
./deploy.sh --checkpoints ./my_checkpoints
```

This SSHs into each cluster node and copies checkpoints from `/var/lib/asteroid/checkpoints/rank-N/` to your local machine:
```
./my_checkpoints/
├── rank-0/
│   ├── ckpt_e0_i100_r0.pt
│   ├── ckpt_e0_i200_r0.pt
│   └── ...
├── rank-1/
│   └── ...
├── rank-2/
│   └── ...
└── rank-3/
    └── ...
```

**2. List available iterations:**
```bash
python -m asteroid.ft.merge_checkpoints -d ./my_checkpoints --list
# Output:
# Available iterations in my_checkpoints:
#   iteration 100: 4 rank(s)
#   iteration 200: 4 rank(s)
#   ...
```

**3. Merge into a single model file:**
```bash
# Via deploy.sh
./deploy.sh --merge-checkpoints ./my_checkpoints 500

# Or directly
python -m asteroid.ft.merge_checkpoints -d ./my_checkpoints -i 500 -o model.pt
```

**Output:** A unified checkpoint file containing:
- `model_state_dict`: Full model weights (all layers merged)
- `config`: Training configuration
- `epoch`, `iteration`: Training progress

**4. Load merged model for inference:**
```python
import torch
from asteroid.model.stage import AsteroidStage
from asteroid.core.config import AsteroidConfig

# Load merged checkpoint
ckpt = torch.load("model.pt", map_location="cpu")
cfg = AsteroidConfig.from_dict(ckpt["config"])

# Create full model (all layers on one device)
model = AsteroidStage(cfg, start_layer=0, end_layer=cfg.num_layers,
                      is_first=True, is_last=True)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

### Phase Details

#### Phase 3: Profiling

Profiling runs `profile_node.py` on each node to measure:
- Per-layer forward/backward times at multiple batch sizes
- Network bandwidth between nodes (via iperf3)

Profiles are saved to `./profiles/profile_<hostname>_rank<N>.json`.

#### Phase 4: Planning

The HPP planner (`run_planner.py`) uses dynamic programming to find optimal:
- Layer partitioning across stages
- Microbatch allocation per stage
- Minimizes total pipeline latency

Output: `hpp_plan.json`

```json
{
  "num_stages": 4,
  "partition_points": [8, 16, 24],
  "device_groups": {"0": [0], "1": [1], "2": [2], "3": [3]},
  "estimated_latency_ms": 245.3,
  "node_mapping": {
    "0": {"hostname": "master", "ip": "10.203.54.11"},
    ...
  }
}
```

#### Phase 5: Build

Builds and distributes the Docker image:

```bash
# Manual build
docker build -t asteroid:latest .

# Export and distribute manually
docker save asteroid:latest -o /tmp/asteroid.tar
scp /tmp/asteroid.tar ubuntu@<node>:/tmp/
ssh ubuntu@<node> "k3s ctr images import /tmp/asteroid.tar"
```

---

## Pipeline Schedules

### GPipe

GPipe executes all forward passes, then all backward passes:

```
Stage 0: F0 F1 F2 F3 ─── B3 B2 B1 B0
Stage 1:    F0 F1 F2 F3 ─── B3 B2 B1 B0
Stage 2:       F0 F1 F2 F3 ─── B3 B2 B1 B0
Stage 3:          F0 F1 F2 F3 ─── B3 B2 B1 B0
```

**Pros**: Simple, stable
**Cons**: Large pipeline bubble (idle time)

### 1F1B (One Forward One Backward)

1F1B interleaves forward and backward passes:

```
Stage 0: F0 F1 F2 F3    B0    B1    B2    B3
Stage 1:    F0 F1 F2 F3    B0    B1    B2    B3
Stage 2:       F0 F1 F2 F3    B0    B1    B2    B3
Stage 3:          F0 F1 F2 F3    B0    B1    B2    B3
```

After warmup forwards, each stage alternates between one forward and one backward, reducing peak memory and bubble time.

**Pros**: Lower memory, smaller bubble
**Cons**: More complex synchronization

**Configuration:**

```yaml
parallelism:
  schedule_type: "1f1b"   # or "gpipe"
```

---

## Supported Models

### GPT-2

```yaml
model:
  model_type: gpt2
  task_type: lm           # or classification
  num_layers: 12          # 12 (small), 24 (medium), 36 (large), 48 (xl)
  embedding_dim: 768      # 768, 1024, 1280, 1600
  num_heads: 12           # 12, 16, 20, 25
  n_kv_heads: 12          # Same as num_heads (MHA)
  d_ff: 3072              # 4 * embedding_dim typically
  vocab_size: 50257
  dropout: 0.1
```

### LLaMA

```yaml
model:
  model_type: llama
  task_type: lm
  num_layers: 32          # 32 (7B), 40 (13B), 60 (33B), 80 (65B)
  embedding_dim: 4096     # 4096, 5120, 6656, 8192
  num_heads: 32           # 32, 40, 52, 64
  n_kv_heads: 32          # GQA heads (32 for 7B/13B, 8 for 70B)
  d_ff: 11008             # SwiGLU dimension
  vocab_size: 32000       # LLaMA tokenizer
  dropout: 0.0
  use_flash_attention: true
```

**LLaMA Features:**
- RMSNorm (instead of LayerNorm)
- Rotary Position Embedding (RoPE)
- SwiGLU activation (instead of GELU)
- Grouped Query Attention (GQA)

### Custom Models

Register custom blocks and heads:

```python
from asteroid.model.stage import register_model, register_task

# Custom block
class MyTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ... your implementation
    
    def forward(self, x):
        # ... your implementation
        return x

# Register
register_model(
    name="my_model",
    block_class=MyTransformerBlock,
    causal=True,
    uses_rope=False
)

# Use in config
# model:
#   model_type: my_model
```

### HuggingFace Adapter

Load pretrained models directly:

```yaml
model:
  model_type: llama
  hf_model_name: "meta-llama/Llama-2-7b"  # Any HF model
```

The adapter automatically:
- Downloads weights from HuggingFace Hub
- Extracts transformer layers
- Wraps them in `AsteroidStage` objects
- Preserves pretrained weights

**Supported HF architectures:**
- GPT-2 (`gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`)
- LLaMA/LLaMA-2 (`meta-llama/Llama-2-7b`, etc.)
- Any model with `.layers`, `.h`, or `.layer` attribute

---

## Monitoring & Debugging

### Live Dashboard

```bash
./deploy.sh --phase monitor
# or
./monitor.sh
```

Shows:
- Per-rank pod status
- GPU utilization
- Training progress (loss, throughput)
- Real-time log streaming

### Status Check

```bash
./deploy.sh --status
# Or with auto-refresh:
./deploy.sh --status --watch
```

### TensorBoard

```bash
./deploy.sh --phase tensorboard
```

Opens TensorBoard at `http://localhost:6006` with per-rank metrics:
- Training loss
- Learning rate
- Throughput (tokens/sec)
- GPU memory usage
- Forward/backward timing

### Kubectl Commands

```bash
# List pods
kubectl get pods -l app=asteroid -o wide

# Stream all logs
kubectl logs -f -l app=asteroid

# Logs from specific rank
kubectl logs -f asteroid-rank-0

# Describe pod (for debugging)
kubectl describe pod asteroid-rank-0

# Execute shell in pod
kubectl exec -it asteroid-rank-0 -- bash
```

### Common Debug Commands

```bash
# Check GPU allocation
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable."nvidia\.com/gpu"

# Check NVIDIA device plugin
kubectl get pods -n kube-system -l name=nvidia-device-plugin-ds

# Check pod events
kubectl get events --sort-by='.lastTimestamp' | grep asteroid

# Check node resources
kubectl describe node <node-name> | grep -A 10 "Allocated resources"
```

---

## Checkpointing

### Configuration

```yaml
fault_tolerance:
  checkpoint_strategy: async    # async (recommended) | sync
  checkpoint_dir: ./checkpoints
  checkpoint_interval: 100      # Save every N iterations (0 = disabled)
```

### Checkpoint Location

Checkpoints are saved to `/var/lib/asteroid/checkpoints/rank-<N>/` on each node.

### Collecting Checkpoints

```bash
./deploy.sh --checkpoints
# or specify directory:
./deploy.sh --checkpoints ./my_checkpoints
```

### Checkpoint Contents

Each `.pt` file contains:
- `epoch`: Training epoch
- `iteration`: Training iteration
- `model_state_dict`: Model weights for this stage
- `optimizer_state_dict`: Optimizer state
- `config`: Model configuration
- `stage_idx`: Pipeline stage index
- `start_layer`, `end_layer`: Layer boundaries for this stage
- `is_first`, `is_last`: Whether this stage has embeddings/head

### Merging Checkpoints

Since each rank saves only its pipeline stage, use the merge utility to combine them:

```bash
# List available iterations
python -m asteroid.ft.merge_checkpoints -d ./my_checkpoints --list

# Merge a specific iteration
./deploy.sh --merge-checkpoints ./my_checkpoints 500
# Or:
python -m asteroid.ft.merge_checkpoints -d ./my_checkpoints -i 500 -o model.pt
```

### Loading Checkpoints

**Single rank checkpoint (for resuming distributed training):**
```python
import torch
checkpoint = torch.load("ckpt_e0_i500_r0.pt")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

**Merged model (for inference):**
```python
import torch
from asteroid.model.stage import AsteroidStage
from asteroid.core.config import AsteroidConfig

ckpt = torch.load("merged_i500.pt")
cfg = AsteroidConfig.from_dict(ckpt["config"])

# Full model on single device
model = AsteroidStage(cfg, start_layer=0, end_layer=cfg.num_layers,
                      is_first=True, is_last=True)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

---

## Troubleshooting

### Common Issues

#### 1. "NCCL error: unhandled cuda error"

**Cause**: NCCL initialization failed, often due to network issues or GPU access.

**Solutions**:
```yaml
# In asteroid.yaml:
deploy:
  nccl_p2p_disable: "1"      # Disable P2P (no NVLink)
  nccl_ib_disable: "1"       # Disable InfiniBand
  nccl_debug: INFO           # Enable debug logging
```

```bash
# Verify GPU access in pods
kubectl exec -it asteroid-rank-0 -- nvidia-smi
```

#### 2. "GPU not found" / "nvidia.com/gpu: 0"

**Cause**: NVIDIA device plugin not running or containerd not configured.

**Solutions**:
```bash
# Check device plugin
kubectl get pods -n kube-system -l name=nvidia-device-plugin-ds
kubectl logs -n kube-system -l name=nvidia-device-plugin-ds

# Restart device plugin
kubectl delete pods -n kube-system -l name=nvidia-device-plugin-ds

# Check containerd config (must have nvidia runtime)
cat /var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl

# Reapply GPU setup
./deploy.sh --phase gpu
```

#### 3. "Connection refused" during distributed init

**Cause**: Master node not reachable or port blocked.

**Solutions**:
```bash
# Check master node is running
kubectl get pods -l app=asteroid -o wide

# Verify network connectivity
kubectl exec -it asteroid-rank-1 -- curl -v telnet://asteroid-rank-0:29500

# Check firewall
sudo ufw status
sudo ufw allow 29500:29600/tcp
```

#### 4. Pods stuck in "Pending"

**Cause**: Insufficient resources or node selector issues.

**Solutions**:
```bash
# Check events
kubectl describe pod asteroid-rank-0

# Common issues:
# - "Insufficient nvidia.com/gpu" → GPU not detected
# - "node(s) didn't match Pod's node affinity" → Wrong hostname in manifest

# Verify node names match
kubectl get nodes
# Compare with hostnames in asteroid.yaml
```

#### 5. "OOM Killed" or "CUDA out of memory"

**Cause**: Microbatch size too large for GPU memory.

**Solutions**:
```yaml
# In asteroid.yaml:
training:
  micro_batch_size: 1       # Reduce microbatch size
  
model:
  max_seq_len: 256          # Reduce sequence length
```

#### 6. Training hangs at iteration 0

**Cause**: Deadlock in pipeline communication, often due to schedule mismatch.

**Solutions**:
```yaml
# In asteroid.yaml:
parallelism:
  schedule_type: "gpipe"    # Use simpler schedule for debugging
```

```bash
# Enable NCCL debug logging
deploy:
  nccl_debug: TRACE
  
# Check logs for where training stops
kubectl logs asteroid-rank-0 | tail -50
kubectl logs asteroid-rank-3 | tail -50  # Last stage
```

#### 7. Slow training / low throughput

**Causes**: Network bottleneck, suboptimal partitioning, or excessive synchronization.

**Solutions**:
```bash
# Check network bandwidth
iperf3 -c <other-node-ip>

# Re-run profiling with fresh profiles
rm -rf profiles/
./deploy.sh --phase profile
./deploy.sh --phase plan
```

#### 8. containerd CNI not initialized

**Cause**: On newly added nodes, the containerd config may be missing CNI paths.

**Solution**: The `config.toml.tmpl` must have version 3 format with proper CNI paths:

```toml
version = 3

[plugins]
  [plugins."io.containerd.grpc.v1.cri"]
    [plugins."io.containerd.grpc.v1.cri".cni]
      bin_dir = "/opt/cni/bin:/var/lib/rancher/k3s/data/current/bin"
      conf_dir = "/var/lib/rancher/k3s/agent/etc/cni/net.d"
```

Copy from a working node:
```bash
# From working node
scp ubuntu@working-node:/var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl /tmp/

# To broken node
scp /tmp/config.toml.tmpl ubuntu@broken-node:/tmp/
ssh ubuntu@broken-node "sudo cp /tmp/config.toml.tmpl /var/lib/rancher/k3s/agent/etc/containerd/"
ssh ubuntu@broken-node "sudo systemctl restart k3s-agent"
```

### Deep Clean / Reset

If things are in a bad state:

```bash
./deploy.sh --stop
```

This will:
- Delete all Kubernetes resources (jobs, pods, services, configmaps)
- Kill GPU processes on all nodes via SSH
- Clear NCCL shared memory (`/dev/shm/nccl-*`, `/dev/shm/torch_*`)
- Remove checkpoints from all nodes
- Clear generated manifests, profiles, and hpp_plan.json
- Kill local monitoring processes
- Reset GPU memory via `torch.cuda.empty_cache()`

### Logs and Diagnostics

```bash
# Full deployment log
./deploy.sh 2>&1 | tee deploy.log

# Per-component logs
kubectl logs asteroid-rank-0 > rank0.log
kubectl logs asteroid-rank-1 > rank1.log
# ... etc

# System logs on nodes
ssh ubuntu@<node> "journalctl -u k3s-agent -n 100"
ssh ubuntu@<node> "dmesg | tail -50"
```

---

## API Reference

### Core Classes

#### AsteroidConfig

```python
from asteroid.core.config import AsteroidConfig

cfg = AsteroidConfig.from_yaml("asteroid.yaml")

# Key attributes
cfg.model_type         # "gpt2", "llama", "encoder"
cfg.task_type          # "lm", "classification"
cfg.num_layers         # Total transformer layers
cfg.embedding_dim      # Hidden dimension
cfg.num_heads          # Attention heads
cfg.n_kv_heads         # GQA key/value heads
cfg.global_batch_size  # Total batch size
cfg.micro_batch_size   # Per-GPU microbatch
cfg.world_size         # Number of workers
cfg.num_stages         # Pipeline stages
cfg.schedule_type      # "gpipe", "1f1b"
```

#### AsteroidStage

```python
from asteroid.model.stage import AsteroidStage

# Create a pipeline stage
stage = AsteroidStage(
    config=cfg,
    start_layer=0,
    end_layer=8,
    is_first=True,   # Has embedding
    is_last=False    # No task head
)

# Forward pass
output = stage(input_tensor)  # [batch, seq_len, dim]
# or with labels (last stage)
loss = stage(input_tensor, labels)
```

#### HPPPlanConfig

```python
from asteroid.core.config import HPPPlanConfig

# Load from planner output
plan = HPPPlanConfig.from_json(json.load(open("hpp_plan.json")))

plan.num_stages          # Number of pipeline stages
plan.partition_points    # Layer boundaries [8, 16, 24]
plan.device_groups       # {stage_idx: [rank_ids]}
plan.estimated_latency_ms
```

#### Pipeline Schedules

```python
from asteroid.pipeline.schedule import build_schedule

# Build schedule for all stages
timeline = build_schedule(
    schedule_type="1f1b",  # or "gpipe"
    num_stages=4,
    num_microbatches=8
)

# timeline[stage_idx] = [(action, microbatch_idx), ...]
# action = "F" (forward) or "B" (backward)
```

### Model Registry

```python
from asteroid.model.stage import MODEL_REGISTRY, register_model

# View registered models
print(MODEL_REGISTRY)
# {'gpt2': {'block_class': GPT2Block, 'causal': True, 'uses_rope': False},
#  'llama': {'block_class': LlamaBlock, 'causal': True, 'uses_rope': True},
#  'encoder': {'block_class': EncoderBlock, 'causal': False, 'uses_rope': False}}

# Register custom model
register_model(
    name="my_model",
    block_class=MyBlock,
    causal=True,
    uses_rope=False
)
```

### Fault Tolerance

```python
from asteroid.ft.fault_tolerance import AsteroidFaultTolerance

ft = AsteroidFaultTolerance(state_manager, checkpoint_dir="./ckpts")

# Start heartbeat monitoring
ft.start_heartbeat(rank=0, interval=2.0)

# Save checkpoint
path = ft.save_checkpoint(epoch=0, iter_id=100, model=model, optimizer=opt)

# Load checkpoint
state = ft.load_latest_checkpoint()

# Detect failure
if ft.detect_failure(prev_iter, timeout_ms=30000):
    ft.handle_passive_timeout(prev_iter)
```

---

## Local Testing

Test locally without a cluster using `mp.spawn`:

```bash
# 2 GPUs, 2 pipeline stages
python worker.py --local --world-size 2 --num-stages 2

# With existing plan
python worker.py --local --world-size 4 --plan hpp_plan.json

# Skip profiling (use defaults)
python worker.py --local --world-size 2 --skip-profiling
```

---

## Contributing

### Development Setup

```bash
git clone https://github.com/your-org/asteroid.git
cd asteroid_project

python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

### Code Style

```bash
# Format
black asteroid/ worker.py
isort asteroid/ worker.py

# Lint
ruff check asteroid/
mypy asteroid/
```

### Testing

```bash
# Run tests
pytest tests/

# Local multi-GPU test
python worker.py --local --world-size 2 --num-stages 2
```

### Adding New Models

1. Create block class in `asteroid/model/`
2. Register in `asteroid/model/stage.py`:
   ```python
   register_model("my_model", MyBlock, causal=True, uses_rope=False)
   ```
3. Update `asteroid.yaml` schema if needed
4. Add tests

---

## License

Apache 2.0

---

## Citation

If you use Asteroid in your research, please cite:

```bibtex
@software{asteroid2026,
  title = {Asteroid: Heterogeneous Pipeline Parallelism for Edge Clusters},
  year = {2026},
  url = {https://github.com/your-org/asteroid}
}
```

---

## Acknowledgments

Asteroid builds on research from:
- **DT-FM** — Distributed training fundamentals
- **Confident** — Fault tolerance with passive timeout detection
- **MegaScale** — Async checkpointing inspiration
- **GPipe** — Pipeline parallelism basics
- **PipeDream** — 1F1B scheduling
