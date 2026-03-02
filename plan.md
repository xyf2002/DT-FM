# Asteroid Multi-Node Edge Cluster Upgrade — Complete Execution Plan

## Executive Summary

**Objective:** Upgrade the Asteroid deep learning codebase from a single-node simulated environment (`mp.spawn`) to a true multi-node, heterogeneous edge cluster execution engine using Heterogeneous Pipeline Parallelism (HPP).

**Architecture:** Four-phase deployment pipeline:
1. **Bootstrap** — Ansible environment setup & security
2. **Profile** — Ansible-driven hardware profiling across nodes
3. **Plan** — Aggregate profiles → `hpp_plan.json` via DP planner
4. **Execute** — Jinja2-templated K8s Jobs with Headless Service discovery

**Key Technical Decisions:**
| Decision | Choice | Rationale |
|----------|--------|-----------|
| Manifest Templating | Jinja2 | Direct integration with DP planner, no Helm CLI on edge |
| K8s Job Strategy | Separate Job per stage | Heterogeneous resource requests, independent lifecycle |
| Service Discovery | Headless Service | Stable DNS for Rank 0 rendezvous |
| Distributed Init | `env://` | Standard K8s pattern, env var injection |
| Network Profiling | Full TCP bandwidth | Accurate cross-node cost modeling |
| Version Management | Runtime detection | Handles mixed x86_64 + ARM64 Jetson cluster |

---

## Project Structure (Final State)
asteroid_project/
├── asteroid/ # Core library (existing)
│ ├── core/
│ │ ├── config.py # Updated: add node_mapping to HPPPlanConfig
│ │ └── state.py
│ ├── planner/
│ │ ├── dp_planner.py # Updated: cross-node bandwidth support
│ │ └── profiler.py # Updated: TCP bandwidth measurement
│ ├── comm/
│ ├── ft/
│ ├── model/
│ ├── optim/
│ └── utils/
├── deploy/ # Deployment infrastructure
│ ├── ansible.cfg # NEW: Ansible configuration
│ ├── secrets.yml # NEW: Vault-encrypted credentials
│ ├── inventory.ini # NEW: Cluster inventory
│ ├── bootstrap_nodes.yaml # NEW: Environment bootstrap playbook
│ ├── profile_and_gather.yaml # NEW: Profiling playbook
│ ├── setup_cluster.yml # Updated: Full deployment workflow
│ ├── vars/
│ │ └── pytorch_sources.yml # NEW: Arch→wheel mappings
│ ├── templates/
│ │ ├── headless_service.yaml.j2 # NEW: K8s Headless Service
│ │ ├── stage_job.yaml.j2 # NEW: K8s Job template
│ │ └── configmap.yaml.j2 # NEW: HPP plan ConfigMap
│ ├── generate_manifests.py # NEW: Jinja2 renderer
│ └── generated/ # Output directory for rendered YAMLs
├── profile_node.py # NEW: Standalone profiler script
├── run_planner.py # Updated: Profile aggregation
├── worker.py # Refactored: env:// init, no mp.spawn
├── Dockerfile # NEW: Container image
├── cluster.conf # Existing: Node definitions
└── requirements.txt # Updated: Dependencies

| `deploy/inventory.ini` | Static inventory with node IPs, NICs, ranks, GPU IDs |
2. deploy/secrets.yml
Create with: ansible-vault create deploy/secrets.yml

3. deploy/inventory.ini
4. deploy/bootstrap_nodes.yaml
Playbook Structure:

5. deploy/vars/pytorch_sources.yml
Verification
Checkpoint 1: Standalone Node Profiler
Commit: feat(profiler): add standalone [profile_node.py](http://_vscodecontentref_/6) for edge hardware characterization

Deliverables
File	Description
profile_node.py	Self-contained profiling script
profiler.py	Updated with TCP bandwidth measurement
Task Breakdown
1. profile_node.py
CLI Interface:

Output Schema (profile_<hostname>.json):

Implementation Outline:

2. Update profiler.py
Add TCP bandwidth measurement method to existing AsteroidProfiler class:

Verification
Checkpoint 2: Ansible Profiling & Data Sync Playbook
Commit: feat(ansible): add profile_and_gather.yaml playbook for cluster-wide profiling

Deliverables
File	Description
deploy/profile_and_gather.yaml	Main profiling playbook
deploy/files/iperf_server.service	Systemd service for iperf3 server
Task Breakdown
1. deploy/profile_and_gather.yaml
2. deploy/files/iperf_server.service
Verification
Checkpoint 3: Planner Aggregation
Commit: feat(planner): update [run_planner.py](http://_vscodecontentref_/12) to aggregate profiles and emit hpp_plan.json

Deliverables
File	Description
run_planner.py	Refactored to ingest profiles and output hpp_plan.json
config.py	Updated HPPPlanConfig with node_mapping
dp_planner.py	Cross-node bandwidth support
Task Breakdown
1. Update config.py
Add node_mapping field to HPPPlanConfig:

2. Refactor run_planner.py
3. Update dp_planner.py
Add cross-node bandwidth awareness to _comm_time_inter_stage:

Verification
Checkpoint 4: Kubernetes Manifest Templates
Commit: feat(k8s): add Jinja2 templates for dynamic Job generation from hpp_plan.json

Deliverables
File	Description
deploy/templates/headless_service.yaml.j2	Headless Service for Rank 0 DNS
deploy/templates/stage_job.yaml.j2	Per-rank K8s Job template
deploy/templates/configmap.yaml.j2	ConfigMap for hpp_plan.json
deploy/generate_manifests.py	Jinja2 renderer script
Task Breakdown
1. deploy/templates/headless_service.yaml.j2
2. deploy/templates/configmap.yaml.j2
3. deploy/templates/stage_job.yaml.j2
4. deploy/generate_manifests.py
Verification
Checkpoint 5: Worker Refactor
Commit: feat(worker): refactor to multi-node with env:// init, drop mp.spawn

Deliverables
File	Description
worker.py	Refactored for multi-node execution
Task Breakdown
1. Remove Hardcoded Environment Variables
Before (lines 1-17):

After:

2. Update init_process_group
Before (line 82-85):

After:

3. Replace mp.spawn Entry Point
Before (line 457):

After:

4. Update Data Loading for Distributed
In data_utils.py, add distributed support:

5. Full Refactored worker.py
Verification
Checkpoint 6: Integration & Deployment
Commit: feat(deploy): add Dockerfile and master execution workflow

Deliverables
File	Description
Dockerfile	Multi-arch edge-aware container image
setup_cluster.yml	Full deployment Ansible playbook
deploy/distribute_image.yaml	Image distribution playbook
Task Breakdown
1. Dockerfile
2. deploy/distribute_image.yaml
3. setup_cluster.yml
Verification
Summary: Git Checkpoint Messages
Checkpoint	Commit Message
0	feat(ansible): add vault, inventory, and dynamic bootstrap_nodes.yaml for heterogeneous edge cluster
1	feat(profiler): add standalone [profile_node.py](http://_vscodecontentref_/30) for edge hardware characterization
2	feat(ansible): add profile_and_gather.yaml playbook for cluster-wide profiling
3	feat(planner): update [run_planner.py](http://_vscodecontentref_/31) to aggregate profiles and emit hpp_plan.json
4	feat(k8s): add Jinja2 templates for dynamic Job generation from hpp_plan.json
5	feat(worker): refactor to multi-node with env:// init, drop mp.spawn
6	feat(deploy): add Dockerfile and master execution workflow
Quick Reference: Full Deployment Workflow

---

## Checkpoint 0: Security & Environment Bootstrap

**Commit:** `feat(ansible): add vault, inventory, and dynamic bootstrap_nodes.yaml for heterogeneous edge cluster`

### Deliverables

| File | Description |
|------|-------------|
| `deploy/ansible.cfg` | Ansible configuration (host key checking, vault, inventory path) |
| `deploy/secrets.yml` | Vault-encrypted SSH and sudo passwords |
| `deploy/inventory.ini` | Static inventory with node IPs, NICs, ranks, GPU IDs |
| `deploy/bootstrap_nodes.yaml` | Dynamic bootstrap playbook with hardware detection |
| `deploy/vars/pytorch_sources.yml` | Architecture and CUDA version to pip index mappings |

### Task Breakdown

#### 1. `deploy/ansible.cfg`
```ini
[defaults]
inventory = inventory.ini
vault_password_file = ~/.asteroid_vault_pass
host_key_checking = False
remote_user = ubuntu
timeout = 30
gathering = smart

[privilege_escalation]
become = True
become_method = sudo

[ssh_connection]
pipelining = True
ssh_args = -o ControlMaster=auto -o ControlPersist=60s
2. deploy/secrets.yml
Create with: ansible-vault create deploy/secrets.yml

# Encrypted contents
ansible_ssh_pass: "<your-ssh-password>"
ansible_become_pass: "<your-sudo-password>"

3. deploy/inventory.ini
[master]
192.168.1.10 nic=eth0 rank=0 gpu_id=0 hostname=master-node

[workers]
192.168.1.11 nic=eth0 rank=1 gpu_id=0 hostname=edge-node-1
192.168.1.12 nic=eth0 rank=2 gpu_id=0 hostname=edge-node-2
192.168.1.13 nic=eth0 rank=3 gpu_id=0 hostname=edge-node-3

[all:vars]
ansible_python_interpreter=/usr/bin/python3
4. deploy/bootstrap_nodes.yaml
Playbook Structure:

---
- name: Bootstrap Asteroid Edge Cluster
  hosts: all
  vars_files:
    - secrets.yml
    - vars/pytorch_sources.yml
  
  tasks:
    # === PHASE 1: Hardware Detection ===
    - name: Detect system architecture
      set_fact:
        node_arch: "{{ ansible_architecture }}"  # x86_64 or aarch64

    - name: Get NVIDIA driver version
      shell: nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1
      register: nvidia_driver_raw
      ignore_errors: true

    - name: Determine CUDA major version from driver
      set_fact:
        cuda_major: >-
          {{ 12 if (nvidia_driver_raw.stdout | default('0') | float) >= 525.0 
             else (11 if (nvidia_driver_raw.stdout | default('0') | float) >= 450.0 
             else 0) }}

    - name: Detect JetPack version (ARM64 only)
      shell: cat /etc/nv_tegra_release 2>/dev/null | grep -oP 'R\K[0-9]+' | head -1
      register: jetpack_raw
      when: node_arch == 'aarch64'
      ignore_errors: true

    - name: Set JetPack version fact
      set_fact:
        jetpack_major: "{{ jetpack_raw.stdout | default('0') }}"
      when: node_arch == 'aarch64'

    # === PHASE 2: System Packages ===
    - name: Update apt cache
      apt:
        update_cache: yes
        cache_valid_time: 3600
      become: true

    - name: Install system prerequisites
      apt:
        name:
          - python3-venv
          - python3-pip
          - python3-dev
          - iperf3
          - curl
          - git
          - sshpass
          - build-essential
        state: present
      become: true

    # === PHASE 3: Python Virtual Environment ===
    - name: Create Asteroid directory
      file:
        path: /opt/asteroid
        state: directory
        mode: '0755'
      become: true

    - name: Create Python virtual environment
      command: python3 -m venv /opt/asteroid/venv
      args:
        creates: /opt/asteroid/venv/bin/python
      become: true

    - name: Upgrade pip in venv
      pip:
        name:
          - pip
          - setuptools
          - wheel
        state: latest
        virtualenv: /opt/asteroid/venv
      become: true

    # === PHASE 4: PyTorch Installation (Conditional) ===
    - name: Install PyTorch (x86_64 + CUDA 12+)
      pip:
        name: torch
        extra_args: --index-url https://download.pytorch.org/whl/cu124
        virtualenv: /opt/asteroid/venv
      when: node_arch == 'x86_64' and cuda_major | int >= 12
      become: true

    - name: Install PyTorch (x86_64 + CUDA 11.x)
      pip:
        name: torch
        extra_args: --index-url https://download.pytorch.org/whl/cu118
        virtualenv: /opt/asteroid/venv
      when: node_arch == 'x86_64' and cuda_major | int == 11
      become: true

    - name: Install PyTorch (ARM64 Jetson - JetPack 5.x)
      pip:
        name: "{{ pytorch_l4t_jp5_url }}"
        virtualenv: /opt/asteroid/venv
      when: node_arch == 'aarch64' and jetpack_major | int >= 32 and jetpack_major | int < 36
      become: true

    - name: Install PyTorch (ARM64 Jetson - JetPack 6.x)
      pip:
        name: "{{ pytorch_l4t_jp6_url }}"
        virtualenv: /opt/asteroid/venv
      when: node_arch == 'aarch64' and jetpack_major | int >= 36
      become: true

    # === PHASE 5: CuPy Installation (Conditional) ===
    - name: Install CuPy (CUDA 12.x)
      pip:
        name: cupy-cuda12x
        virtualenv: /opt/asteroid/venv
      when: cuda_major | int >= 12
      become: true

    - name: Install CuPy (CUDA 11.x)
      pip:
        name: cupy-cuda11x
        virtualenv: /opt/asteroid/venv
      when: cuda_major | int == 11
      become: true

    # === PHASE 6: Other Python Dependencies ===
    - name: Install remaining Python dependencies
      pip:
        name:
          - transformers
          - datasets
          - numpy
          - pyyaml
          - jinja2
        virtualenv: /opt/asteroid/venv
      become: true

    # === PHASE 7: K3s & NVIDIA Runtime Verification ===
    - name: Check K3s agent status
      systemd:
        name: k3s-agent
        state: started
      become: true
      ignore_errors: true
      register: k3s_status

    - name: Verify nvidia-smi is accessible
      command: nvidia-smi
      register: nvidia_smi_check
      ignore_errors: true

    - name: Report node bootstrap status
      debug:
        msg: |
          Node: {{ inventory_hostname }}
          Architecture: {{ node_arch }}
          CUDA Major: {{ cuda_major }}
          K3s Status: {{ 'Running' if k3s_status is succeeded else 'Not Running' }}
          NVIDIA SMI: {{ 'OK' if nvidia_smi_check.rc == 0 else 'Failed' }}

5. deploy/vars/pytorch_sources.yml
---
# PyTorch wheel sources by architecture and CUDA version

# x86_64 pip index URLs (handled via --index-url in tasks)
pytorch_cu124_index: "https://download.pytorch.org/whl/cu124"
pytorch_cu118_index: "https://download.pytorch.org/whl/cu118"

# ARM64 Jetson L4T wheels (direct URLs)
pytorch_l4t_jp5_url: "https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl"
pytorch_l4t_jp6_url: "https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0-cp310-cp310-linux_aarch64.whl"

# CuPy packages
cupy_cuda12: "cupy-cuda12x"
cupy_cuda11: "cupy-cuda11x"


Verification
# 1. Create vault password file
echo "your-vault-password" > ~/.asteroid_vault_pass
chmod 600 ~/.asteroid_vault_pass

# 2. Create encrypted secrets
cd deploy && ansible-vault create secrets.yml

# 3. Test connectivity
ansible -i inventory.ini all -m ping

# 4. Dry run bootstrap
ansible-playbook bootstrap_nodes.yaml --check -v

# 5. Execute bootstrap
ansible-playbook bootstrap_nodes.yaml -v

# 6. Verify PyTorch installation on all nodes
ansible -i inventory.ini all -a "/opt/asteroid/venv/bin/python -c 'import torch; print(f\"CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\")'"


Checkpoint 1: Standalone Node Profiler
Commit: feat(profiler): add standalone [profile_node.py](http://_vscodecontentref_/6) for edge hardware characterization

Deliverables
File	Description
profile_node.py	Self-contained profiling script
profiler.py	Updated with TCP bandwidth measurement
Task Breakdown
1. profile_node.py
CLI Interface:

python profile_node.py \
    --output-dir ./profiles \
    --model-config '{"num_layers": 12, "embedding_dim": 768}' \
    --network-peers "192.168.1.11:5201,192.168.1.12:5201" \
    --batch-sizes "1,2,4,8,16"


Output Schema (profile_<hostname>.json):

{
  "hostname": "edge-node-1",
  "timestamp": "2026-03-01T14:30:00Z",
  "hardware": {
    "architecture": "aarch64",
    "gpu_name": "NVIDIA Tegra Xavier",
    "gpu_memory_mb": 8192,
    "cuda_version": "11.4",
    "driver_version": "510.47"
  },
  "exec_times": {
    "0": {
      "1": {"fwd_ms": 5.2, "bwd_ms": 10.1},
      "2": {"fwd_ms": 8.7, "bwd_ms": 17.3},
      "4": {"fwd_ms": 15.1, "bwd_ms": 30.5},
      "8": {"fwd_ms": 28.9, "bwd_ms": 58.2},
      "16": {"fwd_ms": "OOM", "bwd_ms": "OOM"}
    },
    "1": { ... },
    ...
  },
  "max_memory_per_layer_mb": {
    "0": {"1": 245, "2": 312, "4": 456, "8": 743},
    ...
  },
  "bandwidths_mbps": {
    "192.168.1.10": 94.5,
    "192.168.1.12": 87.2,
    "192.168.1.13": 91.8
  }
}

Implementation Outline:
#!/usr/bin/env python3
"""
Standalone hardware profiler for Asteroid edge nodes.
Measures GPU compute times, memory usage, and network bandwidth to peers.
"""

import argparse
import json
import socket
import time
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def get_hardware_info() -> dict:
    """Collect GPU and system hardware information."""
    info = {
        "architecture": platform.machine(),
        "hostname": socket.gethostname(),
    }
    
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_mb"] = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        info["cuda_version"] = torch.version.cuda
        # Driver version from nvidia-smi would be captured separately
    
    return info


class ProfilerBlock(nn.Module):
    """Lightweight GPT-2 style block for profiling."""
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, d_ff: int = 3072):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, embed_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x


def profile_layer(
    layer: nn.Module,
    batch_size: int,
    seq_len: int = 128,
    embed_dim: int = 768,
    device: torch.device = torch.device("cuda"),
    warmup_iters: int = 3,
    profile_iters: int = 10,
) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    """
    Profile forward and backward pass times for a single layer.
    Returns (fwd_ms, bwd_ms, peak_memory_mb) or (None, None, None) on OOM.
    """
    layer = layer.to(device)
    layer.train()
    
    try:
        torch.cuda.reset_peak_memory_stats(device)
        
        # Create input
        x = torch.randn(batch_size, seq_len, embed_dim, device=device, requires_grad=True)
        target = torch.randn_like(x)
        
        # Warmup
        for _ in range(warmup_iters):
            out = layer(x)
            loss = (out - target).pow(2).mean()
            loss.backward()
            layer.zero_grad()
            x.grad = None
        
        torch.cuda.synchronize(device)
        
        # Profile forward
        fwd_times = []
        for _ in range(profile_iters):
            x = torch.randn(batch_size, seq_len, embed_dim, device=device, requires_grad=True)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            out = layer(x)
            torch.cuda.synchronize(device)
            fwd_times.append((time.perf_counter() - t0) * 1000)
        
        # Profile backward
        bwd_times = []
        for _ in range(profile_iters):
            x = torch.randn(batch_size, seq_len, embed_dim, device=device, requires_grad=True)
            out = layer(x)
            loss = (out - target).pow(2).mean()
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize(device)
            bwd_times.append((time.perf_counter() - t0) * 1000)
            layer.zero_grad()
        
        peak_mem = torch.cuda.max_memory_allocated(device) // (1024 * 1024)
        
        return (
            sum(fwd_times) / len(fwd_times),
            sum(bwd_times) / len(bwd_times),
            peak_mem,
        )
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return None, None, None
        raise


def measure_tcp_bandwidth(target_ip: str, target_port: int = 5201, data_mb: int = 10) -> Optional[float]:
    """
    Measure TCP bandwidth to a peer using raw socket transfer.
    Returns bandwidth in MB/s or None on failure.
    
    Note: Requires iperf3 server running on target: `iperf3 -s -p 5201`
    Falls back to raw socket test if iperf3 unavailable.
    """
    import subprocess
    
    try:
        # Try iperf3 first (more accurate)
        result = subprocess.run(
            ["iperf3", "-c", target_ip, "-p", str(target_port), "-t", "5", "-J"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            # bits_per_second to MB/s
            bps = data["end"]["sum_sent"]["bits_per_second"]
            return bps / 8 / 1024 / 1024
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    
    # Fallback: raw socket test
    try:
        data = b"x" * (data_mb * 1024 * 1024)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(30)
        sock.connect((target_ip, target_port))
        
        t0 = time.perf_counter()
        sock.sendall(data)
        elapsed = time.perf_counter() - t0
        sock.close()
        
        return data_mb / elapsed if elapsed > 0 else None
    except (socket.error, socket.timeout):
        return None


def main():
    parser = argparse.ArgumentParser(description="Asteroid Node Profiler")
    parser.add_argument("--output-dir", type=str, default="./profiles")
    parser.add_argument("--model-config", type=str, default='{"num_layers": 12, "embedding_dim": 768}')
    parser.add_argument("--network-peers", type=str, default="", help="Comma-separated ip:port list")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16")
    parser.add_argument("--seq-len", type=int, default=128)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_cfg = json.loads(args.model_config)
    num_layers = model_cfg.get("num_layers", 12)
    embed_dim = model_cfg.get("embedding_dim", 768)
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    
    # Collect hardware info
    hw_info = get_hardware_info()
    hostname = hw_info["hostname"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Profile each layer at each batch size
    exec_times = {}
    max_memory = {}
    
    print(f"[{hostname}] Profiling {num_layers} layers at batch sizes {batch_sizes}")
    
    for layer_idx in range(num_layers):
        exec_times[layer_idx] = {}
        max_memory[layer_idx] = {}
        
        layer = ProfilerBlock(embed_dim=embed_dim)
        
        for bs in batch_sizes:
            fwd_ms, bwd_ms, mem_mb = profile_layer(
                layer, bs, args.seq_len, embed_dim, device
            )
            
            if fwd_ms is not None:
                exec_times[layer_idx][bs] = {"fwd_ms": round(fwd_ms, 2), "bwd_ms": round(bwd_ms, 2)}
                max_memory[layer_idx][bs] = mem_mb
                print(f"  Layer {layer_idx}, BS {bs}: fwd={fwd_ms:.2f}ms, bwd={bwd_ms:.2f}ms, mem={mem_mb}MB")
            else:
                exec_times[layer_idx][bs] = {"fwd_ms": "OOM", "bwd_ms": "OOM"}
                max_memory[layer_idx][bs] = "OOM"
                print(f"  Layer {layer_idx}, BS {bs}: OOM")
        
        del layer
        torch.cuda.empty_cache()
    
    # Profile network bandwidth
    bandwidths = {}
    if args.network_peers:
        peers = [p.strip() for p in args.network_peers.split(",") if p.strip()]
        print(f"[{hostname}] Profiling network bandwidth to {len(peers)} peers")
        
        for peer in peers:
            if ":" in peer:
                ip, port = peer.split(":")
                port = int(port)
            else:
                ip, port = peer, 5201
            
            bw = measure_tcp_bandwidth(ip, port)
            if bw is not None:
                bandwidths[ip] = round(bw, 2)
                print(f"  -> {ip}: {bw:.2f} MB/s")
            else:
                bandwidths[ip] = None
                print(f"  -> {ip}: FAILED")
    
    # Build output
    profile = {
        "hostname": hostname,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "hardware": hw_info,
        "exec_times": exec_times,
        "max_memory_per_layer_mb": max_memory,
        "bandwidths_mbps": bandwidths,
    }
    
    output_file = output_dir / f"profile_{hostname}.json"
    with open(output_file, "w") as f:
        json.dump(profile, f, indent=2)
    
    print(f"[{hostname}] Profile saved to {output_file}")


if __name__ == "__main__":
    main()


2. Update profiler.py
Add TCP bandwidth measurement method to existing AsteroidProfiler class:

# Add to AsteroidProfiler class

def load_from_profile_json(self, profile_path: str, device_id: int) -> None:
    """Load profiling data from a JSON file generated by profile_node.py."""
    with open(profile_path, "r") as f:
        data = json.load(f)
    
    # Load exec times
    for layer_idx_str, bs_data in data.get("exec_times", {}).items():
        layer_idx = int(layer_idx_str)
        if device_id not in self.exec_times:
            self.exec_times[device_id] = {}
        if layer_idx not in self.exec_times[device_id]:
            self.exec_times[device_id][layer_idx] = {}
        
        for bs_str, times in bs_data.items():
            bs = int(bs_str)
            if isinstance(times.get("fwd_ms"), (int, float)):
                self.exec_times[device_id][layer_idx][bs] = (
                    times["fwd_ms"],
                    times["bwd_ms"],
                )
    
    # Load bandwidths
    for target_ip, bw_mbps in data.get("bandwidths_mbps", {}).items():
        if bw_mbps is not None:
            # Store as (src_device, target_ip) -> bandwidth
            # Will be remapped to (src_device, dst_device) after all profiles loaded
            self._ip_bandwidths[(device_id, target_ip)] = bw_mbps
# 1. Run profiler locally
python profile_node.py --output-dir ./test_profiles --batch-sizes "1,2,4"

# 2. Verify JSON output
cat ./test_profiles/profile_$(hostname).json | python -m json.tool

# 3. Test with network peers (requires iperf3 server on peer)
# On peer: iperf3 -s -p 5201
python profile_node.py --output-dir ./test_profiles --network-peers "192.168.1.11:5201"

Checkpoint 2: Ansible Profiling & Data Sync Playbook
Commit: feat(ansible): add profile_and_gather.yaml playbook for cluster-wide profiling

Deliverables
File	Description
deploy/profile_and_gather.yaml	Main profiling playbook
deploy/files/iperf_server.service	Systemd service for iperf3 server
Task Breakdown
1. deploy/profile_and_gather.yaml
---
- name: Asteroid Cluster Profiling and Data Sync
  hosts: all
  vars_files:
    - secrets.yml
  vars:
    asteroid_src: "{{ playbook_dir }}/.."
    asteroid_dest: /opt/asteroid/src
    profiles_local: "{{ playbook_dir }}/../profiles"
    venv_python: /opt/asteroid/venv/bin/python
    # Build peer list dynamically from inventory
    network_peers: "{{ groups['all'] | map('extract', hostvars, 'ansible_host') | reject('equalto', ansible_host) | map('regex_replace', '$', ':5201') | join(',') }}"
  
  tasks:
    # === PHASE 1: Sync Codebase and Dataset ===
    - name: Sync Asteroid codebase to all nodes
      synchronize:
        src: "{{ asteroid_src }}/"
        dest: "{{ asteroid_dest }}/"
        rsync_opts:
          - "--exclude=__pycache__"
          - "--exclude=*.pyc"
          - "--exclude=profiles"
          - "--exclude=.git"
      become: true

    - name: Create HuggingFace cache directory
      file:
        path: /opt/asteroid/.cache/huggingface
        state: directory
        mode: '0755'
      become: true

    - name: Sync HuggingFace dataset cache (from master)
      synchronize:
        src: "~/.cache/huggingface/datasets/sst2/"
        dest: "/opt/asteroid/.cache/huggingface/datasets/sst2/"
      become: true
      when: inventory_hostname != groups['master'][0]
      delegate_to: "{{ groups['master'][0] }}"

    # === PHASE 2: Setup iperf3 Servers ===
    - name: Copy iperf3 systemd service
      copy:
        src: files/iperf_server.service
        dest: /etc/systemd/system/iperf_server.service
      become: true

    - name: Start iperf3 server
      systemd:
        name: iperf_server
        state: started
        enabled: yes
        daemon_reload: yes
      become: true

    - name: Wait for iperf3 servers to be ready
      wait_for:
        port: 5201
        timeout: 30

    # === PHASE 3: Execute Profiler ===
    - name: Create profiles directory
      file:
        path: "{{ asteroid_dest }}/profiles"
        state: directory
        mode: '0755'
      become: true

    - name: Run profile_node.py
      command: >
        {{ venv_python }} {{ asteroid_dest }}/profile_node.py
        --output-dir {{ asteroid_dest }}/profiles
        --network-peers "{{ network_peers }}"
        --batch-sizes "1,2,4,8,16"
      environment:
        HF_HOME: /opt/asteroid/.cache/huggingface
      register: profile_result
      become: true

    - name: Display profiling output
      debug:
        var: profile_result.stdout_lines

    # === PHASE 4: Gather Profiles to Master ===
    - name: Fetch profile JSON to master
      fetch:
        src: "{{ asteroid_dest }}/profiles/profile_{{ ansible_hostname }}.json"
        dest: "{{ profiles_local }}/"
        flat: yes
      become: true

    # === PHASE 5: Cleanup iperf3 Servers ===
    - name: Stop iperf3 server
      systemd:
        name: iperf_server
        state: stopped
      become: true

2. deploy/files/iperf_server.service
[Unit]
Description=iperf3 Server for Asteroid Network Profiling
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/iperf3 -s -p 5201
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target

Verification
# 1. Ensure master has dataset cached
python -c "from datasets import load_dataset; load_dataset('glue', 'sst2')"

# 2. Run profiling playbook
ansible-playbook deploy/profile_and_gather.yaml -v

# 3. Verify profiles collected
ls -la profiles/
cat profiles/profile_*.json | head -50

Checkpoint 3: Planner Aggregation
Commit: feat(planner): update [run_planner.py](http://_vscodecontentref_/12) to aggregate profiles and emit hpp_plan.json

Deliverables
File	Description
run_planner.py	Refactored to ingest profiles and output hpp_plan.json
config.py	Updated HPPPlanConfig with node_mapping
dp_planner.py	Cross-node bandwidth support
Task Breakdown
1. Update config.py
Add node_mapping field to HPPPlanConfig:

from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class NodeInfo:
    """Physical node information for deployment."""
    hostname: str
    ip: str
    nic: str
    gpu_id: int
    memory_mb: int
    architecture: str = "x86_64"

@dataclass
class HPPPlanConfig:
    """Output of the Asteroid Planner — the HPP execution plan."""
    num_stages: int = 2
    partition_points: List[int] = field(default_factory=list)
    device_groups: Dict[int, List[int]] = field(default_factory=dict)
    micro_batch_alloc: Dict[int, Dict[int, int]] = field(default_factory=dict)
    dominant_step: int = 0
    estimated_latency_ms: float = float('inf')
    # NEW: Maps device_id (rank) to physical node information
    node_mapping: Dict[int, NodeInfo] = field(default_factory=dict)
    
    def to_json(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "num_stages": self.num_stages,
            "partition_points": self.partition_points,
            "device_groups": {str(k): v for k, v in self.device_groups.items()},
            "micro_batch_alloc": {
                str(s): {str(d): samples for d, samples in alloc.items()}
                for s, alloc in self.micro_batch_alloc.items()
            },
            "dominant_step": self.dominant_step,
            "estimated_latency_ms": self.estimated_latency_ms,
            "node_mapping": {
                str(k): {
                    "hostname": v.hostname,
                    "ip": v.ip,
                    "nic": v.nic,
                    "gpu_id": v.gpu_id,
                    "memory_mb": v.memory_mb,
                    "architecture": v.architecture,
                } for k, v in self.node_mapping.items()
            },
            "world_size": sum(len(devs) for devs in self.device_groups.values()),
        }
    
    @classmethod
    def from_json(cls, data: dict) -> "HPPPlanConfig":
        """Deserialize from JSON dict."""
        node_mapping = {}
        for k, v in data.get("node_mapping", {}).items():
            node_mapping[int(k)] = NodeInfo(**v)
        
        return cls(
            num_stages=data["num_stages"],
            partition_points=data["partition_points"],
            device_groups={int(k): v for k, v in data["device_groups"].items()},
            micro_batch_alloc={
                int(s): {int(d): samples for d, samples in alloc.items()}
                for s, alloc in data["micro_batch_alloc"].items()
            },
            dominant_step=data.get("dominant_step", 0),
            estimated_latency_ms=data.get("estimated_latency_ms", float('inf')),
            node_mapping=node_mapping,
        )

2. Refactor run_planner.py
#!/usr/bin/env python3
"""
Asteroid HPP Planner - Aggregates node profiles and generates hpp_plan.json.

Usage:
    python run_planner.py \
        --profiles-dir ./profiles \
        --cluster-conf ./cluster.conf \
        --output ./hpp_plan.json \
        --num-stages 3 \
        --num-layers 12
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from asteroid.core.config import AsteroidConfig, DeviceSpec, HPPPlanConfig, NodeInfo
from asteroid.planner.profiler import AsteroidProfiler
from asteroid.planner.dp_planner import AsteroidPlanner


def parse_cluster_conf(path: str) -> Dict[str, dict]:
    """
    Parse cluster.conf to extract node information.
    Format: IP_ADDRESS  NIC_NAME  RANK  GPU_ID
    Returns: {hostname/IP: {ip, nic, rank, gpu_id}}
    """
    nodes = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                ip, nic, rank, gpu_id = parts[0], parts[1], int(parts[2]), int(parts[3])
                nodes[ip] = {
                    "ip": ip,
                    "nic": nic,
                    "rank": rank,
                    "gpu_id": gpu_id,
                }
    return nodes


def load_profiles(profiles_dir: str) -> Dict[str, dict]:
    """Load all profile_*.json files from directory."""
    profiles = {}
    profiles_path = Path(profiles_dir)
    
    for profile_file in profiles_path.glob("profile_*.json"):
        with open(profile_file, "r") as f:
            data = json.load(f)
            hostname = data.get("hostname", profile_file.stem.replace("profile_", ""))
            profiles[hostname] = data
    
    return profiles


def build_ip_to_rank_mapping(cluster_nodes: Dict[str, dict], profiles: Dict[str, dict]) -> Dict[str, int]:
    """Build mapping from IP address to rank."""
    ip_to_rank = {}
    for ip, info in cluster_nodes.items():
        ip_to_rank[ip] = info["rank"]
    
    # Also map hostnames to ranks if we can match them
    for hostname, profile in profiles.items():
        # Try to find matching IP in cluster conf
        for ip, info in cluster_nodes.items():
            if hostname in ip or ip in hostname:
                ip_to_rank[hostname] = info["rank"]
                break
    
    return ip_to_rank


def aggregate_profiles(
    profiles: Dict[str, dict],
    cluster_nodes: Dict[str, dict],
    profiler: AsteroidProfiler,
) -> Tuple[Dict[int, DeviceSpec], Dict[int, NodeInfo]]:
    """
    Aggregate profile data into AsteroidProfiler and build device specs.
    Returns: (device_specs, node_mapping)
    """
    device_specs = {}
    node_mapping = {}
    
    # Build hostname to cluster info mapping
    hostname_to_cluster = {}
    for ip, info in cluster_nodes.items():
        hostname_to_cluster[ip] = info
    
    for hostname, profile in profiles.items():
        # Find matching cluster node
        cluster_info = None
        for ip, info in cluster_nodes.items():
            # Match by IP or hostname substring
            if ip == hostname or hostname in ip or ip in hostname:
                cluster_info = info
                break
        
        if cluster_info is None:
            print(f"Warning: No cluster.conf entry for {hostname}, skipping")
            continue
        
        rank = cluster_info["rank"]
        hw = profile.get("hardware", {})
        
        # Build DeviceSpec
        device_specs[rank] = DeviceSpec(
            device_id=rank,
            device_type=hw.get("gpu_name", "unknown"),
            memory_budget_mb=hw.get("gpu_memory_mb", 4096),
            cuda_id=cluster_info["gpu_id"],
            compute_capacity=1.0,  # Could be derived from profiled times
        )
        
        # Build NodeInfo
        node_mapping[rank] = NodeInfo(
            hostname=hostname,
            ip=cluster_info["ip"],
            nic=cluster_info["nic"],
            gpu_id=cluster_info["gpu_id"],
            memory_mb=hw.get("gpu_memory_mb", 4096),
            architecture=hw.get("architecture", "x86_64"),
        )
        
        # Load exec times into profiler
        profiler.exec_times[rank] = {}
        for layer_idx_str, bs_data in profile.get("exec_times", {}).items():
            layer_idx = int(layer_idx_str)
            profiler.exec_times[rank][layer_idx] = {}
            for bs_str, times in bs_data.items():
                bs = int(bs_str)
                if isinstance(times.get("fwd_ms"), (int, float)):
                    profiler.exec_times[rank][layer_idx][bs] = (
                        times["fwd_ms"],
                        times["bwd_ms"],
                    )
    
    # Build bandwidth matrix (rank, rank) -> MB/s
    ip_to_rank = {info["ip"]: info["rank"] for info in cluster_nodes.values()}
    
    for hostname, profile in profiles.items():
        src_rank = None
        for ip, info in cluster_nodes.items():
            if ip == hostname or hostname in ip:
                src_rank = info["rank"]
                break
        
        if src_rank is None:
            continue
        
        for target_ip, bw_mbps in profile.get("bandwidths_mbps", {}).items():
            if bw_mbps is not None and target_ip in ip_to_rank:
                dst_rank = ip_to_rank[target_ip]
                profiler.bandwidths[(src_rank, dst_rank)] = bw_mbps
    
    return device_specs, node_mapping


def main():
    parser = argparse.ArgumentParser(description="Asteroid HPP Planner")
    parser.add_argument("--profiles-dir", type=str, required=True, help="Directory with profile_*.json files")
    parser.add_argument("--cluster-conf", type=str, default="./cluster.conf", help="Path to cluster.conf")
    parser.add_argument("--output", type=str, default="./hpp_plan.json", help="Output path for hpp_plan.json")
    parser.add_argument("--num-stages", type=int, default=2, help="Number of pipeline stages")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of model layers")
    parser.add_argument("--micro-batch-size", type=int, default=4, help="Micro-batch size")
    parser.add_argument("--global-batch-size", type=int, default=256, help="Global batch size")
    args = parser.parse_args()
    
    print(f"Loading cluster configuration from {args.cluster_conf}")
    cluster_nodes = parse_cluster_conf(args.cluster_conf)
    print(f"  Found {len(cluster_nodes)} nodes in cluster.conf")
    
    print(f"Loading profiles from {args.profiles_dir}")
    profiles = load_profiles(args.profiles_dir)
    print(f"  Found {len(profiles)} profile files")
    
    # Initialize profiler and aggregate data
    profiler = AsteroidProfiler()
    device_specs, node_mapping = aggregate_profiles(profiles, cluster_nodes, profiler)
    print(f"  Aggregated specs for {len(device_specs)} devices")
    
    # Create config
    config = AsteroidConfig(
        num_layers=args.num_layers,
        world_size=len(device_specs),
        num_stages=args.num_stages,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
    )
    
    # Run planner
    print(f"Running DP planner with {args.num_stages} stages, {args.num_layers} layers")
    planner = AsteroidPlanner(
        config=config,
        profiler=profiler,
        device_specs=list(device_specs.values()),
    )
    
    plan = planner.plan()
    
    # Attach node mapping
    plan.node_mapping = node_mapping
    
    # Save plan
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(plan.to_json(), f, indent=2)
    
    print(f"\nHPP Plan saved to {output_path}")
    print(f"  Stages: {plan.num_stages}")
    print(f"  Partition points: {plan.partition_points}")
    print(f"  Device groups: {dict(plan.device_groups)}")
    print(f"  Estimated latency: {plan.estimated_latency_ms:.2f} ms")


if __name__ == "__main__":
    main()


3. Update dp_planner.py
Add cross-node bandwidth awareness to _comm_time_inter_stage:

# In AsteroidPlanner class, update _comm_time_inter_stage method:

def _comm_time_inter_stage(
    self,
    layer_idx: int,
    src_group: List[int],
    dst_group: List[int],
) -> float:
    """
    Compute communication time for inter-stage activation transfer.
    Uses profiled bandwidth when available, falls back to config default.
    """
    if layer_idx >= len(self.profiler.activation_sizes):
        return 0.0
    
    activation_mb = self.profiler.activation_sizes[layer_idx] / (1024 * 1024)
    
    # Find minimum bandwidth across all src->dst pairs
    min_bandwidth = self.config.d2d_bandwidth_mbps  # Default fallback
    
    for src_dev in src_group:
        for dst_dev in dst_group:
            # Check if we have profiled bandwidth for this pair
            if (src_dev, dst_dev) in self.profiler.bandwidths:
                bw = self.profiler.bandwidths[(src_dev, dst_dev)]
                min_bandwidth = min(min_bandwidth, bw)
    
    return (activation_mb / min_bandwidth) * 1000  # ms


Verification
# 1. Run planner with collected profiles
python run_planner.py \
    --profiles-dir ./profiles \
    --cluster-conf ./cluster.conf \
    --output ./hpp_plan.json \
    --num-stages 3

# 2. Verify plan structure
cat hpp_plan.json | python -m json.tool

# 3. Validate node_mapping present
python -c "import json; d=json.load(open('hpp_plan.json')); print('Nodes:', list(d['node_mapping'].keys()))"


Checkpoint 4: Kubernetes Manifest Templates
Commit: feat(k8s): add Jinja2 templates for dynamic Job generation from hpp_plan.json

Deliverables
File	Description
deploy/templates/headless_service.yaml.j2	Headless Service for Rank 0 DNS
deploy/templates/stage_job.yaml.j2	Per-rank K8s Job template
deploy/templates/configmap.yaml.j2	ConfigMap for hpp_plan.json
deploy/generate_manifests.py	Jinja2 renderer script
Task Breakdown
1. deploy/templates/headless_service.yaml.j2


---
# Headless Service for PyTorch distributed rendezvous
# Provides stable DNS: asteroid-master.{{ namespace }}.svc.cluster.local
apiVersion: v1
kind: Service
metadata:
  name: asteroid-master
  namespace: {{ namespace | default('default') }}
  labels:
    app: asteroid
    component: master
spec:
  clusterIP: None
  selector:
    app: asteroid
    rank: "0"
  ports:
    - name: dist
      port: {{ master_port | default(29500) }}
      targetPort: {{ master_port | default(29500) }}

2. deploy/templates/configmap.yaml.j2      
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: asteroid-config
  namespace: {{ namespace | default('default') }}
  labels:
    app: asteroid
data:
  hpp_plan.json: |
{{ hpp_plan_json | indent(4, first=True) }}

3. deploy/templates/stage_job.yaml.j2

---
# Job for Rank {{ rank }} (Stage {{ stage }})
apiVersion: batch/v1
kind: Job
metadata:
  name: asteroid-rank-{{ rank }}
  namespace: {{ namespace | default('default') }}
  labels:
    app: asteroid
    stage: "{{ stage }}"
    rank: "{{ rank }}"
spec:
  backoffLimit: 3
  ttlSecondsAfterFinished: 3600
  template:
    metadata:
      labels:
        app: asteroid
        stage: "{{ stage }}"
        rank: "{{ rank }}"
    spec:
      restartPolicy: OnFailure
      
      # Pin to specific node from hpp_plan
      nodeSelector:
        kubernetes.io/hostname: {{ node_hostname }}
      
      # Tolerations for GPU nodes
      tolerations:
        - key: "nvidia.com/gpu"
          operator: "Exists"
          effect: "NoSchedule"
      
      containers:
        - name: asteroid-worker
          image: {{ image | default('asteroid:latest') }}
          imagePullPolicy: {{ image_pull_policy | default('IfNotPresent') }}
          
          # Resource requests/limits from profiled memory
          resources:
            requests:
              memory: "{{ memory_mb }}Mi"
              cpu: "{{ cpu_request | default('1000m') }}"
              nvidia.com/gpu: "1"
            limits:
              memory: "{{ (memory_mb * 1.2) | int }}Mi"
              cpu: "{{ cpu_limit | default('4000m') }}"
              nvidia.com/gpu: "1"
          
          # Environment variables for torch.distributed
          env:
            - name: RANK
              value: "{{ rank }}"
            - name: WORLD_SIZE
              value: "{{ world_size }}"
            - name: MASTER_ADDR
              value: "asteroid-master.{{ namespace | default('default') }}.svc.cluster.local"
            - name: MASTER_PORT
              value: "{{ master_port | default(29500) }}"
            - name: LOCAL_RANK
              value: "0"
            - name: NCCL_SOCKET_IFNAME
              value: "{{ nccl_ifname }}"
            - name: CUDA_VISIBLE_DEVICES
              value: "{{ gpu_id }}"
            - name: NCCL_DEBUG
              value: "{{ nccl_debug | default('WARN') }}"
            - name: NCCL_IB_DISABLE
              value: "{{ nccl_ib_disable | default('1') }}"
            - name: HPP_PLAN_PATH
              value: "/config/hpp_plan.json"
            - name: HF_HOME
              value: "/cache/huggingface"
            
            # Optional: Pass stage and micro-batch info
            - name: STAGE_IDX
              value: "{{ stage }}"
            - name: MICRO_BATCH_SIZE
              value: "{{ micro_batch_size }}"
          
          # Mount hpp_plan.json from ConfigMap
          volumeMounts:
            - name: config-volume
              mountPath: /config
              readOnly: true
            - name: cache-volume
              mountPath: /cache
            - name: shm
              mountPath: /dev/shm
          
          # Command
          command:
            - python
            - -u
            - worker.py
          
          # Working directory
          workingDir: /app
      
      volumes:
        - name: config-volume
          configMap:
            name: asteroid-config
        - name: cache-volume
          {% if use_pvc | default(false) %}
          persistentVolumeClaim:
            claimName: asteroid-cache
          {% else %}
          emptyDir: {}
          {% endif %}
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: "{{ shm_size | default('2Gi') }}"

4. deploy/generate_manifests.py
#!/usr/bin/env python3
"""
Generate Kubernetes manifests from hpp_plan.json using Jinja2 templates.

Usage:
    python generate_manifests.py \
        --plan ./hpp_plan.json \
        --output-dir ./deploy/generated \
        --image asteroid:v1.0.0 \
        --namespace asteroid
"""

import argparse
import json
import os
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


def load_plan(path: str) -> dict:
    """Load hpp_plan.json."""
    with open(path, "r") as f:
        return json.load(f)


def generate_manifests(
    plan: dict,
    templates_dir: str,
    output_dir: str,
    image: str,
    namespace: str,
    master_port: int,
    dry_run: bool = False,
) -> None:
    """Generate all K8s manifests from templates."""
    
    # Setup Jinja2
    env = Environment(
        loader=FileSystemLoader(templates_dir),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    world_size = plan.get("world_size", sum(len(v) for v in plan["device_groups"].values()))
    node_mapping = plan.get("node_mapping", {})
    
    generated_files = []
    
    # 1. Generate ConfigMap
    configmap_template = env.get_template("configmap.yaml.j2")
    configmap_content = configmap_template.render(
        namespace=namespace,
        hpp_plan_json=json.dumps(plan, indent=2),
    )
    
    configmap_path = output_path / "00-configmap.yaml"
    if not dry_run:
        configmap_path.write_text(configmap_content)
    generated_files.append(("ConfigMap", configmap_path))
    
    # 2. Generate Headless Service
    service_template = env.get_template("headless_service.yaml.j2")
    service_content = service_template.render(
        namespace=namespace,
        master_port=master_port,
    )
    
    service_path = output_path / "01-headless-service.yaml"
    if not dry_run:
        service_path.write_text(service_content)
    generated_files.append(("Service", service_path))
    
    # 3. Generate Jobs for each rank
    job_template = env.get_template("stage_job.yaml.j2")
    
    for stage_str, devices in plan["device_groups"].items():
        stage = int(stage_str)
        
        for rank in devices:
            rank_str = str(rank)
            node_info = node_mapping.get(rank_str, {})
            
            # Get micro-batch allocation for this device
            stage_alloc = plan.get("micro_batch_alloc", {}).get(stage_str, {})
            micro_batch_size = stage_alloc.get(rank_str, 4)
            
            job_content = job_template.render(
                rank=rank,
                stage=stage,
                world_size=world_size,
                namespace=namespace,
                master_port=master_port,
                image=image,
                node_hostname=node_info.get("hostname", f"node-{rank}"),
                memory_mb=node_info.get("memory_mb", 4096),
                gpu_id=node_info.get("gpu_id", 0),
                nccl_ifname=node_info.get("nic", "eth0"),
                micro_batch_size=micro_batch_size,
            )
            
            job_path = output_path / f"02-job-rank-{rank}.yaml"
            if not dry_run:
                job_path.write_text(job_content)
            generated_files.append((f"Job (Rank {rank})", job_path))
    
    # Print summary
    print(f"\nGenerated {len(generated_files)} manifests:")
    for name, path in generated_files:
        status = "[DRY RUN]" if dry_run else "[CREATED]"
        print(f"  {status} {name}: {path}")
    
    if not dry_run:
        # Generate apply script
        apply_script = output_path / "apply.sh"
        apply_script.write_text(f"""#!/bin/bash
# Apply all Asteroid K8s manifests
set -e
kubectl apply -f {output_path}/*.yaml
echo "All manifests applied successfully"
kubectl get pods -l app=asteroid -n {namespace}
""")
        apply_script.chmod(0o755)
        print(f"\n  Apply all: bash {apply_script}")
        print(f"  Or: kubectl apply -f {output_path}/")


def main():
    parser = argparse.ArgumentParser(description="Generate K8s manifests from hpp_plan.json")
    parser.add_argument("--plan", type=str, required=True, help="Path to hpp_plan.json")
    parser.add_argument("--templates-dir", type=str, default="./deploy/templates", help="Jinja2 templates directory")
    parser.add_argument("--output-dir", type=str, default="./deploy/generated", help="Output directory")
    parser.add_argument("--image", type=str, default="asteroid:latest", help="Docker image")
    parser.add_argument("--namespace", type=str, default="default", help="K8s namespace")
    parser.add_argument("--master-port", type=int, default=29500, help="PyTorch distributed port")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be generated")
    parser.add_argument("--apply", action="store_true", help="Apply manifests after generation")
    args = parser.parse_args()
    
    print(f"Loading plan from {args.plan}")
    plan = load_plan(args.plan)
    
    print(f"Generating manifests to {args.output_dir}")
    generate_manifests(
        plan=plan,
        templates_dir=args.templates_dir,
        output_dir=args.output_dir,
        image=args.image,
        namespace=args.namespace,
        master_port=args.master_port,
        dry_run=args.dry_run,
    )
    
    if args.apply and not args.dry_run:
        import subprocess
        print("\nApplying manifests...")
        subprocess.run(["kubectl", "apply", "-f", args.output_dir], check=True)


if __name__ == "__main__":
    main()



Verification
# 1. Generate manifests (dry run)
python deploy/generate_manifests.py \
    --plan ./hpp_plan.json \
    --dry-run

# 2. Generate actual manifests
python deploy/generate_manifests.py \
    --plan ./hpp_plan.json \
    --image asteroid:v1.0.0 \
    --namespace asteroid

# 3. Validate generated YAML
kubectl apply --dry-run=client -f deploy/generated/

# 4. Inspect a specific job
cat deploy/generated/02-job-rank-0.yaml


Checkpoint 5: Worker Refactor
Commit: feat(worker): refactor to multi-node with env:// init, drop mp.spawn

Deliverables
File	Description
worker.py	Refactored for multi-node execution
Task Breakdown
1. Remove Hardcoded Environment Variables
Before (lines 1-17):

os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
os.environ["NCCL_SOCKET_IFNAME"] = "lo"
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")

After:

import os
import sys

def validate_env():
    """Validate required environment variables are set by K8s."""
    required = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
    missing = [var for var in required if var not in os.environ]
    if missing:
        print(f"ERROR: Missing required environment variables: {missing}")
        print("These should be injected by Kubernetes. Exiting.")
        sys.exit(1)
    
    # Set NCCL config from env (injected by K8s template)
    if "NCCL_SOCKET_IFNAME" not in os.environ:
        os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
    
    # Reasonable defaults for edge clusters
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_SHM_DISABLE", "0")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")


    2. Update init_process_group
Before (line 82-85):

torch.distributed.init_process_group(
    backend='nccl', init_method=cfg.dist_url,
    world_size=cfg.world_size, rank=rank,
    timeout=timedelta(seconds=120))


After:
torch.distributed.init_process_group(
    backend='nccl',
    init_method='env://',  # Reads MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE
    timeout=timedelta(seconds=120)
)


3. Replace mp.spawn Entry Point
Before (line 457):

mp.spawn(worker, args=(cfg, train_data, val_data, plan), nprocs=cfg.world_size, join=True)


def load_hpp_plan(path: str) -> HPPPlanConfig:
    """Load HPP plan from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return HPPPlanConfig.from_json(data)


def main():
    """Main entry point for distributed worker."""
    # Validate environment
    validate_env()
    
    # Get rank from environment
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # Load HPP plan
    plan_path = os.environ.get("HPP_PLAN_PATH", "./hpp_plan.json")
    if os.path.exists(plan_path):
        print(f"[Rank {rank}] Loading HPP plan from {plan_path}")
        plan = load_hpp_plan(plan_path)
    else:
        print(f"[Rank {rank}] No HPP plan found at {plan_path}, using defaults")
        plan = None
    
    # Build config
    cfg = AsteroidConfig(
        world_size=world_size,
        num_stages=plan.num_stages if plan else 2,
    )
    
    # Load data with distributed sampler
    print(f"[Rank {rank}] Loading dataset...")
    train_data, val_data = prepare_sst2(
        cfg.max_seq_len,
        cfg.embedding_dim,
        distributed=True,
        rank=rank,
        world_size=world_size,
    )
    
    # Run worker
    print(f"[Rank {rank}] Starting worker (world_size={world_size})")
    worker(rank, cfg, train_data, val_data, plan)


if __name__ == "__main__":
    main()


4. Update Data Loading for Distributed
In data_utils.py, add distributed support:

def prepare_sst2(
    max_seq_len: int = 128,
    embedding_dim: int = 768,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    """
    Load and prepare SST-2 dataset.
    When distributed=True, returns only the shard for this rank.
    """
    from datasets import load_dataset
    
    # Load dataset
    ds = load_dataset("glue", "sst2", trust_remote_code=True)
    train_ds = ds["train"]
    val_ds = ds["validation"]
    
    if distributed and world_size > 1:
        # Shard dataset across workers
        train_ds = train_ds.shard(num_shards=world_size, index=rank)
        val_ds = val_ds.shard(num_shards=world_size, index=rank)
    
    # ... rest of preprocessing ...


5. Full Refactored worker.py
#!/usr/bin/env python3
"""
Asteroid Worker - Multi-node distributed training worker.

This script is designed to be launched by Kubernetes with environment variables:
  - RANK: Global rank of this worker
  - WORLD_SIZE: Total number of workers
  - MASTER_ADDR: Hostname/IP of rank 0
  - MASTER_PORT: Port for distributed communication
  - NCCL_SOCKET_IFNAME: Network interface for NCCL
  - CUDA_VISIBLE_DEVICES: GPU to use
  - HPP_PLAN_PATH: Path to hpp_plan.json
"""

import os
import sys
import json
import time
from datetime import timedelta
from pathlib import Path
from typing import Optional, Tuple

# === Environment Validation (before any CUDA imports) ===
def validate_env():
    """Validate required environment variables are set."""
    required = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
    missing = [var for var in required if var not in os.environ]
    if missing:
        print(f"ERROR: Missing required environment variables: {missing}", file=sys.stderr)
        sys.exit(1)
    
    # Set reasonable defaults for edge clusters
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_SHM_DISABLE", "0")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")


validate_env()

# Now safe to import CUDA-dependent modules
import torch
import torch.distributed

from asteroid.core.config import AsteroidConfig, HPPPlanConfig
from asteroid.core.state import AsteroidStateManager
from asteroid.utils.logger import EVENT_LOGGER, logger
from asteroid.model.stage import AsteroidStage
from asteroid.optim.optim_utils import flatten_params, get_lr
from asteroid.ft.fault_tolerance import AsteroidFaultTolerance
from asteroid.utils.data_utils import prepare_sst2
from asteroid.planner.profiler import AsteroidProfiler
from asteroid.planner.dp_planner import AsteroidPlanner


def worker(
    rank: int,
    cfg: AsteroidConfig,
    train_data: Tuple[torch.Tensor, torch.Tensor],
    val_data: Tuple[torch.Tensor, torch.Tensor],
    plan: Optional[HPPPlanConfig] = None,
):
    """
    Main worker function for distributed training.
    
    Args:
        rank: Global rank of this worker
        cfg: Asteroid configuration
        train_data: (inputs, labels) training data tensors
        val_data: (inputs, labels) validation data tensors
        plan: HPP execution plan (optional)
    """
    # Setup device
    cuda_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    torch.manual_seed(cfg.seed + rank)
    
    print(f"[Rank {rank}] Device: {device}, CUDA: {torch.cuda.get_device_name(device)}")
    
    # Determine stage assignment from plan
    if plan is not None and plan.device_groups:
        _rank_to_stage, _rank_to_dp_pos, _stage_sizes = {}, {}, {}
        for _s, _devs in plan.device_groups.items():
            _stage_sizes[_s] = len(_devs)
            for _dp, _dev in enumerate(_devs):
                _rank_to_stage[_dev] = _s
                _rank_to_dp_pos[_dev] = _dp
        
        pp_size = plan.num_stages
        pp_rank = _rank_to_stage.get(rank, rank % pp_size)
        dp_rank = _rank_to_dp_pos.get(rank, 0)
        dp_size = _stage_sizes.get(pp_rank, 1)
        print(f"[Rank {rank}] Stage {pp_rank}, DP rank {dp_rank}/{dp_size}")
    else:
        pp_size = cfg.num_stages
        dp_size = cfg.world_size // pp_size
        pp_rank = rank % pp_size
        dp_rank = rank // pp_size
    
    # Initialize state manager
    state = AsteroidStateManager()
    state.global_rank = rank
    state.stage_idx = pp_rank
    state.device = device
    state._dp_rank = dp_rank
    state._dp_size = dp_size
    state.plan = plan
    
    # Initialize distributed process group
    if not torch.distributed.is_initialized():
        print(f"[Rank {rank}] Initializing process group...")
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            timeout=timedelta(seconds=120)
        )
        print(f"[Rank {rank}] Process group initialized")
    
    # ... rest of worker logic (process groups, model, training loop) ...
    # [Keep existing implementation from line 87 onwards]


def load_hpp_plan(path: str) -> HPPPlanConfig:
    """Load HPP plan from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return HPPPlanConfig.from_json(data)


def main():
    """Main entry point for Kubernetes-launched worker."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    print(f"[Rank {rank}] Starting Asteroid worker")
    print(f"[Rank {rank}] MASTER_ADDR={os.environ['MASTER_ADDR']}")
    print(f"[Rank {rank}] MASTER_PORT={os.environ['MASTER_PORT']}")
    print(f"[Rank {rank}] WORLD_SIZE={world_size}")
    print(f"[Rank {rank}] NCCL_SOCKET_IFNAME={os.environ.get('NCCL_SOCKET_IFNAME', 'not set')}")
    
    # Load HPP plan
    plan_path = os.environ.get("HPP_PLAN_PATH", "./hpp_plan.json")
    plan = None
    if os.path.exists(plan_path):
        print(f"[Rank {rank}] Loading HPP plan from {plan_path}")
        plan = load_hpp_plan(plan_path)
        print(f"[Rank {rank}] Plan: {plan.num_stages} stages, partition={plan.partition_points}")
    else:
        print(f"[Rank {rank}] No HPP plan found, using default configuration")
    
    # Build config
    cfg = AsteroidConfig(
        world_size=world_size,
        num_stages=plan.num_stages if plan else 2,
    )
    
    # Load data
    print(f"[Rank {rank}] Loading dataset (shard {rank}/{world_size})...")
    train_data, val_data = prepare_sst2(
        cfg.max_seq_len,
        cfg.embedding_dim,
        distributed=True,
        rank=rank,
        world_size=world_size,
    )
    print(f"[Rank {rank}] Data loaded: train={train_data[0].shape}, val={val_data[0].shape}")
    
    # Run worker
    worker(rank, cfg, train_data, val_data, plan)
    
    print(f"[Rank {rank}] Worker finished")


if __name__ == "__main__":
    main()

Verification
# 1. Local test with environment variables
RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
    python worker.py

# 2. Multi-process local test (simulated)
# Terminal 1:
RANK=0 WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 python worker.py

# Terminal 2:
RANK=1 WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 python worker.py


Checkpoint 6: Integration & Deployment
Commit: feat(deploy): add Dockerfile and master execution workflow

Deliverables
File	Description
Dockerfile	Multi-arch edge-aware container image
setup_cluster.yml	Full deployment Ansible playbook
deploy/distribute_image.yaml	Image distribution playbook
Task Breakdown
1. Dockerfile

# ============================================================================
# Asteroid Edge Training Container
# Supports both x86_64 (CUDA) and ARM64 (Jetson) architectures
# ============================================================================

# Build argument for base image selection
ARG BASE_IMAGE=pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

FROM ${BASE_IMAGE}

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    git \
    curl \
    iperf3 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional dependencies
RUN pip3 install --no-cache-dir \
    cupy-cuda12x \
    transformers \
    datasets \
    pyyaml \
    jinja2

# Copy application code
COPY asteroid/ ./asteroid/
COPY worker.py .
COPY run_planner.py .
COPY profile_node.py .

# Pre-download HuggingFace tokenizer (optional, reduces runtime downloads)
RUN python3 -c "from transformers import GPT2Tokenizer; GPT2Tokenizer.from_pretrained('gpt2')" || true

# Default environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_NCCL_BLOCKING_WAIT=1

# Entry point
ENTRYPOINT ["python3", "-u", "worker.py"]


2. deploy/distribute_image.yaml

---
# Distribute Docker image to edge nodes (for air-gapped/bandwidth-limited clusters)
- name: Distribute Asteroid Docker Image
  hosts: all
  vars_files:
    - secrets.yml
  vars:
    image_name: asteroid
    image_tag: "{{ lookup('env', 'IMAGE_TAG') | default('latest', true) }}"
    image_tarball: /tmp/asteroid_image.tar
    local_tarball: "{{ playbook_dir }}/../asteroid_image.tar"
  
  tasks:
    # === On Master: Build and Export Image ===
    - name: Build Docker image (master only)
      command: >
        docker build 
        -t {{ image_name }}:{{ image_tag }}
        -f Dockerfile
        .
      args:
        chdir: "{{ playbook_dir }}/.."
      when: inventory_hostname == groups['master'][0]
      delegate_to: localhost
      run_once: true

    - name: Export image to tarball (master only)
      command: >
        docker save -o {{ local_tarball }} {{ image_name }}:{{ image_tag }}
      when: inventory_hostname == groups['master'][0]
      delegate_to: localhost
      run_once: true

    # === Distribute to All Nodes ===
    - name: Copy image tarball to nodes
      copy:
        src: "{{ local_tarball }}"
        dest: "{{ image_tarball }}"
      become: true

    - name: Load image on nodes
      command: docker load -i {{ image_tarball }}
      become: true

    - name: Tag image for K3s containerd
      command: >
        ctr -n k8s.io image import {{ image_tarball }}
      become: true
      ignore_errors: true  # May fail if not using containerd

    - name: Verify image loaded
      command: docker images {{ image_name }}:{{ image_tag }} --format "{{ '{{' }}.Repository{{ '}}' }}:{{ '{{' }}.Tag{{ '}}' }}"
      register: image_check
      become: true

    - name: Display loaded image
      debug:
        msg: "Image loaded: {{ image_check.stdout }}"

    # === Cleanup ===
    - name: Remove tarball from nodes
      file:
        path: "{{ image_tarball }}"
        state: absent
      become: true


3. setup_cluster.yml
---
# Master playbook for full Asteroid deployment
- name: Asteroid Full Cluster Deployment
  hosts: localhost
  gather_facts: false
  vars:
    profiles_dir: "{{ playbook_dir }}/../profiles"
    plan_output: "{{ playbook_dir }}/../hpp_plan.json"
    manifests_dir: "{{ playbook_dir }}/generated"
    image_name: asteroid
    image_tag: "{{ lookup('env', 'IMAGE_TAG') | default('latest', true) }}"
    namespace: default
  
  tasks:
    # === Phase 0: Bootstrap (if not already done) ===
    - name: Check if bootstrap completed
      stat:
        path: /opt/asteroid/venv/bin/python
      register: venv_check
      delegate_to: "{{ groups['all'][0] }}"

    - name: Run bootstrap playbook if needed
      command: ansible-playbook bootstrap_nodes.yaml
      args:
        chdir: "{{ playbook_dir }}"
      when: not venv_check.stat.exists

    # === Phase 1: Profile Cluster ===
    - name: Check for existing profiles
      find:
        paths: "{{ profiles_dir }}"
        patterns: "profile_*.json"
      register: existing_profiles

    - name: Run profiling playbook
      command: ansible-playbook profile_and_gather.yaml
      args:
        chdir: "{{ playbook_dir }}"
      when: existing_profiles.matched == 0

    # === Phase 2: Generate HPP Plan ===
    - name: Run planner
      command: >
        python3 run_planner.py
        --profiles-dir {{ profiles_dir }}
        --cluster-conf {{ playbook_dir }}/../cluster.conf
        --output {{ plan_output }}
      args:
        chdir: "{{ playbook_dir }}/.."

    - name: Display generated plan
      command: cat {{ plan_output }}
      register: plan_content

    - name: Show plan summary
      debug:
        msg: "{{ plan_content.stdout | from_json }}"

    # === Phase 3: Build and Distribute Docker Image ===
    - name: Build and distribute image
      command: ansible-playbook distribute_image.yaml
      args:
        chdir: "{{ playbook_dir }}"
      environment:
        IMAGE_TAG: "{{ image_tag }}"

    # === Phase 4: Generate K8s Manifests ===
    - name: Generate K8s manifests
      command: >
        python3 deploy/generate_manifests.py
        --plan {{ plan_output }}
        --image {{ image_name }}:{{ image_tag }}
        --namespace {{ namespace }}
        --output-dir {{ manifests_dir }}
      args:
        chdir: "{{ playbook_dir }}/.."

    - name: List generated manifests
      find:
        paths: "{{ manifests_dir }}"
        patterns: "*.yaml"
      register: manifests

    - name: Display generated manifests
      debug:
        msg: "Generated {{ manifests.matched }} manifests in {{ manifests_dir }}"

    # === Phase 5: Deploy to K8s ===
    - name: Apply K8s manifests
      command: kubectl apply -f {{ manifests_dir }}/
      register: kubectl_apply

    - name: Display apply result
      debug:
        var: kubectl_apply.stdout_lines

    - name: Wait for pods to be ready
      command: >
        kubectl wait --for=condition=Ready pod
        -l app=asteroid
        -n {{ namespace }}
        --timeout=300s
      register: pod_wait
      ignore_errors: true

    - name: Get pod status
      command: kubectl get pods -l app=asteroid -n {{ namespace }} -o wide
      register: pod_status

    - name: Display final status
      debug:
        var: pod_status.stdout_lines


Verification
# 1. Build image locally
docker build -t asteroid:latest .

# 2. Test image
docker run --rm -it asteroid:latest python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 3. Run full deployment
cd deploy
ansible-playbook setup_cluster.yml -v

# 4. Monitor training
kubectl logs -f -l app=asteroid -n default

# 5. Check pod status
kubectl get pods -l app=asteroid -o wide        

Summary: Git Checkpoint Messages
Checkpoint	Commit Message
0	feat(ansible): add vault, inventory, and dynamic bootstrap_nodes.yaml for heterogeneous edge cluster
1	feat(profiler): add standalone [profile_node.py](http://_vscodecontentref_/30) for edge hardware characterization
2	feat(ansible): add profile_and_gather.yaml playbook for cluster-wide profiling
3	feat(planner): update [run_planner.py](http://_vscodecontentref_/31) to aggregate profiles and emit hpp_plan.json
4	feat(k8s): add Jinja2 templates for dynamic Job generation from hpp_plan.json
5	feat(worker): refactor to multi-node with env:// init, drop mp.spawn
6	feat(deploy): add Dockerfile and master execution workflow
Quick Reference: Full Deployment Workflow
Claude Opus 4.5 • 3x

# 1. Create vault password
echo "your-password" > ~/.asteroid_vault_pass && chmod 600 ~/.asteroid_vault_pass

# 2. Create encrypted secrets
cd deploy && ansible-vault create secrets.yml

# 3. Update inventory.ini with your nodes

# 4. Bootstrap cluster
ansible-playbook bootstrap_nodes.yaml

# 5. Profile cluster
ansible-playbook profile_and_gather.yaml

# 6. Generate plan
cd .. && python run_planner.py --profiles-dir ./profiles --output hpp_plan.json

# 7. Generate and apply K8s manifests
python deploy/generate_manifests.py --plan hpp_plan.json --apply

# 8. Monitor
kubectl logs -f -l app=asteroid