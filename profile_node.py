#!/usr/bin/env python3
"""
Asteroid Node Profiler - Standalone hardware profiling script.

Measures GPU compute times, memory usage, and network bandwidth to peers.
Outputs profile_<hostname>.json for aggregation by run_planner.py.

Usage:
    python profile_node.py --output-dir ./profiles
    python profile_node.py --output-dir ./profiles --network-peers "192.168.1.11:5201,192.168.1.12:5201"
    python profile_node.py --output-dir ./profiles --batch-sizes "1,2,4,8" --num-layers 12

Environment:
    HF_HOME: HuggingFace cache directory (optional)
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Check for required packages
try:
    import torch
    import torch.nn as nn
except ImportError:
    print("ERROR: PyTorch not found. Please install PyTorch first.")
    sys.exit(1)


def get_hardware_info() -> Dict[str, Any]:
    """Collect GPU and system hardware information."""
    info = {
        "hostname": socket.gethostname(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }
    
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["gpu_memory_mb"] = props.total_memory // (1024 * 1024)
        info["gpu_compute_capability"] = f"{props.major}.{props.minor}"
        info["cuda_version"] = torch.version.cuda or "unknown"
        info["cudnn_version"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A"
        info["num_gpus"] = torch.cuda.device_count()
    else:
        info["gpu_name"] = None
        info["gpu_memory_mb"] = 0
        info["cuda_version"] = None
        info["num_gpus"] = 0
    
    # Try to get NVIDIA driver version
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            info["driver_version"] = result.stdout.strip().split("\n")[0]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        info["driver_version"] = "unknown"
    
    return info


class ProfilerBlock(nn.Module):
    """
    Lightweight GPT-2 style transformer block for profiling.
    Mirrors the structure in asteroid/model/blocks.py.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture (GPT-2 style)
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


def profile_layer(
    layer: nn.Module,
    batch_size: int,
    seq_len: int = 128,
    embed_dim: int = 768,
    device: torch.device = torch.device("cuda"),
    warmup_iters: int = 5,
    profile_iters: int = 20,
) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    """
    Profile forward and backward pass times for a single layer.
    
    Returns:
        (fwd_ms, bwd_ms, peak_memory_mb) or (None, None, None) on OOM.
    """
    layer = layer.to(device)
    layer.train()
    
    try:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        
        # Create input tensor
        x = torch.randn(batch_size, seq_len, embed_dim, device=device, requires_grad=True)
        
        # Warmup iterations
        for _ in range(warmup_iters):
            out = layer(x)
            loss = out.sum()
            loss.backward()
            layer.zero_grad()
            if x.grad is not None:
                x.grad = None
        
        torch.cuda.synchronize(device)
        
        # Profile forward pass with CUDA events (more accurate)
        stream = torch.cuda.current_stream(device)
        fwd_times = []
        
        for _ in range(profile_iters):
            x = torch.randn(batch_size, seq_len, embed_dim, device=device, requires_grad=True)
            torch.cuda.synchronize(device)
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record(stream)
            out = layer(x)
            end_event.record(stream)
            
            torch.cuda.synchronize(device)
            fwd_times.append(start_event.elapsed_time(end_event))
        
        # Profile backward pass
        bwd_times = []
        target = torch.randn(batch_size, seq_len, embed_dim, device=device)
        
        for _ in range(profile_iters):
            x = torch.randn(batch_size, seq_len, embed_dim, device=device, requires_grad=True)
            layer.zero_grad()
            out = layer(x)
            loss = ((out - target) ** 2).mean()
            
            torch.cuda.synchronize(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record(stream)
            loss.backward()
            end_event.record(stream)
            
            torch.cuda.synchronize(device)
            bwd_times.append(start_event.elapsed_time(end_event))
        
        # Get peak memory
        peak_mem = torch.cuda.max_memory_allocated(device) // (1024 * 1024)
        
        # Take median, skipping first few measurements
        import numpy as np
        fwd_ms = float(np.median(fwd_times[warmup_iters:]))
        bwd_ms = float(np.median(bwd_times[warmup_iters:]))
        
        return fwd_ms, bwd_ms, peak_mem
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            torch.cuda.empty_cache()
            return None, None, None
        raise


def measure_tcp_bandwidth_iperf(
    target_ip: str, 
    target_port: int = 5201, 
    duration: int = 5
) -> Optional[float]:
    """
    Measure TCP bandwidth using iperf3 (preferred method).
    
    Returns:
        Bandwidth in MB/s or None on failure.
    """
    try:
        result = subprocess.run(
            ["iperf3", "-c", target_ip, "-p", str(target_port), 
             "-t", str(duration), "-J"],
            capture_output=True,
            text=True,
            timeout=duration + 30,
        )
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            # Get bits_per_second and convert to MB/s
            bps = data["end"]["sum_sent"]["bits_per_second"]
            return bps / 8 / 1024 / 1024
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"    iperf3 failed: {e}")
    
    return None


def measure_tcp_bandwidth_socket(
    target_ip: str, 
    target_port: int = 5201, 
    data_mb: int = 10
) -> Optional[float]:
    """
    Measure TCP bandwidth using raw socket transfer (fallback method).
    
    Note: Requires a listening server on target. This is less accurate than iperf3.
    
    Returns:
        Bandwidth in MB/s or None on failure.
    """
    try:
        data = b"x" * (data_mb * 1024 * 1024)
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(60)
        sock.connect((target_ip, target_port))
        
        # Send data and measure time
        t0 = time.perf_counter()
        total_sent = 0
        while total_sent < len(data):
            sent = sock.send(data[total_sent:])
            if sent == 0:
                break
            total_sent += sent
        elapsed = time.perf_counter() - t0
        
        sock.close()
        
        if elapsed > 0:
            return (total_sent / (1024 * 1024)) / elapsed
    except (socket.error, socket.timeout) as e:
        print(f"    Socket test failed: {e}")
    
    return None


def measure_network_bandwidth(
    target_ip: str, 
    target_port: int = 5201
) -> Optional[float]:
    """
    Measure network bandwidth to target, trying iperf3 first then socket fallback.
    
    Returns:
        Bandwidth in MB/s or None on failure.
    """
    # Try iperf3 first (more accurate)
    bw = measure_tcp_bandwidth_iperf(target_ip, target_port)
    if bw is not None:
        return bw
    
    # Fallback to socket test (requires server)
    # Note: This typically won't work without a custom server
    # Return None to indicate measurement failed
    return None


def profile_all_layers(
    num_layers: int,
    embed_dim: int,
    num_heads: int,
    d_ff: int,
    seq_len: int,
    batch_sizes: List[int],
    device: torch.device,
) -> Tuple[Dict[int, Dict[int, Dict[str, Any]]], Dict[int, Dict[int, int]]]:
    """
    Profile all transformer layers at multiple batch sizes.
    
    Returns:
        (exec_times, max_memory)
        exec_times: layer_idx -> batch_size -> {fwd_ms, bwd_ms}
        max_memory: layer_idx -> batch_size -> memory_mb
    """
    exec_times = {}
    max_memory = {}
    
    for layer_idx in range(num_layers):
        exec_times[layer_idx] = {}
        max_memory[layer_idx] = {}
        
        # Create fresh layer for each index
        layer = ProfilerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            d_ff=d_ff,
        )
        
        for bs in batch_sizes:
            fwd_ms, bwd_ms, mem_mb = profile_layer(
                layer=layer,
                batch_size=bs,
                seq_len=seq_len,
                embed_dim=embed_dim,
                device=device,
            )
            
            if fwd_ms is not None:
                exec_times[layer_idx][bs] = {
                    "fwd_ms": round(fwd_ms, 3),
                    "bwd_ms": round(bwd_ms, 3),
                }
                max_memory[layer_idx][bs] = mem_mb
                print(f"    Layer {layer_idx:2d}, BS {bs:3d}: fwd={fwd_ms:7.2f}ms, bwd={bwd_ms:7.2f}ms, mem={mem_mb}MB")
            else:
                exec_times[layer_idx][bs] = {"fwd_ms": "OOM", "bwd_ms": "OOM"}
                max_memory[layer_idx][bs] = "OOM"
                print(f"    Layer {layer_idx:2d}, BS {bs:3d}: OOM")
        
        # Clean up
        del layer
        torch.cuda.empty_cache()
    
    return exec_times, max_memory


def profile_network_peers(peers: List[str]) -> Dict[str, Optional[float]]:
    """
    Profile network bandwidth to all specified peers.
    
    Args:
        peers: List of "ip:port" or "ip" strings
        
    Returns:
        Dict mapping IP to bandwidth in MB/s (or None if failed)
    """
    bandwidths = {}
    
    for peer in peers:
        if ":" in peer:
            ip, port = peer.split(":")
            port = int(port)
        else:
            ip, port = peer, 5201
        
        print(f"    Probing {ip}:{port}...")
        bw = measure_network_bandwidth(ip, port)
        
        if bw is not None:
            bandwidths[ip] = round(bw, 2)
            print(f"    -> {ip}: {bw:.2f} MB/s")
        else:
            bandwidths[ip] = None
            print(f"    -> {ip}: FAILED (iperf3 server not running?)")
    
    return bandwidths


def main():
    parser = argparse.ArgumentParser(
        description="Asteroid Node Profiler - Hardware characterization for HPP planning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile local GPU only
  python profile_node.py --output-dir ./profiles

  # Profile with network bandwidth to peers
  python profile_node.py --output-dir ./profiles --network-peers "192.168.1.11:5201,192.168.1.12:5201"

  # Custom model configuration
  python profile_node.py --output-dir ./profiles --num-layers 24 --embed-dim 1024

Network profiling requires iperf3 server running on peers:
  On each peer: iperf3 -s -p 5201
        """
    )
    
    # Output options
    parser.add_argument(
        "--output-dir", type=str, default="./profiles",
        help="Directory to save profile JSON (default: ./profiles)"
    )
    parser.add_argument(
        "--rank", "--rand", dest="rank", type=str, default="",
        help="Node rank/id suffix to avoid filename collisions (e.g. 0,1,2)"
    )
    
    # Model configuration
    parser.add_argument(
        "--num-layers", type=int, default=12,
        help="Number of transformer layers to profile (default: 12)"
    )
    parser.add_argument(
        "--embed-dim", type=int, default=768,
        help="Embedding dimension (default: 768)"
    )
    parser.add_argument(
        "--num-heads", type=int, default=12,
        help="Number of attention heads (default: 12)"
    )
    parser.add_argument(
        "--d-ff", type=int, default=3072,
        help="Feed-forward dimension (default: 3072)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=128,
        help="Sequence length (default: 128)"
    )
    
    # Profiling options
    parser.add_argument(
        "--batch-sizes", type=str, default="1,2,4,8,16",
        help="Comma-separated batch sizes to profile (default: 1,2,4,8,16)"
    )
    parser.add_argument(
        "--network-peers", type=str, default="",
        help="Comma-separated ip:port list for network profiling"
    )
    parser.add_argument(
        "--skip-gpu", action="store_true",
        help="Skip GPU profiling (network only)"
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse batch sizes
    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",")]
    
    # Get hardware info
    print("=" * 60)
    print("ASTEROID NODE PROFILER")
    print("=" * 60)
    
    hw_info = get_hardware_info()
    hostname = hw_info["hostname"]
    
    print(f"\nHostname: {hostname}")
    print(f"Architecture: {hw_info['architecture']}")
    print(f"PyTorch: {hw_info['torch_version']}")
    
    if hw_info.get("gpu_name"):
        print(f"GPU: {hw_info['gpu_name']}")
        print(f"GPU Memory: {hw_info['gpu_memory_mb']} MB")
        print(f"CUDA: {hw_info.get('cuda_version', 'N/A')}")
    else:
        print("GPU: None detected")
    
    # Initialize profile data
    profile = {
        "hostname": hostname,
        "rank": args.rank if args.rank else None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "hardware": hw_info,
        "model_config": {
            "num_layers": args.num_layers,
            "embed_dim": args.embed_dim,
            "num_heads": args.num_heads,
            "d_ff": args.d_ff,
            "seq_len": args.seq_len,
        },
        "batch_sizes_profiled": batch_sizes,
        "exec_times": {},
        "max_memory_per_layer_mb": {},
        "bandwidths_mbps": {},
    }
    
    # GPU Profiling
    if not args.skip_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        
        print(f"\n--- GPU Profiling ({args.num_layers} layers, batch sizes {batch_sizes}) ---")
        
        exec_times, max_memory = profile_all_layers(
            num_layers=args.num_layers,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            seq_len=args.seq_len,
            batch_sizes=batch_sizes,
            device=device,
        )
        
        profile["exec_times"] = exec_times
        profile["max_memory_per_layer_mb"] = max_memory
        
    elif args.skip_gpu:
        print("\n--- Skipping GPU profiling (--skip-gpu) ---")
    else:
        print("\n--- No GPU available, skipping GPU profiling ---")
    
    # Network Profiling
    if args.network_peers:
        peers = [p.strip() for p in args.network_peers.split(",") if p.strip()]
        
        if peers:
            print(f"\n--- Network Profiling ({len(peers)} peers) ---")
            bandwidths = profile_network_peers(peers)
            profile["bandwidths_mbps"] = bandwidths
    else:
        print("\n--- Skipping network profiling (no --network-peers specified) ---")
    
    # Save profile
    filename_suffix = f"_rank{args.rank}" if args.rank else ""
    output_file = output_dir / f"profile_{hostname}{filename_suffix}.json"
    with open(output_file, "w") as f:
        json.dump(profile, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"Profile saved to: {output_file}")
    print(f"{'=' * 60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
