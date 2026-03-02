import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from collections import defaultdict

from ..core.config import AsteroidConfig, DeviceSpec
from ..utils.logger import logger

class AsteroidProfiler:
    """Device profiler — adapted from Confident's ConfidantProfiler.

    Measures per-layer forward/backward time at multiple batch sizes,
    memory footprint, and D2D bandwidth. Implements Section 3.3 of
    the Asteroid paper (non-linear batch-size profiling).
    """

    def __init__(self, cfg: AsteroidConfig, devices: List[DeviceSpec]):
        self.cfg = cfg
        self.devices = {d.device_id: d for d in devices}
        # Profiled data: device_id → layer_idx → batch_size → (fwd_ms, bwd_ms)
        self.exec_times: Dict[int, Dict[int, Dict[int, Tuple[float, float]]]] = \
            defaultdict(lambda: defaultdict(dict))
        # Per-layer output activation size in bytes (for a single sample)
        self.activation_sizes: List[float] = []
        # Per-layer weight size in bytes
        self.weight_sizes: List[float] = []
        # D2D bandwidth matrix: (src, dst) → MB/s
        self.bandwidths: Dict[Tuple[int, int], float] = {}

    def profile_model(self, model_layers: nn.ModuleList,
                      batch_sizes: List[int],
                      device: torch.device, device_id: int,
                      seq_len: int, d_model: int):
        """Profile all layers at multiple batch sizes on a device."""
        torch.cuda.set_device(device)
        num_iters = 20  # more iterations for robust median

        for layer_idx, layer in enumerate(model_layers):
            layer = layer.to(device)
            for bs in batch_sizes:
                x = torch.randn(bs, seq_len, d_model, device=device)
                x.requires_grad_(True)

                # Warmup
                for _ in range(3):
                    with torch.no_grad():
                        _ = layer(x)
                torch.cuda.synchronize(device)

                # Forward timing 
                cur_stream = torch.cuda.current_stream(device)
                fwd_times = []
                for _ in range(num_iters):
                    torch.cuda.synchronize(device)
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record(cur_stream)
                    with torch.no_grad():
                        _ = layer(x)
                    e.record(cur_stream)
                    torch.cuda.synchronize(device)
                    fwd_times.append(s.elapsed_time(e))

                # Backward timing
                bwd_times = []
                for _ in range(num_iters):
                    torch.cuda.synchronize(device)
                    layer.zero_grad()
                    if x.grad is not None:
                        x.grad.zero_()
                    out = layer(x)
                    grad_out = torch.ones_like(out)
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record(cur_stream)
                    out.backward(gradient=grad_out)
                    e.record(cur_stream)
                    torch.cuda.synchronize(device)
                    bwd_times.append(s.elapsed_time(e))

                # Skip first 3 warmup measurements, take median of rest
                fwd_ms = float(np.median(fwd_times[3:]))
                bwd_ms = float(np.median(bwd_times[3:]))
                self.exec_times[device_id][layer_idx][bs] = (fwd_ms, bwd_ms)

            # Peak memory profiling
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            layer.zero_grad()
            x_mem = torch.randn(batch_sizes[-1], seq_len, d_model,
                                device=device, requires_grad=True)
            out = layer(x_mem)
            out.backward(gradient=torch.ones_like(out))
            peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            logger.debug(f"Profile: layer {layer_idx} peak_mem={peak_mem_mb:.1f}MB")

            layer.cpu()
            torch.cuda.empty_cache()

        # Compute activation and weight sizes
        if not self.activation_sizes:
            for layer in model_layers:
                self.activation_sizes.append(seq_len * d_model * 4)
                self.weight_sizes.append(
                    sum(p.numel() * p.element_size() for p in layer.parameters()))

    def profile_bandwidth(self, src_device: torch.device, dst_device: torch.device,
                          src_id: int, dst_id: int, data_size_mb: float = 50.0):
        """Measure P2P bandwidth."""
        n = int(data_size_mb * 1024 * 1024 / 4)
        src_t = torch.empty(n, dtype=torch.float32, device=src_device)
        for _ in range(3):
            _ = src_t.to(dst_device)
        torch.cuda.synchronize()
        times = []
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.time()
            _ = src_t.to(dst_device)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        avg = np.median(times[3:])
        bw = data_size_mb / avg if avg > 0 else float('inf')
        self.bandwidths[(src_id, dst_id)] = bw

    def set_synthetic_profiles(self, num_layers: int, num_devices: int,
                               batch_sizes: List[int]):
        """Generate synthetic profiles for testing without real hardware."""
        for d in range(num_devices):
            cap = self.devices.get(d, DeviceSpec(device_id=d)).compute_capacity
            for l in range(num_layers):
                for bs in batch_sizes:
                    # Non-linear scaling: smaller BS underutilizes GPU
                    fwd = (0.5 + 0.1 * l) * (bs ** 0.85) / cap
                    bwd = fwd * 2.0
                    self.exec_times[d][l][bs] = (fwd, bwd)
        self.activation_sizes = [128 * 768 * 4] * num_layers  # approx
        self.weight_sizes = [768 * 768 * 4 * 4] * num_layers  # approx
        for d1 in range(num_devices):
            for d2 in range(num_devices):
                if d1 != d2:
                    self.bandwidths[(d1, d2)] = 100.0  # 100 MB/s default

    def get_exec_time(self, device_id: int, layer_idx: int, batch_size: int) \
            -> Tuple[float, float]:
        """Get (fwd_ms, bwd_ms) for a layer on a device at given batch size."""
        times = self.exec_times.get(device_id, {}).get(layer_idx, {})
        if batch_size in times:
            return times[batch_size]
        # Interpolate from nearest profiled batch size
        if not times:
            return (1.0, 2.0)  # fallback
        keys = sorted(times.keys())
        if batch_size <= keys[0]:
            return times[keys[0]]
        if batch_size >= keys[-1]:
            # Extrapolate with power law
            ref_bs, (ref_f, ref_b) = keys[-1], times[keys[-1]]
            ratio = (batch_size / ref_bs) ** 0.85
            return (ref_f * ratio, ref_b * ratio)
        # Linear interpolation
        for i in range(len(keys) - 1):
            if keys[i] <= batch_size <= keys[i + 1]:
                lo, hi = keys[i], keys[i + 1]
                alpha = (batch_size - lo) / (hi - lo)
                f = times[lo][0] * (1 - alpha) + times[hi][0] * alpha
                b = times[lo][1] * (1 - alpha) + times[hi][1] * alpha
                return (f, b)
        return (1.0, 2.0)

    def load_from_profile_json(self, profile_path: str, device_id: int) -> None:
        """
        Load profiling data from a JSON file generated by profile_node.py.
        
        Args:
            profile_path: Path to profile_<hostname>.json file
            device_id: Device ID (rank) to assign this profile to
        """
        import json
        
        with open(profile_path, "r") as f:
            data = json.load(f)
        
        # Load execution times
        for layer_idx_str, bs_data in data.get("exec_times", {}).items():
            layer_idx = int(layer_idx_str)
            if device_id not in self.exec_times:
                self.exec_times[device_id] = {}
            if layer_idx not in self.exec_times[device_id]:
                self.exec_times[device_id][layer_idx] = {}
            
            for bs_str, times in bs_data.items():
                bs = int(bs_str)
                # Skip OOM entries
                if isinstance(times.get("fwd_ms"), (int, float)):
                    self.exec_times[device_id][layer_idx][bs] = (
                        float(times["fwd_ms"]),
                        float(times["bwd_ms"]),
                    )
        
        # Load hardware info for memory budget
        hw = data.get("hardware", {})
        memory_mb = hw.get("gpu_memory_mb", 4096)
        
        # Update device spec if available
        if device_id in self.devices:
            self.devices[device_id].memory_budget_mb = memory_mb
        
        logger.info(f"Loaded profile for device {device_id} from {profile_path}")
        return data  # Return for further processing (e.g., bandwidth mapping)

    def load_bandwidth_from_profiles(
        self, 
        profiles: Dict[int, dict], 
        ip_to_rank: Dict[str, int]
    ) -> None:
        """
        Load bandwidth data from multiple profiles.
        
        Args:
            profiles: Dict mapping device_id to profile data dict
            ip_to_rank: Dict mapping IP addresses to device ranks
        """
        for src_rank, profile_data in profiles.items():
            for target_ip, bw_mbps in profile_data.get("bandwidths_mbps", {}).items():
                if bw_mbps is not None and target_ip in ip_to_rank:
                    dst_rank = ip_to_rank[target_ip]
                    self.bandwidths[(src_rank, dst_rank)] = float(bw_mbps)
        
        logger.info(f"Loaded {len(self.bandwidths)} bandwidth measurements")