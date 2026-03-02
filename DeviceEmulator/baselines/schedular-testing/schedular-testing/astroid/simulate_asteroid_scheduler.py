import json
import copy
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Replicated Dataclasses ---
@dataclass
class DeviceSpec:
    device_id: int = 0
    device_type: str = "simulated"
    memory_budget_mb: float = 4096.0
    compute_capacity: float = 1.0

@dataclass
class AsteroidConfig:
    num_layers: int = 12
    num_stages: int = 3
    global_batch_size: int = 32
    micro_batch_size: int = 4
    num_microbatches: int = 8
    d2d_bandwidth_mbps: float = 100.0

@dataclass
class HPPPlanConfig:
    num_stages: int = 2
    partition_points: List[int] = field(default_factory=list)
    device_groups: Dict[int, List[int]] = field(default_factory=dict)
    micro_batch_alloc: Dict[int, Dict[int, int]] = field(default_factory=dict)
    estimated_latency_ms: float = float('inf')


class SimulatedAsteroidProfiler:
    """Mock profiler using Asteroid's non-linear batch scaling logic."""
    def __init__(self, config_data: dict, devices: List[DeviceSpec]):
        self.devices = {d.device_id: d for d in devices}
        self.exec_times = defaultdict(lambda: defaultdict(dict))
        self.activation_sizes = config_data["activation_sizes_bytes"]
        self.weight_sizes = config_data["weight_sizes_bytes"]
        self.bandwidths = {}
        
        # Populate non-linear profiles
        batch_sizes = [1, 2, 4, 8, 16, 32]
        for d_id, d_spec in self.devices.items():
            for l_idx, base_fwd in enumerate(config_data["baseline_fwd_ms"]):
                for bs in batch_sizes:
                    # Non-linear scaling: smaller BS underutilizes GPU (Asteroid Sec 3.3)
                    fwd = base_fwd * (bs ** 0.85) / d_spec.compute_capacity
                    bwd = fwd * 2.0
                    self.exec_times[d_id][l_idx][bs] = (fwd, bwd)
                    
        # --- UPDATE: Heterogeneous Bandwidth Matrix ---
        # Get the heterogeneous bandwidth array, fallback to 100 if missing
        bw_list = config_data.get("node_bandwidths_mbps", [100.0] * len(self.devices))
        
        for d1 in self.devices:
            for d2 in self.devices:
                if d1 != d2:
                    # D2D communication is limited by the slower node's link
                    self.bandwidths[(d1, d2)] = min(bw_list[d1], bw_list[d2])
                else:
                    self.bandwidths[(d1, d2)] = float('inf') # Internal copy is instant

    def get_exec_time(self, device_id: int, layer_idx: int, batch_size: int) -> Tuple[float, float]:
        times = self.exec_times.get(device_id, {}).get(layer_idx, {})
        if batch_size in times:
            return times[batch_size]
        keys = sorted(times.keys())
        if not keys: return (1.0, 2.0)
        if batch_size <= keys[0]: return times[keys[0]]
        if batch_size >= keys[-1]:
            ref_bs, (ref_f, ref_b) = keys[-1], times[keys[-1]]
            ratio = (batch_size / ref_bs) ** 0.85
            return (ref_f * ratio, ref_b * ratio)
        for i in range(len(keys) - 1):
            if keys[i] <= batch_size <= keys[i + 1]:
                lo, hi = keys[i], keys[i + 1]
                alpha = (batch_size - lo) / (hi - lo)
                return (times[lo][0] * (1 - alpha) + times[hi][0] * alpha,
                        times[lo][1] * (1 - alpha) + times[hi][1] * alpha)


class AsteroidPlanner:
    """Asteroid HPP Dynamic Programming algorithm extracted from astroid.py"""
    def __init__(self, profiler: SimulatedAsteroidProfiler, cfg: AsteroidConfig, devices: List[DeviceSpec]):
        self.profiler = profiler
        self.cfg = cfg
        # Asteroid sorts by memory budget to prioritize heavy lifting
        self.devices = sorted(devices, key=lambda d: d.memory_budget_mb, reverse=True)
        self.N = len(devices)
        self.L = cfg.num_layers
        self.M = cfg.num_microbatches
        self.B = cfg.micro_batch_size

    def _memory_footprint(self, stage_idx: int, num_stages: int, start_l: int, end_l: int, batch_size: int) -> float:
        P = num_stages
        K_p = max(1, 2 * (P - stage_idx) - 1)
        weight_bytes = sum(self.profiler.weight_sizes[l] for l in range(start_l, end_l))
        mem_mod = weight_bytes
        mem_opt = weight_bytes * 2
        mem_act = sum(self.profiler.activation_sizes[l] for l in range(start_l, end_l)) * batch_size
        total = mem_mod + mem_opt + K_p * mem_act
        return total / (1024 * 1024)

    def _alloc_microbatch(self, device_ids: List[int], start_l: int, end_l: int, micro_bs: int) -> Tuple[Dict[int, int], float]:
        if not device_ids: return {}, float('inf')
        
        # --- THE FIX IS HERE ---
        # Explicitly search for the device object by its ID rather than using the ID as a list index
        devices_here = [next(d for d in self.devices if d.device_id == did) for did in device_ids]
        # -----------------------
        
        alloc = {d.device_id: 0 for d in devices_here}
        remaining = micro_bs
        active = list(devices_here)

        while remaining > 0 and active:
            total_cap = sum(d.compute_capacity for d in active)
            if total_cap <= 0: break
            new_active = []
            for d in active:
                share = max(1, int(round(d.compute_capacity / total_cap * remaining)))
                mem_needed = self._memory_footprint(0, 1, start_l, end_l, share)
                max_bs = share
                while mem_needed > d.memory_budget_mb and max_bs > 1:
                    max_bs -= 1
                    mem_needed = self._memory_footprint(0, 1, start_l, end_l, max_bs)
                actual = min(share, max_bs, remaining)
                alloc[d.device_id] += actual
                remaining -= actual
                if mem_needed < d.memory_budget_mb * 0.95:
                    new_active.append(d)
            active = new_active

        def exec_time(did, bs):
            if bs <= 0: return 0.0
            return sum(sum(self.profiler.get_exec_time(did, l, bs)) for l in range(start_l, end_l))

        for _ in range(5):
            times = {did: exec_time(did, bs) for did, bs in alloc.items() if bs > 0}
            if not times: break
            slowest = max(times, key=times.get)
            fastest = min(times, key=times.get)
            if slowest == fastest or alloc[slowest] <= 1: break
            old_time = times[slowest]
            alloc[slowest] -= 1
            alloc[fastest] += 1
            new_time = max(exec_time(slowest, alloc[slowest]), exec_time(fastest, alloc[fastest]))
            if new_time >= old_time:
                alloc[slowest] += 1
                alloc[fastest] -= 1
                break

        straggler_time = max(exec_time(did, bs) for did, bs in alloc.items()) if any(bs > 0 for bs in alloc.values()) else float('inf')
        return alloc, straggler_time

    def _comm_time_inter_stage(self, layer_idx: int, src_group: List[int], dst_group: List[int], batch_size: int = 0) -> float:
        if not src_group or not dst_group: return 0.0
        bs = batch_size if batch_size > 0 else self.B
        act_size = self.profiler.activation_sizes[min(layer_idx, len(self.profiler.activation_sizes) - 1)] * bs
        act_size_mb = act_size / (1024 * 1024)
        min_bw = min(self.profiler.bandwidths.get((s, d), self.cfg.d2d_bandwidth_mbps) for s in src_group for d in dst_group)
        if min_bw <= 0: return float('inf')
        return 2 * act_size_mb / min_bw * 1000

    def _allreduce_time(self, device_group: List[int], start_l: int, end_l: int) -> float:
        g_size = len(device_group)
        if g_size <= 1: return 0.0
        weight_bytes = sum(self.profiler.weight_sizes[l] for l in range(start_l, end_l))
        min_bw = min(self.profiler.bandwidths.get((d1, d2), self.cfg.d2d_bandwidth_mbps) for d1 in device_group for d2 in device_group if d1 != d2)
        if min_bw <= 0: return float('inf')
        vol_mb = 2 * (g_size - 1) / g_size * weight_bytes / (1024 * 1024)
        return vol_mb / min_bw * 1000

    def plan(self) -> HPPPlanConfig:
        L, N = self.L, self.N
        device_ids = [d.device_id for d in self.devices]
        best_plan, best_latency = HPPPlanConfig(), float('inf')

        max_stages = min(L, N, self.cfg.num_stages + 2)
        for P in range(1, max_stages + 1):
            result = self._dp_plan(L, N, P, device_ids)
            if result and result.estimated_latency_ms < best_latency:
                best_latency = result.estimated_latency_ms
                best_plan = result
        return best_plan

    def _dp_plan(self, L: int, N: int, P: int, device_ids: List[int]) -> Optional[HPPPlanConfig]:
        if P > L or P > N: return None
        INF = float('inf')
        Q = [[[INF for _ in range(P + 1)] for _ in range(N + 1)] for _ in range(L + 1)]
        Config = [[[None for _ in range(P + 1)] for _ in range(N + 1)] for _ in range(L + 1)]

        for l in range(1, L + 1):
            for n in range(1, N + 1):
                group = device_ids[N - n:]
                alloc, exec_t = self._alloc_microbatch(group, L - l, L, self.B)
                ar_t = self._allreduce_time(group, L - l, L)
                lat = self.M * exec_t + ar_t
                if lat < Q[l][n][1]:
                    Q[l][n][1] = lat
                    Config[l][n][1] = {'partition': [L - l], 'groups': {0: group}, 'allocs': {0: alloc}}

        for p in range(2, P + 1):
            for l in range(p, L + 1):
                for n in range(p, N + 1):
                    for l_prime in range(p - 1, l):
                        for n_prime in range(p - 1, n):
                            if Q[l_prime][n_prime][p - 1] >= INF: continue
                            new_group = device_ids[N - n:N - n_prime]
                            if not new_group: continue
                            start_l, end_l = L - l, L - l_prime

                            alloc, exec_t = self._alloc_microbatch(new_group, start_l, end_l, self.B)
                            
                            mem_ok = True
                            for did, bs in alloc.items():
                                if bs > 0:
                                    dev = next(d for d in self.devices if d.device_id == did)
                                    if self._memory_footprint(P - p, P, start_l, end_l, bs) > dev.memory_budget_mb:
                                        mem_ok = False
                                        break
                            if not mem_ok: continue

                            prev_cfg = Config[l_prime][n_prime][p - 1]
                            prev_last_group = prev_cfg['groups'][p - 2] if prev_cfg else []
                            comm_t = self._comm_time_inter_stage(end_l - 1, new_group, prev_last_group, sum(alloc.values())) if prev_last_group else 0.0
                            ar_t = self._allreduce_time(new_group, start_l, end_l)

                            total_lat = max(Q[l_prime][n_prime][p - 1], self.M * exec_t + comm_t) + ar_t

                            if total_lat < Q[l][n][p]:
                                Q[l][n][p] = total_lat
                                new_config = copy.deepcopy(prev_cfg)
                                new_config['partition'] = [start_l] + new_config['partition']
                                groups_new = {0: new_group}
                                allocs_new = {0: alloc}
                                for si, g in prev_cfg['groups'].items(): groups_new[si + 1] = g
                                for si, a in prev_cfg['allocs'].items(): allocs_new[si + 1] = a
                                Config[l][n][p] = {'partition': new_config['partition'], 'groups': groups_new, 'allocs': allocs_new}

        if Q[L][N][P] >= INF: return None
        return HPPPlanConfig(P, Config[L][N][P]['partition'], Config[L][N][P]['groups'], Config[L][N][P]['allocs'], Q[L][N][P])


if __name__ == "__main__":
    with open("./asteroid_config.json", "r") as f:
        cfg_data = json.load(f)

    # 1. Setup Config and Device Specs
    num_microbatches = cfg_data["global_batch_size"] // cfg_data["micro_batch_size"]
    
    # --- UPDATE: Safely handle missing generic bandwidth ---
    sys_cfg = AsteroidConfig(
        num_layers=cfg_data["num_layers"],
        num_stages=cfg_data["num_stages"],
        micro_batch_size=cfg_data["micro_batch_size"],
        num_microbatches=num_microbatches,
        d2d_bandwidth_mbps=cfg_data.get("d2d_bandwidth_mbps", 100.0) 
    )
    
    device_specs = []
    for idx, name in enumerate(cfg_data["simulated_hardware"]):
        device_specs.append(DeviceSpec(
            device_id=idx,
            device_type=name,
            compute_capacity=cfg_data["computing_capacities"][idx],
            memory_budget_mb=cfg_data["available_memory_mb"][idx]
        ))

    # 2. Run Planner
    profiler = SimulatedAsteroidProfiler(cfg_data, device_specs)
    planner = AsteroidPlanner(profiler, sys_cfg, device_specs)
    
    print("\n--- Running Asteroid HPP Scheduler Simulation ---")
    plan = planner.plan()

    # 3. Print Hybrid Pipeline Output
    if plan.estimated_latency_ms == float('inf'):
        print("Failed to find a valid partition under memory constraints.")
    else:
        print(f"\n✅ Optimal Plan Found: {plan.num_stages} Stages | Latency: {plan.estimated_latency_ms:.2f}ms")
        for stage_idx in range(plan.num_stages):
            start_l = plan.partition_points[stage_idx]
            end_l = plan.partition_points[stage_idx+1] if stage_idx+1 < len(plan.partition_points) else sys_cfg.num_layers
            group = plan.device_groups[stage_idx]
            
            print(f"\n[Stage {stage_idx}] Layers {start_l} to {end_l-1}")
            for d_id in group:
                d_name = cfg_data["simulated_hardware"][d_id]
                alloc = plan.micro_batch_alloc[stage_idx][d_id]
                print(f"  ↳ Device {d_id} ({d_name}) computes {alloc} micro-batches")