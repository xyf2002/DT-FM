import copy
from typing import List, Dict, Tuple, Optional

from ..core.config import AsteroidConfig, DeviceSpec, HPPPlanConfig
from .profiler import AsteroidProfiler
from ..utils.logger import logger

class AsteroidPlanner:
    """Dynamic Programming HPP Planner — implements Section 3.3 of the paper.

    Finds optimal: model partition, device grouping, micro-batch allocation
    to minimize HPP-Round Latency under memory constraints.
    """

    def __init__(self, profiler: AsteroidProfiler, cfg: AsteroidConfig,
                 devices: List[DeviceSpec]):
        self.profiler = profiler
        self.cfg = cfg
        self.devices = sorted(devices, key=lambda d: d.memory_budget_mb, reverse=True)
        self.N = len(devices)
        self.L = cfg.num_layers
        self.M = cfg.num_microbatches
        self.B = cfg.micro_batch_size

    def _memory_footprint(self, stage_idx: int, num_stages: int,
                          start_l: int, end_l: int, batch_size: int) -> float:
        """Compute memory footprint for a stage (Eq. 3 in paper)."""
        P = num_stages
        K_p = max(1, 2 * (P - stage_idx) - 1)

        weight_bytes = sum(self.profiler.weight_sizes[l]
                          for l in range(start_l, end_l)
                          if l < len(self.profiler.weight_sizes))
        mem_mod = weight_bytes
        mem_opt = weight_bytes * 2  # Adam states

        mem_act = sum(self.profiler.activation_sizes[l]
                     for l in range(start_l, end_l)
                     if l < len(self.profiler.activation_sizes)) * batch_size

        total = mem_mod + mem_opt + K_p * mem_act
        return total / (1024 * 1024)  # Convert to MB

    def _alloc_microbatch(self, device_ids: List[int], start_l: int, end_l: int,
                          micro_bs: int) -> Tuple[Dict[int, int], float]:
        """Algorithm 1: Memory-aware micro-batch allocation within a device group."""
        if not device_ids:
            return {}, float('inf')

        devices_here = [self.devices[i] if i < len(self.devices)
                        else DeviceSpec(device_id=i) for i in device_ids]

        # Phase 1: Memory-aware balancing
        alloc = {d.device_id: 0 for d in devices_here}
        remaining = micro_bs
        active = list(devices_here)

        while remaining > 0 and active:
            total_cap = sum(d.compute_capacity for d in active)
            if total_cap <= 0:
                break
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

        # Phase 2: Straggler offloading
        def exec_time(did, bs):
            if bs <= 0:
                return 0.0
            total = 0.0
            for l in range(start_l, end_l):
                f, b = self.profiler.get_exec_time(did, l, bs)
                total += f + b
            return total

        for _ in range(5):  # max offload iterations
            times = {did: exec_time(did, bs) for did, bs in alloc.items() if bs > 0}
            if not times:
                break
            slowest = max(times, key=times.get)
            fastest = min(times, key=times.get)
            if slowest == fastest or alloc[slowest] <= 1:
                break
            old_time = times[slowest]
            alloc[slowest] -= 1
            alloc[fastest] += 1
            new_time = max(exec_time(slowest, alloc[slowest]),
                          exec_time(fastest, alloc[fastest]))
            if new_time >= old_time:
                alloc[slowest] += 1
                alloc[fastest] -= 1
                break

        straggler_time = max(exec_time(did, bs) for did, bs in alloc.items()) \
            if any(bs > 0 for bs in alloc.values()) else float('inf')
        return alloc, straggler_time

    def _comm_time_inter_stage(self, layer_idx: int, src_group: List[int],
                                dst_group: List[int],
                                batch_size: int = 0) -> float:
        """Communication time for inter-stage activation transfer."""
        if not src_group or not dst_group:
            return 0.0
        bs = batch_size if batch_size > 0 else self.B
        act_size = self.profiler.activation_sizes[min(layer_idx,
            len(self.profiler.activation_sizes) - 1)] * bs
        act_size_mb = act_size / (1024 * 1024)

        min_bw = float('inf')
        for s in src_group:
            for d in dst_group:
                bw = self.profiler.bandwidths.get((s, d), self.cfg.d2d_bandwidth_mbps)
                min_bw = min(min_bw, bw)
        if min_bw <= 0:
            return float('inf')
        return 2 * act_size_mb / min_bw * 1000  # ms (fwd + bwd)

    def _allreduce_time(self, device_group: List[int], start_l: int,
                         end_l: int) -> float:
        """AllReduce time for gradient sync within a device group."""
        g_size = len(device_group)
        if g_size <= 1:
            return 0.0
        weight_bytes = sum(self.profiler.weight_sizes[l]
                          for l in range(start_l, end_l)
                          if l < len(self.profiler.weight_sizes))
        min_bw = float('inf')
        for i, d1 in enumerate(device_group):
            for d2 in device_group:
                if d1 != d2:
                    bw = self.profiler.bandwidths.get((d1, d2),
                        self.cfg.d2d_bandwidth_mbps)
                    min_bw = min(min_bw, bw)
        if min_bw <= 0:
            return float('inf')
        vol_mb = 2 * (g_size - 1) / g_size * weight_bytes / (1024 * 1024)
        return vol_mb / min_bw * 1000  # ms

    def plan(self) -> HPPPlanConfig:
        """Dynamic Programming HPP Planning — Algorithm 2 from the paper."""
        L, N = self.L, self.N
        device_ids = [d.device_id for d in self.devices]
        best_plan = HPPPlanConfig()
        best_latency = float('inf')

        max_stages = min(L, N, self.cfg.num_stages + 2)
        for P in range(1, max_stages + 1):
            result = self._dp_plan(L, N, P, device_ids)
            if result is not None and result.estimated_latency_ms < best_latency:
                best_latency = result.estimated_latency_ms
                best_plan = result

        if best_plan.estimated_latency_ms == float('inf'):
            best_plan = self._fallback_plan(device_ids)

        logger.info(f"Planner: {best_plan.num_stages} stages, "
                    f"partition={best_plan.partition_points}, "
                    f"latency={best_plan.estimated_latency_ms:.1f}ms")
        return best_plan

    def _dp_plan(self, L: int, N: int, P: int, device_ids: List[int]) \
            -> Optional[HPPPlanConfig]:
        """Core DP: find optimal partition of L layers into P stages across N devices."""
        if P > L or P > N:
            return None

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
                    Config[l][n][1] = {
                        'partition': [L - l],
                        'groups': {0: group},
                        'allocs': {0: alloc}
                    }

        for p in range(2, P + 1):
            for l in range(p, L + 1):
                for n in range(p, N + 1):
                    for l_prime in range(p - 1, l):
                        for n_prime in range(p - 1, n):
                            if Q[l_prime][n_prime][p - 1] >= INF:
                                continue
                            new_group = device_ids[N - n:N - n_prime]
                            if not new_group:
                                continue
                            start_l = L - l
                            end_l = L - l_prime

                            alloc, exec_t = self._alloc_microbatch(
                                new_group, start_l, end_l, self.B)

                            global_stage_idx = P - p
                            mem_ok = True
                            for did, bs in alloc.items():
                                if bs > 0:
                                    mem = self._memory_footprint(
                                        global_stage_idx, P, start_l, end_l, bs)
                                    dev = self.devices[did] if did < len(self.devices) \
                                        else DeviceSpec(device_id=did)
                                    if mem > dev.memory_budget_mb:
                                        mem_ok = False
                                        break
                            if not mem_ok:
                                continue

                            prev_cfg = Config[l_prime][n_prime][p - 1]
                            prev_last_group = prev_cfg['groups'][p - 2] \
                                if prev_cfg and p - 2 in prev_cfg['groups'] else []
                            total_alloc_bs = sum(alloc.values())
                            comm_t = self._comm_time_inter_stage(
                                end_l - 1, new_group, prev_last_group,
                                batch_size=total_alloc_bs) \
                                if prev_last_group else 0.0

                            ar_t = self._allreduce_time(new_group, start_l, end_l)

                            sub_lat = Q[l_prime][n_prime][p - 1]
                            new_step_lat = self.M * exec_t
                            total_lat = max(sub_lat, new_step_lat + comm_t) + ar_t

                            if total_lat < Q[l][n][p]:
                                Q[l][n][p] = total_lat
                                new_config = copy.deepcopy(prev_cfg) if prev_cfg else {
                                    'partition': [], 'groups': {}, 'allocs': {}}
                                new_config['partition'] = [start_l] + \
                                    new_config.get('partition', [])
                                new_config['groups'][p - 1] = new_group
                                groups_new = {}
                                groups_new[0] = new_group
                                for si, g in prev_cfg['groups'].items():
                                    groups_new[si + 1] = g
                                allocs_new = {}
                                allocs_new[0] = alloc
                                for si, a in prev_cfg.get('allocs', {}).items():
                                    allocs_new[si + 1] = a
                                Config[l][n][p] = {
                                    'partition': [start_l] + prev_cfg.get('partition', []),
                                    'groups': groups_new,
                                    'allocs': allocs_new
                                }

        best_lat = Q[L][N][P]
        best_cfg = Config[L][N][P]
        if best_lat >= INF or best_cfg is None:
            return None

        raw_partition = best_cfg['partition']
        if raw_partition and raw_partition[0] == 0:
            raw_partition = raw_partition[1:]

        plan = HPPPlanConfig(
            num_stages=P,
            partition_points=raw_partition,
            device_groups=best_cfg['groups'],
            micro_batch_alloc=best_cfg['allocs'],
            estimated_latency_ms=best_lat
        )
        return plan

    def _fallback_plan(self, device_ids: List[int]) -> HPPPlanConfig:
        """Simple equal partition fallback."""
        P = min(self.cfg.num_stages, len(device_ids), self.L)
        layers_per = self.L // P
        partition = [i * layers_per for i in range(1, P)]
        groups = {}
        devices_per = max(1, len(device_ids) // P)
        for s in range(P):
            start_d = s * devices_per
            end_d = start_d + devices_per if s < P - 1 else len(device_ids)
            groups[s] = device_ids[start_d:end_d]
        allocs = {s: {d: self.B // max(1, len(groups[s]))
                      for d in groups[s]} for s in range(P)}
        return HPPPlanConfig(
            num_stages=P, partition_points=partition,
            device_groups=groups, micro_batch_alloc=allocs,
            estimated_latency_ms=float('inf'))