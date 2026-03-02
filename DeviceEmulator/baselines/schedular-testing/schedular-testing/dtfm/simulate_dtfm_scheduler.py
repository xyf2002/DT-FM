import json
import logging
import itertools
import random
import numpy as np
from typing import List, Tuple, Dict
from scipy.optimize import linear_sum_assignment

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# STAGE 1: GCMA Scheduler (Extracted exactly from dtfm_gpt2_train.py)
# =============================================================================
class DTFMGCMAScheduler:
    def __init__(self, num_devices: int, pp_size: int, dp_size: int,
                 peer_delay: np.ndarray, peer_bandwidth: np.ndarray,
                 send_gradient_size: float, send_activation_size: float):
        self.num_devices = num_devices
        self.way = pp_size
        self.partition_size = dp_size
        self.peer_delay = peer_delay
        self.peer_bandwidth = peer_bandwidth
        self.send_gradient_size = send_gradient_size
        self.send_activation_size = send_activation_size
        assert num_devices == pp_size * dp_size

    def compute_data_parallel_cost(self, candidate_partition: List[tuple]) -> float:
        data_parallel_cost = float('-inf')
        for partition in candidate_partition:
            within_cost = [0.0] * self.partition_size
            for i in range(self.partition_size):
                for j in range(self.partition_size):
                    if i != j:
                        within_cost[i] += 2 * (
                            self.peer_delay[partition[i], partition[j]] / 1e3
                            + self.send_gradient_size * 8
                            / (self.peer_bandwidth[partition[i], partition[j]] * self.partition_size)
                        )
            if data_parallel_cost < np.max(within_cost):
                data_parallel_cost = np.max(within_cost)
        return data_parallel_cost

    def compute_pipeline_parallel_cost(self, candidate_partition: List[tuple]) -> Tuple[float, List[int], List[List]]:
        way, psz = self.way, self.partition_size

        def bipartite_matching(part_0: tuple, part_1: tuple):
            cost_mat = np.zeros((psz, psz))
            for i in range(psz):
                for j in range(psz):
                    cost_mat[i, j] = (
                        self.peer_delay[part_0[i], part_1[j]] / 1e3
                        + self.send_activation_size * 8 / self.peer_bandwidth[part_0[i], part_1[j]]
                    )
            descending = np.argsort(cost_mat.flatten())[::-1]
            inf_weight = 1e6
            for idx in descending:
                r, c = idx // psz, idx % psz
                cur_max = cost_mat[r, c]
                cost_mat[r, c] = inf_weight
                row_ind, col_ind = linear_sum_assignment(cost_mat)
                if cost_mat[row_ind, col_ind].sum() >= inf_weight:
                    return cur_max, list(zip(row_ind, col_ind))
            return 0.0, []

        cross_cost = np.zeros((way, way))
        match_matrix = [[None] * way for _ in range(way)]

        for i in range(way):
            for j in range(i + 1, way):
                cost, match = bipartite_matching(candidate_partition[i], candidate_partition[j])
                cross_cost[i, j] = cost
                cross_cost[j, i] = cost
                match_matrix[i][j] = match
                match_matrix[j][i] = [(c, r) for r, c in match]

        best_cost = float('inf')
        best_path = None

        for start in range(way):
            dp_table = np.full((way, 1 << way), np.inf)
            trace = np.zeros((way, 1 << way), dtype=int)

            def _bitmask(nodes):
                return sum(1 << n for n in nodes)

            def _solve(node, future):
                if not future: return 0.0
                bm = _bitmask(future)
                if dp_table[node][bm] < np.inf: return dp_table[node][bm]
                best_d = np.inf
                best_next = future[0]
                for nxt in future:
                    nxt_future = [f for f in future if f != nxt]
                    nxt_bm = _bitmask(nxt_future)
                    d = cross_cost[node][nxt] + (_solve(nxt, nxt_future) if dp_table[nxt][nxt_bm] == np.inf else dp_table[nxt][nxt_bm])
                    if d < best_d:
                        best_d = d
                        best_next = nxt
                dp_table[node][bm] = best_d
                trace[node][bm] = best_next
                return best_d

            future = [n for n in range(way) if n != start]
            cost = _solve(start, future)
            if cost < best_cost:
                best_cost = cost
                path = [start]
                cur = start
                remaining = list(future)
                while remaining:
                    bm = _bitmask(remaining)
                    nxt = int(trace[cur][bm])
                    path.append(nxt)
                    remaining.remove(nxt)
                    cur = nxt
                best_path = path

        return best_cost, best_path, match_matrix

    def gcma(self, population_size: int = 50, trails: int = 500) -> Tuple[List[List[int]], List[float], List[float]]:
        # Abridged version of GCMA for simulation speed
        nd, way, psz = self.num_devices, self.way, self.partition_size
        
        nodes = list(range(nd))
        partitions = []
        scores = []
        min_scores = []

        # Population Init
        for i in range(population_size):
            cur = nodes.copy()
            random.seed(i)
            random.shuffle(cur)
            partitions.append(cur)

        for part in partitions:
            cp = [tuple(part[i:i + psz]) for i in range(0, nd, psz)]
            dp_cost = self.compute_data_parallel_cost(cp)
            pp_cost, _, _ = self.compute_pipeline_parallel_cost(cp)
            scores.append(dp_cost + 2 * pp_cost)
            min_scores.append(np.min(scores))

        # Basic Evolutionary search
        for i in range(trails):
            p1_idx, p2_idx = np.random.randint(population_size, size=2)
            # Simplified crossover (shuffle)
            off_list = partitions[p1_idx].copy()
            random.shuffle(off_list)
            
            off_cp = [tuple(off_list[i:i + psz]) for i in range(0, nd, psz)]
            off_score = self.compute_data_parallel_cost(off_cp) + 2 * self.compute_pipeline_parallel_cost(off_cp)[0]

            if off_score < max(scores[p1_idx], scores[p2_idx]):
                replaced = p1_idx if scores[p1_idx] > scores[p2_idx] else p2_idx
                partitions[replaced] = off_list
                scores[replaced] = off_score
            min_scores.append(np.min(scores))

        return partitions, scores, min_scores

    def get_pipelines(self, candidate_partition, path, match_matrix):
        pipeline = np.zeros((self.way, self.partition_size), dtype=int)
        for stage_idx, part_idx in enumerate(path):
            if stage_idx > 0:
                last_part_idx = path[stage_idx - 1]
                bm = match_matrix[last_part_idx][part_idx]
                for match in bm:
                    for i in range(self.partition_size):
                        if pipeline[stage_idx - 1][i] == match[0]:
                            pipeline[stage_idx][i] = match[1]
            else:
                next_part_idx = path[1] if self.way > 1 else 0
                bm = match_matrix[part_idx][next_part_idx]
                for i, match in enumerate(bm):
                    pipeline[0][i] = match[0]

        for stage_idx, part_idx in enumerate(path):
            for i in range(self.partition_size):
                pipeline[stage_idx][i] = candidate_partition[part_idx][pipeline[stage_idx][i]]
        return pipeline

    def build_gpu_map(self, pipeline_matrix):
        gpu_map = {}
        for p in range(self.partition_size):
            for s in range(self.way):
                global_rank = p * self.way + s
                gpu_map[global_rank] = int(pipeline_matrix[s, p])
        return gpu_map

    def run(self):
        partitions, scores, _ = self.gcma()
        best_idx = int(np.argmin(scores))
        best_cp = [tuple(partitions[best_idx][i:i + self.partition_size]) for i in range(0, self.num_devices, self.partition_size)]
        
        dp_cost = self.compute_data_parallel_cost(best_cp)
        pp_cost, pp_path, pp_match = self.compute_pipeline_parallel_cost(best_cp)
        
        pipeline_matrix = self.get_pipelines(best_cp, pp_path, pp_match)
        return self.build_gpu_map(pipeline_matrix)


# =============================================================================
# STAGE 2: DP Scheduler (Extracted exactly from dtfm_gpt2_train.py)
# =============================================================================
class SimulatedDTFMProfiler:
    def __init__(self, cfg, gpu_map):
        self.cfg = cfg
        self.gpu_map = gpu_map
        self.pp_size = cfg["pp_size"]
        self.time_intervals = {d: {} for d in range(self.pp_size)}
        
        for s in range(self.pp_size):
            cuda_id = gpu_map[s] # global_rank = s (for DP replica 0)
            cap = cfg["computing_capacities"][cuda_id]
            n = cfg["num_layers"]
            for start in range(n):
                for end in range(start, n):
                    fwd = sum(cfg["baseline_fwd_ms"][i] / cap for i in range(start, end + 1))
                    bwd = sum(cfg["baseline_bwd_ms"][i] / cap for i in range(start, end + 1))
                    self.time_intervals[s][(start, end, 0)] = fwd
                    self.time_intervals[s][(start, end, 1)] = bwd

    def get_time_interval(self, device_id, start, end, phase):
        return self.time_intervals[device_id].get((start, end, phase), 0.0)

    def get_output_size(self, layer_idx): return self.cfg["output_sizes_mb"][layer_idx]
    
    def get_bandwidth(self, device_id):
        # Bandwidth from stage device_id to device_id + 1
        if device_id >= self.pp_size - 1: return 100.0
        src_cuda = self.gpu_map[device_id]
        dst_cuda = self.gpu_map[device_id + 1]
        return self.cfg["peer_bandwidth_gbps"][src_cuda][dst_cuda] * 1000 / 8

    def get_computing_capacity(self, device_id): return 1.0 # Handled in intervals

class DTFMScheduler:
    def __init__(self, profiler, num_layers, num_devices):
        self.profiler = profiler
        self.num_layers = num_layers
        self.num_devices = num_devices

    def _get_time(self, device_id, start, end):
        return self.profiler.get_time_interval(device_id, start, end, 0) + \
               self.profiler.get_time_interval(device_id, start, end, 1)

    def _get_comm_time(self, layer_idx, device_id):
        bw = self.profiler.get_bandwidth(device_id)
        return self.profiler.get_output_size(layer_idx) / bw if bw > 0 else 0.0

    def calculate_partition_point(self) -> List[int]:
        n, k = self.num_layers, self.num_devices
        INF = float('inf')
        dp = [[INF] * k for _ in range(n)]
        split = [[-1] * k for _ in range(n)]

        for i in range(n):
            dp[i][0] = self._get_time(0, 0, i)

        for j in range(1, k):
            for i in range(j, n):
                for m in range(j - 1, i):
                    cost = max(dp[m][j - 1], self._get_time(j, m + 1, i) + self._get_comm_time(m, j - 1))
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        split[i][j] = m

        points = []
        i, j = n - 1, k - 1
        while j > 0:
            m = split[i][j]
            points.append(m)
            i, j = m, j - 1
        points.reverse()
        logger.info(f"DT-FM DP Bottleneck Time: {dp[n - 1][k - 1]:.2f}ms")
        return points

def partition_points_to_layers(points, num_layers, pp_size):
    boundaries = points + [num_layers - 1]
    counts, prev = [], -1
    for b in boundaries:
        counts.append(b - prev)
        prev = b
    return counts

# =============================================================================
# EXECUTION
# =============================================================================
# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__":
    with open("./dtfm_config.json", "r") as f:
        cfg = json.load(f)

    print("\n--- Running DT-FM Scheduler Simulation ---")
    
    # 1. GCMA
    print("\n[Stage 1] Running GCMA (Device Assignment)...")
    gcma = DTFMGCMAScheduler(
        cfg["num_devices"], cfg["pp_size"], cfg["dp_size"],
        np.array(cfg["peer_delay_ms"]), np.array(cfg["peer_bandwidth_gbps"]),
        cfg["send_gradient_size_gb"], cfg["send_activation_size_gb"]
    )
    gpu_map = gcma.run()
    
    # --- UPDATED: Grouped DP Replica Printing ---
    print("\n[Stage 1 Output] GCMA GPU Map (Grouped by DP Replica):")
    for dp_rank in range(cfg["dp_size"]):
        print(f"  DP Replica {dp_rank} (Pipeline {dp_rank + 1}):")
        for pp_rank in range(cfg["pp_size"]):
            # DT-FM rank math: global_rank = dp_rank * pp_size + pp_rank
            global_rank = dp_rank * cfg["pp_size"] + pp_rank
            node_idx = gpu_map[global_rank]
            hw_name = cfg["simulated_hardware"][node_idx]
            print(f"    Stage {pp_rank}: Node {node_idx} ({hw_name})")

    # 2. DP Partitioner
    print("\n[Stage 2] Running DP Partitioner (Layer Boundaries)...")
    profiler = SimulatedDTFMProfiler(cfg, gpu_map)
    scheduler = DTFMScheduler(profiler, cfg["num_layers"], cfg["pp_size"])
    
    points = scheduler.calculate_partition_point()
    layer_counts = partition_points_to_layers(points, cfg["num_layers"], cfg["pp_size"])
    
    # --- UPDATED: Show how the strict symmetry forces layers onto specific devices ---
    print("\n[Stage 2 Output] Pipeline Layer Assignment (Enforced symmetrically):")
    for pp_stage in range(cfg["pp_size"]):
        print(f"  Stage {pp_stage} strictly gets {layer_counts[pp_stage]} layers:")
        for dp_rank in range(cfg["dp_size"]):
            global_rank = dp_rank * cfg["pp_size"] + pp_stage
            node_idx = gpu_map[global_rank]
            hw_name = cfg["simulated_hardware"][node_idx]
            print(f"    -> [DP {dp_rank}] forces {hw_name} to compute {layer_counts[pp_stage]} layers")