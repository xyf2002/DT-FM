"""DT-FM strategy — GCMA topology search + DP partitioning.

Ported from scheduler_testing/dtfm_scheduler.py reference implementation.
First runs GCMA evolutionary search (with bipartite matching + DP-TSP)
to assign devices to pipeline stages, then uses DP to partition layers.
"""
from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from asteroid.core.config import AsteroidConfig, DeviceSpec, DeviceTopology

from .base import ParallelismPlan, ParallelismStrategy

logger = logging.getLogger(__name__)


class DTFMStrategy(ParallelismStrategy):
    """DT-FM baseline with GCMA topology search and DP partitioning."""

    def __init__(
        self,
        pp_size: int = 2,
        dp_size: int = 1,
        population_size: int = 100,
        gcma_trails: int = 4900,
    ) -> None:
        self.pp_size = max(1, pp_size)
        self.dp_size = max(1, dp_size)
        self.population_size = max(2, population_size)
        self.gcma_trails = max(1, gcma_trails)

    # ── ParallelismStrategy interface ───────────────────────────────────────

    def create_plan(
        self,
        config: AsteroidConfig,
        device_topology: DeviceTopology,
        profiler: Any | None = None,
    ) -> ParallelismPlan:
        topology = self._normalize_topology(device_topology)
        num_devices = len(topology.device_specs)
        num_layers = max(1, config.num_layers)

        best_plan: ParallelismPlan | None = None
        # Auto-search: try every valid (pp, dp) factorisation of
        # num_devices and pick the one with lowest pipeline latency.
        for pp in range(1, min(num_devices, num_layers) + 1):
            if num_devices % pp != 0:
                continue
            dp = num_devices // pp

            # Temporarily set for _run_gcma / _assignment_to_groups
            saved_pp, saved_dp = self.pp_size, self.dp_size
            self.pp_size, self.dp_size = pp, dp
            try:
                if num_devices > 1 and pp > 1:
                    device_groups = self._run_gcma(topology)
                else:
                    device_ids = [s.device_id for s in topology.device_specs]
                    device_groups = {0: device_ids}

                partition_points, est_latency = self._dp_partition(
                    config, topology, device_groups, profiler,
                )
            finally:
                self.pp_size, self.dp_size = saved_pp, saved_dp

            if est_latency == float("inf") or est_latency < 0:
                continue

            # Model full pipeline schedule: bottleneck × (PP + ceil(M/dp) - 1)
            # With DP, each pipeline only processes M/dp micro-batches,
            # reducing fill/drain bubble overhead.
            M = max(1, config.global_batch_size // max(1, config.micro_batch_size))
            micro_per_pipe = max(1, -(-M // dp))        # ceil(M / dp)
            schedule_factor = pp + micro_per_pipe - 1
            effective_latency = est_latency * schedule_factor

            if best_plan is None or effective_latency < best_plan.estimated_latency_ms:
                best_plan = ParallelismPlan(
                    partition_points=partition_points,
                    device_groups=device_groups,
                    micro_batch_alloc={},
                    schedule_type=self.get_schedule_type(),
                    estimated_latency_ms=effective_latency,
                )
                logger.debug(
                    "DTFM candidate pp=%d dp=%d step=%.2fms batch=%.2fms (M=%d, M/dp=%d)",
                    pp, dp, est_latency, effective_latency, M, micro_per_pipe,
                )

        if best_plan is None:
            device_ids = [s.device_id for s in topology.device_specs]
            best_plan = ParallelismPlan(
                device_groups={0: device_ids},
                schedule_type=self.get_schedule_type(),
            )

        logger.info(
            "DTFM plan: %d stages, points=%s latency=%.2fms",
            len(best_plan.device_groups),
            best_plan.partition_points,
            best_plan.estimated_latency_ms,
        )
        return best_plan

    def get_schedule_type(self) -> str:
        return "gpipe"

    # ── GCMA topology search (matches reference dtfm_scheduler.py) ─────────

    def _run_gcma(self, topology: DeviceTopology) -> dict[int, list[int]]:
        """Run GCMA evolutionary search with bipartite matching + DP-TSP."""
        specs = topology.device_specs
        device_ids = [s.device_id for s in specs]
        num_devices = len(device_ids)
        pp_size = max(1, min(self.pp_size, num_devices))
        dp_size = max(1, self.dp_size)
        if pp_size == 1:
            return {0: device_ids}

        # Build peer matrices as numpy arrays (reference format)
        peer_delay, peer_bw = self._build_np_matrices(topology)

        # Approximate gradient/activation sizes for GCMA cost function
        send_gradient_size_gb = 0.01   # placeholder ~10 MB
        send_activation_size_gb = 0.01

        # Population init
        pop_n = min(self.population_size, max(num_devices * 2, 4))
        population: list[list[int]] = []
        for seed_i in range(pop_n):
            order = list(range(num_devices))
            random.seed(seed_i)
            random.shuffle(order)
            population.append(order)

        def to_candidate_partition(order: list[int]) -> list[tuple[int, ...]]:
            return [tuple(order[i:i + dp_size]) for i in range(0, num_devices, dp_size)]

        def score(order: list[int]) -> float:
            cp = to_candidate_partition(order)
            dp_cost = self._compute_data_parallel_cost(
                cp, dp_size, peer_delay, peer_bw, send_gradient_size_gb)
            pp_cost = self._compute_pipeline_parallel_cost_value(
                cp, pp_size, dp_size, peer_delay, peer_bw, send_activation_size_gb)
            return dp_cost + 2.0 * pp_cost

        scores = [score(p) for p in population]

        # Evolutionary search
        for _ in range(self.gcma_trails):
            p1, p2 = random.randrange(pop_n), random.randrange(pop_n)
            if p1 == p2:
                continue
            child = population[p1].copy()
            random.shuffle(child)  # simplified crossover from reference
            child_score = score(child)
            replaced = p1 if scores[p1] > scores[p2] else p2
            if child_score < max(scores[p1], scores[p2]):
                population[replaced] = child
                scores[replaced] = child_score

        # Best partition
        best_idx = int(np.argmin(scores))
        best_cp = to_candidate_partition(population[best_idx])

        # Get optimal stage ordering via DP-TSP + bipartite matching
        pp_cost, pp_path, pp_match = self._compute_pipeline_parallel_cost(
            best_cp, pp_size, dp_size, peer_delay, peer_bw, send_activation_size_gb)

        # Build device_groups using pipeline matrix (reference get_pipelines)
        pipeline = self._get_pipelines(best_cp, pp_path, pp_match, pp_size, dp_size)
        device_groups: dict[int, list[int]] = {}
        for stage in range(pp_size):
            device_groups[stage] = [int(pipeline[stage, p]) for p in range(dp_size)]

        logger.debug("GCMA best score %.6f, path=%s", scores[best_idx], pp_path)
        return device_groups

    # ── GCMA cost functions (from reference dtfm_scheduler.py) ──────────────

    def _compute_data_parallel_cost(
        self,
        candidate_partition: list[tuple[int, ...]],
        dp_size: int,
        peer_delay: np.ndarray,
        peer_bw: np.ndarray,
        send_gradient_size_gb: float,
    ) -> float:
        """DP cost: max over all partitions of ring-allreduce cost."""
        data_parallel_cost = float("-inf")
        for partition in candidate_partition:
            within_cost = [0.0] * dp_size
            for i in range(dp_size):
                for j in range(dp_size):
                    if i != j:
                        within_cost[i] += 2 * (
                            peer_delay[partition[i], partition[j]] / 1e3
                            + send_gradient_size_gb * 8
                            / (peer_bw[partition[i], partition[j]] * dp_size)
                        )
            if data_parallel_cost < max(within_cost):
                data_parallel_cost = max(within_cost)
        return data_parallel_cost

    def _bipartite_matching(
        self,
        part_0: tuple[int, ...],
        part_1: tuple[int, ...],
        dp_size: int,
        peer_delay: np.ndarray,
        peer_bw: np.ndarray,
        send_activation_size_gb: float,
    ) -> tuple[float, list[tuple[int, int]]]:
        """Hungarian-based bottleneck bipartite matching (reference algorithm)."""
        cost_mat = np.zeros((dp_size, dp_size))
        for i in range(dp_size):
            for j in range(dp_size):
                cost_mat[i, j] = (
                    peer_delay[part_0[i], part_1[j]] / 1e3
                    + send_activation_size_gb * 8 / max(peer_bw[part_0[i], part_1[j]], 1e-6)
                )
        descending = np.argsort(cost_mat.flatten())[::-1]
        inf_weight = 1e6
        for idx in descending:
            r, c = int(idx // dp_size), int(idx % dp_size)
            cur_max = cost_mat[r, c]
            cost_mat[r, c] = inf_weight
            row_ind, col_ind = linear_sum_assignment(cost_mat)
            if cost_mat[row_ind, col_ind].sum() >= inf_weight:
                return float(cur_max), list(zip(row_ind.tolist(), col_ind.tolist()))
        return 0.0, []

    def _compute_pipeline_parallel_cost(
        self,
        candidate_partition: list[tuple[int, ...]],
        pp_size: int,
        dp_size: int,
        peer_delay: np.ndarray,
        peer_bw: np.ndarray,
        send_activation_size_gb: float,
    ) -> tuple[float, list[int], list[list[Any]]]:
        """Full PP cost with DP-TSP stage ordering + bipartite matching."""
        # Cross-cost matrix between all stage pairs
        cross_cost = np.zeros((pp_size, pp_size))
        match_matrix: list[list[Any]] = [[None] * pp_size for _ in range(pp_size)]

        for i in range(pp_size):
            for j in range(i + 1, pp_size):
                cost, match = self._bipartite_matching(
                    candidate_partition[i], candidate_partition[j],
                    dp_size, peer_delay, peer_bw, send_activation_size_gb)
                cross_cost[i, j] = cost
                cross_cost[j, i] = cost
                match_matrix[i][j] = match
                match_matrix[j][i] = [(c, r) for r, c in match]

        # DP-TSP: find optimal stage ordering
        best_cost = float("inf")
        best_path: list[int] = list(range(pp_size))

        for start in range(pp_size):
            dp_table = np.full((pp_size, 1 << pp_size), np.inf)
            trace = np.zeros((pp_size, 1 << pp_size), dtype=int)

            def _bitmask(nodes: list[int]) -> int:
                return sum(1 << n for n in nodes)

            def _solve(node: int, future: list[int]) -> float:
                if not future:
                    return 0.0
                bm = _bitmask(future)
                if dp_table[node][bm] < np.inf:
                    return float(dp_table[node][bm])
                best_d = np.inf
                best_next = future[0]
                for nxt in future:
                    nxt_future = [f for f in future if f != nxt]
                    d = cross_cost[node][nxt] + _solve(nxt, nxt_future)
                    if d < best_d:
                        best_d = d
                        best_next = nxt
                dp_table[node][bm] = best_d
                trace[node][bm] = best_next
                return float(best_d)

            future = [n for n in range(pp_size) if n != start]
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

    def _compute_pipeline_parallel_cost_value(
        self,
        candidate_partition: list[tuple[int, ...]],
        pp_size: int,
        dp_size: int,
        peer_delay: np.ndarray,
        peer_bw: np.ndarray,
        send_activation_size_gb: float,
    ) -> float:
        """PP cost value only (without full matching data), for scoring."""
        cost, _, _ = self._compute_pipeline_parallel_cost(
            candidate_partition, pp_size, dp_size,
            peer_delay, peer_bw, send_activation_size_gb)
        return cost

    def _get_pipelines(
        self,
        candidate_partition: list[tuple[int, ...]],
        path: list[int],
        match_matrix: list[list[Any]],
        pp_size: int,
        dp_size: int,
    ) -> np.ndarray:
        """Build pipeline matrix from GCMA result (reference get_pipelines)."""
        pipeline = np.zeros((pp_size, dp_size), dtype=int)
        for stage_idx, part_idx in enumerate(path):
            if stage_idx > 0:
                last_part_idx = path[stage_idx - 1]
                bm = match_matrix[last_part_idx][part_idx]
                if bm:
                    for match in bm:
                        for i in range(dp_size):
                            if pipeline[stage_idx - 1][i] == match[0]:
                                pipeline[stage_idx][i] = match[1]
            else:
                next_part_idx = path[1] if pp_size > 1 else 0
                bm = match_matrix[part_idx][next_part_idx]
                if bm:
                    for i, match in enumerate(bm):
                        if i < dp_size:
                            pipeline[0][i] = match[0]

        # Map local indices to global device IDs
        for stage_idx, part_idx in enumerate(path):
            for i in range(dp_size):
                pipeline[stage_idx][i] = candidate_partition[part_idx][pipeline[stage_idx][i]]
        return pipeline

    def _build_np_matrices(
        self,
        topology: DeviceTopology,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build numpy peer_delay and peer_bandwidth matrices."""
        specs = topology.device_specs
        n = len(specs)
        peer_delay = np.ones((n, n)) * 1.0   # default 1ms
        peer_bw = np.ones((n, n)) * 1.0      # default 1 Gbps
        for i in range(n):
            peer_delay[i, i] = 0.0
            peer_bw[i, i] = 100.0
        for i, src in enumerate(specs):
            for j, dst in enumerate(specs):
                if i == j:
                    continue
                d = self._lookup_metric(topology.latencies, src.device_id, dst.device_id, 1.0)
                b = self._lookup_metric(topology.bandwidths, src.device_id, dst.device_id, 1.0)
                peer_delay[i, j] = max(d, 1e-6)
                # Convert from MB/s to Gbps for reference compatibility
                peer_bw[i, j] = max(b * 8.0 / 1000.0, 1e-6)
        return peer_delay, peer_bw

    @staticmethod
    def _lookup_metric(table: dict, src: int, dst: int, default: float) -> float:
        direct = table.get((src, dst))
        if direct is not None:
            return float(direct)
        reverse = table.get((dst, src))
        if reverse is not None:
            return float(reverse)
        return default

    # ── DP partition (after GCMA assigns devices to stages) ─────────────────

    def _dp_partition(
        self,
        config: AsteroidConfig,
        topology: DeviceTopology,
        device_groups: dict[int, list[int]],
        profiler: Any | None,
    ) -> tuple[list[int], float]:
        num_layers = max(1, config.num_layers)
        stage_ids = sorted(device_groups)
        num_stages = min(len(stage_ids), num_layers)
        if num_stages <= 1:
            # Single stage: bottleneck = slowest device in the group
            if device_groups and stage_ids:
                spec_by_id = {s.device_id: s for s in topology.device_specs}
                worst = float("-inf")
                for did in device_groups[stage_ids[0]]:
                    spec = spec_by_id.get(did, DeviceSpec(device_id=did))
                    total = sum(
                        self._layer_time(li, did, spec, config, profiler)
                        for li in range(num_layers)
                    )
                    worst = max(worst, total)
                return [], worst
            return [], 0.0

        spec_by_id = {s.device_id: s for s in topology.device_specs}
        stage_prefix: list[list[float]] = []
        for si in range(num_stages):
            stage = stage_ids[si]
            # Use slowest device in group as bottleneck representative
            group = device_groups[stage]
            rep_id = group[0]
            rep_time = float("-inf")
            for did in group:
                spec = spec_by_id.get(did, DeviceSpec(device_id=did))
                t = sum(self._layer_time(li, did, spec, config, profiler)
                        for li in range(num_layers))
                if t > rep_time:
                    rep_time = t
                    rep_id = did
            spec = spec_by_id.get(rep_id, DeviceSpec(device_id=rep_id))
            prefix = [0.0]
            for li in range(num_layers):
                lt = self._layer_time(li, rep_id, spec, config, profiler)
                prefix.append(prefix[-1] + lt)
            stage_prefix.append(prefix)

        dp = [[float("inf")] * num_stages for _ in range(num_layers)]
        split_arr = [[-1] * num_stages for _ in range(num_layers)]

        def range_cost(si: int, start: int, end: int) -> float:
            return stage_prefix[si][end + 1] - stage_prefix[si][start]

        for i in range(num_layers):
            dp[i][0] = range_cost(0, 0, i)

        for si in range(1, num_stages):
            for end in range(si, num_layers):
                for cut in range(si - 1, end):
                    comp = range_cost(si, cut + 1, end)
                    comm = self._boundary_comm_time(
                        cut,
                        device_groups[stage_ids[si - 1]],
                        device_groups[stage_ids[si]],
                        config, topology, profiler,
                    )
                    candidate = max(dp[cut][si - 1], comp + comm)
                    if candidate < dp[end][si]:
                        dp[end][si] = candidate
                        split_arr[end][si] = cut

        points: list[int] = []
        end = num_layers - 1
        si = num_stages - 1
        while si > 0:
            cut = split_arr[end][si]
            if cut < 0:
                break
            points.append(cut)
            end = cut
            si -= 1
        points.reverse()

        bottleneck = dp[num_layers - 1][num_stages - 1]
        if bottleneck == float("inf"):
            points = self._fallback_points(num_layers, num_stages)
            bottleneck = 0.0
        return points, bottleneck

    def _layer_time(
        self,
        layer_idx: int,
        device_id: int,
        spec: DeviceSpec,
        config: AsteroidConfig,
        profiler: Any | None,
    ) -> float:
        if profiler is not None:
            try:
                fwd = profiler.get_time_interval(device_id, layer_idx, layer_idx, 0)
                bwd = profiler.get_time_interval(device_id, layer_idx, layer_idx, 1)
                if fwd > 0.0 and bwd > 0.0:
                    return fwd + bwd
            except Exception:
                pass
        dim_factor = max(1.0, config.embedding_dim / 1024.0)
        cap_factor = 1.0 / max(spec.compute_capacity, 0.1)
        return (1.0 + 0.015 * layer_idx) * dim_factor * cap_factor

    def _boundary_comm_time(
        self,
        boundary_layer: int,
        left_group: list[int],
        right_group: list[int],
        config: AsteroidConfig,
        topology: DeviceTopology,
        profiler: Any | None,
    ) -> float:
        if not left_group or not right_group:
            return 0.0
        if profiler is not None:
            try:
                out_sz = max(profiler.get_output_size(boundary_layer), 1e-6)
                left_bw = max(profiler.get_bandwidth(left_group[0]), 1e-6)
                right_bw = max(profiler.get_bandwidth(right_group[0]), 1e-6)
                return out_sz / min(left_bw, right_bw)
            except Exception:
                pass
        output_mb = self._default_output_mb(config)
        min_bw = float("inf")
        max_lat = 0.0
        for src in left_group:
            for dst in right_group:
                bw = self._lookup_metric(topology.bandwidths, src, dst, 1000.0)
                lat = self._lookup_metric(topology.latencies, src, dst, 0.1)
                min_bw = min(min_bw, max(bw, 1e-6))
                max_lat = max(max_lat, lat)
        return max_lat + output_mb / max(min_bw, 1e-6) * 1000.0

    @staticmethod
    def _default_output_mb(config: AsteroidConfig) -> float:
        total = config.max_seq_len * config.embedding_dim
        return max(1e-6, total * 4.0 / (1024.0 * 1024.0))

    def _normalize_topology(self, topology: DeviceTopology) -> DeviceTopology:
        if topology.device_specs:
            return topology
        count = max(1, self.pp_size * self.dp_size)
        specs = [DeviceSpec(device_id=i) for i in range(count)]
        return DeviceTopology(
            device_specs=specs,
            bandwidths=dict(topology.bandwidths),
            latencies=dict(topology.latencies),
        )

    @staticmethod
    def _fallback_points(num_layers: int, num_stages: int) -> list[int]:
        if num_stages <= 1:
            return []
        step = max(1, num_layers // num_stages)
        points: list[int] = []
        for si in range(1, num_stages):
            point = min(num_layers - 2, si * step - 1)
            points.append(point)
        return sorted(set(points))


__all__ = ["DTFMStrategy"]
