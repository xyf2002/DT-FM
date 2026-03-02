from __future__ import annotations

import logging
import random
from typing_extensions import override

from baselines.core.config import (
    DeviceConfig,
    DeviceTopology,
    FaultToleranceConfig,
    ModelConfig,
    ParallelConfig,
    ParallelismPlan,
)
from baselines.core.profiler import ProfilerBackend
from baselines.core.scheduler import SchedulerBackend

from .base import ParallelismStrategy

logger = logging.getLogger(__name__)


class DTFMStrategy(ParallelismStrategy):
    """DT-FM baseline with GCMA topology search and DP partitioning."""

    def __init__(
        self,
        pp_size: int,
        dp_size: int,
        population_size: int = 100,
        gcma_trails: int = 4900,
    ) -> None:
        self.pp_size: int = max(1, pp_size)
        self.dp_size: int = max(1, dp_size)
        self.population_size: int = max(2, population_size)
        self.gcma_trails: int = max(1, gcma_trails)
        self.parallel_config: ParallelConfig = ParallelConfig(
            pp_size=self.pp_size,
            dp_size=self.dp_size,
            schedule_type=self.get_schedule_type(),
        )
        self.scheduler_backend: SchedulerBackend | None = None

    @override
    def create_plan(
        self,
        model_config: ModelConfig,
        device_topology: DeviceTopology,
        profiler: ProfilerBackend | None = None,
    ) -> ParallelismPlan:
        topology = self._normalize_topology(device_topology)
        if len(topology.device_specs) > 1:
            device_groups = self._run_gcma(topology)
        else:
            did = topology.device_specs[0].device_id
            device_groups = {0: [did]}

        partition_points, est_latency = self._dp_partition(
            model_config,
            topology,
            device_groups,
            profiler,
        )
        logger.info(
            "DTFM strategy plan points=%s latency=%.2fms",
            partition_points,
            est_latency,
        )
        return ParallelismPlan(
            partition_points=partition_points,
            device_groups=device_groups,
            micro_batch_alloc={},
            schedule_type=self.get_schedule_type(),
            estimated_latency_ms=est_latency,
        )

    @override
    def get_schedule_type(self) -> str:
        return "gpipe"

    @override
    def get_fault_tolerance_config(self) -> FaultToleranceConfig:
        return FaultToleranceConfig(
            checkpoint_dir="./checkpoints/dtfm",
            checkpoint_interval=100,
            heartbeat_interval_s=0.0,
            heartbeat_timeout_s=0.0,
            backward_timeout_ms=0.0,
            replication_mode="none",
            replication_interval=0,
            ft_check_interval=10,
        )

    def _normalize_topology(self, topology: DeviceTopology) -> DeviceTopology:
        if topology.device_specs:
            return topology
        count = max(1, self.pp_size * self.dp_size)
        specs = [DeviceConfig(device_id=i) for i in range(count)]
        return DeviceTopology(
            device_specs=specs,
            bandwidths=dict(topology.bandwidths),
            latencies=dict(topology.latencies),
        )

    def _run_gcma(self, topology: DeviceTopology) -> dict[int, list[int]]:
        specs = topology.device_specs
        device_ids = [spec.device_id for spec in specs]
        num_devices = len(device_ids)
        stage_count = max(1, min(self.pp_size, num_devices))
        if stage_count == 1:
            return {0: device_ids}

        peer_delay, peer_bandwidth, idx_of = self._build_peer_matrices(topology)
        target_width = max(1, self.dp_size)
        rng = random.Random(0)

        population: list[list[int]] = []
        pop_n = min(self.population_size, max(num_devices * 2, 4))
        for seed in range(pop_n):
            order = device_ids[:]
            rng.seed(seed)
            rng.shuffle(order)
            population.append(order)

        best_order = device_ids[:]
        best_groups = self._assignment_to_groups(
            best_order,
            stage_count,
            target_width,
        )
        best_score = self._score_grouping(
            best_groups,
            peer_delay,
            peer_bandwidth,
            idx_of,
        )

        scores = [
            self._score_grouping(
                self._assignment_to_groups(order, stage_count, target_width),
                peer_delay,
                peer_bandwidth,
                idx_of,
            )
            for order in population
        ]

        for _ in range(self.gcma_trails):
            p1 = rng.randrange(len(population))
            p2 = rng.randrange(len(population))
            if p1 == p2:
                continue
            child = self._crossover(population[p1], population[p2], rng)
            child = self._mutate(child, rng)
            child_groups = self._assignment_to_groups(
                child,
                stage_count,
                target_width,
            )
            child_score = self._score_grouping(
                child_groups,
                peer_delay,
                peer_bandwidth,
                idx_of,
            )

            replace = p1 if scores[p1] >= scores[p2] else p2
            if child_score < scores[replace]:
                population[replace] = child
                scores[replace] = child_score
                if child_score < best_score:
                    best_score = child_score
                    best_groups = child_groups

        logger.debug("GCMA best score %.6f", best_score)
        return best_groups

    def _assignment_to_groups(
        self,
        order: list[int],
        stage_count: int,
        target_width: int,
    ) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = {
            stage: [] for stage in range(stage_count)
        }
        cursor = 0
        for stage in range(stage_count):
            if cursor < len(order):
                groups[stage].append(order[cursor])
                cursor += 1
        for stage in range(stage_count):
            while len(groups[stage]) < target_width and cursor < len(order):
                groups[stage].append(order[cursor])
                cursor += 1
        while cursor < len(order):
            stage = cursor % stage_count
            groups[stage].append(order[cursor])
            cursor += 1
        return groups

    def _score_grouping(
        self,
        groups: dict[int, list[int]],
        peer_delay: list[list[float]],
        peer_bandwidth: list[list[float]],
        idx_of: dict[int, int],
    ) -> float:
        stage_ids = sorted(groups)
        data_parallel_cost = 0.0
        for stage in stage_ids:
            group = groups[stage]
            if len(group) <= 1:
                continue
            group_cost = 0.0
            for src in group:
                acc = 0.0
                for dst in group:
                    if src == dst:
                        continue
                    acc += self._transfer_cost(
                        src,
                        dst,
                        1.0,
                        peer_delay,
                        peer_bandwidth,
                        idx_of,
                    )
                group_cost = max(group_cost, acc)
            data_parallel_cost = max(data_parallel_cost, group_cost)

        pipeline_parallel_cost = 0.0
        for i in range(len(stage_ids) - 1):
            left = groups[stage_ids[i]]
            right = groups[stage_ids[i + 1]]
            pipeline_parallel_cost += self._inter_group_cost(
                left,
                right,
                peer_delay,
                peer_bandwidth,
                idx_of,
            )
        return data_parallel_cost + 2.0 * pipeline_parallel_cost

    def _inter_group_cost(
        self,
        left: list[int],
        right: list[int],
        peer_delay: list[list[float]],
        peer_bandwidth: list[list[float]],
        idx_of: dict[int, int],
    ) -> float:
        if not left or not right:
            return float("inf")
        used: set[int] = set()
        max_cost = 0.0
        for src in left:
            best_cost = float("inf")
            best_dst = right[0]
            for dst in right:
                if dst in used:
                    continue
                cur = self._transfer_cost(
                    src,
                    dst,
                    1.0,
                    peer_delay,
                    peer_bandwidth,
                    idx_of,
                )
                if cur < best_cost:
                    best_cost = cur
                    best_dst = dst
            if best_cost == float("inf"):
                best_cost = min(
                    self._transfer_cost(
                        src,
                        dst,
                        1.0,
                        peer_delay,
                        peer_bandwidth,
                        idx_of,
                    )
                    for dst in right
                )
            else:
                used.add(best_dst)
            max_cost = max(max_cost, best_cost)
        return max_cost

    def _crossover(
        self,
        parent_a: list[int],
        parent_b: list[int],
        rng: random.Random,
    ) -> list[int]:
        n = len(parent_a)
        if n <= 1:
            return parent_a[:]
        left, right = sorted(rng.sample(range(n), 2))
        child = [-1] * n
        child[left : right + 1] = parent_a[left : right + 1]
        used = set(child[left : right + 1])

        write = 0
        for value in parent_b:
            if value in used:
                continue
            while write < n and child[write] != -1:
                write += 1
            if write < n:
                child[write] = value
        return [value if value != -1 else 0 for value in child]

    def _mutate(self, order: list[int], rng: random.Random) -> list[int]:
        if len(order) > 1 and rng.random() < 0.35:
            i, j = rng.sample(range(len(order)), 2)
            order[i], order[j] = order[j], order[i]
        return order

    def _build_peer_matrices(
        self,
        topology: DeviceTopology,
    ) -> tuple[list[list[float]], list[list[float]], dict[int, int]]:
        specs = topology.device_specs
        num_devices = len(specs)
        idx_of = {spec.device_id: idx for idx, spec in enumerate(specs)}
        peer_delay: list[list[float]] = []
        peer_bandwidth: list[list[float]] = []

        for i, src in enumerate(specs):
            delay_row: list[float] = []
            bw_row: list[float] = []
            for j, dst in enumerate(specs):
                if i == j:
                    delay_row.append(0.0)
                    bw_row.append(max(src.compute_capacity, 1.0))
                    continue
                delay = self._lookup_metric(
                    topology.latencies,
                    src.device_id,
                    dst.device_id,
                    1.0,
                )
                bw = self._lookup_metric(
                    topology.bandwidths,
                    src.device_id,
                    dst.device_id,
                    1000.0,
                )
                delay_row.append(max(delay, 1e-6))
                bw_row.append(max(bw, 1e-6))
            peer_delay.append(delay_row)
            peer_bandwidth.append(bw_row)

        if num_devices == 0:
            return [[0.0]], [[1.0]], {0: 0}
        return peer_delay, peer_bandwidth, idx_of

    def _lookup_metric(
        self,
        table: dict[tuple[int, int], float],
        src: int,
        dst: int,
        default: float,
    ) -> float:
        direct = table.get((src, dst))
        if direct is not None:
            return float(direct)
        reverse = table.get((dst, src))
        if reverse is not None:
            return float(reverse)
        return default

    def _transfer_cost(
        self,
        src: int,
        dst: int,
        payload_gb: float,
        peer_delay: list[list[float]],
        peer_bandwidth: list[list[float]],
        idx_of: dict[int, int],
    ) -> float:
        src_idx = idx_of[src]
        dst_idx = idx_of[dst]
        delay_ms = peer_delay[src_idx][dst_idx]
        bw_gbps = max(peer_bandwidth[src_idx][dst_idx], 1e-6)
        return delay_ms / 1e3 + (payload_gb * 8.0 / bw_gbps)

    def _dp_partition(
        self,
        model_config: ModelConfig,
        topology: DeviceTopology,
        device_groups: dict[int, list[int]],
        profiler: ProfilerBackend | None,
    ) -> tuple[list[int], float]:
        num_layers = max(1, model_config.num_layers)
        stage_ids = sorted(device_groups)
        num_stages = min(len(stage_ids), num_layers)
        if num_stages <= 1:
            return [], 0.0

        spec_by_id = {
            spec.device_id: spec for spec in topology.device_specs
        }
        stage_prefix: list[list[float]] = []
        for stage_idx in range(num_stages):
            stage = stage_ids[stage_idx]
            rep_id = device_groups[stage][0]
            spec = spec_by_id.get(rep_id, DeviceConfig(device_id=rep_id))
            cap = max(spec.compute_capacity, 1e-6)
            prefix = [0.0]
            for layer_idx in range(num_layers):
                layer_t = self._layer_time(
                    layer_idx,
                    rep_id,
                    spec,
                    model_config,
                    profiler,
                )
                prefix.append(prefix[-1] + layer_t / cap)
            stage_prefix.append(prefix)

        dp = [
            [float("inf")] * num_stages
            for _ in range(num_layers)
        ]
        split = [[-1] * num_stages for _ in range(num_layers)]

        def range_cost(stage_idx: int, start: int, end: int) -> float:
            prefix = stage_prefix[stage_idx]
            return prefix[end + 1] - prefix[start]

        for i in range(num_layers):
            dp[i][0] = range_cost(0, 0, i)

        for stage_idx in range(1, num_stages):
            for end in range(stage_idx, num_layers):
                for cut in range(stage_idx - 1, end):
                    comp = range_cost(stage_idx, cut + 1, end)
                    comm = self._boundary_comm_time(
                        boundary_layer=cut,
                        left_group=device_groups[stage_ids[stage_idx - 1]],
                        right_group=device_groups[stage_ids[stage_idx]],
                        model_config=model_config,
                        topology=topology,
                        profiler=profiler,
                    )
                    candidate = max(dp[cut][stage_idx - 1], comp + comm)
                    if candidate < dp[end][stage_idx]:
                        dp[end][stage_idx] = candidate
                        split[end][stage_idx] = cut

        points: list[int] = []
        end = num_layers - 1
        stage_idx = num_stages - 1
        while stage_idx > 0:
            cut = split[end][stage_idx]
            if cut < 0:
                break
            points.append(cut)
            end = cut
            stage_idx -= 1
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
        device_cfg: DeviceConfig,
        model_config: ModelConfig,
        profiler: ProfilerBackend | None,
    ) -> float:
        if profiler is not None:
            try:
                fwd = profiler.get_time_interval(
                    device_id,
                    layer_idx,
                    layer_idx,
                    0,
                )
                bwd = profiler.get_time_interval(
                    device_id,
                    layer_idx,
                    layer_idx,
                    1,
                )
                if fwd > 0.0 and bwd > 0.0:
                    return fwd + bwd
            except Exception:
                logger.debug(
                    "Profiler layer lookup failed for device %s layer %s",
                    device_id,
                    layer_idx,
                )
        dim_factor = max(1.0, model_config.embedding_dim / 1024.0)
        cap_factor = 1.0 / max(device_cfg.compute_capacity, 0.1)
        return (1.0 + 0.015 * layer_idx) * dim_factor * cap_factor

    def _boundary_comm_time(
        self,
        boundary_layer: int,
        left_group: list[int],
        right_group: list[int],
        model_config: ModelConfig,
        topology: DeviceTopology,
        profiler: ProfilerBackend | None,
    ) -> float:
        if not left_group or not right_group:
            return 0.0

        if profiler is not None:
            try:
                output_size = max(
                    profiler.get_output_size(boundary_layer),
                    1e-6,
                )
                left_bw = max(profiler.get_bandwidth(left_group[0]), 1e-6)
                right_bw = max(profiler.get_bandwidth(right_group[0]), 1e-6)
                return output_size / min(left_bw, right_bw)
            except Exception:
                logger.debug(
                    "Profiler comm lookup failed at boundary %s",
                    boundary_layer,
                )

        output_mb = self._default_output_mb(model_config)
        min_bw = float("inf")
        max_lat = 0.0
        for src in left_group:
            for dst in right_group:
                bw = self._lookup_metric(topology.bandwidths, src, dst, 1000.0)
                lat = self._lookup_metric(topology.latencies, src, dst, 0.1)
                min_bw = min(min_bw, max(bw, 1e-6))
                max_lat = max(max_lat, lat)
        return max_lat + output_mb / max(min_bw, 1e-6) * 1000.0

    def _default_output_mb(self, model_config: ModelConfig) -> float:
        bytes_per_token = 4.0
        total = model_config.seq_length * model_config.embedding_dim
        return max(1e-6, total * bytes_per_token / (1024.0 * 1024.0))

    def _fallback_points(self, num_layers: int, num_stages: int) -> list[int]:
        if num_stages <= 1:
            return []
        step = max(1, num_layers // num_stages)
        points: list[int] = []
        for stage_idx in range(1, num_stages):
            point = min(num_layers - 2, stage_idx * step - 1)
            points.append(point)
        return sorted(set(points))


__all__ = ["DTFMStrategy"]
