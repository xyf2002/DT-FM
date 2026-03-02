"""Confident strategy — bottleneck-minimising DP stage partition.

Ported from DeviceEmulator/baselines/strategies/confident_strategy.py.
Uses dynamic programming to find partition points that minimise the
bottleneck stage time across heterogeneous devices.
"""
from __future__ import annotations

import logging
from typing import Any

from asteroid.core.config import AsteroidConfig, DeviceSpec, DeviceTopology

from .base import ParallelismPlan, ParallelismStrategy

logger = logging.getLogger(__name__)


class ConfidentStrategy(ParallelismStrategy):
    """Confident baseline with DP bottleneck-minimising stage partition."""

    def __init__(self, pp_size: int = 2, dp_size: int = 1) -> None:
        self.pp_size = max(1, pp_size)
        self.dp_size = max(1, dp_size)

    # ── ParallelismStrategy interface ───────────────────────────────────────

    def create_plan(
        self,
        config: AsteroidConfig,
        device_topology: DeviceTopology,
        profiler: Any | None = None,
    ) -> ParallelismPlan:
        topology = self._normalize_topology(device_topology)
        num_layers = max(1, config.num_layers)
        num_stages = max(1, min(self.pp_size, len(topology.device_specs), num_layers))
        device_groups = {si: [si] for si in range(num_stages)}

        partition_points, est_latency = self._dp_partition(
            config, topology, num_stages, profiler,
        )
        logger.info(
            "Confident plan points=%s latency=%.2fms",
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

    def get_schedule_type(self) -> str:
        return "1f1b"

    # ── DP partition ────────────────────────────────────────────────────────

    def _dp_partition(
        self,
        config: AsteroidConfig,
        topology: DeviceTopology,
        num_stages: int,
        profiler: Any | None,
    ) -> tuple[list[int], float]:
        num_layers = max(1, config.num_layers)
        if num_stages <= 1:
            return [], 0.0

        # Build prefix-sum per-stage
        stage_prefix: list[list[float]] = []
        for si in range(num_stages):
            spec = topology.device_specs[si]
            cap = max(spec.compute_capacity, 0.1)
            prefix = [0.0]
            for li in range(num_layers):
                val = self._layer_time(config, spec, si, li, profiler)
                prefix.append(prefix[-1] + val / cap)
            stage_prefix.append(prefix)

        dp = [[float("inf")] * num_stages for _ in range(num_layers)]
        split = [[-1] * num_stages for _ in range(num_layers)]

        def range_cost(si: int, start: int, end: int) -> float:
            return stage_prefix[si][end + 1] - stage_prefix[si][start]

        for end in range(num_layers):
            dp[end][0] = range_cost(0, 0, end)

        for si in range(1, num_stages):
            for end in range(si, num_layers):
                for cut in range(si - 1, end):
                    stage_time = range_cost(si, cut + 1, end)
                    comm = self._comm_time(config, topology, cut, si - 1, si, profiler)
                    candidate = max(dp[cut][si - 1], stage_time + comm)
                    if candidate < dp[end][si]:
                        dp[end][si] = candidate
                        split[end][si] = cut

        # Backtrace
        points: list[int] = []
        end = num_layers - 1
        si = num_stages - 1
        while si > 0:
            cut = split[end][si]
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

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _normalize_topology(self, topology: DeviceTopology) -> DeviceTopology:
        specs = list(topology.device_specs)
        target = max(self.pp_size, 1)
        if not specs:
            specs = [DeviceSpec(device_id=i) for i in range(target)]
        elif len(specs) < target:
            for idx in range(len(specs), target):
                specs.append(DeviceSpec(device_id=idx))
        return DeviceTopology(
            device_specs=specs,
            bandwidths=dict(topology.bandwidths),
            latencies=dict(topology.latencies),
        )

    def _layer_time(
        self,
        config: AsteroidConfig,
        spec: DeviceSpec,
        device_id: int,
        layer_idx: int,
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
        dim = max(1.0, config.embedding_dim / 1024.0)
        cap = max(spec.compute_capacity, 0.1)
        return (1.0 + 0.02 * layer_idx) * dim / cap

    def _comm_time(
        self,
        config: AsteroidConfig,
        topology: DeviceTopology,
        boundary_layer: int,
        left_stage: int,
        right_stage: int,
        profiler: Any | None,
    ) -> float:
        if profiler is not None:
            try:
                out_size = max(profiler.get_output_size(boundary_layer), 1e-6)
                left_bw = max(profiler.get_bandwidth(left_stage), 1e-6)
                right_bw = max(profiler.get_bandwidth(right_stage), 1e-6)
                return out_size / min(left_bw, right_bw)
            except Exception:
                pass

        left_id = topology.device_specs[left_stage].device_id
        right_id = topology.device_specs[right_stage].device_id
        bw = self._lookup_link(topology.bandwidths, left_id, right_id, 1000.0)
        lat = self._lookup_link(topology.latencies, left_id, right_id, 0.1)
        payload = config.max_seq_len * config.embedding_dim * 4.0
        payload_mb = payload / (1024.0 * 1024.0)
        return lat + payload_mb / max(bw, 1e-6) * 1000.0

    @staticmethod
    def _lookup_link(table: dict, src: int, dst: int, default: float) -> float:
        val = table.get((src, dst))
        if val is not None:
            return float(val)
        rev = table.get((dst, src))
        if rev is not None:
            return float(rev)
        return default

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


__all__ = ["ConfidentStrategy"]
