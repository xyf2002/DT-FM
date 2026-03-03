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
        num_devices = len(topology.device_specs)

        best_plan: ParallelismPlan | None = None

        # Auto-search: try every valid (PP, DP) factorisation and pick
        # the one with lowest bottleneck latency.
        for pp in range(1, min(num_devices, num_layers) + 1):
            if num_devices % pp != 0:
                continue
            dp = num_devices // pp

            # Assign devices to stages in order: stage 0 gets first dp
            # devices, stage 1 gets next dp devices, etc.
            device_groups: dict[int, list[int]] = {}
            specs = topology.device_specs
            for si in range(pp):
                group_specs = specs[si * dp : (si + 1) * dp]
                device_groups[si] = [s.device_id for s in group_specs]

            partition_points, est_latency = self._dp_partition(
                config, topology, device_groups, profiler,
            )
            if est_latency == float("inf") or est_latency < 0:
                continue

            # Model full pipeline schedule: bottleneck × (PP + ceil(M/dp) - 1)
            # With DP, each pipeline only processes M/dp micro-batches.
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
                    "Confident candidate pp=%d dp=%d step=%.2fms batch=%.2fms (M=%d, M/dp=%d)",
                    pp, dp, est_latency, effective_latency, M, micro_per_pipe,
                )

        if best_plan is None:
            # Fallback: single stage with all devices
            device_ids = [s.device_id for s in topology.device_specs]
            best_plan = ParallelismPlan(
                device_groups={0: device_ids},
                schedule_type=self.get_schedule_type(),
            )

        logger.info(
            "Confident plan: %d stages, points=%s latency=%.2fms",
            len(best_plan.device_groups),
            best_plan.partition_points,
            best_plan.estimated_latency_ms,
        )
        return best_plan

    def get_schedule_type(self) -> str:
        return "1f1b"

    # ── DP partition ────────────────────────────────────────────────────────

    def _dp_partition(
        self,
        config: AsteroidConfig,
        topology: DeviceTopology,
        device_groups: dict[int, list[int]],
        profiler: Any | None,
    ) -> tuple[list[int], float]:
        num_layers = max(1, config.num_layers)
        stage_ids = sorted(device_groups)
        num_stages = len(stage_ids)
        spec_by_id = {s.device_id: s for s in topology.device_specs}

        if num_stages <= 1:
            # Single stage: bottleneck = slowest device in the group
            group = device_groups[stage_ids[0]] if stage_ids else []
            if not group:
                return [], 0.0
            worst = float("-inf")
            for did in group:
                spec = spec_by_id.get(did, DeviceSpec(device_id=did))
                total = sum(
                    self._layer_time(config, spec, did, li, profiler)
                    for li in range(num_layers)
                )
                worst = max(worst, total)
            return [], worst

        # For each stage use the slowest device in the group (bottleneck)
        stage_prefix: list[list[float]] = []
        stage_rep_id: list[int] = []
        for si_idx in range(num_stages):
            stage = stage_ids[si_idx]
            group = device_groups[stage]
            # Pick slowest device as representative (bottleneck for DP)
            slowest_id = group[0]
            slowest_time = float("-inf")
            for did in group:
                spec = spec_by_id.get(did, DeviceSpec(device_id=did))
                t = sum(self._layer_time(config, spec, did, li, profiler)
                        for li in range(num_layers))
                if t > slowest_time:
                    slowest_time = t
                    slowest_id = did
            spec = spec_by_id.get(slowest_id, DeviceSpec(device_id=slowest_id))
            stage_rep_id.append(slowest_id)
            prefix = [0.0]
            for li in range(num_layers):
                val = self._layer_time(config, spec, slowest_id, li, profiler)
                prefix.append(prefix[-1] + val)
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
                    comm = self._comm_time_groups(
                        config, topology, cut,
                        device_groups[stage_ids[si - 1]],
                        device_groups[stage_ids[si]],
                        profiler,
                    )
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
        if not specs:
            target = max(self.pp_size * self.dp_size, 1)
            specs = [DeviceSpec(device_id=i) for i in range(target)]
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

    def _comm_time_groups(
        self,
        config: AsteroidConfig,
        topology: DeviceTopology,
        boundary_layer: int,
        left_group: list[int],
        right_group: list[int],
        profiler: Any | None = None,
    ) -> float:
        """Inter-stage comm cost.  Matches reference: output_size / bandwidth."""
        # Try profiler first (reference behaviour)
        if profiler is not None:
            try:
                out_size = max(profiler.get_output_size(boundary_layer), 1e-6)
                # Use worst (slowest) bandwidth among sending group devices
                worst_bw = float("inf")
                for did in left_group:
                    worst_bw = min(worst_bw, max(profiler.get_bandwidth(did), 1e-6))
                return out_size / worst_bw
            except Exception:
                pass
        # Fallback: topology-based
        payload = config.max_seq_len * config.embedding_dim * 4.0
        payload_mb = payload / (1024.0 * 1024.0)
        worst = 0.0
        for src in left_group:
            for dst in right_group:
                bw = self._lookup_link(topology.bandwidths, src, dst, 1000.0)
                lat = self._lookup_link(topology.latencies, src, dst, 0.1)
                cost = lat + payload_mb / max(bw, 1e-6) * 1000.0
                worst = max(worst, cost)
        return worst

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
