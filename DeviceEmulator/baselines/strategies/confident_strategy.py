from __future__ import annotations

import logging
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

from .base import ParallelismStrategy

logger = logging.getLogger(__name__)


class ConfidentStrategy(ParallelismStrategy):
    """Confident baseline with DP bottleneck-minimizing stage partition."""

    def __init__(self, pp_size: int, dp_size: int) -> None:
        self.pp_size: int = max(1, pp_size)
        self.dp_size: int = max(1, dp_size)
        self.parallel_config: ParallelConfig = ParallelConfig(
            pp_size=self.pp_size,
            dp_size=self.dp_size,
            schedule_type=self.get_schedule_type(),
        )

    @override
    def create_plan(
        self,
        model_config: ModelConfig,
        device_topology: DeviceTopology,
        profiler: ProfilerBackend | None = None,
    ) -> ParallelismPlan:
        topology = self._normalize_topology(device_topology)
        num_layers = max(1, model_config.num_layers)
        num_stages = min(self.pp_size, len(topology.device_specs), num_layers)
        num_stages = max(1, num_stages)
        device_groups = {
            stage_idx: [stage_idx] for stage_idx in range(num_stages)
        }

        partition_points, est_latency = self._dp_partition(
            model_config,
            topology,
            num_stages,
            profiler,
        )
        logger.info(
            "Confident strategy plan points=%s latency=%.2fms",
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
        return "1f1b"

    @override
    def get_fault_tolerance_config(self) -> FaultToleranceConfig:
        return FaultToleranceConfig(
            checkpoint_dir="./checkpoints/confident",
            checkpoint_interval=100,
            heartbeat_interval_s=0.0,
            heartbeat_timeout_s=0.0,
            backward_timeout_ms=30000.0,
            replication_mode="local_global",
            replication_interval=20,
            ft_check_interval=5,
        )

    def _normalize_topology(self, topology: DeviceTopology) -> DeviceTopology:
        specs = list(topology.device_specs)
        target = max(self.pp_size, 1)
        if not specs:
            specs = [DeviceConfig(device_id=i) for i in range(target)]
        elif len(specs) < target:
            start = len(specs)
            for idx in range(start, target):
                specs.append(DeviceConfig(device_id=idx))
        return DeviceTopology(
            device_specs=specs,
            bandwidths=dict(topology.bandwidths),
            latencies=dict(topology.latencies),
        )

    def _dp_partition(
        self,
        model_config: ModelConfig,
        topology: DeviceTopology,
        num_stages: int,
        profiler: ProfilerBackend | None,
    ) -> tuple[list[int], float]:
        num_layers = max(1, model_config.num_layers)
        if num_stages <= 1:
            return [], 0.0

        stage_prefix: list[list[float]] = []
        for stage_idx in range(num_stages):
            spec = topology.device_specs[stage_idx]
            cap = max(spec.compute_capacity, 0.1)
            prefix = [0.0]
            for layer_idx in range(num_layers):
                value = self._layer_time(
                    model_config,
                    spec,
                    stage_idx,
                    layer_idx,
                    profiler,
                )
                prefix.append(prefix[-1] + value / cap)
            stage_prefix.append(prefix)

        dp = [
            [float("inf")] * num_stages
            for _ in range(num_layers)
        ]
        split = [[-1] * num_stages for _ in range(num_layers)]

        def range_cost(stage_idx: int, start: int, end: int) -> float:
            prefix = stage_prefix[stage_idx]
            return prefix[end + 1] - prefix[start]

        for end in range(num_layers):
            dp[end][0] = range_cost(0, 0, end)

        for stage_idx in range(1, num_stages):
            for end in range(stage_idx, num_layers):
                for cut in range(stage_idx - 1, end):
                    stage_time = range_cost(stage_idx, cut + 1, end)
                    comm = self._comm_time(
                        model_config,
                        topology,
                        cut,
                        stage_idx - 1,
                        stage_idx,
                        profiler,
                    )
                    candidate = max(dp[cut][stage_idx - 1], stage_time + comm)
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
        model_config: ModelConfig,
        spec: DeviceConfig,
        device_id: int,
        layer_idx: int,
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
                logger.debug("Confident profiler miss layer=%s device=%s",
                             layer_idx, device_id)
        dim = max(1.0, model_config.embedding_dim / 1024.0)
        cap = max(spec.compute_capacity, 0.1)
        return (1.0 + 0.02 * layer_idx) * dim / cap

    def _comm_time(
        self,
        model_config: ModelConfig,
        topology: DeviceTopology,
        boundary_layer: int,
        left_stage: int,
        right_stage: int,
        profiler: ProfilerBackend | None,
    ) -> float:
        if profiler is not None:
            try:
                out_size = max(profiler.get_output_size(boundary_layer), 1e-6)
                left_bw = max(profiler.get_bandwidth(left_stage), 1e-6)
                right_bw = max(profiler.get_bandwidth(right_stage), 1e-6)
                return out_size / min(left_bw, right_bw)
            except Exception:
                logger.debug(
                    "Confident profiler comm miss boundary=%s",
                    boundary_layer,
                )

        left_id = topology.device_specs[left_stage].device_id
        right_id = topology.device_specs[right_stage].device_id
        bw = self._lookup_link(topology.bandwidths, left_id, right_id, 1000.0)
        lat = self._lookup_link(topology.latencies, left_id, right_id, 0.1)
        payload = model_config.seq_length * model_config.embedding_dim * 4.0
        payload_mb = payload / (1024.0 * 1024.0)
        return lat + payload_mb / max(bw, 1e-6) * 1000.0

    def _lookup_link(
        self,
        table: dict[tuple[int, int], float],
        src: int,
        dst: int,
        default: float,
    ) -> float:
        val = table.get((src, dst))
        if val is not None:
            return float(val)
        rev = table.get((dst, src))
        if rev is not None:
            return float(rev)
        return default

    def _fallback_points(self, num_layers: int, num_stages: int) -> list[int]:
        if num_stages <= 1:
            return []
        step = max(1, num_layers // num_stages)
        points: list[int] = []
        for stage_idx in range(1, num_stages):
            point = min(num_layers - 2, stage_idx * step - 1)
            points.append(point)
        return sorted(set(points))


__all__ = ["ConfidentStrategy"]
