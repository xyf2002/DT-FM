"""Asteroid strategy — HPP DP planning with micro-batch allocation.

Ported from DeviceEmulator/baselines/strategies/asteroid_strategy.py.
Uses dynamic programming to find optimal layer partition across
heterogeneous devices, with memory-aware micro-batch allocation
that targets the Mem_p(β) = Mem_MOD + Mem_OPT + K_p × Mem_ACT(β)
formula from the ASTEROID paper.
"""
from __future__ import annotations

import logging
from typing import Any, TypedDict

from asteroid.core.config import AsteroidConfig, DeviceSpec, DeviceTopology

from .base import ParallelismPlan, ParallelismStrategy

logger = logging.getLogger(__name__)


class _DPState(TypedDict):
    ranges: list[tuple[int, int]]
    groups: list[list[int]]
    allocs: list[dict[int, int]]


class AsteroidStrategy(ParallelismStrategy):
    """ASTEROID baseline using HPP DP planning for hybrid pipelines."""

    def __init__(
        self,
        num_stages: int = 2,
        micro_batch_size: int = 4,
        num_microbatches: int = 8,
    ) -> None:
        self.num_stages = max(1, num_stages)
        self.micro_batch_size = max(1, micro_batch_size)
        self.num_microbatches = max(1, num_microbatches)

    # ── ParallelismStrategy interface ───────────────────────────────────────

    def create_plan(
        self,
        config: AsteroidConfig,
        device_topology: DeviceTopology,
        profiler: Any | None = None,
    ) -> ParallelismPlan:
        topology = self._normalize_topology(device_topology)
        exec_profiles = self._build_exec_profiles(config, topology, profiler)
        max_stages = min(
            max(1, self.num_stages + 2),
            len(topology.device_specs),
            max(1, config.num_layers),
        )

        best_plan: ParallelismPlan | None = None
        for stages in range(1, max_stages + 1):
            candidate = self._dp_plan(config, topology, exec_profiles, stages)
            if candidate is None:
                continue
            if best_plan is None or candidate.estimated_latency_ms < best_plan.estimated_latency_ms:
                best_plan = candidate

        if best_plan is None:
            best_plan = self._fallback_plan(config, topology, exec_profiles)

        logger.info(
            "Asteroid plan points=%s latency=%.2fms",
            best_plan.partition_points,
            best_plan.estimated_latency_ms,
        )
        return best_plan

    def get_schedule_type(self) -> str:
        return "1f1b"

    # ── Topology normalisation ──────────────────────────────────────────────

    def _normalize_topology(self, topology: DeviceTopology) -> DeviceTopology:
        if topology.device_specs:
            return topology
        specs = [DeviceSpec(device_id=i) for i in range(self.num_stages)]
        return DeviceTopology(
            device_specs=specs,
            bandwidths=dict(topology.bandwidths),
            latencies=dict(topology.latencies),
        )

    # ── Execution profiles ──────────────────────────────────────────────────

    def _build_exec_profiles(
        self,
        config: AsteroidConfig,
        topology: DeviceTopology,
        profiler: Any | None,
    ) -> dict[int, list[tuple[float, float]]]:
        profiles: dict[int, list[tuple[float, float]]] = {}
        num_layers = max(1, config.num_layers)
        for spec in topology.device_specs:
            layer_times: list[tuple[float, float]] = []
            for layer_idx in range(num_layers):
                if profiler is not None:
                    try:
                        fwd = profiler.get_time_interval(spec.device_id, layer_idx, layer_idx, 0)
                        bwd = profiler.get_time_interval(spec.device_id, layer_idx, layer_idx, 1)
                        if fwd > 0.0 and bwd > 0.0:
                            layer_times.append((fwd, bwd))
                            continue
                    except Exception:
                        pass
                layer_times.append(self._synthetic_layer_time(config, spec, layer_idx))
            profiles[spec.device_id] = layer_times
        return profiles

    def _synthetic_layer_time(
        self,
        config: AsteroidConfig,
        spec: DeviceSpec,
        layer_idx: int,
    ) -> tuple[float, float]:
        cap = max(spec.compute_capacity, 0.1)
        dim = max(1.0, config.embedding_dim / 1024.0)
        base = (0.9 + 0.01 * layer_idx) * dim / cap
        return base, 2.0 * base

    # ── DP planning ─────────────────────────────────────────────────────────

    def _dp_plan(
        self,
        config: AsteroidConfig,
        topology: DeviceTopology,
        exec_profiles: dict[int, list[tuple[float, float]]],
        num_stages: int,
    ) -> ParallelismPlan | None:
        num_layers = max(1, config.num_layers)
        device_ids = [spec.device_id for spec in topology.device_specs]
        num_devices = len(device_ids)
        if num_stages > num_layers or num_stages > num_devices:
            return None

        inf = float("inf")
        latency = [
            [[inf] * (num_stages + 1) for _ in range(num_devices + 1)]
            for _ in range(num_layers + 1)
        ]
        dp_config: list[list[list[_DPState | None]]] = [
            [[None] * (num_stages + 1) for _ in range(num_devices + 1)]
            for _ in range(num_layers + 1)
        ]

        # Base case: single stage
        for layers_tail in range(1, num_layers + 1):
            for devices_tail in range(1, num_devices + 1):
                group = device_ids[num_devices - devices_tail:]
                start = num_layers - layers_tail
                end = num_layers
                alloc, stage_exec = self._alloc_microbatch(
                    config, topology, exec_profiles,
                    num_stages - 1, num_stages, group, start, end,
                )
                if not alloc:
                    continue
                ar = self._allreduce_time(config, topology, group, start, end)
                stage_latency = self.num_microbatches * stage_exec + ar
                latency[layers_tail][devices_tail][1] = stage_latency
                dp_config[layers_tail][devices_tail][1] = {
                    "ranges": [(start, end)],
                    "groups": [group],
                    "allocs": [alloc],
                }

        # Multi-stage DP
        for stages_tail in range(2, num_stages + 1):
            for layers_tail in range(stages_tail, num_layers + 1):
                for devices_tail in range(stages_tail, num_devices + 1):
                    for prev_lt in range(stages_tail - 1, layers_tail):
                        for prev_dt in range(stages_tail - 1, devices_tail):
                            prev_lat = latency[prev_lt][prev_dt][stages_tail - 1]
                            if prev_lat >= inf:
                                continue
                            group = device_ids[num_devices - devices_tail: num_devices - prev_dt]
                            if not group:
                                continue
                            start = num_layers - layers_tail
                            end = num_layers - prev_lt
                            stage_idx = num_stages - stages_tail
                            alloc, stage_exec = self._alloc_microbatch(
                                config, topology, exec_profiles,
                                stage_idx, num_stages, group, start, end,
                            )
                            if not alloc:
                                continue
                            prev_cfg = dp_config[prev_lt][prev_dt][stages_tail - 1]
                            if prev_cfg is None:
                                continue
                            next_group = prev_cfg["groups"][0]
                            comm = self._comm_time_inter_stage(
                                config, topology, end - 1, group, next_group, sum(alloc.values()),
                            )
                            ar = self._allreduce_time(config, topology, group, start, end)
                            step_lat = self.num_microbatches * stage_exec
                            total = max(prev_lat, step_lat + comm) + ar

                            if total < latency[layers_tail][devices_tail][stages_tail]:
                                latency[layers_tail][devices_tail][stages_tail] = total
                                dp_config[layers_tail][devices_tail][stages_tail] = {
                                    "ranges": [(start, end)] + prev_cfg["ranges"],
                                    "groups": [group] + prev_cfg["groups"],
                                    "allocs": [alloc] + prev_cfg["allocs"],
                                }

        best = latency[num_layers][num_devices][num_stages]
        best_cfg = dp_config[num_layers][num_devices][num_stages]
        if best >= inf or best_cfg is None:
            return None

        ranges = best_cfg["ranges"]
        groups = best_cfg["groups"]
        allocs = best_cfg["allocs"]
        partition_points = [rng[1] - 1 for rng in ranges[:-1]]
        device_groups = {si: g for si, g in enumerate(groups)}
        micro_batch_alloc = {si: a for si, a in enumerate(allocs)}
        return ParallelismPlan(
            partition_points=partition_points,
            device_groups=device_groups,
            micro_batch_alloc=micro_batch_alloc,
            schedule_type=self.get_schedule_type(),
            estimated_latency_ms=best,
        )

    # ── Micro-batch allocation ──────────────────────────────────────────────

    def _alloc_microbatch(
        self,
        config: AsteroidConfig,
        topology: DeviceTopology,
        exec_profiles: dict[int, list[tuple[float, float]]],
        stage_idx: int,
        num_stages: int,
        device_group: list[int],
        start_l: int,
        end_l: int,
    ) -> tuple[dict[int, int], float]:
        if not device_group:
            return {}, float("inf")

        spec_by_id = {s.device_id: s for s in topology.device_specs}
        alloc = {did: 0 for did in device_group}
        remaining = self.micro_batch_size

        while remaining > 0:
            best_id: int | None = None
            best_cost = float("inf")
            for did in device_group:
                spec = spec_by_id.get(did, DeviceSpec(device_id=did))
                next_bs = alloc[did] + 1
                mem = self._memory_footprint(config, stage_idx, num_stages, start_l, end_l, next_bs)
                if mem > spec.memory_budget_mb:
                    continue
                projected = self._device_exec_time(exec_profiles, did, start_l, end_l, next_bs, spec.compute_capacity)
                if projected < best_cost:
                    best_cost = projected
                    best_id = did
            if best_id is None:
                break
            alloc[best_id] += 1
            remaining -= 1

        if remaining > 0:
            fastest = max(
                device_group,
                key=lambda did: spec_by_id.get(did, DeviceSpec(device_id=did)).compute_capacity,
            )
            alloc[fastest] += remaining

        active = {did: bs for did, bs in alloc.items() if bs > 0}
        if not active:
            return {}, float("inf")
        straggler = max(
            self._device_exec_time(exec_profiles, did, start_l, end_l, bs, spec_by_id.get(did, DeviceSpec(device_id=did)).compute_capacity)
            for did, bs in active.items()
        )
        return alloc, straggler

    def _memory_footprint(
        self,
        config: AsteroidConfig,
        stage_idx: int,
        num_stages: int,
        start_l: int,
        end_l: int,
        batch_size: int,
    ) -> float:
        """Mem_p(β) = Mem_MOD + Mem_OPT + K_p × Mem_ACT(β)."""
        k_p = max(1, 2 * (num_stages - stage_idx) - 1)
        mem_mod = self._weights_mb(config, start_l, end_l)
        mem_opt = mem_mod * 2.0
        mem_act = self._activations_mb(config, start_l, end_l) * max(batch_size, 0)
        return mem_mod + mem_opt + k_p * mem_act

    def _device_exec_time(
        self,
        exec_profiles: dict[int, list[tuple[float, float]]],
        device_id: int,
        start_l: int,
        end_l: int,
        batch_size: int,
        capacity: float,
    ) -> float:
        profile = exec_profiles.get(device_id, [])
        total = 0.0
        for layer_idx in range(start_l, end_l):
            if not profile:
                fwd, bwd = 1.0, 2.0
            else:
                idx = min(layer_idx, len(profile) - 1)
                fwd, bwd = profile[idx]
            total += (fwd + bwd) * max(batch_size, 0)
        return total / max(capacity, 0.1)

    def _weights_mb(self, config: AsteroidConfig, start_l: int, end_l: int) -> float:
        span = max(1, end_l - start_l)
        num_layers = max(1, config.num_layers)
        params = config.embedding_dim * max(1, config.d_ff) * 2
        per_layer_mb = params * 4.0 / (1024.0 * 1024.0 * num_layers)
        return max(1e-6, per_layer_mb * span)

    def _activations_mb(self, config: AsteroidConfig, start_l: int, end_l: int) -> float:
        span = max(1, end_l - start_l)
        per_layer = config.max_seq_len * config.embedding_dim
        per_layer_mb = per_layer * 4.0 / (1024.0 * 1024.0)
        return max(1e-6, per_layer_mb * span)

    # ── Communication cost helpers ──────────────────────────────────────────

    def _allreduce_time(
        self,
        config: AsteroidConfig,
        topology: DeviceTopology,
        device_group: list[int],
        start_l: int,
        end_l: int,
    ) -> float:
        group_size = len(device_group)
        if group_size <= 1:
            return 0.0
        min_bw = float("inf")
        for src in device_group:
            for dst in device_group:
                if src == dst:
                    continue
                bw = self._lookup_link(topology.bandwidths, src, dst, 1000.0)
                min_bw = min(min_bw, max(bw, 1e-6))
        weights_mb = self._weights_mb(config, start_l, end_l)
        ring_factor = 2.0 * (group_size - 1) / group_size
        return ring_factor * weights_mb / max(min_bw, 1e-6) * 1000.0

    def _comm_time_inter_stage(
        self,
        config: AsteroidConfig,
        topology: DeviceTopology,
        boundary_layer: int,
        src_group: list[int],
        dst_group: list[int],
        batch_size: int,
    ) -> float:
        if not src_group or not dst_group:
            return 0.0
        act_mb = self._activations_mb(config, boundary_layer, boundary_layer + 1) * max(batch_size, 1)
        min_bw = float("inf")
        max_lat = 0.0
        for src in src_group:
            for dst in dst_group:
                bw = self._lookup_link(topology.bandwidths, src, dst, 1000.0)
                lat = self._lookup_link(topology.latencies, src, dst, 0.1)
                min_bw = min(min_bw, max(bw, 1e-6))
                max_lat = max(max_lat, lat)
        return 2.0 * act_mb / max(min_bw, 1e-6) * 1000.0 + max_lat

    @staticmethod
    def _lookup_link(table: dict, src: int, dst: int, default: float) -> float:
        val = table.get((src, dst))
        if val is not None:
            return float(val)
        rev = table.get((dst, src))
        if rev is not None:
            return float(rev)
        return default

    # ── Fallback plan ───────────────────────────────────────────────────────

    def _fallback_plan(
        self,
        config: AsteroidConfig,
        topology: DeviceTopology,
        exec_profiles: dict[int, list[tuple[float, float]]],
    ) -> ParallelismPlan:
        num_layers = max(1, config.num_layers)
        stages = max(1, min(self.num_stages, len(topology.device_specs), num_layers))
        ranges: list[tuple[int, int]] = []
        start = 0
        for si in range(stages):
            remaining = num_layers - start
            rem_stages = stages - si
            span = max(1, remaining // rem_stages)
            end = start + span if si < stages - 1 else num_layers
            ranges.append((start, end))
            start = end

        device_ids = [s.device_id for s in topology.device_specs]
        groups: dict[int, list[int]] = {}
        cursor = 0
        for si in range(stages):
            remaining = len(device_ids) - cursor
            rem_stages = stages - si
            width = max(1, remaining // rem_stages)
            end = cursor + width if si < stages - 1 else len(device_ids)
            groups[si] = device_ids[cursor:end]
            cursor = end

        allocs: dict[int, dict[int, int]] = {}
        est = 0.0
        for si, (s, e) in enumerate(ranges):
            alloc, stage_exec = self._alloc_microbatch(
                config, topology, exec_profiles, si, stages, groups[si], s, e,
            )
            allocs[si] = alloc
            est = max(est, stage_exec)
        partition_points = [rng[1] - 1 for rng in ranges[:-1]]
        return ParallelismPlan(
            partition_points=partition_points,
            device_groups=groups,
            micro_batch_alloc=allocs,
            schedule_type=self.get_schedule_type(),
            estimated_latency_ms=self.num_microbatches * est,
        )


__all__ = ["AsteroidStrategy"]
