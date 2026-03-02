from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _DeviceSpec:
    device_id: int
    memory_budget_mb: float = 80_000.0
    compute_capacity: float = 1.0


class AsteroidHPPPlanner:
    """Dynamic Programming HPP Planner (Asteroid Section 3.3).

    Finds optimal: model partition, device grouping, micro-batch allocation
    to minimize HPP-Round Latency under memory constraints.
    """

    def __init__(
        self,
        num_layers: int,
        num_devices: int,
        num_microbatches: int,
        micro_batch_size: int,
        profiler_data: dict[str, object] | None = None,
        device_specs: list[dict[str, float | int]] | None = None,
    ) -> None:
        self.num_layers = int(max(0, num_layers))
        self.num_devices = int(max(0, num_devices))
        self.num_microbatches = int(max(1, num_microbatches))
        self.micro_batch_size = int(max(1, micro_batch_size))

        data = profiler_data or {}
        self.exec_times = self._parse_exec_times(data.get("exec_times", {}))
        self.activation_sizes = self._to_float_list(
            data.get("activation_sizes", [])
        )
        self.weight_sizes = self._to_float_list(data.get("weight_sizes", []))
        self.bandwidths = self._parse_bandwidths(data.get("bandwidths", {}))
        self.default_bandwidth_mbps = self._infer_default_bandwidth()

        self.devices = self._prepare_devices(device_specs)
        self.device_by_id = {
            device.device_id: device for device in self.devices
        }

    def _prepare_devices(
        self,
        specs: list[dict[str, float | int]] | None,
    ) -> list[_DeviceSpec]:
        devices: list[_DeviceSpec] = []
        if specs:
            for idx, spec in enumerate(specs):
                device_id = int(spec.get("device_id", idx))
                memory_budget_mb = float(spec.get("memory_budget_mb", 80_000.0))
                compute_capacity = float(spec.get("compute_capacity", 1.0))
                devices.append(
                    _DeviceSpec(
                        device_id=device_id,
                        memory_budget_mb=max(0.0, memory_budget_mb),
                        compute_capacity=max(1e-6, compute_capacity),
                    )
                )

        existing_ids = {device.device_id for device in devices}
        for device_id in range(self.num_devices):
            if device_id in existing_ids:
                continue
            devices.append(_DeviceSpec(device_id=device_id))

        devices.sort(key=lambda item: item.memory_budget_mb, reverse=True)
        return devices[: self.num_devices]

    def _infer_default_bandwidth(self) -> float:
        values = [bw for bw in self.bandwidths.values() if bw > 0]
        if not values:
            return 12_000.0
        return float(min(values))

    def _device_or_default(self, device_id: int) -> _DeviceSpec:
        return self.device_by_id.get(
            device_id,
            _DeviceSpec(device_id=device_id),
        )

    @staticmethod
    def _to_float_list(value: object) -> list[float]:
        if isinstance(value, dict):
            return []
        if isinstance(value, (str, bytes, bytearray)):
            return []
        if not isinstance(value, list):
            return []

        parsed: list[float] = []
        for item in value:
            if not isinstance(item, (int, float, str)):
                continue
            try:
                parsed.append(float(item))
            except (TypeError, ValueError):
                continue
        return parsed

    @staticmethod
    def _parse_bandwidths(value: object) -> dict[tuple[int, int], float]:
        parsed: dict[tuple[int, int], float] = {}
        if not isinstance(value, dict):
            return parsed

        for key, raw_bw in value.items():
            if not isinstance(key, tuple) or len(key) != 2:
                continue
            try:
                src = int(key[0])
                dst = int(key[1])
                parsed[(src, dst)] = float(raw_bw)
            except (TypeError, ValueError):
                continue
        return parsed

    @staticmethod
    def _parse_exec_times(
        value: object,
    ) -> dict[int, dict[int, dict[int, tuple[float, float]]]]:
        parsed: dict[int, dict[int, dict[int, tuple[float, float]]]] = {}
        if not isinstance(value, dict):
            return parsed

        for raw_dev, raw_layers in value.items():
            if not isinstance(raw_layers, dict):
                continue
            try:
                dev_id = int(raw_dev)
            except (TypeError, ValueError):
                continue

            layer_map: dict[int, dict[int, tuple[float, float]]] = {}
            for raw_layer, raw_bs_map in raw_layers.items():
                if not isinstance(raw_bs_map, dict):
                    continue
                try:
                    layer_idx = int(raw_layer)
                except (TypeError, ValueError):
                    continue

                bs_map: dict[int, tuple[float, float]] = {}
                for raw_bs, raw_pair in raw_bs_map.items():
                    if not isinstance(raw_pair, (list, tuple)):
                        continue
                    if len(raw_pair) != 2:
                        continue
                    try:
                        batch_size = int(raw_bs)
                        fwd_ms = float(raw_pair[0])
                        bwd_ms = float(raw_pair[1])
                    except (TypeError, ValueError):
                        continue
                    bs_map[batch_size] = (fwd_ms, bwd_ms)

                if bs_map:
                    layer_map[layer_idx] = bs_map

            if layer_map:
                parsed[dev_id] = layer_map

        return parsed

    def _get_exec_time(
        self,
        device_id: int,
        layer_idx: int,
        batch_size: int,
    ) -> tuple[float, float]:
        times = self.exec_times.get(device_id, {}).get(layer_idx, {})
        if not times:
            return 1.0, 2.0

        normalized = {int(k): v for k, v in times.items()}
        if batch_size in normalized:
            fwd_ms, bwd_ms = normalized[batch_size]
            return float(fwd_ms), float(bwd_ms)

        keys = sorted(normalized.keys())
        if batch_size <= keys[0]:
            fwd_ms, bwd_ms = normalized[keys[0]]
            return float(fwd_ms), float(bwd_ms)

        if batch_size >= keys[-1]:
            ref_bs = keys[-1]
            ref_fwd, ref_bwd = normalized[ref_bs]
            ratio = (batch_size / ref_bs) ** 0.85
            return float(ref_fwd) * ratio, float(ref_bwd) * ratio

        for idx in range(len(keys) - 1):
            lo = keys[idx]
            hi = keys[idx + 1]
            if lo <= batch_size <= hi:
                alpha = (batch_size - lo) / (hi - lo)
                lo_fwd, lo_bwd = normalized[lo]
                hi_fwd, hi_bwd = normalized[hi]
                fwd_ms = float(lo_fwd) * (1 - alpha) + float(hi_fwd) * alpha
                bwd_ms = float(lo_bwd) * (1 - alpha) + float(hi_bwd) * alpha
                return fwd_ms, bwd_ms

        return 1.0, 2.0

    def _memory_footprint(
        self,
        stage_idx: int,
        num_stages: int,
        start_l: int,
        end_l: int,
        batch_size: int,
    ) -> float:
        k_p = max(1, 2 * (num_stages - stage_idx) - 1)

        weight_bytes = sum(
            self.weight_sizes[layer_idx]
            for layer_idx in range(start_l, end_l)
            if layer_idx < len(self.weight_sizes)
        )
        mem_mod = weight_bytes
        mem_opt = weight_bytes * 2

        mem_act = (
            sum(
                self.activation_sizes[layer_idx]
                for layer_idx in range(start_l, end_l)
                if layer_idx < len(self.activation_sizes)
            )
            * batch_size
        )
        total_bytes = mem_mod + mem_opt + k_p * mem_act
        return total_bytes / (1024 * 1024)

    def _alloc_microbatch(
        self,
        device_ids: list[int],
        start_l: int,
        end_l: int,
        micro_bs: int,
    ) -> tuple[dict[int, int], float]:
        if not device_ids:
            return {}, float("inf")

        devices_here = [self._device_or_default(did) for did in device_ids]
        alloc = {device.device_id: 0 for device in devices_here}
        remaining = micro_bs
        active = list(devices_here)

        while remaining > 0 and active:
            total_cap = sum(device.compute_capacity for device in active)
            if total_cap <= 0:
                break

            new_active: list[_DeviceSpec] = []
            for device in active:
                share = max(
                    1,
                    int(round(device.compute_capacity / total_cap * remaining)),
                )
                mem_needed = self._memory_footprint(
                    0,
                    1,
                    start_l,
                    end_l,
                    share,
                )
                max_bs = share
                while mem_needed > device.memory_budget_mb and max_bs > 1:
                    max_bs -= 1
                    mem_needed = self._memory_footprint(
                        0,
                        1,
                        start_l,
                        end_l,
                        max_bs,
                    )

                actual = min(share, max_bs, remaining)
                alloc[device.device_id] += actual
                remaining -= actual

                if mem_needed < device.memory_budget_mb * 0.95:
                    new_active.append(device)

            active = new_active

        if remaining > 0:
            ordered = sorted(
                devices_here,
                key=lambda dev: dev.compute_capacity,
                reverse=True,
            )
            progress = True
            while remaining > 0 and progress:
                progress = False
                for device in ordered:
                    candidate = alloc[device.device_id] + 1
                    mem_needed = self._memory_footprint(
                        0,
                        1,
                        start_l,
                        end_l,
                        candidate,
                    )
                    if mem_needed <= device.memory_budget_mb:
                        alloc[device.device_id] = candidate
                        remaining -= 1
                        progress = True
                        if remaining == 0:
                            break

        def exec_time(device_id: int, bs: int) -> float:
            if bs <= 0:
                return 0.0

            total = 0.0
            for layer_idx in range(start_l, end_l):
                fwd_ms, bwd_ms = self._get_exec_time(device_id, layer_idx, bs)
                total += fwd_ms + bwd_ms
            return total

        for _ in range(5):
            times: dict[int, float] = {
                device_id: exec_time(device_id, bs)
                for device_id, bs in alloc.items()
                if bs > 0
            }
            if not times:
                break

            slowest = max(times, key=lambda device_id: times[device_id])
            fastest = min(times, key=lambda device_id: times[device_id])
            if slowest == fastest or alloc[slowest] <= 1:
                break

            old_time = times[slowest]
            alloc[slowest] -= 1
            alloc[fastest] += 1
            new_time = max(
                exec_time(slowest, alloc[slowest]),
                exec_time(fastest, alloc[fastest]),
            )
            if new_time >= old_time:
                alloc[slowest] += 1
                alloc[fastest] -= 1
                break

        if any(bs > 0 for bs in alloc.values()):
            straggler_time = max(
                exec_time(did, bs) for did, bs in alloc.items()
            )
        else:
            straggler_time = float("inf")

        return alloc, straggler_time

    def _comm_time_inter_stage(
        self,
        layer_idx: int,
        src_group: list[int],
        dst_group: list[int],
        batch_size: int,
    ) -> float:
        if not src_group or not dst_group or not self.activation_sizes:
            return 0.0

        bs = batch_size if batch_size > 0 else self.micro_batch_size
        idx = min(max(layer_idx, 0), len(self.activation_sizes) - 1)
        act_size = self.activation_sizes[idx] * bs
        act_size_mb = act_size / (1024 * 1024)

        min_bw = float("inf")
        for src in src_group:
            for dst in dst_group:
                bw = self.bandwidths.get(
                    (src, dst),
                    self.default_bandwidth_mbps,
                )
                min_bw = min(min_bw, bw)

        if min_bw <= 0 or min_bw == float("inf"):
            return float("inf")

        return 2 * act_size_mb / min_bw * 1000

    def _allreduce_time(
        self,
        device_group: list[int],
        start_l: int,
        end_l: int,
    ) -> float:
        g_size = len(device_group)
        if g_size <= 1:
            return 0.0

        weight_bytes = sum(
            self.weight_sizes[layer_idx]
            for layer_idx in range(start_l, end_l)
            if layer_idx < len(self.weight_sizes)
        )

        min_bw = float("inf")
        for d1 in device_group:
            for d2 in device_group:
                if d1 == d2:
                    continue
                bw = self.bandwidths.get((d1, d2), self.default_bandwidth_mbps)
                min_bw = min(min_bw, bw)

        if min_bw <= 0 or min_bw == float("inf"):
            return float("inf")

        vol_mb = 2 * (g_size - 1) / g_size * weight_bytes / (1024 * 1024)
        return vol_mb / min_bw * 1000

    def plan(self) -> dict[str, object]:
        l_total = self.num_layers
        n_total = min(self.num_devices, len(self.devices))
        if l_total <= 0 or n_total <= 0:
            return self._fallback_plan([])

        device_ids = [device.device_id for device in self.devices[:n_total]]
        best_plan: dict[str, object] | None = None
        best_latency = float("inf")

        max_stages = min(l_total, n_total)
        for p in range(1, max_stages + 1):
            result = self._dp_plan(l_total, n_total, p, device_ids)
            if result is None:
                continue

            latency_obj = result.get("latency")
            if not isinstance(latency_obj, (int, float)):
                continue
            latency = float(latency_obj)
            if latency < best_latency:
                best_latency = latency
                best_plan = result

        if best_plan is None:
            best_plan = self._fallback_plan(device_ids)

        _LOGGER.info(
            "Asteroid planner: stages=%s partition=%s latency=%.3fms",
            best_plan["num_stages"],
            best_plan["partition_points"],
            best_plan["latency"],
        )
        return best_plan

    def _normalize_partition_points(self, points: list[int]) -> list[int]:
        normalized = sorted(set(int(point) for point in points))
        return [point for point in normalized if 0 < point < self.num_layers]

    def _dp_plan(
        self,
        L: int,
        N: int,
        P: int,
        device_ids: list[int],
    ) -> dict[str, object] | None:
        if P > L or P > N:
            return None

        inf = float("inf")
        q = [
            [[inf for _ in range(P + 1)] for _ in range(N + 1)]
            for _ in range(L + 1)
        ]
        configs: list[list[list[dict[str, object] | None]]] = [
            [[None for _ in range(P + 1)] for _ in range(N + 1)]
            for _ in range(L + 1)
        ]

        for l in range(1, L + 1):
            for n in range(1, N + 1):
                group = device_ids[N - n :]
                alloc, exec_t = self._alloc_microbatch(
                    group,
                    L - l,
                    L,
                    self.micro_batch_size,
                )
                ar_t = self._allreduce_time(group, L - l, L)
                lat = self.num_microbatches * exec_t + ar_t
                if lat < q[l][n][1]:
                    q[l][n][1] = lat
                    configs[l][n][1] = {
                        "partition": [L - l],
                        "groups": {0: group},
                        "allocs": {0: alloc},
                    }

        for p in range(2, P + 1):
            for l in range(p, L + 1):
                for n in range(p, N + 1):
                    for l_prime in range(p - 1, l):
                        for n_prime in range(p - 1, n):
                            if q[l_prime][n_prime][p - 1] >= inf:
                                continue

                            prev_cfg = configs[l_prime][n_prime][p - 1]
                            if prev_cfg is None:
                                continue

                            new_group = device_ids[N - n : N - n_prime]
                            if not new_group:
                                continue

                            start_l = L - l
                            end_l = L - l_prime
                            alloc, exec_t = self._alloc_microbatch(
                                new_group,
                                start_l,
                                end_l,
                                self.micro_batch_size,
                            )

                            global_stage_idx = P - p
                            mem_ok = True
                            for did, bs in alloc.items():
                                if bs <= 0:
                                    continue
                                mem = self._memory_footprint(
                                    global_stage_idx,
                                    P,
                                    start_l,
                                    end_l,
                                    bs,
                                )
                                device = self._device_or_default(did)
                                if mem > device.memory_budget_mb:
                                    mem_ok = False
                                    break
                            if not mem_ok:
                                continue

                            groups_obj = prev_cfg.get("groups", {})
                            if not isinstance(groups_obj, dict):
                                continue

                            prev_last_group_obj = groups_obj.get(p - 2, [])
                            if isinstance(prev_last_group_obj, list):
                                prev_last_group = [
                                    int(device_id)
                                    for device_id in prev_last_group_obj
                                ]
                            else:
                                prev_last_group = []

                            total_alloc_bs = sum(alloc.values())
                            if prev_last_group:
                                comm_t = self._comm_time_inter_stage(
                                    end_l - 1,
                                    new_group,
                                    prev_last_group,
                                    batch_size=total_alloc_bs,
                                )
                            else:
                                comm_t = 0.0

                            ar_t = self._allreduce_time(
                                new_group,
                                start_l,
                                end_l,
                            )
                            sub_lat = q[l_prime][n_prime][p - 1]
                            new_step_lat = self.num_microbatches * exec_t
                            total_lat = max(
                                sub_lat,
                                new_step_lat + comm_t,
                            ) + ar_t

                            if total_lat < q[l][n][p]:
                                prev_groups = copy.deepcopy(groups_obj)
                                allocs_obj = prev_cfg.get("allocs", {})
                                if not isinstance(allocs_obj, dict):
                                    continue
                                prev_allocs = copy.deepcopy(allocs_obj)
                                groups_new = {0: list(new_group)}
                                allocs_new = {0: dict(alloc)}
                                for stage_idx, group in prev_groups.items():
                                    if not isinstance(group, list):
                                        continue
                                    groups_new[int(stage_idx) + 1] = [
                                        int(device_id) for device_id in group
                                    ]
                                for stage_idx, stage_alloc in (
                                    prev_allocs.items()
                                ):
                                    if not isinstance(stage_alloc, dict):
                                        continue
                                    allocs_new[int(stage_idx) + 1] = dict(
                                        stage_alloc
                                    )

                                partition_obj = prev_cfg.get("partition", [])
                                if isinstance(partition_obj, list):
                                    partition_new = [start_l] + [
                                        int(point) for point in partition_obj
                                    ]
                                else:
                                    partition_new = [start_l]

                                q[l][n][p] = total_lat
                                configs[l][n][p] = {
                                    "partition": partition_new,
                                    "groups": groups_new,
                                    "allocs": allocs_new,
                                }

        best_lat = q[L][N][P]
        best_cfg = configs[L][N][P]
        if best_lat >= inf or best_cfg is None:
            return None

        groups = best_cfg.get("groups", {})
        if not isinstance(groups, dict):
            groups = {}

        allocs = best_cfg.get("allocs", {})
        if not isinstance(allocs, dict):
            allocs = {}

        partition_obj = best_cfg.get("partition", [])
        if isinstance(partition_obj, list):
            partition_raw = [int(point) for point in partition_obj]
        else:
            partition_raw = []

        partition_points = self._normalize_partition_points(partition_raw)
        return {
            "num_stages": P,
            "partition_points": partition_points,
            "device_groups": groups,
            "micro_batch_alloc": allocs,
            "latency": float(best_lat),
        }

    def _fallback_plan(self, device_ids: list[int]) -> dict[str, object]:
        if not device_ids or self.num_layers <= 0:
            return {
                "num_stages": 0,
                "partition_points": [],
                "device_groups": {},
                "micro_batch_alloc": {},
                "latency": float("inf"),
            }

        p = min(self.num_layers, len(device_ids))
        layers_per = max(1, self.num_layers // p)
        partition_points = [idx * layers_per for idx in range(1, p)]
        partition_points = [
            point for point in partition_points if point < self.num_layers
        ]

        groups: dict[int, list[int]] = {}
        devices_per = max(1, len(device_ids) // p)
        for stage_idx in range(p):
            start_d = stage_idx * devices_per
            end_d = start_d + devices_per
            if stage_idx == p - 1:
                end_d = len(device_ids)
            groups[stage_idx] = device_ids[start_d:end_d]

        allocs: dict[int, dict[int, int]] = {}
        for stage_idx, group in groups.items():
            if not group:
                allocs[stage_idx] = {}
                continue
            base = self.micro_batch_size // len(group)
            rem = self.micro_batch_size % len(group)
            allocs[stage_idx] = {
                did: base + (1 if idx < rem else 0)
                for idx, did in enumerate(group)
            }

        return {
            "num_stages": p,
            "partition_points": partition_points,
            "device_groups": groups,
            "micro_batch_alloc": allocs,
            "latency": float("inf"),
        }


__all__ = ["AsteroidHPPPlanner"]
