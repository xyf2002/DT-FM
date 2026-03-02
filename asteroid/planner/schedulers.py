"""DP-based layer partitioner shared by DT-FM and Confident.

Given per-layer execution times and inter-device bandwidths, finds the
optimal partition of N layers across K pipeline stages to minimize
bottleneck stage time.

This is a standalone, config-agnostic partitioner.  The ``AsteroidPlanner``
in ``asteroid.planner.dp_planner`` wraps this with Asteroid-specific
memory constraints and micro-batch allocation.

Ported from DeviceEmulator/baselines/schedulers/dp_partitioner.py.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

logger = logging.getLogger(__name__)


class DPPartitioner:
    """DP partition of *num_layers* across *num_devices* stages."""

    def __init__(
        self,
        num_layers: int,
        num_devices: int,
        profiler_data: dict[str, object] | list[float] | None = None,
    ) -> None:
        self.num_layers = int(max(0, num_layers))
        self.num_devices = int(max(0, num_devices))
        self.profiler_data = profiler_data

        self.time_intervals: dict[tuple[int, int, int, int], float] = {}
        self.output_sizes: list[float] = []
        self.bandwidths: list[float] = []
        self.computing_capacities: list[float] = []
        self.layer_times: list[float] | None = None

        if isinstance(profiler_data, Sequence) and not isinstance(profiler_data, dict):
            self.layer_times = [float(v) for v in profiler_data]
        elif isinstance(profiler_data, dict):
            intervals_obj = profiler_data.get("time_intervals", {})
            if isinstance(intervals_obj, dict):
                parsed: dict[tuple[int, int, int, int], float] = {}
                for raw_key, raw_value in intervals_obj.items():
                    if not isinstance(raw_key, tuple) or len(raw_key) != 4:
                        continue
                    try:
                        parsed[tuple(int(k) for k in raw_key)] = float(raw_value)  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        continue
                self.time_intervals = parsed

            self.output_sizes = _to_float_list(profiler_data.get("output_sizes", []))
            self.bandwidths = _to_float_list(profiler_data.get("bandwidths", []))
            self.computing_capacities = _to_float_list(
                profiler_data.get("computing_capacities", []))

            lt = profiler_data.get("layer_times")
            if lt is not None:
                self.layer_times = _to_float_list(lt)

        if self.num_layers == 0 and self.layer_times:
            self.num_layers = len(self.layer_times)

    # ---- internal --------------------------------------------------------

    def _effective_sizes(self) -> tuple[int, int]:
        n = self.num_layers or (len(self.layer_times) if self.layer_times else 0)
        return (n, max(1, min(self.num_devices, n))) if n > 0 else (0, 0)

    def _capacity(self, device_id: int, is_average: bool) -> float:
        if not is_average:
            return 1.0
        if 0 <= device_id < len(self.computing_capacities):
            c = self.computing_capacities[device_id]
            if c > 0:
                return c
        return 1.0

    def _get_time(self, device_id: int, start: int, end: int) -> float:
        if end < start:
            return 0.0
        if self.time_intervals:
            fwd = self.time_intervals.get((device_id, start, end, 0))
            bwd = self.time_intervals.get((device_id, start, end, 1))
            if fwd is not None and bwd is not None:
                return fwd + bwd
            total, found = 0.0, False
            for li in range(start, end + 1):
                lf = self.time_intervals.get((device_id, li, li, 0))
                lb = self.time_intervals.get((device_id, li, li, 1))
                if lf is None and lb is None:
                    continue
                total += (lf or 0.0) + (lb or 0.0)
                found = True
            if found:
                return total
        if self.layer_times:
            lo, hi = max(0, start), min(end, len(self.layer_times) - 1)
            if lo <= hi:
                return sum(self.layer_times[lo:hi + 1])
        return float(max(0, end - start + 1))

    def _get_comm_time(self, layer_idx: int, device_id: int) -> float:
        if layer_idx < 0 or layer_idx >= len(self.output_sizes):
            return 0.0
        if device_id < 0 or device_id >= len(self.bandwidths):
            return 0.0
        bw = self.bandwidths[device_id]
        return self.output_sizes[layer_idx] / bw * 1000.0 if bw > 0 else 0.0

    def _equal_partition_points(self, n: int, k: int) -> list[int]:
        if k <= 1:
            return []
        pts, prev = [], -1
        for s in range(1, k):
            end = (s * n) // k - 1
            end = max(prev + 1, min(end, n - (k - s) - 1))
            pts.append(end)
            prev = end
        return pts

    # ---- public API -------------------------------------------------------

    def partition(self, is_average: bool = True) -> list[int]:
        """Return split points (layer indices) for the optimal partition."""
        n, k = self._effective_sizes()
        if n <= 0 or k <= 1:
            return []

        INF = float("inf")
        dp = [[INF] * k for _ in range(n)]
        split = [[-1] * k for _ in range(n)]

        cap0 = self._capacity(0, is_average)
        for i in range(n):
            dp[i][0] = self._get_time(0, 0, i) / cap0

        for j in range(1, k):
            cj = self._capacity(j, is_average)
            for i in range(j, n):
                for m in range(j - 1, i):
                    cost = max(dp[m][j - 1],
                               self._get_time(j, m + 1, i) / cj
                               + self._get_comm_time(m, j - 1))
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        split[i][j] = m

        pts: list[int] = []
        i, j = n - 1, k - 1
        while j > 0:
            m = split[i][j]
            if m < 0:
                logger.warning("DP backtrack failed; falling back to equal partition")
                return self._equal_partition_points(n, k)
            pts.append(m)
            i, j = m, j - 1
        pts.reverse()
        return pts

    def partition_with_memory(
        self,
        memory_budgets: list[float],
        is_average: bool = True,
    ) -> list[int]:
        """Memory-constrained DP partition."""
        n, k = self._effective_sizes()
        if n <= 0 or k <= 1:
            return []
        if not memory_budgets:
            return self.partition(is_average=is_average)
        k = min(k, len(memory_budgets))

        INF = float("inf")
        dp = [[INF] * k for _ in range(n)]
        split = [[-1] * k for _ in range(n)]

        cap0 = self._capacity(0, is_average)
        for i in range(n):
            seg_mem = sum(self.output_sizes[max(0, l):min(l + 1, len(self.output_sizes))]
                         for l in range(0, i + 1)) if self.output_sizes else 0.0
            if seg_mem <= memory_budgets[0]:
                dp[i][0] = self._get_time(0, 0, i) / cap0

        for j in range(1, k):
            cj = self._capacity(j, is_average)
            for i in range(j, n):
                for m in range(j - 1, i):
                    seg_mem = sum(
                        self.output_sizes[l]
                        for l in range(m + 1, i + 1)
                        if l < len(self.output_sizes)
                    ) if self.output_sizes else 0.0
                    if seg_mem > memory_budgets[j]:
                        continue
                    cost = max(dp[m][j - 1],
                               self._get_time(j, m + 1, i) / cj
                               + self._get_comm_time(m, j - 1))
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        split[i][j] = m

        if dp[n - 1][k - 1] == INF:
            logger.warning("No feasible memory partition; falling back to plain DP")
            return self.partition(is_average=is_average)

        pts: list[int] = []
        i, j = n - 1, k - 1
        while j > 0:
            m = split[i][j]
            if m < 0:
                return self.partition(is_average=is_average)
            pts.append(m)
            i, j = m, j - 1
        pts.reverse()
        return pts


# ---------------------------------------------------------------------------
# Confident scheduler (DPPartitioner + failure re-planning)
# ---------------------------------------------------------------------------

class ConfidentScheduler:
    """Confident-style scheduler: DP partition + failure-aware replanning."""

    def __init__(
        self,
        num_layers: int,
        num_devices: int,
        profiler_data: dict[str, object] | list[float] | None = None,
    ) -> None:
        self.num_layers = int(max(0, num_layers))
        self.num_devices = int(max(0, num_devices))
        self.profiler_data = profiler_data
        self._partitioner = DPPartitioner(
            num_layers=self.num_layers,
            num_devices=self.num_devices,
            profiler_data=self.profiler_data,
        )

    def partition(self, is_average: bool = True) -> list[int]:
        return self._partitioner.partition(is_average=is_average)

    def replan_after_failure(
        self,
        failed_devices: list[int],
        current_partition: list[int],
    ) -> list[int]:
        """Re-partition layers among surviving devices."""
        failed = set(int(d) for d in failed_devices)
        if not failed:
            return list(current_partition)

        survivors = [d for d in range(self.num_devices) if d not in failed]
        if not survivors:
            raise ValueError("No surviving devices after failure")

        logger.warning(
            "Replanning: failed=%s survivors=%s", sorted(failed), survivors,
        )

        remapped = self._remap_profiler_data(survivors)
        replanner = DPPartitioner(
            num_layers=self.num_layers,
            num_devices=len(survivors),
            profiler_data=remapped,
        )
        return replanner.partition(is_average=True)

    def _remap_profiler_data(
        self, survivors: list[int],
    ) -> dict[str, object] | list[float] | None:
        if isinstance(self.profiler_data, list):
            return [float(v) for v in self.profiler_data]
        if not isinstance(self.profiler_data, dict):
            return self.profiler_data

        mapping = {old: new for new, old in enumerate(survivors)}
        remapped: dict[str, object] = {}

        intervals = self.profiler_data.get("time_intervals", {})
        if isinstance(intervals, dict):
            new_int: dict[tuple[int, int, int, int], float] = {}
            for key, val in intervals.items():
                if not isinstance(key, tuple) or len(key) != 4:
                    continue
                did = int(key[0])
                if did not in mapping:
                    continue
                new_int[(mapping[did], int(key[1]), int(key[2]), int(key[3]))] = float(val)
            remapped["time_intervals"] = new_int

        for k in ("output_sizes", "layer_times"):
            if k in self.profiler_data:
                remapped[k] = _to_float_list(self.profiler_data.get(k, []))

        bw = self.profiler_data.get("bandwidths", [])
        if isinstance(bw, list):
            remapped["bandwidths"] = [
                float(bw[d]) for d in survivors if 0 <= d < len(bw)
            ]
        cap = self.profiler_data.get("computing_capacities", [])
        if isinstance(cap, list):
            remapped["computing_capacities"] = [
                float(cap[d]) for d in survivors if 0 <= d < len(cap)
            ]
        return remapped


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float_list(value: object) -> list[float]:
    if isinstance(value, (dict, str, bytes, bytearray)):
        return []
    if not isinstance(value, Sequence):
        return []
    parsed: list[float] = []
    for item in value:
        try:
            parsed.append(float(item))
        except (TypeError, ValueError):
            continue
    return parsed


__all__ = ["DPPartitioner", "ConfidentScheduler"]
