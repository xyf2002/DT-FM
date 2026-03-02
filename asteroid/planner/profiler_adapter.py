"""Profiler adapter — converts AsteroidProfiler output to formats
expected by different schedulers (DPPartitioner, ConfidentScheduler, GCMA).

Each scheduler expects profiler data in a slightly different shape.
This adapter bridges the gap.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class ProfilerAdapter:
    """Adapt ``AsteroidProfiler`` data for use with various schedulers."""

    def __init__(self, profiler: Any) -> None:
        """
        Args:
            profiler: An ``AsteroidProfiler`` instance (or anything with
                      ``exec_times``, ``activation_sizes``, ``weight_sizes``,
                      ``bandwidths`` attributes).
        """
        self.profiler = profiler

    # ---- For DPPartitioner / ConfidentScheduler --------------------------

    def to_layer_times(self, device_id: int = 0,
                       batch_size: int | None = None) -> list[float]:
        """Return a flat list of (fwd+bwd) ms per layer for *device_id*.

        If *batch_size* is ``None``, uses the smallest available.
        """
        dev_data = self.profiler.exec_times.get(device_id, {})
        if not dev_data:
            return []

        n_layers = max(dev_data.keys()) + 1 if dev_data else 0
        result: list[float] = []
        for li in range(n_layers):
            bs_data = dev_data.get(li, {})
            if not bs_data:
                result.append(1.0)
                continue
            if batch_size is not None and batch_size in bs_data:
                fwd, bwd = bs_data[batch_size]
            else:
                # Use smallest profiled batch size
                bs_key = min(bs_data.keys())
                fwd, bwd = bs_data[bs_key]
            result.append(fwd + bwd)
        return result

    def to_dp_profiler_data(
        self,
        num_devices: int,
        batch_size: int | None = None,
    ) -> dict[str, object]:
        """Return a dict suitable for ``DPPartitioner(profiler_data=...)``.

        Includes ``time_intervals``, ``output_sizes``, ``bandwidths``,
        and ``computing_capacities``.
        """
        # Time intervals: (device, start, end, fwd/bwd) -> time
        intervals: dict[tuple[int, int, int, int], float] = {}
        for did in range(num_devices):
            dev_data = self.profiler.exec_times.get(did, {})
            for li, bs_data in dev_data.items():
                if batch_size is not None and batch_size in bs_data:
                    fwd, bwd = bs_data[batch_size]
                elif bs_data:
                    fwd, bwd = bs_data[min(bs_data.keys())]
                else:
                    fwd, bwd = 1.0, 2.0
                intervals[(did, li, li, 0)] = fwd
                intervals[(did, li, li, 1)] = bwd

        # Output sizes (activation per layer in MB)
        output_sizes = [
            s / (1024 * 1024) for s in self.profiler.activation_sizes
        ] if self.profiler.activation_sizes else []

        # Bandwidths per device (use average outbound)
        bw_list: list[float] = []
        for did in range(num_devices):
            outbound = [
                bw for (s, d), bw in self.profiler.bandwidths.items()
                if s == did
            ]
            bw_list.append(sum(outbound) / len(outbound) if outbound else 100.0)

        # Computing capacities
        caps: list[float] = []
        for did in range(num_devices):
            dev = self.profiler.devices.get(did)
            caps.append(dev.compute_capacity if dev else 1.0)

        return {
            "time_intervals": intervals,
            "output_sizes": output_sizes,
            "bandwidths": bw_list,
            "computing_capacities": caps,
        }

    # ---- For GCMA --------------------------------------------------------

    def to_gcma_data(
        self,
        num_devices: int,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Return data dict for the GCMA scheduler.

        Returns a dict with:
        - ``layer_times``: list of per-layer (fwd+bwd) ms
        - ``bandwidths``: device-pair bandwidth matrix (MBps)
        - ``memory_budgets``: per-device memory (MB)
        - ``computing_capacities``: per-device capacity
        """
        layer_times = self.to_layer_times(device_id=0, batch_size=batch_size)

        bw_matrix: dict[tuple[int, int], float] = {}
        for (s, d), bw in self.profiler.bandwidths.items():
            bw_matrix[(s, d)] = bw

        mem_budgets: list[float] = []
        caps: list[float] = []
        for did in range(num_devices):
            dev = self.profiler.devices.get(did)
            mem_budgets.append(dev.memory_budget_mb if dev else 4096.0)
            caps.append(dev.compute_capacity if dev else 1.0)

        return {
            "layer_times": layer_times,
            "bandwidths": bw_matrix,
            "memory_budgets": mem_budgets,
            "computing_capacities": caps,
        }


__all__ = ["ProfilerAdapter"]
