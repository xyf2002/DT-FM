from __future__ import annotations

import logging

from .dp_partitioner import DPPartitioner

_LOGGER = logging.getLogger(__name__)


class ConfidentScheduler:
    """Confident DP-based bottleneck minimization scheduler.

    Similar to DPPartitioner but with passive FT awareness.
    After failure, re-partitions among surviving devices.
    """

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

    def _remap_profiler_data(
        self,
        survivors: list[int],
    ) -> dict[str, object] | list[float] | None:
        if isinstance(self.profiler_data, list):
            return [float(v) for v in self.profiler_data]
        if not isinstance(self.profiler_data, dict):
            return self.profiler_data

        mapping = {old: new for new, old in enumerate(survivors)}
        remapped: dict[str, object] = {}

        intervals = self.profiler_data.get("time_intervals", {})
        if isinstance(intervals, dict):
            new_intervals: dict[tuple[int, int, int, int], float] = {}
            for key, value in intervals.items():
                if not isinstance(key, tuple) or len(key) != 4:
                    continue
                did = int(key[0])
                if did not in mapping:
                    continue
                new_key = (
                    mapping[did],
                    int(key[1]),
                    int(key[2]),
                    int(key[3]),
                )
                new_intervals[new_key] = float(value)
            remapped["time_intervals"] = new_intervals

        if "output_sizes" in self.profiler_data:
            remapped["output_sizes"] = self._to_float_list(
                self.profiler_data.get("output_sizes", [])
            )
        if "layer_times" in self.profiler_data:
            remapped["layer_times"] = self._to_float_list(
                self.profiler_data.get("layer_times", [])
            )

        bandwidths = self.profiler_data.get("bandwidths", [])
        if isinstance(bandwidths, list):
            remapped["bandwidths"] = [
                float(bandwidths[did])
                for did in survivors
                if 0 <= did < len(bandwidths)
            ]

        capacities = self.profiler_data.get("computing_capacities", [])
        if isinstance(capacities, list):
            remapped["computing_capacities"] = [
                float(capacities[did])
                for did in survivors
                if 0 <= did < len(capacities)
            ]

        return remapped

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
            try:
                parsed.append(float(item))
            except (TypeError, ValueError):
                continue
        return parsed

    def replan_after_failure(
        self,
        failed_devices: list[int],
        current_partition: list[int],
    ) -> list[int]:
        failed = {int(device) for device in failed_devices}
        if not failed:
            return list(current_partition)

        survivors = [
            device_id
            for device_id in range(self.num_devices)
            if device_id not in failed
        ]
        if not survivors:
            raise ValueError("No surviving devices after failure")

        _LOGGER.warning(
            "Replanning after failures. failed=%s survivors=%s",
            sorted(failed),
            survivors,
        )

        remapped_profiler = self._remap_profiler_data(survivors)
        replanner = DPPartitioner(
            num_layers=self.num_layers,
            num_devices=len(survivors),
            profiler_data=remapped_profiler,
        )
        return replanner.partition(is_average=True)


__all__ = ["ConfidentScheduler"]
