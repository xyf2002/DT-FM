from __future__ import annotations

from abc import ABC, abstractmethod


class SchedulerBackend(ABC):
    """Abstract scheduler that computes PP partition points."""

    @abstractmethod
    def calculate_partition_point(self, is_average: bool) -> list[int]: ...

    @abstractmethod
    def calculate_partition_point_memory(
        self,
        is_average: bool,
    ) -> list[int]: ...


__all__ = ["SchedulerBackend"]
