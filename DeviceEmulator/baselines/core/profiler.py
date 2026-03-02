from __future__ import annotations

from abc import ABC, abstractmethod


class ProfilerBackend(ABC):
    """Abstract profiler interface consumed by schedulers and planners."""

    @abstractmethod
    def profile_layer(
        self,
        model: object,
        input_data: object,
        num_iterations: int,
    ) -> tuple[float, float, float]: ...

    @abstractmethod
    def profile_bandwidth(
        self,
        src_device: object,
        dst_device: object,
        data_size_mb: float,
    ) -> float: ...

    @abstractmethod
    def get_memory_info(self, device: object) -> tuple[float, float]: ...

    @abstractmethod
    def get_time_interval(
        self,
        device_id: int,
        start: int,
        end: int,
        phase: int,
    ) -> float: ...

    @abstractmethod
    def get_output_size(self, layer_idx: int) -> float: ...

    @abstractmethod
    def get_bandwidth(self, device_id: int) -> float: ...

    @abstractmethod
    def get_computing_capacity(self, device_id: int) -> float: ...

    @abstractmethod
    def get_available_memory(self, device_id: int) -> float: ...


__all__ = ["ProfilerBackend"]
