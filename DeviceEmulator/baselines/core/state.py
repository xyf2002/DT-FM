from __future__ import annotations

from abc import ABC, abstractmethod


class StateManager(ABC):
    """Abstract runtime state holder for PP/DP ranks, device, and comms."""

    @abstractmethod
    def get_pipeline_parallel_rank(self) -> int: ...

    @abstractmethod
    def set_pipeline_parallel_rank(self, rank: int) -> None: ...

    @abstractmethod
    def get_pipeline_parallel_world_size(self) -> int: ...

    @abstractmethod
    def set_pipeline_parallel_world_size(self, size: int) -> None: ...

    @abstractmethod
    def get_data_parallel_rank(self) -> int: ...

    @abstractmethod
    def set_data_parallel_rank(self, rank: int) -> None: ...

    @abstractmethod
    def get_data_parallel_world_size(self) -> int: ...

    @abstractmethod
    def set_data_parallel_world_size(self, size: int) -> None: ...

    @abstractmethod
    def get_device(self) -> object: ...

    @abstractmethod
    def set_device(self, device: object) -> None: ...

    @abstractmethod
    def get_global_rank(self) -> int: ...

    @abstractmethod
    def set_global_rank(self, rank: int) -> None: ...

    @abstractmethod
    def get_pipeline_comm(self) -> object: ...

    @abstractmethod
    def set_pipeline_comm(self, comm: object) -> None: ...

    @abstractmethod
    def get_data_parallel_comm(self) -> object: ...

    @abstractmethod
    def set_data_parallel_comm(self, comm: object) -> None: ...

    @abstractmethod
    def get_current_epoch(self) -> int: ...

    @abstractmethod
    def set_current_epoch(self, epoch: int) -> None: ...

    @abstractmethod
    def get_current_iter(self) -> int: ...

    @abstractmethod
    def set_current_iter(self, it: int) -> None: ...


__all__ = ["StateManager"]
