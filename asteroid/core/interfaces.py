"""Abstract base classes for the Asteroid distributed-training system.

Ported from DeviceEmulator/baselines/core/ — these ABCs define the
extension points that schedulers, communication backends, fault-tolerance
modules, and strategy implementations program against.

All classes are pure interfaces (no concrete state). Concrete
implementations live in their respective subpackages:
  * ``asteroid.comm``   — CommunicationBackend implementations
  * ``asteroid.ft``     — FaultToleranceBackend / CheckpointStrategy
  * ``asteroid.planner``— SchedulerBackend implementations
  * ``asteroid.model``  — ComputeBackend
"""
from __future__ import annotations

from abc import ABC, abstractmethod


# ── Scheduler ───────────────────────────────────────────────────────────────

class SchedulerBackend(ABC):
    """Abstract scheduler that computes PP partition points."""

    @abstractmethod
    def calculate_partition_point(self, is_average: bool) -> list[int]: ...

    @abstractmethod
    def calculate_partition_point_memory(
        self,
        is_average: bool,
    ) -> list[int]: ...


# ── Communication ───────────────────────────────────────────────────────────

class CommunicationBackend(ABC):
    """Abstract communication backend (NCCL / Gloo / torch.distributed)."""

    @abstractmethod
    def send(self, tensor: object, dst: int) -> None: ...

    @abstractmethod
    def recv(self, tensor: object, src: int) -> object: ...

    @abstractmethod
    def allreduce(self, tensor: object) -> object: ...

    @abstractmethod
    def broadcast(self, tensor: object, src: int) -> object: ...

    @abstractmethod
    def barrier(self) -> None: ...


# ── Fault Tolerance ─────────────────────────────────────────────────────────

class FaultToleranceBackend(ABC):
    """Abstract fault-tolerance backend for health checks and recovery."""

    @abstractmethod
    def save_checkpoint(self, state: dict[str, object], path: str) -> None: ...

    @abstractmethod
    def load_checkpoint(self, path: str) -> dict[str, object] | None: ...

    @abstractmethod
    def health_check(self) -> bool: ...

    @abstractmethod
    def on_failure(self, error: Exception) -> bool: ...


class CheckpointStrategy(ABC):
    """Abstract save/load policy separate from failure detection logic."""

    @abstractmethod
    def save(self, state: dict[str, object], path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> dict[str, object]: ...

    @abstractmethod
    def should_checkpoint(self, iter_id: int) -> bool: ...


# ── Compute ─────────────────────────────────────────────────────────────────

class ComputeBackend(ABC):
    """Abstract compute backend for forward / backward stage execution."""

    @abstractmethod
    def get_model(self) -> object: ...

    @abstractmethod
    def get_parameters(self) -> list[object]: ...

    @abstractmethod
    def forward(
        self,
        input_data: object,
        micro_batch_id: int,
        target: object | None = None,
    ) -> object: ...

    @abstractmethod
    def backward(
        self,
        micro_batch_id: int,
        grad: object | None = None,
        target: object | None = None,
    ) -> object | None: ...

    @abstractmethod
    def zero_input_grad(self) -> None: ...

    @abstractmethod
    def half(self) -> None: ...


# ── Profiler ────────────────────────────────────────────────────────────────

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


# ── Optimizer ───────────────────────────────────────────────────────────────

class OptimizerBackend(ABC):
    """Abstract optimizer backend with creation and step primitives."""

    @abstractmethod
    def create_optimizer(self, model: object, **kwargs: object) -> object: ...

    @abstractmethod
    def zero_grad(self) -> None: ...

    @abstractmethod
    def step(self) -> bool: ...


# ── Orchestrator ────────────────────────────────────────────────────────────

class Orchestrator(ABC):
    """Abstract orchestration layer for training and worker execution."""

    @abstractmethod
    def run_training(self) -> None: ...

    @abstractmethod
    def run_worker(self, rank: int, *args: object, **kwargs: object) -> None: ...


# ── State Manager ──────────────────────────────────────────────────────────

class StateManagerInterface(ABC):
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
    def set_current_iter(self, iter_id: int) -> None: ...


__all__ = [
    "SchedulerBackend",
    "CommunicationBackend",
    "FaultToleranceBackend",
    "CheckpointStrategy",
    "ComputeBackend",
    "ProfilerBackend",
    "OptimizerBackend",
    "Orchestrator",
    "StateManagerInterface",
]
