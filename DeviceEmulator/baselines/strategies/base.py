from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from baselines.core.config import (
    DeviceConfig,
    DeviceTopology,
    FaultToleranceConfig,
    ModelConfig,
    ParallelConfig,
    ParallelismPlan,
)
from baselines.core.profiler import ProfilerBackend

logger = logging.getLogger(__name__)
_CONFIG_TYPES = (DeviceConfig, ParallelConfig)


class ParallelismStrategy(ABC):
    """Produces a model parallelism plan for a given model + device topology."""

    @abstractmethod
    def create_plan(
        self,
        model_config: ModelConfig,
        device_topology: DeviceTopology,
        profiler: ProfilerBackend | None = None,
    ) -> ParallelismPlan:
        ...

    @abstractmethod
    def get_schedule_type(self) -> str:
        ...

    @abstractmethod
    def get_fault_tolerance_config(self) -> FaultToleranceConfig:
        ...


__all__ = ["ParallelismStrategy"]
