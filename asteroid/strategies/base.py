"""Base class for parallelism strategies.

Ported from DeviceEmulator/baselines/strategies/base.py.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from asteroid.core.config import AsteroidConfig, DeviceTopology

logger = logging.getLogger(__name__)


# ── Plan dataclass ──────────────────────────────────────────────────────────

@dataclass
class ParallelismPlan:
    """Output of a strategy's :meth:`create_plan`.

    Attributes
    ----------
    partition_points:
        Layer indices at which to split the pipeline. An empty list
        means a single stage covering all layers.
    device_groups:
        ``{stage_idx: [device_id, ...]}`` mapping.
    micro_batch_alloc:
        ``{stage_idx: {device_id: micro_batch_count, ...}}`` — only
        populated by strategies that do heterogeneous allocation
        (e.g. Asteroid).
    schedule_type:
        Pipeline schedule to use (``"gpipe"`` or ``"1f1b"``).
    estimated_latency_ms:
        Estimated per-step latency from the planner.
    """

    partition_points: List[int] = field(default_factory=list)
    device_groups: Dict[int, List[int]] = field(default_factory=dict)
    micro_batch_alloc: Dict[int, Dict[int, int]] = field(default_factory=dict)
    schedule_type: str = "1f1b"
    estimated_latency_ms: float = 0.0


# ── ABC ─────────────────────────────────────────────────────────────────────

class ParallelismStrategy(ABC):
    """Produces a model-parallelism plan for a given model + device topology."""

    @abstractmethod
    def create_plan(
        self,
        config: AsteroidConfig,
        device_topology: DeviceTopology,
        profiler: Any | None = None,
    ) -> ParallelismPlan:
        """Compute partition points, device groups, and schedule type."""
        ...

    @abstractmethod
    def get_schedule_type(self) -> str:
        """Return the pipeline-schedule identifier this strategy prefers."""
        ...


__all__ = ["ParallelismStrategy", "ParallelismPlan"]
