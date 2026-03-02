from __future__ import annotations

from abc import ABC, abstractmethod


class Orchestrator(ABC):
    """Abstract orchestration layer for training and worker execution."""

    @abstractmethod
    def run_training(self) -> None: ...

    @abstractmethod
    def run_worker(self, rank: int, *args: object, **kwargs: object) -> None: ...


__all__ = ["Orchestrator"]
