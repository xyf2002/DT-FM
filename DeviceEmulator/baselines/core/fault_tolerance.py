from __future__ import annotations

from abc import ABC, abstractmethod


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


__all__ = ["FaultToleranceBackend", "CheckpointStrategy"]
