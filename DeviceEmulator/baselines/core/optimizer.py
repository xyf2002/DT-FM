from __future__ import annotations

from abc import ABC, abstractmethod


class OptimizerBackend(ABC):
    """Abstract optimizer backend with creation and step primitives."""

    @abstractmethod
    def create_optimizer(
        self,
        model: object,
        **kwargs: object,
    ) -> object: ...

    @abstractmethod
    def zero_grad(self) -> None: ...

    @abstractmethod
    def step(self) -> bool: ...


__all__ = ["OptimizerBackend"]
