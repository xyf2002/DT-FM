from __future__ import annotations

from abc import ABC, abstractmethod


class ComputeBackend(ABC):
    """Abstract compute backend for forward and backward stage execution."""

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


__all__ = ["ComputeBackend"]
