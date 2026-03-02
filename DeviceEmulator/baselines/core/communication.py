from __future__ import annotations

from abc import ABC, abstractmethod


class CommunicationBackend(ABC):
    """Abstract communication backend (NCCL/Gloo/torch.distributed)."""

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


__all__ = ["CommunicationBackend"]
