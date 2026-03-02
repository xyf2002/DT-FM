from __future__ import annotations

import importlib
import logging
from typing import Any, cast

from .torch_dist import TorchDistBackend

logger = logging.getLogger(__name__)


class GlooBackend(TorchDistBackend):
    """Gloo-specific backend for CPU communication."""

    def __init__(self, rank: int, world_size: int) -> None:
        super().__init__(rank=rank, world_size=world_size, backend="gloo")

    def send(
        self,
        tensor: object,
        dst_rank: int,
        group: object | None = None,
    ) -> None:
        tensor_any = cast(Any, tensor)
        cpu_tensor = tensor_any.detach().cpu() if tensor_any.is_cuda else tensor
        super().send(cpu_tensor, dst_rank=dst_rank, group=group)

    def recv(
        self,
        tensor: object,
        src_rank: int,
        group: object | None = None,
    ) -> object:
        tensor_any = cast(Any, tensor)
        if not tensor_any.is_cuda:
            return super().recv(tensor_any, src_rank=src_rank, group=group)

        torch = importlib.import_module("torch")
        cpu_tensor = torch.empty_like(tensor_any, device="cpu")
        super().recv(cpu_tensor, src_rank=src_rank, group=group)
        tensor_any.copy_(cpu_tensor)
        return tensor_any


__all__ = ["GlooBackend"]
