from __future__ import annotations

import importlib
import logging
from datetime import timedelta
from typing import Any

logger = logging.getLogger(__name__)


def _dist() -> Any:
    try:
        return importlib.import_module("torch.distributed")
    except ImportError as exc:
        raise RuntimeError("torch.distributed is required.") from exc


class TorchDistBackend:
    """torch.distributed wrappers for send/recv/allreduce."""

    rank: int
    world_size: int
    backend: str

    def __init__(
        self,
        rank: int,
        world_size: int,
        backend: str = "nccl",
    ) -> None:
        backend_name = backend.lower()
        if backend_name not in {"nccl", "gloo"}:
            raise ValueError("backend must be 'nccl' or 'gloo'.")
        self.rank = rank
        self.world_size = world_size
        self.backend = backend_name

    def send(
        self,
        tensor: object,
        dst_rank: int,
        group: object | None = None,
    ) -> None:
        dist = _dist()
        dist.send(tensor=tensor, dst=dst_rank, group=group)

    def recv(
        self,
        tensor: object,
        src_rank: int,
        group: object | None = None,
    ) -> object:
        dist = _dist()
        dist.recv(tensor=tensor, src=src_rank, group=group)
        return tensor

    def allreduce(
        self,
        tensor: object,
        group: object | None = None,
        op: str = "sum",
    ) -> object:
        dist = _dist()
        op_name = op.lower()
        op_map = {
            "sum": dist.ReduceOp.SUM,
            "max": dist.ReduceOp.MAX,
            "min": dist.ReduceOp.MIN,
            "prod": dist.ReduceOp.PRODUCT,
            "product": dist.ReduceOp.PRODUCT,
        }
        if op_name not in op_map:
            raise ValueError(f"Unsupported allreduce op: {op}")
        dist.all_reduce(tensor=tensor, op=op_map[op_name], group=group)
        return tensor

    def broadcast(
        self,
        tensor: object,
        src_rank: int,
        group: object | None = None,
    ) -> object:
        dist = _dist()
        dist.broadcast(tensor=tensor, src=src_rank, group=group)
        return tensor

    def barrier(self, group: object | None = None) -> None:
        dist = _dist()
        dist.barrier(group=group)

    def init_process_group(
        self,
        init_method: str,
        timeout_s: float = 120.0,
    ) -> None:
        dist = _dist()
        if dist.is_initialized():
            logger.debug("Process group already initialized.")
            return
        dist.init_process_group(
            backend=self.backend,
            init_method=init_method,
            rank=self.rank,
            world_size=self.world_size,
            timeout=timedelta(seconds=timeout_s),
        )

    def new_group(self, ranks: list[int]) -> object:
        dist = _dist()
        return dist.new_group(ranks=ranks, backend=self.backend)


__all__ = ["TorchDistBackend"]
