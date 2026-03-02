"""Communication backend implementations.

Provides three backends implementing the ``CommunicationBackend`` ABC:

- ``TorchDistComm`` — wraps ``torch.distributed`` (NCCL or Gloo)
- ``GlooComm`` — Gloo-specific with automatic CPU tensoring
- ``NCCLComm`` — CuPy NCCL with P2P send/recv (from DT-FM)

Use ``create_comm_backend()`` factory to instantiate by name.
"""
from __future__ import annotations

import importlib
import logging
from datetime import timedelta
from typing import Any, cast

from ..core.interfaces import CommunicationBackend

logger = logging.getLogger(__name__)


def _dist() -> Any:
    return importlib.import_module("torch.distributed")


# ---------------------------------------------------------------------------
# torch.distributed backend (NCCL or Gloo under the hood)
# ---------------------------------------------------------------------------

class TorchDistComm(CommunicationBackend):
    """Wrapper around ``torch.distributed`` send/recv/allreduce."""

    def __init__(self, rank: int, world_size: int, backend: str = "nccl") -> None:
        if backend.lower() not in ("nccl", "gloo"):
            raise ValueError("backend must be 'nccl' or 'gloo'")
        self.rank = rank
        self.world_size = world_size
        self.backend = backend.lower()

    def send(self, tensor: object, dst_rank: int,
             group: object | None = None) -> None:
        _dist().send(tensor=tensor, dst=dst_rank, group=group)

    def recv(self, tensor: object, src_rank: int,
             group: object | None = None) -> object:
        _dist().recv(tensor=tensor, src=src_rank, group=group)
        return tensor

    def allreduce(self, tensor: object, group: object | None = None,
                  op: str = "sum") -> object:
        dist = _dist()
        op_map = {
            "sum": dist.ReduceOp.SUM,
            "max": dist.ReduceOp.MAX,
            "min": dist.ReduceOp.MIN,
            "prod": dist.ReduceOp.PRODUCT,
        }
        dist.all_reduce(tensor=tensor, op=op_map[op.lower()], group=group)
        return tensor

    def broadcast(self, tensor: object, src_rank: int,
                  group: object | None = None) -> object:
        _dist().broadcast(tensor=tensor, src=src_rank, group=group)
        return tensor

    def barrier(self, group: object | None = None) -> None:
        _dist().barrier(group=group)

    def init_process_group(self, init_method: str,
                           timeout_s: float = 120.0) -> None:
        dist = _dist()
        if dist.is_initialized():
            return
        dist.init_process_group(
            backend=self.backend,
            init_method=init_method,
            rank=self.rank,
            world_size=self.world_size,
            timeout=timedelta(seconds=timeout_s),
        )

    def new_group(self, ranks: list[int]) -> object:
        return _dist().new_group(ranks=ranks, backend=self.backend)


# ---------------------------------------------------------------------------
# Gloo-specific backend (CPU tensors for sends)
# ---------------------------------------------------------------------------

class GlooComm(TorchDistComm):
    """Gloo backend — automatically moves tensors to CPU for send/recv."""

    def __init__(self, rank: int, world_size: int) -> None:
        super().__init__(rank=rank, world_size=world_size, backend="gloo")

    def send(self, tensor: object, dst_rank: int,
             group: object | None = None) -> None:
        t = cast(Any, tensor)
        cpu_t = t.detach().cpu() if t.is_cuda else t
        super().send(cpu_t, dst_rank=dst_rank, group=group)

    def recv(self, tensor: object, src_rank: int,
             group: object | None = None) -> object:
        t = cast(Any, tensor)
        if not t.is_cuda:
            return super().recv(t, src_rank=src_rank, group=group)
        torch = importlib.import_module("torch")
        cpu_buf = torch.empty_like(t, device="cpu")
        super().recv(cpu_buf, src_rank=src_rank, group=group)
        t.copy_(cpu_buf)
        return t


# ---------------------------------------------------------------------------
# CuPy NCCL backend (direct P2P)
# ---------------------------------------------------------------------------

class NCCLComm(CommunicationBackend):
    """CuPy NCCL P2P + AllReduce backend (from DT-FM / existing nccl_utils)."""

    def __init__(self, rank: int, world_size: int, cuda_id: int) -> None:
        self.rank = rank
        self.world_size = world_size
        self.cuda_id = cuda_id
        self.pp_nccl_comm: object | None = None
        self.dp_nccl_comm: object | None = None

    def send(self, tensor: object, dst_rank: int,
             group: object | None = None) -> None:
        if self.pp_nccl_comm is None:
            raise RuntimeError("PP comm not initialized")
        from ..comm.nccl_utils import _nccl_send
        import torch
        stream = torch.cuda.current_stream(self.cuda_id)
        _nccl_send(cast(Any, tensor), dst_rank, self.pp_nccl_comm, stream)

    def recv(self, tensor: object, src_rank: int,
             group: object | None = None) -> object:
        if self.pp_nccl_comm is None:
            raise RuntimeError("PP comm not initialized")
        from ..comm.nccl_utils import _nccl_recv
        import torch
        stream = torch.cuda.current_stream(self.cuda_id)
        _nccl_recv(cast(Any, tensor), src_rank, self.pp_nccl_comm, stream)
        return tensor

    def allreduce(self, tensor: object, group: object | None = None,
                  op: str = "sum") -> object:
        comm = self.dp_nccl_comm or self.pp_nccl_comm
        if comm is None:
            raise RuntimeError("No NCCL comm initialized")
        from ..comm.nccl_utils import _nccl_allreduce
        import torch
        stream = torch.cuda.current_stream(self.cuda_id)
        _nccl_allreduce(cast(Any, tensor), comm, stream)
        return tensor

    def broadcast(self, tensor: object, src_rank: int,
                  group: object | None = None) -> object:
        # Fallback to torch.distributed for broadcast
        _dist().broadcast(tensor=tensor, src=src_rank, group=group)
        return tensor

    def barrier(self, group: object | None = None) -> None:
        _dist().barrier(group=group)

    def setup_communicators(
        self,
        pp_rank: int, dp_rank: int,
        pp_size: int, dp_size: int,
        dist_store: object | None = None,
    ) -> None:
        """Initialize CuPy NCCL communicators (delegates to nccl_utils)."""
        from ..core.config import AsteroidConfig
        from ..comm.nccl_utils import setup_nccl_communicators

        # Build a minimal config for the setup function
        cfg = AsteroidConfig.__new__(AsteroidConfig)
        cfg.num_stages = pp_size
        cfg.world_size = self.world_size

        self.pp_nccl_comm, self.dp_nccl_comm = setup_nccl_communicators(
            rank=self.rank, cfg=cfg,
            pp_rank=pp_rank, dp_rank=dp_rank,
            cuda_id=self.cuda_id, dist_store=dist_store,
            pp_size_override=pp_size, dp_size_override=dp_size,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_comm_backend(
    backend: str,
    rank: int,
    world_size: int,
    cuda_id: int = 0,
) -> CommunicationBackend:
    """Create a communication backend by name.

    Args:
        backend: ``"torch_dist"`` | ``"nccl"`` | ``"gloo"``
        rank: This process's rank.
        world_size: Total number of processes.
        cuda_id: CUDA device ordinal (only used by ``nccl``).
    """
    b = backend.lower().replace("-", "_")
    if b in ("torch_dist", "torch"):
        return TorchDistComm(rank, world_size, backend="nccl")
    if b == "gloo":
        return GlooComm(rank, world_size)
    if b == "nccl":
        return NCCLComm(rank, world_size, cuda_id)
    raise ValueError(
        f"Unknown comm backend '{backend}'. Use 'torch_dist', 'nccl', or 'gloo'."
    )


__all__ = [
    "TorchDistComm",
    "GlooComm",
    "NCCLComm",
    "create_comm_backend",
]
