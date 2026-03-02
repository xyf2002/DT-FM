from __future__ import annotations

import importlib
import os
import random
from typing import Protocol, cast

import numpy as np


class _GeneratorLike(Protocol):
    def manual_seed(self, seed: int) -> object: ...


class _CudaNamespace(Protocol):
    def is_available(self) -> bool: ...

    def manual_seed(self, seed: int) -> None: ...

    def manual_seed_all(self, seed: int) -> None: ...


class _CudnnNamespace(Protocol):
    deterministic: bool
    benchmark: bool


class _BackendsNamespace(Protocol):
    cudnn: _CudnnNamespace


class _TorchModule(Protocol):
    cuda: _CudaNamespace
    backends: _BackendsNamespace

    def manual_seed(self, seed: int) -> None: ...

    def use_deterministic_algorithms(
        self,
        mode: bool,
        *,
        warn_only: bool = False,
    ) -> None: ...

    def Generator(self) -> _GeneratorLike: ...


def _load_torch_module() -> _TorchModule:
    module = cast(object, importlib.import_module("torch"))
    return cast(_TorchModule, module)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Set Python, NumPy, and Torch RNG seeds for reproducibility."""

    torch = _load_torch_module()
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def make_rank_seed(base_seed: int, rank: int) -> int:
    """Match baseline per-rank seed derivation: seed + rank."""

    return int(base_seed) + int(rank)


def make_microbatch_seed(
    base_seed: int,
    dp_rank: int,
    iter_num: int,
    micro_idx: int,
) -> int:
    """Deterministic hash seed used for (dp_rank, iter, micro-batch)."""

    return (
        int(base_seed) * 1000003
        + int(dp_rank) * 100003
        + int(iter_num) * 997
        + int(micro_idx)
    )


def seeded_generator(seed: int) -> _GeneratorLike:
    """Create a torch.Generator with deterministic seed state."""

    torch = _load_torch_module()
    generator = torch.Generator()
    _ = generator.manual_seed(seed)
    return generator


__all__ = [
    "seed_everything",
    "make_rank_seed",
    "make_microbatch_seed",
    "seeded_generator",
]
