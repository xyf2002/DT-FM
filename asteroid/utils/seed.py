"""Deterministic seeding utilities for distributed training.

Provides ``seed_everything()`` to set all RNG seeds once, and helpers
for per-rank / per-microbatch seed derivation to guarantee
reproducibility in pipeline-parallel settings.

Usage::

    from asteroid.utils.seed import seed_everything, make_rank_seed
    seed_everything(42)
    rank_seed = make_rank_seed(42, rank=1)
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Set Python, NumPy, and PyTorch RNG seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        torch.use_deterministic_algorithms(True, warn_only=True)


def make_rank_seed(base_seed: int, rank: int) -> int:
    """Derive a per-rank seed: ``base_seed + rank``."""
    return int(base_seed) + int(rank)


def make_microbatch_seed(
    base_seed: int,
    dp_rank: int,
    iter_num: int,
    micro_idx: int,
) -> int:
    """Deterministic hash seed for a specific (dp_rank, iter, micro-batch)."""
    return (
        int(base_seed) * 1_000_003
        + int(dp_rank) * 100_003
        + int(iter_num) * 997
        + int(micro_idx)
    )


def seeded_generator(seed: int) -> torch.Generator:
    """Create a ``torch.Generator`` with a fixed seed."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


__all__ = [
    "seed_everything",
    "make_rank_seed",
    "make_microbatch_seed",
    "seeded_generator",
]
