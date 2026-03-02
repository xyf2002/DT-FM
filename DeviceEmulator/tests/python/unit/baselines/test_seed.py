# pyright: reportMissingImports=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownParameterType=false
# pyright: reportAny=false

from __future__ import annotations

import random

import numpy as np
import torch

from baselines.utils.seed import (
    make_microbatch_seed,
    make_rank_seed,
    seed_everything,
    seeded_generator,
)

CPU_DEVICE = torch.device("cpu")


def test_seed_everything_reproducible() -> None:
    """seed_everything should make torch random draws reproducible."""
    seed_everything(42)
    first = torch.randn(5, device=CPU_DEVICE)

    seed_everything(42)
    second = torch.randn(5, device=CPU_DEVICE)

    assert torch.allclose(first, second)


def test_seed_everything_numpy() -> None:
    """seed_everything should make NumPy random draws reproducible."""
    seed_everything(42)
    first = np.random.rand(5)

    seed_everything(42)
    second = np.random.rand(5)

    assert np.allclose(first, second)


def test_seed_everything_python() -> None:
    """seed_everything should make Python random draws reproducible."""
    seed_everything(42)
    first = random.random()

    seed_everything(42)
    second = random.random()

    assert first == second


def test_make_rank_seed_different() -> None:
    """make_rank_seed should produce different values per rank."""
    seed_rank0 = make_rank_seed(42, 0)
    seed_rank1 = make_rank_seed(42, 1)

    assert seed_rank0 != seed_rank1


def test_make_rank_seed_deterministic() -> None:
    """make_rank_seed should return stable values for the same input."""
    assert make_rank_seed(42, 3) == make_rank_seed(42, 3)


def test_make_microbatch_seed_deterministic() -> None:
    """make_microbatch_seed should be deterministic for fixed arguments."""
    seed_a = make_microbatch_seed(42, 0, 1, 0)
    seed_b = make_microbatch_seed(42, 0, 1, 0)

    assert seed_a == seed_b


def test_make_microbatch_seed_varies() -> None:
    """make_microbatch_seed should change with rank, iter, or micro index."""
    base = make_microbatch_seed(42, 0, 1, 0)
    different_rank = make_microbatch_seed(42, 1, 1, 0)
    different_iter = make_microbatch_seed(42, 0, 2, 0)
    different_micro = make_microbatch_seed(42, 0, 1, 1)

    assert base != different_rank
    assert base != different_iter
    assert base != different_micro


def test_seeded_generator() -> None:
    """seeded_generator should make generator-based draws reproducible."""
    gen_a = seeded_generator(42)
    first = torch.randn(5, generator=gen_a, device=CPU_DEVICE)

    gen_b = seeded_generator(42)
    second = torch.randn(5, generator=gen_b, device=CPU_DEVICE)

    assert torch.allclose(first, second)
