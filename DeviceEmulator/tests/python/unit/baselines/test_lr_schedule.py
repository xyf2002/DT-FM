# pyright: reportUnknownMemberType=false

from __future__ import annotations

import pytest

from baselines.core.config import BaseConfig, TrainingConfig
from baselines.utils.lr_schedule import get_lr


def _make_training_config() -> TrainingConfig:
    return TrainingConfig(
        lr=1.0,
        min_lr=0.1,
        warmup_iters=10,
        max_iters=100,
        batch_size=4,
        global_batch_size=4,
        micro_batch_size=2,
        seed=42,
    )


def test_warmup_linear() -> None:
    """Warmup phase should increase learning rate linearly by step."""
    cfg = _make_training_config()

    assert get_lr(0, cfg) == pytest.approx(0.1)
    assert get_lr(4, cfg) == pytest.approx(0.5)
    assert get_lr(9, cfg) == pytest.approx(1.0)


def test_post_warmup_decay() -> None:
    """Cosine decay should begin at base lr and stay above min lr."""
    cfg = _make_training_config()

    assert get_lr(10, cfg) == pytest.approx(cfg.lr)
    mid_lr = get_lr(55, cfg)
    assert cfg.min_lr <= mid_lr <= cfg.lr


def test_beyond_max_iters() -> None:
    """Iterations after max_iters should clamp to min learning rate."""
    cfg = _make_training_config()

    assert get_lr(101, cfg) == pytest.approx(cfg.min_lr)
    assert get_lr(200, cfg) == pytest.approx(cfg.min_lr)


def test_at_max_iters() -> None:
    """Learning rate at max_iters should be at the cosine minimum."""
    cfg = _make_training_config()

    assert get_lr(100, cfg) == pytest.approx(cfg.min_lr)


def test_with_base_config(small_base_config: BaseConfig) -> None:
    """get_lr should read training settings from BaseConfig objects."""
    lr_from_base = get_lr(0, small_base_config)
    lr_from_training = get_lr(0, small_base_config.training)

    assert lr_from_base == pytest.approx(lr_from_training)


def test_monotonic_decay() -> None:
    """Learning rates from warmup end to max_iters should not increase."""
    cfg = _make_training_config()
    values = [
        get_lr(iteration, cfg)
        for iteration in range(cfg.warmup_iters, cfg.max_iters + 1)
    ]

    assert all(
        values[index] >= values[index + 1] - 1e-12
        for index in range(len(values) - 1)
    )
