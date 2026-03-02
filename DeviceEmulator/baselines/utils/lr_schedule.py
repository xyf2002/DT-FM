from __future__ import annotations

import math
from typing import Protocol, cast


class _ScheduleConfig(Protocol):
    lr: float
    warmup_iters: int
    max_iters: int
    min_lr: float


class _NestedScheduleConfig(Protocol):
    training: _ScheduleConfig


def _resolve_schedule(cfg: object) -> _ScheduleConfig:
    if hasattr(cfg, "training"):
        nested = cast(_NestedScheduleConfig, cfg)
        return nested.training
    return cast(_ScheduleConfig, cfg)


def get_lr(it: int, cfg: object) -> float:
    """Cosine learning-rate schedule with linear warmup."""

    schedule = _resolve_schedule(cfg)
    lr = float(schedule.lr)
    warmup_iters = int(schedule.warmup_iters)
    max_iters = int(schedule.max_iters)
    min_lr = float(schedule.min_lr)

    if it < warmup_iters:
        return lr * (it + 1) / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


__all__ = ["get_lr"]
