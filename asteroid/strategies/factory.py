"""Strategy factory — instantiate a parallelism strategy by name.

Usage::

    from asteroid.strategies import create_strategy

    strategy = create_strategy("asteroid", num_stages=3, micro_batch_size=4)
    plan = strategy.create_plan(config, topology)
"""
from __future__ import annotations

import logging
from typing import Any

from .base import ParallelismStrategy

logger = logging.getLogger(__name__)

# Registry populated lazily to avoid circular imports
_REGISTRY: dict[str, type[ParallelismStrategy]] = {}


def _ensure_registry() -> None:
    if _REGISTRY:
        return
    from .asteroid_strategy import AsteroidStrategy
    from .confident_strategy import ConfidentStrategy
    from .dtfm_strategy import DTFMStrategy

    _REGISTRY["asteroid"] = AsteroidStrategy
    _REGISTRY["confident"] = ConfidentStrategy
    _REGISTRY["dtfm"] = DTFMStrategy


def create_strategy(name: str, **kwargs: Any) -> ParallelismStrategy:
    """Instantiate a strategy by *name*.

    Parameters
    ----------
    name:
        One of ``"asteroid"``, ``"confident"``, ``"dtfm"``.
    **kwargs:
        Forwarded to the strategy constructor (e.g. ``num_stages``,
        ``pp_size``, ``dp_size``, ``population_size``).

    Returns
    -------
    ParallelismStrategy
        A concrete strategy instance.

    Raises
    ------
    ValueError
        If *name* is unknown.
    """
    _ensure_registry()
    key = name.lower().strip()
    cls = _REGISTRY.get(key)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown strategy {name!r}.  Available: {available}"
        )
    logger.info("Creating strategy %r with %s", key, kwargs or "{}")
    return cls(**kwargs)


def list_strategies() -> list[str]:
    """Return available strategy names."""
    _ensure_registry()
    return sorted(_REGISTRY)


__all__ = ["create_strategy", "list_strategies"]
