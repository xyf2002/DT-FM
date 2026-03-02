"""Parallelism strategies for the Asteroid distributed-training system.

Provides three concrete strategies ported from DeviceEmulator/baselines/:
  * ``AsteroidStrategy``  — HPP DP planning with micro-batch allocation
  * ``ConfidentStrategy`` — bottleneck-minimising DP stage partition
  * ``DTFMStrategy``      — GCMA topology search + DP partitioning

Use the factory function :func:`create_strategy` to instantiate by name.
"""
from .base import ParallelismStrategy
from .factory import create_strategy

__all__ = ["ParallelismStrategy", "create_strategy"]
