from __future__ import annotations

from baselines.core.config import (
    DeviceConfig,
    DeviceTopology,
    FaultToleranceConfig,
    ModelConfig,
    ParallelismPlan,
)
from baselines.strategies.asteroid_strategy import AsteroidStrategy
from baselines.strategies.confident_strategy import ConfidentStrategy
from baselines.strategies.dtfm_strategy import DTFMStrategy


def _make_topology(n_devices: int) -> DeviceTopology:
    specs = [DeviceConfig(device_id=i) for i in range(n_devices)]
    return DeviceTopology(device_specs=specs)


def _make_model_config() -> ModelConfig:
    return ModelConfig(
        model_type="gpt2",
        num_layers=8,
        embedding_dim=64,
        num_heads=4,
        d_ff=128,
        max_seq_len=32,
        vocab_size=100,
        dropout=0.0,
        use_flash_attention=False,
    )


def test_dtfm_construction() -> None:
    """DTFMStrategy should preserve constructor parallelism sizes."""
    strategy = DTFMStrategy(pp_size=2, dp_size=2)
    assert strategy.pp_size == 2
    assert strategy.dp_size == 2


def test_dtfm_schedule_type() -> None:
    """DTFMStrategy should expose the GPipe schedule type."""
    strategy = DTFMStrategy(pp_size=2, dp_size=2)
    assert strategy.get_schedule_type() == "gpipe"


def test_dtfm_ft_config() -> None:
    """DTFMStrategy should provide a fault tolerance configuration."""
    strategy = DTFMStrategy(pp_size=2, dp_size=2)
    ft_config = strategy.get_fault_tolerance_config()
    assert isinstance(ft_config, FaultToleranceConfig)


def test_dtfm_create_plan() -> None:
    """DTFMStrategy should create a valid plan on four devices."""
    strategy = DTFMStrategy(pp_size=2, dp_size=2)
    model_config = _make_model_config()
    topology = _make_topology(4)

    plan = strategy.create_plan(model_config, topology)

    assert isinstance(plan, ParallelismPlan)
    assert isinstance(plan.partition_points, list)
    assert len(plan.partition_points) > 0


def test_dtfm_single_device() -> None:
    """DTFMStrategy should still return a plan on one device."""
    strategy = DTFMStrategy(pp_size=1, dp_size=1)
    model_config = _make_model_config()
    topology = _make_topology(1)

    plan = strategy.create_plan(model_config, topology)

    assert isinstance(plan, ParallelismPlan)
    assert isinstance(plan.partition_points, list)


def test_asteroid_construction() -> None:
    """AsteroidStrategy should preserve constructor stage count."""
    strategy = AsteroidStrategy(
        num_stages=2,
        micro_batch_size=4,
        num_microbatches=4,
    )
    assert strategy.num_stages == 2


def test_asteroid_schedule_type() -> None:
    """AsteroidStrategy should expose the 1F1B schedule type."""
    strategy = AsteroidStrategy(
        num_stages=2,
        micro_batch_size=4,
        num_microbatches=4,
    )
    assert strategy.get_schedule_type() == "1f1b"


def test_asteroid_ft_config() -> None:
    """AsteroidStrategy should provide a fault tolerance configuration."""
    strategy = AsteroidStrategy(
        num_stages=2,
        micro_batch_size=4,
        num_microbatches=4,
    )
    ft_config = strategy.get_fault_tolerance_config()
    assert isinstance(ft_config, FaultToleranceConfig)


def test_asteroid_create_plan() -> None:
    """AsteroidStrategy should create a valid plan for two devices."""
    strategy = AsteroidStrategy(
        num_stages=2,
        micro_batch_size=4,
        num_microbatches=4,
    )
    model_config = _make_model_config()
    topology = _make_topology(2)

    plan = strategy.create_plan(model_config, topology)

    assert isinstance(plan, ParallelismPlan)
    assert isinstance(plan.partition_points, list)


def test_confident_construction() -> None:
    """ConfidentStrategy should preserve constructor pipeline size."""
    strategy = ConfidentStrategy(pp_size=2, dp_size=1)
    assert strategy.pp_size == 2


def test_confident_schedule_type() -> None:
    """ConfidentStrategy should expose the GPipe schedule type."""
    strategy = ConfidentStrategy(pp_size=2, dp_size=1)
    assert strategy.get_schedule_type() == "1f1b"


def test_confident_ft_config() -> None:
    """ConfidentStrategy should provide a fault tolerance configuration."""
    strategy = ConfidentStrategy(pp_size=2, dp_size=1)
    ft_config = strategy.get_fault_tolerance_config()
    assert isinstance(ft_config, FaultToleranceConfig)


def test_confident_create_plan() -> None:
    """ConfidentStrategy should create a valid plan for two devices."""
    strategy = ConfidentStrategy(pp_size=2, dp_size=1)
    model_config = _make_model_config()
    topology = _make_topology(2)

    plan = strategy.create_plan(model_config, topology)

    assert isinstance(plan, ParallelismPlan)
    assert isinstance(plan.partition_points, list)
