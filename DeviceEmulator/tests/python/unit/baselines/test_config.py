from __future__ import annotations

import math

from baselines.core.config import (
    BaseConfig,
    DeviceConfig,
    DeviceTopology,
    DistributedConfig,
    FaultToleranceConfig,
    ModelConfig,
    ParallelConfig,
    ParallelismPlan,
    TrainingConfig,
)


def _model_name(config: ModelConfig) -> str:
    return getattr(config, "model_name", getattr(config, "model_type", ""))


def _plan_latency(plan: ParallelismPlan) -> float:
    for field_name in ("latency", "estimated_latency", "latency_ms"):
        latency = getattr(plan, field_name, None)
        if latency is not None:
            return float(latency)
    for value in vars(plan).values():
        if isinstance(value, (int, float)) and math.isinf(float(value)):
            return float(value)
    raise AssertionError("ParallelismPlan latency field was not found")


def test_device_config_defaults() -> None:
    """DeviceConfig should initialize with expected default settings."""
    config = DeviceConfig()
    assert config.use_cuda is True
    assert config.cuda_id == 0


def test_distributed_config_defaults() -> None:
    """DistributedConfig should expose standard single-process defaults."""
    config = DistributedConfig()
    assert config.dist_backend == "nccl"
    assert config.world_size == 1
    assert config.rank == 0
    assert getattr(config, "local_rank", 0) == 0


def test_model_config_defaults() -> None:
    """ModelConfig should match expected GPT-style baseline defaults."""
    config = ModelConfig()
    assert _model_name(config) == "gpt2"
    assert config.embedding_dim == 768


def test_training_config_defaults() -> None:
    """TrainingConfig should initialize baseline optimization defaults."""
    config = TrainingConfig()
    assert config.batch_size == 16
    assert config.lr == 3e-4


def test_training_config_post_init_global_lt_micro() -> None:
    """Global batch should be clamped to micro batch when too small."""
    config = TrainingConfig(global_batch_size=2, micro_batch_size=4)
    assert config.global_batch_size == 4


def test_training_config_post_init_batch_zero() -> None:
    """Zero batch_size should be replaced by global_batch_size."""
    config = TrainingConfig(
        batch_size=0,
        global_batch_size=8,
        micro_batch_size=2,
    )
    assert config.batch_size == 8


def test_training_config_post_init_num_microbatches() -> None:
    """num_microbatches should use integer global/micro ratio with floor 1."""
    config = TrainingConfig(global_batch_size=10, micro_batch_size=4)
    assert config.num_microbatches == max(1, 10 // 4)


def test_parallel_config_defaults() -> None:
    """ParallelConfig should default to no-op parallelism layout."""
    config = ParallelConfig()
    assert config.pp_size == 1
    assert config.dp_size == 1
    assert config.num_stages == 1


def test_parallel_config_post_init_stages() -> None:
    """num_stages should default to pp_size when initialized to zero."""
    config = ParallelConfig(num_stages=0, pp_size=4)
    assert config.num_stages == 4


def test_base_config_nesting() -> None:
    """BaseConfig should compose all nested baseline config sections."""
    config = BaseConfig()
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.training, TrainingConfig)
    assert isinstance(config.device, DeviceConfig)
    assert isinstance(config.distributed, DistributedConfig)
    assert isinstance(config.parallel, ParallelConfig)
    assert isinstance(config.fault_tolerance, FaultToleranceConfig)


def test_parallelism_plan_defaults() -> None:
    """ParallelismPlan should start empty with infinite latency estimate."""
    plan = ParallelismPlan()
    assert len(plan.partition_points) == 0
    assert math.isinf(_plan_latency(plan))


def test_device_topology_defaults() -> None:
    """DeviceTopology should initialize with no discovered device specs."""
    topology = DeviceTopology()
    assert len(topology.device_specs) == 0
