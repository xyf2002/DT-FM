"""Shared fixtures for baselines unit tests."""
from __future__ import annotations

import pytest
import torch


@pytest.fixture()
def cpu_device() -> torch.device:
    """Force CPU device for all tests."""
    return torch.device("cpu")


@pytest.fixture()
def small_model_config():
    """Minimal ModelConfig for fast tests."""
    from baselines.core.config import ModelConfig

    return ModelConfig(
        model_name="gpt2-test",
        model_type="gpt2",
        task_type="classification",
        seq_length=32,
        max_seq_len=32,
        embedding_dim=64,
        num_layers=4,
        num_heads=4,
        d_ff=128,
        vocab_size=100,
        num_classes=2,
        dropout=0.0,
        use_flash_attention=False,
    )


@pytest.fixture()
def small_base_config(small_model_config):
    """Minimal BaseConfig wrapping small_model_config."""
    from baselines.core.config import (
        BaseConfig,
        DistributedConfig,
        FaultToleranceConfig,
        ParallelConfig,
        TrainingConfig,
    )

    return BaseConfig(
        model=small_model_config,
        training=TrainingConfig(
            batch_size=4,
            global_batch_size=4,
            micro_batch_size=2,
            lr=1e-3,
            min_lr=1e-5,
            warmup_iters=10,
            max_iters=100,
            seed=42,
        ),
        distributed=DistributedConfig(world_size=1),
        parallel=ParallelConfig(pp_size=2, dp_size=1, num_stages=2),
        fault_tolerance=FaultToleranceConfig(
            checkpoint_dir="/tmp/test_ckpt",
            checkpoint_interval=10,
        ),
    )
