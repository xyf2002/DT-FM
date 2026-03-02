from __future__ import annotations

import importlib
from dataclasses import replace
from uuid import uuid4

import pytest

from baselines.models.registry import (
    MODEL_REGISTRY,
    TASK_REGISTRY,
    create_block,
    create_head,
    register_model,
    register_task,
)

torch = importlib.import_module("torch")
nn = torch.nn


class DummyBlock(nn.Module):
    def forward(self, x):
        return x


class DummyHead(nn.Module):
    def forward(self, x, targets=None):
        return x


def test_default_model_registry_entries() -> None:
    """MODEL_REGISTRY should include built-in baseline block families."""
    assert "gpt2" in MODEL_REGISTRY
    assert "encoder" in MODEL_REGISTRY
    assert "llama" in MODEL_REGISTRY


def test_default_task_registry_entries() -> None:
    """TASK_REGISTRY should include classification and LM task heads."""
    assert "classification" in TASK_REGISTRY
    assert "lm" in TASK_REGISTRY


def test_register_model() -> None:
    """register_model should add a custom model entry to the registry."""
    name = f"test_model_{uuid4().hex}"
    register_model(name, DummyBlock, causal=False)
    assert name in MODEL_REGISTRY


def test_register_task() -> None:
    """register_task should add a custom task head entry to registry."""
    name = f"test_task_{uuid4().hex}"
    register_task(name, DummyHead)
    assert name in TASK_REGISTRY


def test_create_block_gpt2(small_model_config, cpu_device) -> None:
    """create_block should build a GPT-2 block module from ModelConfig."""
    config = replace(small_model_config, model_type="gpt2")
    block = create_block(config)
    block = block.to(cpu_device)
    assert isinstance(block, nn.Module)


def test_create_block_llama(small_model_config, cpu_device) -> None:
    """create_block should build a LLaMA block using available head fields."""
    if hasattr(small_model_config, "num_kv_heads"):
        config = replace(
            small_model_config,
            model_type="llama",
            num_kv_heads=small_model_config.num_heads,
        )
    else:
        config = replace(small_model_config, model_type="llama")

    block = create_block(config)
    block = block.to(cpu_device)
    assert isinstance(block, nn.Module)


def test_create_block_encoder(small_model_config, cpu_device) -> None:
    """create_block should build an encoder block module from ModelConfig."""
    config = replace(small_model_config, model_type="encoder")
    block = create_block(config)
    block = block.to(cpu_device)
    assert isinstance(block, nn.Module)


def test_create_head_classification(small_model_config, cpu_device) -> None:
    """create_head should build a classification head for class logits."""
    config = replace(
        small_model_config,
        task_type="classification",
        embedding_dim=64,
        num_classes=5,
    )
    head = create_head(config)
    head = head.to(cpu_device)
    assert isinstance(head, nn.Module)


def test_create_head_lm(small_model_config, cpu_device) -> None:
    """create_head should build a language-modeling output head."""
    config = replace(
        small_model_config,
        task_type="lm",
        embedding_dim=64,
        vocab_size=100,
    )
    head = create_head(config)
    head = head.to(cpu_device)
    assert isinstance(head, nn.Module)


def test_create_block_unknown_raises(small_model_config) -> None:
    """create_block should raise ValueError for unknown model types."""
    config = replace(small_model_config, model_type="nonexistent")
    with pytest.raises(ValueError):
        create_block(config)


def test_create_head_unknown_raises(small_model_config) -> None:
    """create_head should raise ValueError for unknown task types."""
    config = replace(small_model_config, task_type="nonexistent")
    with pytest.raises(ValueError):
        create_head(config)
