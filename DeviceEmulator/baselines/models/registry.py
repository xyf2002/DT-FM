from __future__ import annotations

from typing import Any

from torch import nn

from baselines.core.config import BaseConfig, ModelConfig

from .bert import EncoderBlock
from .gpt2 import ClassificationHead, GPT2Block, LMHead
from .llama import LlamaBlock

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "gpt2": {"block": GPT2Block, "causal": True},
    "encoder": {"block": EncoderBlock, "causal": False},
    "llama": {"block": LlamaBlock, "causal": True},
}

TASK_REGISTRY: dict[str, type] = {
    "classification": ClassificationHead,
    "lm": LMHead,
}

ModelRegistry = MODEL_REGISTRY
TaskRegistry = TASK_REGISTRY


def _coerce_model_config(model_config: ModelConfig | BaseConfig) -> ModelConfig:
    if isinstance(model_config, BaseConfig):
        return model_config.model
    return model_config


def register_model(
    name: str,
    block_cls: type[nn.Module],
    causal: bool = True,
) -> None:
    """Register a transformer block class in the model registry."""
    MODEL_REGISTRY[name] = {"block": block_cls, "causal": causal}


def register_task(name: str, head_cls: type[nn.Module]) -> None:
    """Register a task head class in the task registry."""
    TASK_REGISTRY[name] = head_cls


def create_block(model_config: ModelConfig) -> nn.Module:
    """Instantiate a transformer block from ModelConfig."""
    cfg = _coerce_model_config(model_config)
    entry = MODEL_REGISTRY.get(cfg.model_type)
    if entry is None:
        names = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model_type '{cfg.model_type}'. Available: {names}."
        )
    block_cls = entry["block"]
    kwargs: dict[str, Any] = {
        "use_flash": cfg.use_flash_attention,
    }
    if cfg.model_type == "llama":
        kwargs["n_kv_heads"] = getattr(cfg, "num_kv_heads", cfg.num_heads)
    return block_cls(
        cfg.embedding_dim,
        cfg.num_heads,
        cfg.d_ff,
        cfg.max_seq_len,
        cfg.dropout,
        **kwargs,
    )


def create_head(model_config: ModelConfig) -> nn.Module:
    """Instantiate a task head from ModelConfig."""
    cfg = _coerce_model_config(model_config)
    head_cls = TASK_REGISTRY.get(cfg.task_type)
    if head_cls is None:
        names = list(TASK_REGISTRY.keys())
        raise ValueError(
            f"Unknown task_type '{cfg.task_type}'. Available: {names}."
        )
    if cfg.task_type == "lm":
        return head_cls(cfg.embedding_dim, cfg.vocab_size)
    return head_cls(cfg.embedding_dim, cfg.num_classes)


__all__ = [
    "ModelRegistry",
    "TaskRegistry",
    "MODEL_REGISTRY",
    "TASK_REGISTRY",
    "register_model",
    "register_task",
    "create_block",
    "create_head",
]
