# pyright: reportMissingImports=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownParameterType=false
# pyright: reportAny=false

from __future__ import annotations

from dataclasses import replace

import torch

from baselines.core.config import ModelConfig
from baselines.models.stage import PipelineStage

CPU_DEVICE = torch.device("cpu")


def _num_layers(model_config: ModelConfig) -> int:
    """Infer model depth from the most common config attributes."""
    for attr in ("num_layers", "n_layers", "n_layer", "num_hidden_layers"):
        if hasattr(model_config, attr):
            value = getattr(model_config, attr)
            if isinstance(value, int) and value > 1:
                return value
    return 4


def _build_three_stage_pipeline(
    model_config: ModelConfig,
) -> tuple[PipelineStage, PipelineStage, PipelineStage]:
    """Build a first/middle/last pipeline split for a model config."""
    num_layers = _num_layers(model_config)
    first_end = max(1, num_layers // 3)
    middle_end = max(first_end + 1, (2 * num_layers) // 3)
    middle_end = min(middle_end, num_layers - 1)

    first = PipelineStage(
        model_config,
        0,
        first_end,
        is_first=True,
    ).to(CPU_DEVICE)
    middle = PipelineStage(model_config, first_end, middle_end).to(CPU_DEVICE)
    last = PipelineStage(
        model_config,
        middle_end,
        num_layers,
        is_last=True,
    ).to(CPU_DEVICE)
    return first, middle, last


def test_first_stage_has_embeddings(small_model_config: ModelConfig) -> None:
    """First stage should expose token and positional embeddings."""
    stage = PipelineStage(small_model_config, 0, 2, is_first=True).to(CPU_DEVICE)
    assert hasattr(stage, "embedding")
    assert hasattr(stage, "pos_embedding")


def test_last_stage_has_head(small_model_config: ModelConfig) -> None:
    """Last stage should include the task head module."""
    stage = PipelineStage(small_model_config, 2, 4, is_last=True).to(CPU_DEVICE)
    assert hasattr(stage, "head")


def test_middle_stage_no_embed_no_head(small_model_config: ModelConfig) -> None:
    """Middle stage should not expose embedding or head modules."""
    stage = PipelineStage(small_model_config, 1, 3).to(CPU_DEVICE)
    assert not hasattr(stage, "embedding")
    assert not hasattr(stage, "head")


def test_first_stage_forward(small_model_config: ModelConfig) -> None:
    """First stage should map token ids to hidden-state tensors."""
    stage = PipelineStage(small_model_config, 0, 2, is_first=True).to(CPU_DEVICE)
    token_ids = torch.randint(
        0,
        small_model_config.vocab_size,
        (2, 16),
        device=CPU_DEVICE,
        dtype=torch.long,
    )

    output = stage(token_ids)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 16, small_model_config.embedding_dim)
    assert output.dtype.is_floating_point


def test_middle_stage_forward(small_model_config: ModelConfig) -> None:
    """Middle stage should preserve hidden-state tensor shape."""
    stage = PipelineStage(small_model_config, 1, 3).to(CPU_DEVICE)
    hidden = torch.randn(
        2,
        16,
        small_model_config.embedding_dim,
        device=CPU_DEVICE,
    )

    output = stage(hidden)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 16, small_model_config.embedding_dim)


def test_full_pipeline_forward(small_model_config: ModelConfig) -> None:
    """Three-stage classification pipeline should return a scalar loss."""
    first, middle, last = _build_three_stage_pipeline(small_model_config)
    token_ids = torch.randint(
        0,
        small_model_config.vocab_size,
        (2, 16),
        device=CPU_DEVICE,
        dtype=torch.long,
    )
    targets = torch.randint(
        0,
        small_model_config.num_classes,
        (2,),
        device=CPU_DEVICE,
        dtype=torch.long,
    )

    hidden = middle(first(token_ids))
    loss = last(hidden, targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_num_params(small_model_config: ModelConfig) -> None:
    """num_params should return a positive integer parameter count."""
    stage = PipelineStage(small_model_config, 0, 2, is_first=True).to(CPU_DEVICE)
    num_params = stage.num_params()
    assert isinstance(num_params, int)
    assert num_params > 0


def test_weight_size_bytes(small_model_config: ModelConfig) -> None:
    """weight_size_bytes should return a positive integer size in bytes."""
    stage = PipelineStage(small_model_config, 0, 2, is_first=True).to(CPU_DEVICE)
    size_bytes = stage.weight_size_bytes()
    assert isinstance(size_bytes, int)
    assert size_bytes > 0


def test_lm_pipeline_forward(small_model_config: ModelConfig) -> None:
    """Three-stage LM pipeline should return a scalar token-level loss."""
    try:
        lm_config = replace(small_model_config, task_type="lm")
    except TypeError:
        lm_config = ModelConfig(
            model_type=small_model_config.model_type,
            embedding_dim=small_model_config.embedding_dim,
            num_heads=small_model_config.num_heads,
            d_ff=small_model_config.d_ff,
            seq_length=small_model_config.seq_length,
            max_seq_len=small_model_config.max_seq_len,
            vocab_size=small_model_config.vocab_size,
            num_classes=small_model_config.num_classes,
            dropout=small_model_config.dropout,
            use_flash_attention=small_model_config.use_flash_attention,
            task_type="lm",
        )

    first, middle, last = _build_three_stage_pipeline(lm_config)
    token_ids = torch.randint(
        0,
        lm_config.vocab_size,
        (2, 16),
        device=CPU_DEVICE,
        dtype=torch.long,
    )
    targets = torch.randint(
        0,
        lm_config.vocab_size,
        (2, 16),
        device=CPU_DEVICE,
        dtype=torch.long,
    )

    hidden = middle(first(token_ids))
    loss = last(hidden, targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
