from __future__ import annotations

import textwrap
from dataclasses import replace
from pathlib import Path

import pytest

from baselines.core.config import BaseConfig
from baselines.utils.config_loader import load_config, save_config


def _write_yaml(path: Path, text: str) -> Path:
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")
    return path


def _model_name(config: BaseConfig) -> str:
    model = config.model
    return getattr(model, "model_name", getattr(model, "model_type", ""))


def test_load_valid_yaml(tmp_path: Path) -> None:
    """load_config should parse valid YAML fields into BaseConfig."""
    config_path = _write_yaml(
        tmp_path / "valid.yaml",
        """
        model:
          model_name: test
          model_type: test
        training:
          lr: 0.001
        """,
    )
    config = load_config(config_path)
    assert _model_name(config) == "test"
    assert config.training.lr == pytest.approx(0.001)


def test_load_unknown_keys_ignored(tmp_path: Path) -> None:
    """Unknown keys in YAML should be ignored without raising errors."""
    config_path = _write_yaml(
        tmp_path / "unknown.yaml",
        """
        model:
          model_name: gpt2
          unknown_field: 123
        """,
    )
    config = load_config(config_path)
    assert _model_name(config) == "gpt2"
    assert not hasattr(config.model, "unknown_field")


def test_load_missing_sections(
    tmp_path: Path,
    small_model_config,
) -> None:
    """Missing sections should be filled by BaseConfig defaults."""
    updated_embedding_dim = small_model_config.embedding_dim + 1
    config_path = _write_yaml(
        tmp_path / "missing_sections.yaml",
        f"""
        model:
          embedding_dim: {updated_embedding_dim}
        """,
    )
    config = load_config(config_path)
    defaults = BaseConfig()
    assert config.model.embedding_dim == updated_embedding_dim
    assert config.training.lr == defaults.training.lr
    assert config.device.cuda_id == defaults.device.cuda_id


def test_load_empty_yaml(tmp_path: Path) -> None:
    """An empty YAML file should produce a default BaseConfig."""
    config_path = _write_yaml(tmp_path / "empty.yaml", "")
    config = load_config(config_path)
    defaults = BaseConfig()
    assert config.training.batch_size == defaults.training.batch_size
    assert _model_name(config) == _model_name(defaults)


def test_save_and_load_roundtrip(
    tmp_path: Path,
) -> None:
    """save_config + load_config should preserve all customized values."""
    from baselines.core.config import (
        BaseConfig,
        ModelConfig,
        ParallelConfig,
        TrainingConfig,
    )

    config = BaseConfig(
        model=ModelConfig(embedding_dim=96, model_name="test-rt"),
        training=TrainingConfig(lr=5e-4, betas=(0.9, 0.95)),
        parallel=ParallelConfig(pp_size=2),
    )

    config_path = tmp_path / "roundtrip.yaml"
    save_config(config, config_path)

    # No manual patching needed — save_config now uses
    # safe_dump with tuple-to-list sanitization.
    loaded = load_config(config_path)

    assert loaded.model.embedding_dim == 96
    assert loaded.model.model_name == "test-rt"
    assert loaded.training.lr == pytest.approx(5e-4)
    # parallel round-trips via the "parallel" key fallback
    assert loaded.parallel.pp_size == 2


def test_load_parallelism_key(tmp_path: Path) -> None:
    """The YAML key 'parallelism' should map to ParallelConfig values."""
    config_path = _write_yaml(
        tmp_path / "parallelism.yaml",
        """
        parallelism:
          pp_size: 3
          dp_size: 2
        """,
    )
    config = load_config(config_path)
    parallel = getattr(config, "parallel", getattr(config, "parallelism", None))
    assert parallel is not None
    assert parallel.pp_size == 3
