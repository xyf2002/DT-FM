"""YAML config loader that maps to BaseConfig dataclasses."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TypeVar, cast

from baselines.core.config import (
    BaseConfig,
    DeviceConfig,
    DistributedConfig,
    FaultToleranceConfig,
    ModelConfig,
    ParallelConfig,
    TrainingConfig,
)

logger = logging.getLogger(__name__)
_DataclassT = TypeVar("_DataclassT")


def load_config(path: str | Path) -> BaseConfig:
    """Load a YAML config file and return a BaseConfig."""
    import yaml

    path = Path(path)
    with open(path, encoding="utf-8") as f:
        loaded_obj = cast(object, yaml.safe_load(f))
    raw: dict[str, object] = {}
    if isinstance(loaded_obj, dict):
        for key, value in cast(dict[object, object], loaded_obj).items():
            if isinstance(key, str):
                raw[key] = value
    return _parse_config(raw)


def _parse_config(raw: dict[str, object]) -> BaseConfig:
    """Parse raw YAML dict into BaseConfig."""
    # Accept both "parallelism" (YAML convention) and
    # "parallel" (dataclass attribute name from asdict).
    parallel_raw = raw.get("parallelism", raw.get("parallel", {}))
    return BaseConfig(
        device=_build_dataclass(DeviceConfig, raw.get("device", {})),
        distributed=_build_dataclass(DistributedConfig, raw.get("distributed", {})),
        model=_build_dataclass(ModelConfig, raw.get("model", {})),
        training=_build_dataclass(TrainingConfig, raw.get("training", {})),
        parallel=_build_dataclass(ParallelConfig, parallel_raw),
        fault_tolerance=_build_dataclass(
            FaultToleranceConfig,
            raw.get("fault_tolerance", {}),
        ),
    )


def _build_dataclass(
    cls: type[_DataclassT],
    data: object,
) -> _DataclassT:
    """Build a dataclass from a dict, ignoring unknown keys."""
    if not isinstance(data, dict):
        return cls()
    dataclass_fields = getattr(cls, "__dataclass_fields__", {})
    valid_fields = set(dataclass_fields.keys())
    filtered: dict[str, object] = {}
    for key, value in cast(dict[object, object], data).items():
        if isinstance(key, str) and key in valid_fields:
            filtered[key] = value
    return cls(**filtered)


def save_config(config: BaseConfig, path: str | Path) -> None:
    """Save a BaseConfig to a YAML file."""
    import dataclasses
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _sanitize_for_yaml(dataclasses.asdict(config))
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, default_flow_style=False, sort_keys=False)


def _sanitize_for_yaml(obj: object) -> object:
    """Convert tuples to lists so yaml.safe_dump can serialize them."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_yaml(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_yaml(item) for item in obj]
    return obj


__all__ = ["load_config", "save_config"]
