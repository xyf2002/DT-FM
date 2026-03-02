from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

from baselines.core.fault_tolerance import CheckpointStrategy

logger = logging.getLogger(__name__)


def _torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError("torch is required for checkpointing.") from exc


class BasicCheckpoint(CheckpointStrategy):
    """Synchronous torch.save/load checkpoint strategy."""

    checkpoint_dir: Path
    interval: int

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        interval: int = 100,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.interval = max(1, interval)

    def _resolve_path(self, path: str) -> Path:
        raw_path = Path(path)
        if raw_path.is_absolute():
            return raw_path
        return self.checkpoint_dir / raw_path

    def save(self, state: dict[str, object], path: str) -> None:
        torch = _torch()
        full_path = self._resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, full_path)
        logger.debug("Saved checkpoint to %s", full_path)

    def load(self, path: str) -> dict[str, object]:
        torch = _torch()
        full_path = self._resolve_path(path)
        logger.debug("Loading checkpoint from %s", full_path)
        return torch.load(full_path, map_location="cpu")

    def should_checkpoint(self, iter_id: int) -> bool:
        return iter_id % self.interval == 0


__all__ = ["BasicCheckpoint"]
