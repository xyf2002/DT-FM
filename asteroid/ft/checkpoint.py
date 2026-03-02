"""Checkpoint strategies — async (MegaScale) and synchronous.

Both implement the ``CheckpointStrategy`` ABC from ``asteroid.core.interfaces``.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, cast

import torch

from ..core.interfaces import CheckpointStrategy as _CheckpointABC

logger = logging.getLogger(__name__)


class AsyncCheckpoint(_CheckpointABC):
    """Async CPU-offload checkpoint (MegaScale-inspired).

    Clones tensors to CPU in the foreground, then writes them to disk
    in a background thread so that training can continue immediately.
    """

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        interval: int = 100,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.interval = max(1, interval)

        self._lock = threading.Lock()
        self._pending_save: threading.Thread | None = None
        self._pending_error: Exception | None = None

    # ---- internal helpers ------------------------------------------------

    def _resolve_path(self, path: str) -> Path:
        raw = Path(path)
        return raw if raw.is_absolute() else self.checkpoint_dir / raw

    @staticmethod
    def _clone_to_cpu(value: object) -> object:
        val = cast(Any, value)
        if hasattr(val, "detach") and hasattr(val, "cpu"):
            return val.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: AsyncCheckpoint._clone_to_cpu(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            cloned = [AsyncCheckpoint._clone_to_cpu(v) for v in value]
            return type(value)(cloned)
        return value

    def _collect_error(self) -> None:
        with self._lock:
            error = self._pending_error
            self._pending_error = None
        if error is not None:
            raise RuntimeError("Background checkpoint save failed.") from error

    def _wait_pending(self) -> None:
        with self._lock:
            pending = self._pending_save
        if pending is not None and pending.is_alive():
            pending.join()
        with self._lock:
            if self._pending_save is pending:
                self._pending_save = None
        self._collect_error()

    # ---- CheckpointStrategy interface ------------------------------------

    def save(self, state: dict[str, object], path: str) -> None:
        with self._lock:
            if self._pending_save is not None and self._pending_save.is_alive():
                logger.warning("Skipping checkpoint; previous save still running.")
                return

        self._collect_error()

        full_path = self._resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        cpu_state = self._clone_to_cpu(state)

        def _save_worker(snapshot: object, out: Path) -> None:
            try:
                torch.save(snapshot, out)
                logger.debug("Async checkpoint saved to %s", out)
            except Exception as exc:
                with self._lock:
                    self._pending_error = exc
            finally:
                with self._lock:
                    self._pending_save = None

        worker = threading.Thread(
            target=_save_worker, args=(cpu_state, full_path), daemon=True,
        )
        with self._lock:
            self._pending_save = worker
        worker.start()

    def load(self, path: str) -> dict[str, object]:
        self._wait_pending()
        full_path = self._resolve_path(path)
        logger.debug("Loading checkpoint from %s", full_path)
        return torch.load(full_path, map_location="cpu")  # type: ignore[return-value]

    def should_checkpoint(self, iter_id: int) -> bool:
        return iter_id > 0 and iter_id % self.interval == 0


class BasicCheckpoint(_CheckpointABC):
    """Synchronous ``torch.save`` / ``torch.load`` checkpoint strategy."""

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        interval: int = 100,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.interval = max(1, interval)

    def _resolve_path(self, path: str) -> Path:
        raw = Path(path)
        return raw if raw.is_absolute() else self.checkpoint_dir / raw

    def save(self, state: dict[str, object], path: str) -> None:
        full_path = self._resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, full_path)
        logger.debug("Saved checkpoint to %s", full_path)

    def load(self, path: str) -> dict[str, object]:
        full_path = self._resolve_path(path)
        logger.debug("Loading checkpoint from %s", full_path)
        return torch.load(full_path, map_location="cpu")  # type: ignore[return-value]

    def should_checkpoint(self, iter_id: int) -> bool:
        return iter_id > 0 and iter_id % self.interval == 0


def create_checkpoint_strategy(
    strategy: str = "async",
    checkpoint_dir: str = "./checkpoints",
    interval: int = 100,
) -> _CheckpointABC:
    """Factory for checkpoint strategies."""
    if strategy == "async":
        return AsyncCheckpoint(checkpoint_dir, interval)
    if strategy in ("sync", "basic"):
        return BasicCheckpoint(checkpoint_dir, interval)
    raise ValueError(f"Unknown checkpoint strategy '{strategy}'. Use 'async' or 'sync'.")


__all__ = ["AsyncCheckpoint", "BasicCheckpoint", "create_checkpoint_strategy"]
