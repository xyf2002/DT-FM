from __future__ import annotations

import importlib
import logging
import threading
from pathlib import Path
from typing import Any, cast

from baselines.core.fault_tolerance import CheckpointStrategy

logger = logging.getLogger(__name__)


def _torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError("torch is required for checkpointing.") from exc


class AsyncCheckpoint(CheckpointStrategy):
    """Async CPU offload checkpoint (MegaScale-inspired)."""

    checkpoint_dir: Path
    interval: int
    _lock: threading.Lock
    _pending_save: threading.Thread | None
    _pending_error: Exception | None

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

    def _resolve_path(self, path: str) -> Path:
        raw_path = Path(path)
        if raw_path.is_absolute():
            return raw_path
        return self.checkpoint_dir / raw_path

    def _clone_to_cpu(self, value: object) -> object:
        value_any = cast(Any, value)
        if hasattr(value_any, "detach") and hasattr(value_any, "cpu"):
            return value_any.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: self._clone_to_cpu(val) for key, val in value.items()}
        if isinstance(value, list):
            return [self._clone_to_cpu(val) for val in value]
        if isinstance(value, tuple):
            return tuple(self._clone_to_cpu(val) for val in value)
        return value

    def _collect_error(self) -> None:
        with self._lock:
            error = self._pending_error
            self._pending_error = None
        if error is not None:
            raise RuntimeError("Background checkpoint save failed.") from error

    def _wait_pending_save(self) -> None:
        with self._lock:
            pending = self._pending_save
        if pending is not None and pending.is_alive():
            pending.join()
        with self._lock:
            if self._pending_save is pending:
                self._pending_save = None
        self._collect_error()

    def save(self, state: dict[str, object], path: str) -> None:
        torch = _torch()
        with self._lock:
            if self._pending_save is not None and self._pending_save.is_alive():
                logger.warning(
                    "Skipping checkpoint save; previous save still running."
                )
                return

        self._collect_error()

        full_path = self._resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        cpu_state = self._clone_to_cpu(state)

        def _save_worker(snapshot: object, output_path: Path) -> None:
            try:
                torch.save(snapshot, output_path)
                logger.debug("Asynchronous checkpoint saved to %s", output_path)
            except Exception as exc:
                with self._lock:
                    self._pending_error = exc
            finally:
                with self._lock:
                    self._pending_save = None

        worker = threading.Thread(
            target=_save_worker,
            args=(cpu_state, full_path),
            daemon=True,
        )
        with self._lock:
            self._pending_save = worker
        worker.start()

    def load(self, path: str) -> dict[str, object]:
        torch = _torch()
        self._wait_pending_save()
        full_path = self._resolve_path(path)
        logger.debug("Loading checkpoint from %s", full_path)
        return torch.load(full_path, map_location="cpu")

    def should_checkpoint(self, iter_id: int) -> bool:
        return iter_id % self.interval == 0


__all__ = ["AsyncCheckpoint"]
