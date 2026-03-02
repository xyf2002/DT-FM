"""Active heartbeat + passive timeout failure detection.

``HeartbeatDetector`` sends periodic timestamps via
``torch.distributed`` store; ``PassiveTimeoutDetector`` fires
when a backward gradient is not received within a deadline.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any, cast

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Active heartbeat detector (via torch.distributed store)
# ---------------------------------------------------------------------------

class HeartbeatDetector:
    """Active heartbeat sender / checker using the distributed store."""

    def __init__(
        self,
        device_id: int,
        interval_s: float = 2.0,
        timeout_s: float = 8.0,
    ) -> None:
        self.device_id = device_id
        self.interval_s = interval_s
        self.timeout_s = timeout_s

        self._dist_store: object | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Fallback for non-distributed testing
        self._local_heartbeats: dict[int, float] = {}
        self._local_lock = threading.Lock()

    # ---- helpers ---------------------------------------------------------

    @staticmethod
    def _default_store() -> object | None:
        try:
            import torch.distributed as dist
        except ImportError:
            return None
        if not dist.is_available() or not dist.is_initialized():
            return None
        try:
            return dist.distributed_c10d._get_default_store()
        except Exception:
            return None

    # ---- public API -------------------------------------------------------

    def start(self, dist_store: object | None = None) -> None:
        """Start the heartbeat sender thread."""
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return

        self._dist_store = dist_store or self._default_store()
        self._stop_event.clear()

        def _beat() -> None:
            while not self._stop_event.is_set():
                ts = time.time()
                ts_bytes = str(ts).encode("utf-8")
                if self._dist_store is not None:
                    cast(Any, self._dist_store).set(
                        f"hb_{self.device_id}", ts_bytes,
                    )
                else:
                    with self._local_lock:
                        self._local_heartbeats[self.device_id] = ts
                self._stop_event.wait(self.interval_s)

        self._heartbeat_thread = threading.Thread(target=_beat, daemon=True)
        self._heartbeat_thread.start()
        logger.debug("Heartbeat sender started for device %s", self.device_id)

    def stop(self) -> None:
        """Stop the heartbeat sender thread."""
        self._stop_event.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)
        logger.debug("Heartbeat sender stopped for device %s", self.device_id)

    def check_alive(self, target_device_id: int) -> bool:
        """Return ``True`` if *target_device_id* has sent a heartbeat
        within ``self.timeout_s`` seconds."""
        if self._dist_store is not None:
            try:
                ts_bytes = cast(Any, self._dist_store).get(
                    f"hb_{target_device_id}",
                )
                last_seen = float(ts_bytes.decode("utf-8"))
                return (time.time() - last_seen) < self.timeout_s
            except Exception:
                return True  # key not yet written → assume alive
        with self._local_lock:
            last_seen_local = self._local_heartbeats.get(target_device_id)
        if last_seen_local is None:
            return True
        return (time.time() - last_seen_local) < self.timeout_s


# ---------------------------------------------------------------------------
# Passive backward-timeout detector (from Confident)
# ---------------------------------------------------------------------------

class PassiveTimeoutDetector:
    """Backward-timeout failure detection (Confident-style)."""

    def __init__(self, timeout_ms: float = 30_000.0) -> None:
        self.timeout_ms = timeout_ms
        self._missing_since_ms: dict[int, float] = {}

    def detect_failure(
        self,
        iter_id: int,
        received_iter_ids: set[int] | list[int],
    ) -> bool:
        """Return ``True`` if *iter_id* has been missing longer than
        ``self.timeout_ms``."""
        now_ms = time.monotonic() * 1000.0
        if iter_id in received_iter_ids:
            self._missing_since_ms.pop(iter_id, None)
            return False

        first_missing_ms = self._missing_since_ms.setdefault(iter_id, now_ms)
        if now_ms - first_missing_ms >= self.timeout_ms:
            logger.warning(
                "Backward timeout for iter %s after %.1fms",
                iter_id, self.timeout_ms,
            )
            return True
        return False

    def handle_timeout(
        self,
        iter_id: int,
        callback: Callable[[int], None] | None = None,
    ) -> None:
        """Acknowledge the timeout and optionally invoke *callback*."""
        self._missing_since_ms.pop(iter_id, None)
        if callback is not None:
            callback(iter_id)


__all__ = ["HeartbeatDetector", "PassiveTimeoutDetector"]
