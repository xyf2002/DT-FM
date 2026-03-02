from __future__ import annotations

import importlib
import logging
import threading
import time
from typing import Any, cast

logger = logging.getLogger(__name__)


class HeartbeatDetector:
    """Active heartbeat failure detection via distributed store."""

    device_id: int
    interval_s: float
    timeout_s: float
    _dist_store: object | None
    _heartbeat_thread: threading.Thread | None
    _stop_event: threading.Event
    _local_heartbeats: dict[int, float]
    _local_lock: threading.Lock

    def __init__(
        self,
        device_id: int,
        interval_s: float = 5.0,
        timeout_s: float = 15.0,
    ) -> None:
        self.device_id = device_id
        self.interval_s = interval_s
        self.timeout_s = timeout_s

        self._dist_store: object | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._local_heartbeats: dict[int, float] = {}
        self._local_lock = threading.Lock()

    def _default_store(self) -> object | None:
        try:
            dist = importlib.import_module("torch.distributed")
        except ImportError:
            return None

        if not dist.is_available() or not dist.is_initialized():
            return None

        try:
            return dist.distributed_c10d._get_default_store()
        except Exception:
            return None

    def start(self, dist_store: object | None = None) -> None:
        if (
            self._heartbeat_thread is not None
            and self._heartbeat_thread.is_alive()
        ):
            return

        self._dist_store = dist_store if dist_store is not None else None
        if self._dist_store is None:
            self._dist_store = self._default_store()

        self._stop_event.clear()

        def _beat() -> None:
            while not self._stop_event.is_set():
                ts = time.time()
                ts_bytes = str(ts).encode("utf-8")
                if self._dist_store is not None:
                    store = cast(Any, self._dist_store)
                    store.set(f"hb_{self.device_id}", ts_bytes)
                else:
                    with self._local_lock:
                        self._local_heartbeats[self.device_id] = ts
                _ = self._stop_event.wait(self.interval_s)

        self._heartbeat_thread = threading.Thread(target=_beat, daemon=True)
        self._heartbeat_thread.start()
        logger.debug("Heartbeat sender started for device %s", self.device_id)

    def stop(self) -> None:
        self._stop_event.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)
        logger.debug("Heartbeat sender stopped for device %s", self.device_id)

    def check_alive(self, target_device_id: int) -> bool:
        if self._dist_store is not None:
            try:
                store = cast(Any, self._dist_store)
                ts_bytes = store.get(f"hb_{target_device_id}")
                last_seen = float(ts_bytes.decode("utf-8"))
                return (time.time() - last_seen) < self.timeout_s
            except Exception:
                return True

        with self._local_lock:
            last_seen_local = self._local_heartbeats.get(target_device_id)
        if last_seen_local is None:
            return True
        return (time.time() - last_seen_local) < self.timeout_s


__all__ = ["HeartbeatDetector"]
