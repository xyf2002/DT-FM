from __future__ import annotations

import logging
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


class PassiveTimeoutDetector:
    """Backward-timeout failure detection (from Confident)."""

    timeout_ms: float
    _missing_since_ms: dict[int, float]

    def __init__(self, timeout_ms: float = 30000.0) -> None:
        self.timeout_ms = timeout_ms
        self._missing_since_ms: dict[int, float] = {}

    def detect_failure(
        self,
        iter_id: int,
        received_iter_ids: set[int] | list[int],
    ) -> bool:
        now_ms = time.monotonic() * 1000.0
        if iter_id in received_iter_ids:
            _ = self._missing_since_ms.pop(iter_id, None)
            return False

        first_missing_ms = self._missing_since_ms.setdefault(iter_id, now_ms)
        if now_ms - first_missing_ms >= self.timeout_ms:
            logger.warning(
                "Backward timeout for iter %s after %.1fms",
                iter_id,
                self.timeout_ms,
            )
            return True
        return False

    def handle_timeout(
        self,
        iter_id: int,
        callback: Callable[[int], None] | None = None,
    ) -> None:
        _ = self._missing_since_ms.pop(iter_id, None)
        if callback is not None:
            callback(iter_id)


__all__ = ["PassiveTimeoutDetector"]
