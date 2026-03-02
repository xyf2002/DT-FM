from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class TrainingEvent:
    """Single timed event emitted during training."""

    timestamp: float
    device_id: int
    event_type: str
    iter_id: int
    phase: str
    duration_ms: float
    metadata: dict[str, object] = field(default_factory=dict)


class EventLogger:
    """Thread-safe event logger with Chrome trace and summary exports."""

    def __init__(self, enabled: bool = True) -> None:
        self._events: list[TrainingEvent] = []
        self._lock: threading.Lock = threading.Lock()
        self._epoch_start_time: float = 0.0
        self._enabled: bool = enabled

    def set_epoch_start(self, timestamp: float) -> None:
        self._epoch_start_time = timestamp

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @contextmanager
    def log_event(
        self,
        device_id: int,
        event_type: str,
        iter_id: int,
        phase: str = "",
        **metadata: object,
    ) -> Iterator[None]:
        if not self._enabled:
            yield
            return

        start = time.time()
        yield
        duration_ms = (time.time() - start) * 1000
        event = TrainingEvent(
            timestamp=start,
            device_id=device_id,
            event_type=event_type,
            iter_id=iter_id,
            phase=phase,
            duration_ms=duration_ms,
            metadata=metadata,
        )
        with self._lock:
            self._events.append(event)

    def record_event(
        self,
        device_id: int,
        event_type: str,
        iter_id: int,
        phase: str,
        duration_ms: float,
        timestamp: float | None = None,
        **metadata: object,
    ) -> None:
        if not self._enabled:
            return

        event = TrainingEvent(
            timestamp=timestamp or time.time(),
            device_id=device_id,
            event_type=event_type,
            iter_id=iter_id,
            phase=phase,
            duration_ms=duration_ms,
            metadata=metadata,
        )
        with self._lock:
            self._events.append(event)

    def get_events(self) -> list[TrainingEvent]:
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    def to_chrome_trace(self, filepath: str = "trace.json") -> None:
        events = self.get_events()
        t0 = self._epoch_start_time or (
            min(event.timestamp for event in events) if events else 0
        )

        trace_events: list[dict[str, object]] = []
        for event in events:
            trace_events.append(
                {
                    "name": event.event_type,
                    "ph": "X",
                    "pid": event.device_id,
                    "tid": event.event_type,
                    "ts": (event.timestamp - t0) * 1e6,
                    "dur": event.duration_ms * 1000,
                    "args": {
                        "micro-batch": event.iter_id,
                        **event.metadata,
                    },
                }
            )

        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(trace_events, handle, indent=2)

    def get_summary(
        self,
    ) -> dict[tuple[int, str], dict[str, int | float]]:
        stats: dict[tuple[int, str], dict[str, int | float]] = {}

        for event in self.get_events():
            key = (event.device_id, event.event_type)
            if key not in stats:
                stats[key] = {
                    "count": 0,
                    "total_ms": 0.0,
                    "min_ms": float("inf"),
                    "max_ms": 0.0,
                    "avg_ms": 0.0,
                }
            summary = stats[key]
            summary["count"] = int(summary["count"]) + 1
            summary["total_ms"] = float(summary["total_ms"]) + event.duration_ms
            summary["min_ms"] = min(float(summary["min_ms"]), event.duration_ms)
            summary["max_ms"] = max(float(summary["max_ms"]), event.duration_ms)

        for summary in stats.values():
            count = int(summary["count"])
            total = float(summary["total_ms"])
            summary["avg_ms"] = total / count if count else 0.0

        return stats


__all__ = ["EventLogger", "TrainingEvent"]
