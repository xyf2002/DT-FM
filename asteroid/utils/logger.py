import time
import json
import threading
from typing import Dict, Any, List
from collections import defaultdict
from dataclasses import dataclass, field

@dataclass
class TrainingEvent:
    timestamp: float
    device_id: int
    event_type: str
    iter_id: int
    phase: str
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class AsteroidEventLogger:
    """Thread-safe event logger with Chrome trace export."""

    def __init__(self):
        self._events: List[TrainingEvent] = []
        self._lock = threading.Lock()
        self._epoch_start: float = 0.0

    def set_epoch_start(self, t): self._epoch_start = t

    class _ContextManager:
        def __init__(self, logger, device_id, event_type, iter_id, phase, metadata):
            self.logger = logger
            self.device_id = device_id
            self.event_type = event_type
            self.iter_id = iter_id
            self.phase = phase
            self.metadata = metadata

        def __enter__(self):
            self.start = time.time()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            dur = (time.time() - self.start) * 1000
            ev = TrainingEvent(self.start, self.device_id, self.event_type, 
                               self.iter_id, self.phase, dur, self.metadata)
            with self.logger._lock:
                self.logger._events.append(ev)

    def log_event(self, device_id, event_type, iter_id, phase="", **metadata):
        return self._ContextManager(self, device_id, event_type, iter_id, phase, metadata)

    def record_event(self, device_id, event_type, iter_id, phase, duration_ms,
                     timestamp=None, **meta):
        ev = TrainingEvent(timestamp or time.time(), device_id, event_type,
                           iter_id, phase, duration_ms, meta)
        with self._lock:
            self._events.append(ev)

    def to_chrome_trace(self, filepath):
        t0 = self._epoch_start or (
            min(e.timestamp for e in self._events) if self._events else 0)
        trace = [{"name": e.event_type, "ph": "X", "pid": e.device_id,
                  "tid": e.event_type, "ts": (e.timestamp - t0) * 1e6,
                  "dur": e.duration_ms * 1000,
                  "args": {"micro-batch": e.iter_id, **e.metadata}}
                 for e in self._events]
        with open(filepath, 'w') as f:
            json.dump(trace, f, indent=2)

    def get_summary(self) -> Dict:
        stats = defaultdict(lambda: {"count": 0, "total_ms": 0.0})
        for e in self._events:
            k = (e.device_id, e.event_type)
            stats[k]["count"] += 1
            stats[k]["total_ms"] += e.duration_ms
        return dict(stats)

# Global singleton logger
EVENT_LOGGER = AsteroidEventLogger()

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger("Asteroid")