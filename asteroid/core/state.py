import threading
from typing import Optional, List, Dict
import torch
from .config import HPPPlanConfig

class AsteroidStateManager:
    """Runtime state for each worker — reuses DT-FM StateManager pattern."""

    def __init__(self):
        self._global_rank: int = 0
        self._stage_idx: int = 0
        self._device_group: List[int] = []
        self._dp_rank: int = 0     # rank within device group
        self._dp_size: int = 1     # size of device group
        self._device: torch.device = torch.device("cpu")
        self._plan: Optional[HPPPlanConfig] = None
        self._system_status: str = "NORMAL"
        self._lock = threading.Lock()
        # ── Confident FT state tracking ──
        self._received_iter_ids: set = set()
        self._partition_point: List[int] = []
        self._workers: Dict[int, str] = {}  # device_id → address
        self._backward_timeout_ms: float = 30000.0

    # Getters/setters (DT-FM pattern)
    @property
    def global_rank(self): return self._global_rank
    @global_rank.setter
    def global_rank(self, v): self._global_rank = v

    @property
    def stage_idx(self): return self._stage_idx
    @stage_idx.setter
    def stage_idx(self, v): self._stage_idx = v

    @property
    def device(self): return self._device
    @device.setter
    def device(self, v): self._device = v

    @property
    def plan(self): return self._plan
    @plan.setter
    def plan(self, v): self._plan = v

    @property
    def system_status(self):
        with self._lock:
            return self._system_status
    @system_status.setter
    def system_status(self, v):
        with self._lock:
            self._system_status = v

    def record_backward(self, iter_id: int):
        """Record that backward was received for this iteration."""
        with self._lock:
            self._received_iter_ids.add(iter_id)

    def get_received_iter_ids(self) -> set:
        with self._lock:
            return set(self._received_iter_ids)

    def get_partition_point(self) -> List[int]:
        with self._lock:
            return list(self._partition_point)

    def set_partition_point(self, pts: List[int]):
        with self._lock:
            self._partition_point = list(pts)

    def get_workers(self) -> Dict[int, str]:
        with self._lock:
            return dict(self._workers)

    def set_workers(self, w: Dict[int, str]):
        with self._lock:
            self._workers = dict(w)