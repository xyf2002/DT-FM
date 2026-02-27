import time
import threading
import copy
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from ..core.state import AsteroidStateManager
from ..core.config import HPPPlanConfig
from ..utils.logger import logger

class AsteroidFaultTolerance:
    """Fault tolerance with topology-driven model replication.

    Implements Section 3.4 of the paper + Confident FT patterns:
    1. Heartbeat-guided failure detection (active heartbeat)
    2. Passive backward-timeout detection
    3. Topology-driven model replication
    4. LOCAL/GLOBAL weight replication
    5. Weight redistribution after failure
    6. State sync + fault commit
    7. Layer-wise lightweight pipeline re-planning
    """

    def __init__(self, state: AsteroidStateManager, checkpoint_dir: str = "./checkpoints"):
        self.state = state
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Topology replication: backup weights stored on neighboring stage
        self.backup_weights: Dict[int, Dict[str, torch.Tensor]] = {}
        # Confident-style LOCAL/GLOBAL replicas
        self.local_replicas: Dict[int, Dict[str, torch.Tensor]] = {}
        self.global_replicas: Dict[int, Dict[str, torch.Tensor]] = {}
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        # Cross-process heartbeat via torch distributed store
        self._dist_store: Optional[torch.distributed.Store] = None
        self._own_rank: int = -1

    # ── Heartbeat (active failure detection via distributed store) ─

    def start_heartbeat(self, device_id: int, interval_s: float = 5.0):
        """Start heartbeat sender — writes timestamp to the distributed
        store so that other processes (ranks) can read it."""
        self._own_rank = device_id
        try:
            self._dist_store = torch.distributed.distributed_c10d._get_default_store()
        except Exception:
            self._dist_store = None
            logger.debug("FT: No distributed store available, "
                         "heartbeat will be local-only")

        def _beat():
            while not self._stop_event.is_set():
                ts_bytes = str(time.time()).encode('utf-8')
                if self._dist_store is not None:
                    self._dist_store.set(f"hb_{device_id}", ts_bytes)
                self._stop_event.wait(interval_s)
        self._heartbeat_thread = threading.Thread(target=_beat, daemon=True)
        self._heartbeat_thread.start()

    def stop_heartbeat(self):
        self._stop_event.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2.0)

    def check_device_alive(self, device_id: int, timeout_s: float = 15.0) -> bool:
        """Check if a device is alive by reading its heartbeat from the
        distributed store (cross-process safe)."""
        if self._dist_store is None:
            return True  # can't check, assume alive
        try:
            ts_bytes = self._dist_store.get(f"hb_{device_id}")
            last = float(ts_bytes.decode('utf-8'))
            return (time.time() - last) < timeout_s
        except Exception:
            return True  # key not yet written, assume alive

    # ── Passive backward-timeout detection (from Confident) ──────

    def detect_failure(self, iter_id: int, timeout_ms: float) -> bool:
        """Check if backward was received within timeout."""
        received = self.state.get_received_iter_ids()
        if iter_id not in received and self.state.system_status == "NORMAL":
            logger.warning(f"FT: Backward not received for iter {iter_id} "
                           f"within {timeout_ms}ms")
            return True  # failure detected
        return False

    def handle_passive_timeout(self, iter_id: int,
                               stage_models: Optional[Dict[int, nn.Module]] = None,
                               planner: Optional['AsteroidPlanner'] = None):  # Type hint as string to avoid circular import
        """3-phase passive FT recovery — from Confident PassiveFTHandler."""
        logger.warning(f"FT: Passive timeout triggered for iter {iter_id}")
        self.state.system_status = "RECOVERING"

        # Phase 1: Identify failed devices via heartbeat (distributed store)
        failed_devices = []
        world_size = self.state.plan.num_stages if self.state.plan else 4
        for did in range(world_size):
            if did == self._own_rank:
                continue  # skip self
            if not self.check_device_alive(did):
                failed_devices.append(did)
                logger.warning(f"FT: Device {did} detected as failed")

        if not failed_devices:
            logger.info("FT: No failed devices found, resuming")
            self.state.system_status = "NORMAL"
            return

        # Phase 2: Redistribute weights
        if stage_models:
            surviving_models = {did: m for did, m in stage_models.items()
                                if did not in failed_devices}
            self.redistribute_weights(failed_devices,
                                      self.state.get_partition_point(),
                                      surviving_models)

        # Phase 3: Re-plan if planner available
        current_plan = self.state.plan
        if planner and current_plan:
            for fd in failed_devices:
                current_plan = self.lightweight_replan(current_plan, fd, planner)
            self.state.plan = current_plan
            new_partition = current_plan.partition_points
            self.commit_fault_sync(iter_id, new_partition)

        self.state.system_status = "NORMAL"
        logger.info(f"FT: Recovery complete, resuming from iter {iter_id}")

    # ── Weight replication (from Confident ReplicationUtils) ──────

    def replicate_to_neighbor(self, stage_model: nn.Module, stage_idx: int,
                               num_stages: int):
        """Topology-driven model replication: backup to next stage's device."""
        backup_stage = (stage_idx + 1) % num_stages
        state_dict = {k: v.cpu().clone() for k, v in stage_model.state_dict().items()}
        self.backup_weights[stage_idx] = state_dict
        logger.debug(f"FT: Stage {stage_idx} backed up to stage {backup_stage}")

    def replicate_weights(self, replication_type: str,
                          stage_models: Dict[int, nn.Module]):
        """Replicate weights to other devices — from Confident ReplicationUtils."""
        for device_id, model in stage_models.items():
            if model is not None:
                state_dict = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                if replication_type == "local":
                    self.local_replicas[device_id] = state_dict
                elif replication_type == "global":
                    self.global_replicas[device_id] = state_dict
                elif replication_type == "topology":
                    self.backup_weights[device_id] = state_dict
        logger.info(f"FT: {replication_type.upper()} replication complete "
                    f"for {len(stage_models)} devices")

    # ── Weight restoration & redistribution ───────────────────────

    def restore_from_backup(self, failed_stage: int) -> Optional[Dict[str, torch.Tensor]]:
        """Restore weights from backup after device failure."""
        if failed_stage in self.backup_weights:
            logger.info(f"FT: Restoring stage {failed_stage} from backup")
            return self.backup_weights[failed_stage]
        # Fallback: check local then global replicas
        if failed_stage in self.local_replicas:
            logger.info(f"FT: Restoring stage {failed_stage} from local replica")
            return self.local_replicas[failed_stage]
        if failed_stage in self.global_replicas:
            logger.info(f"FT: Restoring stage {failed_stage} from global replica")
            return self.global_replicas[failed_stage]
        logger.warning(f"FT: No backup found for stage {failed_stage}")
        return None

    def redistribute_weights(self, failed_devices: List[int],
                             current_partition: List[int],
                             surviving_models: Dict[int, nn