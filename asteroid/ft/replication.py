"""Weight replication for fault tolerance.

Supports three replication modes:
  - **topology**: backup to the next stage's device (ring)
  - **local**: CPU-side replica per device  
  - **global**: full global copies (highest memory, fastest recovery)
  - **all**: combine topology + local + global  
  - **none**: no replication
"""
from __future__ import annotations

import logging
from typing import Any, cast

logger = logging.getLogger(__name__)


class WeightReplication:
    """Weight replication across pipeline stages for fault tolerance."""

    VALID_MODES = {"topology", "local", "global", "all", "none"}

    def __init__(self, mode: str = "all") -> None:
        mode = mode.lower()
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")
        self.mode = mode
        self.backup_weights: dict[int, dict[str, object]] = {}
        self.local_replicas: dict[int, dict[str, object]] = {}
        self.global_replicas: dict[int, dict[str, object]] = {}
        self._backup_targets: dict[int, int] = {}
        self._redistributed_weights: dict[int, dict[str, object]] = {}

    # ---- helpers ---------------------------------------------------------

    @staticmethod
    def _cpu_clone_state(model: object) -> dict[str, object]:
        model_any = cast(Any, model)
        state_dict = cast(dict[str, Any], model_any.state_dict())
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in state_dict.items()
        }

    # ---- public API -------------------------------------------------------

    def replicate(self, stage_models: dict[int, object]) -> None:
        """Store CPU clones of each stage's weights according to *mode*."""
        if self.mode == "none":
            return

        num_stages = max(1, len(stage_models))
        for stage_idx, model in stage_models.items():
            state = self._cpu_clone_state(model)
            if self.mode in ("topology", "all"):
                backup_stage = (stage_idx + 1) % num_stages
                self.backup_weights[stage_idx] = state
                self._backup_targets[stage_idx] = backup_stage
            if self.mode in ("local", "all"):
                self.local_replicas[stage_idx] = state
            if self.mode in ("global", "all"):
                self.global_replicas[stage_idx] = state

        logger.info(
            "%s replication complete for %s stage(s)",
            self.mode.upper(), len(stage_models),
        )

    def restore(self, failed_stage: int) -> dict[str, object] | None:
        """Restore weights for a failed stage from whichever replica exists."""
        for store in (self.backup_weights, self.local_replicas, self.global_replicas):
            if failed_stage in store:
                return store[failed_stage]
        return None

    def redistribute(
        self,
        failed_devices: list[int],
        surviving_models: dict[int, object],
    ) -> dict[int, dict[str, object]]:
        """Collect surviving weights and restore failed stages."""
        collected: dict[int, dict[str, object]] = {}
        for device_id, model in surviving_models.items():
            collected[device_id] = self._cpu_clone_state(model)

        for fd in failed_devices:
            restored = self.restore(fd)
            if restored is None:
                logger.warning("No replicated weights for failed stage %s", fd)
                continue
            collected[fd] = {
                k: cast(Any, v).detach().cpu().clone()
                for k, v in restored.items()
            }

        self._redistributed_weights = collected
        return collected


__all__ = ["WeightReplication"]
