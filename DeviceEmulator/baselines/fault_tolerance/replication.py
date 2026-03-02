from __future__ import annotations

import logging
from typing import Any, cast

logger = logging.getLogger(__name__)


class WeightReplication:
    """Weight replication across devices for fault tolerance."""

    mode: str
    backup_weights: dict[int, dict[str, object]]
    local_replicas: dict[int, dict[str, object]]
    global_replicas: dict[int, dict[str, object]]
    _backup_targets: dict[int, int]
    _redistributed_weights: dict[int, dict[str, object]]

    def __init__(self, mode: str = "topology") -> None:
        mode_name = mode.lower()
        if mode_name not in {"topology", "local", "global"}:
            raise ValueError("mode must be one of: topology, local, global")

        self.mode = mode_name
        self.backup_weights = {}
        self.local_replicas = {}
        self.global_replicas = {}
        self._backup_targets = {}
        self._redistributed_weights = {}

    def _cpu_clone_state(self, model: object) -> dict[str, object]:
        model_any = cast(Any, model)
        state_dict = cast(dict[str, Any], model_any.state_dict())
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in state_dict.items()
        }

    def replicate(self, stage_models: dict[int, object]) -> None:
        num_stages = max(1, len(stage_models))
        for stage_idx, model in stage_models.items():
            state_dict = self._cpu_clone_state(model)
            if self.mode == "topology":
                backup_stage = (stage_idx + 1) % num_stages
                self.backup_weights[stage_idx] = state_dict
                self._backup_targets[stage_idx] = backup_stage
            elif self.mode == "local":
                self.local_replicas[stage_idx] = state_dict
            else:
                self.global_replicas[stage_idx] = state_dict

        logger.info(
            "%s replication complete for %s devices",
            self.mode.upper(),
            len(stage_models),
        )

    def restore(self, failed_stage: int) -> dict[str, object] | None:
        if failed_stage in self.backup_weights:
            return self.backup_weights[failed_stage]
        if failed_stage in self.local_replicas:
            return self.local_replicas[failed_stage]
        if failed_stage in self.global_replicas:
            return self.global_replicas[failed_stage]
        return None

    def redistribute(
        self,
        failed_devices: list[int],
        surviving_models: dict[int, object],
    ) -> dict[int, dict[str, object]]:
        collected: dict[int, dict[str, object]] = {}
        for device_id, model in surviving_models.items():
            collected[device_id] = self._cpu_clone_state(model)

        for failed_device in failed_devices:
            restored = self.restore(failed_device)
            if restored is None:
                logger.warning(
                    "No replicated weights found for failed stage %s",
                    failed_device,
                )
                continue

            collected[failed_device] = {
                key: cast(Any, value).detach().cpu().clone()
                for key, value in restored.items()
            }

        self._redistributed_weights = collected
        return collected


__all__ = ["WeightReplication"]
