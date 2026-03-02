"""NVIDIA MPS (Multi-Process Service) lifecycle manager.

Provides per-GPU MPS daemon start/stop/check, plus module-level
convenience functions backed by a singleton ``MPSManager``.

Typical usage in a launcher::

    from baselines.utils.mps import start_mps, stop_all_mps
    from baselines.core.config import DeviceConfig

    cfg = DeviceConfig(mps_enabled=True, mps_active_thread_percentage=50)
    for gpu in range(4):
        start_mps(gpu, cfg)
    try:
        # ... spawn workers ...
    finally:
        stop_all_mps()

Workers inject client env vars before any CUDA call::

    from baselines.utils.mps import get_mps_client_env
    env = get_mps_client_env(gpu_id, cfg)
    for k, v in env.items():
        os.environ[k] = v
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from typing import Dict, Optional

from baselines.core.config import DeviceConfig

logger = logging.getLogger(__name__)

_PID_FILENAME = "nvidia-mps.pid"


def _resolve_pipe_dir(
    gpu_id: int, config: DeviceConfig
) -> str:
    if config.mps_pipe_directory:
        return f"{config.mps_pipe_directory}-{gpu_id}"
    return f"/tmp/nvidia-mps-{gpu_id}"


def _resolve_log_dir(
    gpu_id: int, config: DeviceConfig
) -> str:
    if config.mps_log_directory:
        return f"{config.mps_log_directory}-{gpu_id}"
    return f"/tmp/nvidia-log-{gpu_id}"


class MPSManager:
    """Manage one NVIDIA MPS daemon per GPU.

    Tracks started daemons so ``stop_all_mps`` can tear them
    down reliably.  All subprocess calls target
    ``nvidia-cuda-mps-control``.
    """

    def __init__(self) -> None:
        # gpu_id -> {"pipe_dir": ..., "log_dir": ...}
        self._managed: Dict[int, Dict[str, str]] = {}

    # ── Public API ────────────────────────────────────────

    def start_mps(
        self,
        gpu_id: int,
        config: DeviceConfig,
    ) -> None:
        """Start MPS daemon for *gpu_id*.  Idempotent."""
        pipe_dir = _resolve_pipe_dir(gpu_id, config)
        log_dir = _resolve_log_dir(gpu_id, config)

        if self.is_mps_running(gpu_id, pipe_dir):
            logger.info(
                "MPS already running for GPU %d (%s)",
                gpu_id,
                pipe_dir,
            )
            self._managed[gpu_id] = {
                "pipe_dir": pipe_dir,
                "log_dir": log_dir,
            }
            return

        os.makedirs(pipe_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["CUDA_MPS_PIPE_DIRECTORY"] = pipe_dir
        env["CUDA_MPS_LOG_DIRECTORY"] = log_dir
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(
            config.mps_active_thread_percentage
        )

        logger.info(
            "Starting MPS daemon for GPU %d "
            "(pipe=%s, log=%s, thread_pct=%d)",
            gpu_id,
            pipe_dir,
            log_dir,
            config.mps_active_thread_percentage,
        )

        subprocess.run(
            ["nvidia-cuda-mps-control", "-d"],
            env=env,
            check=True,
            capture_output=True,
        )

        # Wait for PID file to appear.
        self._wait_for_pid(pipe_dir, appear=True)

        self._managed[gpu_id] = {
            "pipe_dir": pipe_dir,
            "log_dir": log_dir,
        }

        # Optionally set pinned device memory limit.
        if config.mps_pinned_device_mem_limit:
            self._send_command(
                pipe_dir,
                log_dir,
                (
                    "set_default_device_pinned_mem_limit"
                    f" 0 {config.mps_pinned_device_mem_limit}"
                ),
            )

        logger.info(
            "MPS daemon started for GPU %d", gpu_id
        )

    def stop_mps(self, gpu_id: int) -> None:
        """Stop MPS daemon for *gpu_id*.  No-op if not running."""
        info = self._managed.get(gpu_id)
        if info is None:
            pipe_dir = f"/tmp/nvidia-mps-{gpu_id}"
            log_dir = f"/tmp/nvidia-log-{gpu_id}"
        else:
            pipe_dir = info["pipe_dir"]
            log_dir = info["log_dir"]

        if not self.is_mps_running(gpu_id, pipe_dir):
            self._managed.pop(gpu_id, None)
            return

        logger.info(
            "Stopping MPS daemon for GPU %d (%s)",
            gpu_id,
            pipe_dir,
        )

        try:
            self._send_command(pipe_dir, log_dir, "quit")
        except subprocess.CalledProcessError:
            pass  # best-effort

        try:
            self._wait_for_pid(
                pipe_dir, appear=False, timeout=10.0
            )
        except TimeoutError:
            # Force-kill as last resort.
            self._force_kill(pipe_dir)

        self._managed.pop(gpu_id, None)
        logger.info(
            "MPS daemon stopped for GPU %d", gpu_id
        )

    def is_mps_running(
        self,
        gpu_id: int,
        pipe_dir: Optional[str] = None,
    ) -> bool:
        """Check PID-file liveness for the MPS daemon."""
        if pipe_dir is None:
            info = self._managed.get(gpu_id)
            if info is not None:
                pipe_dir = info["pipe_dir"]
            else:
                pipe_dir = f"/tmp/nvidia-mps-{gpu_id}"
        pid_path = os.path.join(pipe_dir, _PID_FILENAME)
        if not os.path.exists(pid_path):
            return False
        try:
            with open(pid_path) as fh:
                pid = int(fh.read().strip())
            os.kill(pid, 0)  # signal-0 = existence check
            return True
        except (
            ValueError,
            ProcessLookupError,
            PermissionError,
            OSError,
        ):
            return False

    def stop_all_mps(self) -> None:
        """Stop all managed MPS daemons."""
        for gpu_id in list(self._managed):
            self.stop_mps(gpu_id)
        self._managed.clear()

    def get_client_env(
        self, gpu_id: int
    ) -> Dict[str, str]:
        """Return env vars a worker needs to connect to MPS.

        IMPORTANT: ``CUDA_VISIBLE_DEVICES`` is intentionally
        **not** set — it must remain unset in client
        processes when MPS is active (the daemon controls
        device routing).
        """
        info = self._managed.get(gpu_id)
        if info is not None:
            return {
                "CUDA_MPS_PIPE_DIRECTORY": info[
                    "pipe_dir"
                ],
                "CUDA_MPS_LOG_DIRECTORY": info["log_dir"],
            }
        return {
            "CUDA_MPS_PIPE_DIRECTORY": (
                f"/tmp/nvidia-mps-{gpu_id}"
            ),
            "CUDA_MPS_LOG_DIRECTORY": (
                f"/tmp/nvidia-log-{gpu_id}"
            ),
        }

    # ── Context manager ───────────────────────────────────

    def __enter__(self) -> "MPSManager":
        return self

    def __exit__(self, *_: object) -> None:
        self.stop_all_mps()

    # ── Internal helpers ──────────────────────────────────

    @staticmethod
    def _wait_for_pid(
        pipe_dir: str,
        appear: bool = True,
        timeout: float = 10.0,
    ) -> None:
        """Poll until PID file appears/disappears."""
        pid_path = os.path.join(pipe_dir, _PID_FILENAME)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            exists = os.path.exists(pid_path)
            if appear and exists:
                return
            if not appear and not exists:
                return
            time.sleep(0.1)
        action = "appear" if appear else "disappear"
        raise TimeoutError(
            f"MPS PID file did not {action} within"
            f" {timeout}s: {pid_path}"
        )

    @staticmethod
    def _send_command(
        pipe_dir: str,
        log_dir: str,
        cmd: str,
    ) -> str:
        env = os.environ.copy()
        env["CUDA_MPS_PIPE_DIRECTORY"] = pipe_dir
        env["CUDA_MPS_LOG_DIRECTORY"] = log_dir
        result = subprocess.run(
            ["nvidia-cuda-mps-control"],
            input=cmd + "\n",
            env=env,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    @staticmethod
    def _force_kill(pipe_dir: str) -> None:
        pid_path = os.path.join(pipe_dir, _PID_FILENAME)
        try:
            with open(pid_path) as fh:
                pid = int(fh.read().strip())
            os.kill(pid, signal.SIGKILL)
            logger.warning(
                "Force-killed MPS daemon (pid=%d)", pid
            )
        except Exception:
            pass


# ── Module-level singleton ────────────────────────────────

_MANAGER = MPSManager()


def start_mps(gpu_id: int, config: DeviceConfig) -> None:
    """Start MPS daemon for *gpu_id* (singleton)."""
    _MANAGER.start_mps(gpu_id, config)


def stop_mps(gpu_id: int) -> None:
    """Stop MPS daemon for *gpu_id* (singleton)."""
    _MANAGER.stop_mps(gpu_id)


def stop_all_mps() -> None:
    """Stop all managed MPS daemons (singleton)."""
    _MANAGER.stop_all_mps()


def is_mps_running(gpu_id: int) -> bool:
    """Check if MPS daemon is active for *gpu_id*."""
    return _MANAGER.is_mps_running(gpu_id)


def get_mps_client_env(
    gpu_id: int, config: DeviceConfig
) -> Dict[str, str]:
    """Return env dict for a worker connecting to MPS.

    Resolves pipe/log directories from *config* or
    from the singleton manager's tracked state.
    Does NOT include ``CUDA_VISIBLE_DEVICES``.
    """
    # Prefer tracked state from a prior start_mps call.
    env = _MANAGER.get_client_env(gpu_id)
    if env["CUDA_MPS_PIPE_DIRECTORY"] != (
        f"/tmp/nvidia-mps-{gpu_id}"
    ):
        return env
    # Fall back to config resolution.
    return {
        "CUDA_MPS_PIPE_DIRECTORY": _resolve_pipe_dir(
            gpu_id, config
        ),
        "CUDA_MPS_LOG_DIRECTORY": _resolve_log_dir(
            gpu_id, config
        ),
    }


__all__ = [
    "MPSManager",
    "start_mps",
    "stop_mps",
    "stop_all_mps",
    "is_mps_running",
    "get_mps_client_env",
]
