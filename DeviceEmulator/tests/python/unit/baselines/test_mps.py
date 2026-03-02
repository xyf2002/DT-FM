from __future__ import annotations

import os
import signal
from pathlib import Path
from typing import Callable, cast
from unittest.mock import Mock, call, patch

import pytest

import baselines.utils.mps as mps
from baselines.core.config import DeviceConfig
from baselines.utils.mps import MPSManager

PID_FILENAME = "nvidia-mps.pid"


def _resolve_pipe(gpu_id: int, cfg: DeviceConfig) -> str:
    fn = cast(
        Callable[[int, DeviceConfig], str],
        getattr(mps, "_resolve_pipe_dir"),
    )
    return fn(gpu_id, cfg)


def _resolve_log(gpu_id: int, cfg: DeviceConfig) -> str:
    fn = cast(
        Callable[[int, DeviceConfig], str],
        getattr(mps, "_resolve_log_dir"),
    )
    return fn(gpu_id, cfg)


def _capture_run(
    stdout: str = "",
) -> tuple[Callable[..., Mock], dict[str, object]]:
    record: dict[str, object] = {}

    def _fake_run(cmd: list[str], **kwargs: object) -> Mock:
        record["cmd"] = cmd
        record["kwargs"] = dict(kwargs)
        return Mock(stdout=stdout)

    return _fake_run, record


@pytest.fixture()
def fresh_singleton_manager(monkeypatch: pytest.MonkeyPatch) -> MPSManager:
    manager = MPSManager()
    monkeypatch.setattr(mps, "_MANAGER", manager)
    return manager


def _train_source_text() -> str:
    root = Path(__file__).resolve().parents[4]
    train_path = root / "baselines" / "train.py"
    return train_path.read_text(encoding="utf-8")


def test_device_config_mps_defaults() -> None:
    cfg = DeviceConfig()
    assert cfg.mps_enabled is False
    assert cfg.mps_active_thread_percentage == 100
    assert cfg.mps_pipe_directory == ""
    assert cfg.mps_log_directory == ""
    assert cfg.mps_pinned_device_mem_limit == ""


def test_device_config_validate_accepts_valid_thread_pct() -> None:
    cfg = DeviceConfig(mps_active_thread_percentage=37)
    cfg.validate()


@pytest.mark.parametrize("invalid_pct", [0, 101])
def test_device_config_validate_rejects_invalid_thread_pct(
    invalid_pct: int,
) -> None:
    cfg = DeviceConfig(mps_active_thread_percentage=invalid_pct)
    with pytest.raises(ValueError, match="mps_active_thread_percentage"):
        cfg.validate()


def test_resolve_pipe_dir_defaults() -> None:
    cfg = DeviceConfig()
    assert _resolve_pipe(4, cfg) == "/tmp/nvidia-mps-4"


def test_resolve_pipe_dir_custom_prefix() -> None:
    cfg = DeviceConfig(mps_pipe_directory="/custom/mps-pipe")
    assert _resolve_pipe(1, cfg) == "/custom/mps-pipe-1"


def test_resolve_log_dir_defaults() -> None:
    cfg = DeviceConfig()
    assert _resolve_log(4, cfg) == "/tmp/nvidia-log-4"


def test_resolve_log_dir_custom_prefix() -> None:
    cfg = DeviceConfig(mps_log_directory="/custom/mps-log")
    assert _resolve_log(1, cfg) == "/custom/mps-log-1"


def test_start_mps_invokes_daemon_with_expected_env(
    tmp_path: Path,
) -> None:
    manager = MPSManager()
    cfg = DeviceConfig(
        mps_active_thread_percentage=63,
        mps_pipe_directory=str(tmp_path / "pipe"),
        mps_log_directory=str(tmp_path / "log"),
    )
    pipe_dir = f"{cfg.mps_pipe_directory}-2"
    log_dir = f"{cfg.mps_log_directory}-2"

    fake_run, record = _capture_run()
    with patch.object(manager, "is_mps_running", return_value=False), patch.object(
        manager,
        "_wait_for_pid",
    ) as mock_wait, patch(
        "baselines.utils.mps.subprocess.run",
        side_effect=fake_run,
    ):
        manager.start_mps(2, cfg)

    assert "cmd" in record
    assert "kwargs" in record
    cmd = cast(list[str], record["cmd"])
    kwargs = cast(dict[str, object], record["kwargs"])
    assert cmd == ["nvidia-cuda-mps-control", "-d"]
    assert kwargs["check"] is True
    assert kwargs["capture_output"] is True
    env = cast(dict[str, str], kwargs["env"])
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["CUDA_MPS_PIPE_DIRECTORY"] == pipe_dir
    assert env["CUDA_MPS_LOG_DIRECTORY"] == log_dir
    assert env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "63"
    mock_wait.assert_called_once_with(pipe_dir, appear=True)
    client_env = manager.get_client_env(2)
    assert client_env["CUDA_MPS_PIPE_DIRECTORY"] == pipe_dir
    assert client_env["CUDA_MPS_LOG_DIRECTORY"] == log_dir


def test_start_mps_idempotent_when_already_running() -> None:
    manager = MPSManager()
    cfg = DeviceConfig(mps_pipe_directory="/tmp/custom-pipe")

    with patch.object(manager, "is_mps_running", return_value=True), patch.object(
        manager,
        "_wait_for_pid",
    ) as mock_wait, patch("baselines.utils.mps.subprocess.run") as mock_run:
        manager.start_mps(7, cfg)

    mock_run.assert_not_called()
    mock_wait.assert_not_called()
    client_env = manager.get_client_env(7)
    assert client_env["CUDA_MPS_PIPE_DIRECTORY"] == "/tmp/custom-pipe-7"


def test_start_mps_timeout_when_pid_never_appears(
    tmp_path: Path,
) -> None:
    manager = MPSManager()
    cfg = DeviceConfig(
        mps_pipe_directory=str(tmp_path / "pipe"),
        mps_log_directory=str(tmp_path / "log"),
    )
    real_exists = os.path.exists

    def _exists(path: str) -> bool:
        if path.endswith(PID_FILENAME):
            return False
        return real_exists(path)

    with patch.object(manager, "is_mps_running", return_value=False), patch(
        "baselines.utils.mps.subprocess.run"
    ) as mock_run, patch(
        "baselines.utils.mps.os.path.exists", side_effect=_exists
    ), patch("baselines.utils.mps.time.sleep") as mock_sleep, patch(
        "baselines.utils.mps.time.monotonic",
        side_effect=[0.0, 0.1, 0.2, 10.1],
    ):
        mock_run.return_value = Mock(stdout="")
        with pytest.raises(TimeoutError, match="did not appear"):
            manager.start_mps(0, cfg)

    assert mock_sleep.called


def test_start_mps_sends_pinned_mem_limit_command(
    tmp_path: Path,
) -> None:
    manager = MPSManager()
    cfg = DeviceConfig(
        mps_pipe_directory=str(tmp_path / "pipe"),
        mps_log_directory=str(tmp_path / "log"),
        mps_pinned_device_mem_limit="2048M",
    )
    pipe_dir = f"{cfg.mps_pipe_directory}-0"
    log_dir = f"{cfg.mps_log_directory}-0"

    with patch.object(manager, "is_mps_running", return_value=False), patch.object(
        manager,
        "_wait_for_pid",
    ), patch.object(manager, "_send_command") as mock_send, patch(
        "baselines.utils.mps.subprocess.run"
    ) as mock_run:
        mock_run.return_value = Mock(stdout="")
        manager.start_mps(0, cfg)

    mock_run.assert_called_once()
    mock_send.assert_called_once_with(
        pipe_dir,
        log_dir,
        "set_default_device_pinned_mem_limit 0 2048M",
    )


def test_stop_mps_sends_quit_via_stdin(tmp_path: Path) -> None:
    manager = MPSManager()
    cfg = DeviceConfig(
        mps_pipe_directory=str(tmp_path / "pipe"),
        mps_log_directory=str(tmp_path / "log"),
    )
    pipe_dir = _resolve_pipe(3, cfg)
    log_dir = _resolve_log(3, cfg)

    with patch.object(manager, "is_mps_running", return_value=False), patch.object(
        manager,
        "_wait_for_pid",
    ), patch("baselines.utils.mps.subprocess.run", return_value=Mock(stdout="")):
        manager.start_mps(3, cfg)

    fake_run, record = _capture_run(stdout="ok")

    with patch.object(manager, "is_mps_running", return_value=True), patch.object(
        manager,
        "_wait_for_pid",
    ) as mock_wait, patch(
        "baselines.utils.mps.subprocess.run",
        side_effect=fake_run,
    ):
        manager.stop_mps(3)

    assert "cmd" in record
    assert "kwargs" in record
    cmd = cast(list[str], record["cmd"])
    kwargs = cast(dict[str, object], record["kwargs"])
    assert cmd == ["nvidia-cuda-mps-control"]
    assert kwargs["input"] == "quit\n"
    assert kwargs["text"] is True
    env = cast(dict[str, str], kwargs["env"])
    assert env["CUDA_MPS_PIPE_DIRECTORY"] == pipe_dir
    assert env["CUDA_MPS_LOG_DIRECTORY"] == log_dir
    mock_wait.assert_called_once_with(
        pipe_dir,
        appear=False,
        timeout=10.0,
    )


def test_stop_mps_force_kills_when_pid_not_gone(
    tmp_path: Path,
) -> None:
    manager = MPSManager()
    cfg = DeviceConfig(
        mps_pipe_directory=str(tmp_path / "pipe"),
        mps_log_directory=str(tmp_path / "log"),
    )
    with patch.object(manager, "is_mps_running", return_value=False), patch.object(
        manager,
        "_wait_for_pid",
    ), patch("baselines.utils.mps.subprocess.run", return_value=Mock(stdout="")):
        manager.start_mps(0, cfg)

    pipe_dir = Path(_resolve_pipe(0, cfg))
    pipe_dir.mkdir(parents=True, exist_ok=True)
    pid_path = pipe_dir / PID_FILENAME
    _ = pid_path.write_text("4242", encoding="utf-8")

    with patch.object(manager, "is_mps_running", return_value=True), patch.object(
        manager,
        "_wait_for_pid",
        side_effect=TimeoutError("stuck"),
    ), patch("baselines.utils.mps.subprocess.run") as mock_run, patch(
        "baselines.utils.mps.os.kill"
    ) as mock_kill:
        mock_run.return_value = Mock(stdout="")
        manager.stop_mps(0)

    mock_run.assert_called_once()
    mock_kill.assert_called_once_with(4242, signal.SIGKILL)


def test_is_mps_running_true_when_pid_exists_and_alive(
    tmp_path: Path,
) -> None:
    manager = MPSManager()
    pipe_dir = tmp_path / "pipe"
    pipe_dir.mkdir()
    _ = (pipe_dir / PID_FILENAME).write_text("123", encoding="utf-8")

    with patch("baselines.utils.mps.os.kill") as mock_kill:
        mock_kill.return_value = None
        assert manager.is_mps_running(0, str(pipe_dir)) is True
    mock_kill.assert_called_once_with(123, 0)


def test_is_mps_running_false_when_pid_file_missing(
    tmp_path: Path,
) -> None:
    manager = MPSManager()
    assert manager.is_mps_running(1, str(tmp_path / "no-pid")) is False


def test_is_mps_running_false_when_process_dead(
    tmp_path: Path,
) -> None:
    manager = MPSManager()
    pipe_dir = tmp_path / "pipe"
    pipe_dir.mkdir()
    _ = (pipe_dir / PID_FILENAME).write_text("999", encoding="utf-8")

    with patch(
        "baselines.utils.mps.os.kill",
        side_effect=ProcessLookupError,
    ):
        assert manager.is_mps_running(0, str(pipe_dir)) is False


def test_get_mps_client_env_falls_back_to_config_without_visible_devices(
    fresh_singleton_manager: MPSManager,
) -> None:
    assert isinstance(fresh_singleton_manager, MPSManager)
    cfg = DeviceConfig(
        mps_pipe_directory="/custom/pipe",
        mps_log_directory="/custom/log",
    )

    env = mps.get_mps_client_env(5, cfg)

    assert env["CUDA_MPS_PIPE_DIRECTORY"] == "/custom/pipe-5"
    assert env["CUDA_MPS_LOG_DIRECTORY"] == "/custom/log-5"
    assert "CUDA_VISIBLE_DEVICES" not in env


def test_get_mps_client_env_prefers_managed_state_without_visible_devices(
    fresh_singleton_manager: MPSManager,
) -> None:
    managed_cfg = DeviceConfig(
        mps_pipe_directory="/tmp/managed-pipe",
        mps_log_directory="/tmp/managed-log",
    )
    with patch.object(
        fresh_singleton_manager,
        "is_mps_running",
        return_value=False,
    ), patch.object(fresh_singleton_manager, "_wait_for_pid"), patch(
        "baselines.utils.mps.subprocess.run",
        return_value=Mock(stdout=""),
    ):
        fresh_singleton_manager.start_mps(2, managed_cfg)

    cfg = DeviceConfig(
        mps_pipe_directory="/custom/pipe",
        mps_log_directory="/custom/log",
    )

    env = mps.get_mps_client_env(2, cfg)

    assert env["CUDA_MPS_PIPE_DIRECTORY"] == "/tmp/managed-pipe-2"
    assert env["CUDA_MPS_LOG_DIRECTORY"] == "/tmp/managed-log-2"
    assert "CUDA_VISIBLE_DEVICES" not in env


def test_stop_all_mps_stops_all_started_gpus(
    tmp_path: Path,
) -> None:
    manager = MPSManager()
    cfg = DeviceConfig(
        mps_pipe_directory=str(tmp_path / "pipe"),
        mps_log_directory=str(tmp_path / "log"),
    )

    with patch.object(manager, "is_mps_running", return_value=False), patch.object(
        manager,
        "_wait_for_pid",
    ), patch("baselines.utils.mps.subprocess.run"):
        manager.start_mps(0, cfg)
        manager.start_mps(1, cfg)

    env0 = manager.get_client_env(0)
    env1 = manager.get_client_env(1)
    assert env0["CUDA_MPS_PIPE_DIRECTORY"] == _resolve_pipe(0, cfg)
    assert env1["CUDA_MPS_PIPE_DIRECTORY"] == _resolve_pipe(1, cfg)

    with patch.object(manager, "stop_mps") as mock_stop:
        manager.stop_all_mps()

    mock_stop.assert_has_calls([call(0), call(1)], any_order=True)
    default0 = manager.get_client_env(0)
    default1 = manager.get_client_env(1)
    assert default0["CUDA_MPS_PIPE_DIRECTORY"] == "/tmp/nvidia-mps-0"
    assert default1["CUDA_MPS_PIPE_DIRECTORY"] == "/tmp/nvidia-mps-1"


def test_manager_context_manager_calls_stop_all_on_exit() -> None:
    manager = MPSManager()

    with patch.object(manager, "stop_all_mps") as mock_stop:
        with manager as entered:
            assert entered is manager

    mock_stop.assert_called_once_with()


def test_module_start_mps_delegates_to_singleton(
    fresh_singleton_manager: MPSManager,
) -> None:
    cfg = DeviceConfig()
    with patch.object(fresh_singleton_manager, "start_mps") as mock_start:
        mps.start_mps(9, cfg)
    mock_start.assert_called_once_with(9, cfg)


def test_module_stop_mps_delegates_to_singleton(
    fresh_singleton_manager: MPSManager,
) -> None:
    with patch.object(fresh_singleton_manager, "stop_mps") as mock_stop:
        mps.stop_mps(9)
    mock_stop.assert_called_once_with(9)


def test_module_stop_all_mps_delegates_to_singleton(
    fresh_singleton_manager: MPSManager,
) -> None:
    with patch.object(
        fresh_singleton_manager,
        "stop_all_mps",
    ) as mock_stop_all:
        mps.stop_all_mps()
    mock_stop_all.assert_called_once_with()


def test_module_is_mps_running_delegates_to_singleton(
    fresh_singleton_manager: MPSManager,
) -> None:
    with patch.object(
        fresh_singleton_manager,
        "is_mps_running",
        return_value=True,
    ) as mock_is_running:
        result = mps.is_mps_running(6)
    mock_is_running.assert_called_once_with(6)
    assert result is True


def test_train_cli_declares_mps_flags() -> None:
    source = _train_source_text()
    assert "--enable-mps" in source
    assert "--mps-thread-pct" in source


def test_train_main_contains_mps_override_logic() -> None:
    source = _train_source_text()
    assert "if args.enable_mps:" in source
    assert "cfg.device.mps_enabled = True" in source
    assert "if args.mps_thread_pct is not None:" in source
    assert "cfg.device.mps_active_thread_percentage" in source
    assert "if cfg.device.mps_enabled:" in source
    assert "cfg.device.validate()" in source
