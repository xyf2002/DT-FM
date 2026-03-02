from __future__ import annotations

import time

from baselines.fault_tolerance.heartbeat import HeartbeatDetector


def test_construction() -> None:
    """HeartbeatDetector should preserve constructor parameters."""
    hb = HeartbeatDetector(device_id=0, interval_s=1.0, timeout_s=3.0)

    assert hb.device_id == 0
    assert hb.interval_s == 1.0
    assert hb.timeout_s == 3.0


def test_start_stop() -> None:
    """HeartbeatDetector should start and stop without errors."""
    hb = HeartbeatDetector(device_id=0, interval_s=0.05, timeout_s=1.0)

    hb.start()
    hb.stop()


def test_check_alive_self() -> None:
    """HeartbeatDetector should report the local device as alive."""
    hb = HeartbeatDetector(device_id=0, interval_s=0.05, timeout_s=1.0)

    hb.start()
    try:
        time.sleep(0.2)
        assert hb.check_alive(0) is True
    finally:
        hb.stop()


def test_check_alive_unknown() -> None:
    """HeartbeatDetector should conservatively treat unknown devices as alive."""
    hb = HeartbeatDetector(0)

    hb.start()
    try:
        assert hb.check_alive(99) is True
    finally:
        hb.stop()


def test_double_start_noop() -> None:
    """HeartbeatDetector.start should be a no-op when called twice."""
    hb = HeartbeatDetector(0)

    hb.start()
    hb.start()
    hb.stop()


def test_stop_without_start() -> None:
    """HeartbeatDetector.stop should be safe before start."""
    hb = HeartbeatDetector(0)
    hb.stop()
