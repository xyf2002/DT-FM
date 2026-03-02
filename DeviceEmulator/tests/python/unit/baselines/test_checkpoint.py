from __future__ import annotations

import importlib
from pathlib import Path
from typing import Protocol, cast

from baselines.fault_tolerance.basic_checkpoint import BasicCheckpoint


class _TensorLike(Protocol):
    def to(self, device: object) -> _TensorLike:
        ...


class _TorchLike(Protocol):
    def device(self, value: str) -> object:
        ...

    def randn(self, size: int, *, device: object) -> _TensorLike:
        ...

    def allclose(self, left: _TensorLike, right: _TensorLike) -> bool:
        ...


torch = cast(_TorchLike, cast(object, importlib.import_module("torch")))
CPU_DEVICE = torch.device("cpu")


def test_save_creates_file(tmp_path: Path) -> None:
    """BasicCheckpoint.save should create the target file."""
    checkpoint = BasicCheckpoint(checkpoint_dir=str(tmp_path), interval=10)
    checkpoint.save({"a": 1}, "test.pt")

    assert (tmp_path / "test.pt").exists()


def test_load_recovers_state(tmp_path: Path) -> None:
    """BasicCheckpoint.load should recover saved Python state."""
    checkpoint = BasicCheckpoint(checkpoint_dir=str(tmp_path), interval=10)
    state: dict[str, object] = {"a": 1, "b": [1, 2, 3]}
    checkpoint.save(state, "state.pt")

    loaded = checkpoint.load("state.pt")

    assert loaded == state


def test_round_trip_tensor(tmp_path: Path) -> None:
    """BasicCheckpoint should preserve tensor values across save/load."""
    checkpoint = BasicCheckpoint(checkpoint_dir=str(tmp_path), interval=10)
    state: dict[str, object] = {"weights": torch.randn(10, device=CPU_DEVICE)}
    checkpoint.save(state, "tensor.pt")

    loaded = checkpoint.load("tensor.pt")
    loaded_weights = cast(_TensorLike, loaded["weights"])
    original_weights = cast(_TensorLike, state["weights"])

    assert torch.allclose(loaded_weights.to(CPU_DEVICE), original_weights)


def test_should_checkpoint_at_interval(tmp_path: Path) -> None:
    """BasicCheckpoint should trigger on exact interval boundaries."""
    checkpoint = BasicCheckpoint(checkpoint_dir=str(tmp_path), interval=10)

    assert checkpoint.should_checkpoint(0) is True
    assert checkpoint.should_checkpoint(10) is True
    assert checkpoint.should_checkpoint(100) is True


def test_should_not_checkpoint_off_interval(tmp_path: Path) -> None:
    """BasicCheckpoint should not trigger between interval boundaries."""
    checkpoint = BasicCheckpoint(checkpoint_dir=str(tmp_path), interval=10)

    assert checkpoint.should_checkpoint(1) is False
    assert checkpoint.should_checkpoint(5) is False
    assert checkpoint.should_checkpoint(99) is False


def test_checkpoint_dir_created(tmp_path: Path) -> None:
    """BasicCheckpoint constructor should create missing directories."""
    new_dir = tmp_path / "test_ckpt_new_dir_xyz" / "sub"
    assert not new_dir.exists()

    _ = BasicCheckpoint(checkpoint_dir=str(new_dir), interval=10)

    assert new_dir.exists()
    assert new_dir.is_dir()


def test_interval_clamped(tmp_path: Path) -> None:
    """BasicCheckpoint should clamp interval values below one to one."""
    checkpoint = BasicCheckpoint(checkpoint_dir=str(tmp_path), interval=0)

    assert checkpoint.interval == 1
