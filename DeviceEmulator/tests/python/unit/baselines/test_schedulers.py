from __future__ import annotations

import importlib
from numbers import Integral
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
import pytest

from baselines.schedulers.confident_scheduler import ConfidentScheduler
from baselines.schedulers.dp_partitioner import DPPartitioner
from baselines.schedulers.gcma import GCMAScheduler


class _NumpyLike(Protocol):
    def zeros(
        self,
        shape: tuple[int, int],
        dtype: type[np.float64],
    ) -> npt.NDArray[np.float64]:
        ...

    def ones(
        self,
        shape: tuple[int, int],
        dtype: type[np.float64],
    ) -> npt.NDArray[np.float64]:
        ...


numpy = cast(_NumpyLike, cast(object, importlib.import_module("numpy")))
HAS_REPLAN_AFTER_FAILURE = hasattr(ConfidentScheduler, "replan_after_failure")


class _SupportsReplanAfterFailure(Protocol):
    def replan_after_failure(self, survivors: list[int]) -> None:
        ...


def test_dp_partition_uniform() -> None:
    """DPPartitioner should return two splits for 12 layers on 3 devices."""
    partitioner = DPPartitioner(12, 3, [1.0] * 12)
    split_points = partitioner.partition()

    assert isinstance(split_points, list)
    assert len(split_points) == 2


def test_dp_partition_single_device() -> None:
    """DPPartitioner should return no split points for a single device."""
    partitioner = DPPartitioner(12, 1)
    split_points = partitioner.partition()

    assert split_points == []


def test_dp_partition_layers_eq_devices() -> None:
    """DPPartitioner should return devices-1 points when layers==devices."""
    partitioner = DPPartitioner(3, 3, [1.0] * 3)
    split_points = partitioner.partition()

    assert isinstance(split_points, list)
    assert len(split_points) == 2


def test_dp_partition_no_profiler() -> None:
    """DPPartitioner should fall back to default timing without profiler data."""
    partitioner = DPPartitioner(8, 2)
    split_points = partitioner.partition()

    assert isinstance(split_points, list)
    assert len(split_points) == 1


def test_dp_partition_nonuniform() -> None:
    """DPPartitioner should cut early when the first layer is very expensive."""
    layer_times = [10.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    partitioner = DPPartitioner(6, 2, layer_times)
    split_points = partitioner.partition()

    assert isinstance(split_points, list)
    assert len(split_points) == 1
    assert split_points[0] < 3


def test_gcma_validates_dimensions() -> None:
    """GCMAScheduler should reject num_devices != pp_size * dp_size."""
    peer_delay = numpy.zeros((3, 3), dtype=np.float64)
    peer_bandwidth = numpy.zeros((3, 3), dtype=np.float64)

    with pytest.raises(ValueError):
        _ = GCMAScheduler(
            num_devices=3,
            pp_size=2,
            dp_size=2,
            peer_delay=peer_delay,
            peer_bandwidth=peer_bandwidth,
            send_gradient_size=0.1,
            send_activation_size=0.1,
        )


def test_gcma_constructs() -> None:
    """GCMAScheduler should construct with consistent dimensions."""
    peer_delay = numpy.ones((4, 4), dtype=np.float64)
    peer_bandwidth = numpy.ones((4, 4), dtype=np.float64)

    scheduler = GCMAScheduler(
        num_devices=4,
        pp_size=2,
        dp_size=2,
        peer_delay=peer_delay,
        peer_bandwidth=peer_bandwidth * 10.0,
        send_gradient_size=0.1,
        send_activation_size=0.1,
    )

    assert isinstance(scheduler, GCMAScheduler)


def test_confident_partition() -> None:
    """ConfidentScheduler should return two splits for 12 layers, 3 devices."""
    scheduler = ConfidentScheduler(12, 3, [1.0] * 12)
    split_points = scheduler.partition()

    assert isinstance(split_points, list)
    assert len(split_points) == 2


@pytest.mark.skipif(
    not HAS_REPLAN_AFTER_FAILURE,
    reason="ConfidentScheduler does not implement replan_after_failure",
)
def test_confident_replan() -> None:
    """ConfidentScheduler should support replan_after_failure if available."""
    scheduler = ConfidentScheduler(12, 4, [1.0] * 12)
    original_points = scheduler.partition()

    # replan_after_failure(failed_devices, current_partition)
    new_points = scheduler.replan_after_failure([3], original_points)

    assert isinstance(original_points, list)
    assert all(isinstance(point, Integral) for point in original_points)
    assert isinstance(new_points, list)
    assert all(isinstance(point, Integral) for point in new_points)
