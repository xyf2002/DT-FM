# pyright: reportMissingImports=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownParameterType=false
# pyright: reportAny=false

from __future__ import annotations

import pytest
import torch

from baselines.utils.flatten import flatten_params

CPU_DEVICE = torch.device("cpu")


def _build_linear_stack() -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(8, 6),
        torch.nn.Linear(6, 4),
        torch.nn.Linear(4, 2),
    ).to(CPU_DEVICE)


def test_flatten_creates_flat_param() -> None:
    """flatten_params should return a Parameter with data and grad buffers."""
    model = _build_linear_stack()
    flat_param = flatten_params(model.parameters())

    assert isinstance(flat_param, torch.nn.Parameter)
    assert flat_param.data is not None
    assert flat_param.grad is not None


def test_flatten_storage_sharing() -> None:
    """Flattened data buffer should share storage with original params."""
    model = _build_linear_stack()
    params = list(model.parameters())
    flat_param = flatten_params(params)

    expected_value = params[0].data.view(-1)[0].item() + 3.0
    with torch.no_grad():
        flat_param.data[0] = expected_value

    actual_value = params[0].data.view(-1)[0].item()
    assert actual_value == pytest.approx(expected_value)


def test_flatten_grad_allocated() -> None:
    """flatten_params should allocate a gradient buffer for all elements."""
    model = _build_linear_stack()
    params = list(model.parameters())
    flat_param = flatten_params(params)
    total_numel = sum(param.numel() for param in params)

    assert flat_param.grad is not None
    assert tuple(flat_param.grad.shape) == (total_numel,)


def test_flatten_with_chunk() -> None:
    """Chunked flattening should pad total elements to chunk multiple."""
    model = _build_linear_stack()
    params = list(model.parameters())
    flat_param = flatten_params(params, chunk=64)

    assert flat_param.data.numel() % 64 == 0


def test_flatten_values_preserved() -> None:
    """Flattening should preserve the sum of original parameter values."""
    model = _build_linear_stack()
    params = list(model.parameters())
    sum_before = sum(param.detach().sum().item() for param in params)

    flat_param = flatten_params(params)
    sum_after = flat_param.data.sum().item()

    assert sum_after == pytest.approx(sum_before)
