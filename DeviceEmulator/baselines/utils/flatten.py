from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import Protocol, cast


class _TensorLike(Protocol):
    data: "_TensorLike"
    dtype: object
    device: object
    shape: object

    def storage(self) -> object: ...

    def reshape(self, *shape: int) -> "_TensorLike": ...

    def set_(self, storage: object, offset: int, shape: object) -> object: ...

    def __setitem__(self, key: object, value: object) -> None: ...


class _ParameterLike(Protocol):
    data: _TensorLike
    grad: _TensorLike | None
    shape: object

    def numel(self) -> int: ...


class _NoGradContext(Protocol):
    def __enter__(self) -> object: ...

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> bool | None: ...


class _NNNamespace(Protocol):
    def Parameter(
        self,
        data: _TensorLike,
        requires_grad: bool = True,
    ) -> _ParameterLike: ...


class _TorchModule(Protocol):
    nn: _NNNamespace

    def zeros(
        self,
        size: int,
        *,
        dtype: object,
        device: object,
    ) -> _TensorLike: ...

    def zeros_like(self, value: _TensorLike) -> _TensorLike: ...

    def no_grad(self) -> _NoGradContext: ...


def _load_torch_module() -> object:
    return importlib.import_module("torch")


def flatten_params(
    param_set: Iterable[object],
    chunk: int | None = None,
) -> object:
    """Flatten model parameters into contiguous parameter/gradient buffers."""

    torch = cast(_TorchModule, _load_torch_module())
    params = [cast(_ParameterLike, param) for param in param_set]
    weights = [p.data for p in params]
    grads = [
        p.grad.data if p.grad is not None else torch.zeros_like(p.data)
        for p in params
    ]
    sizes = [p.numel() for p in params]
    total_size = sum(sizes)
    if chunk:
        total_size = ((total_size + chunk - 1) // chunk) * chunk
    flatten_weights = torch.zeros(
        total_size,
        dtype=weights[0].dtype,
        device=weights[0].device,
    )
    flatten_grads = torch.zeros(
        total_size,
        dtype=weights[0].dtype,
        device=weights[0].device,
    )
    fw_storage = flatten_weights.storage()
    fg_storage = flatten_grads.storage()

    def _set_storage(
        param: _ParameterLike,
        w_storage: object,
        g_storage: object,
        offset: int,
    ) -> None:
        with torch.no_grad():
            z = torch.zeros_like(param.data)
            _ = z.set_(w_storage, offset, param.shape)
            param.data = z
            t = torch.zeros_like(param.data)
            _ = t.set_(g_storage, offset, param.shape)
            param.grad = t

    offset = 0
    for i, p in enumerate(params):
        flatten_weights[offset : offset + sizes[i]] = weights[i].reshape(-1)
        flatten_grads[offset : offset + sizes[i]] = grads[i].reshape(-1)
        _set_storage(p, fw_storage, fg_storage, offset)
        offset += sizes[i]
    with torch.no_grad():
        flat_param = torch.nn.Parameter(flatten_weights, requires_grad=False)
        flat_param.grad = flatten_grads
        return flat_param
