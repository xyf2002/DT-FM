import functools
from typing import List, Union

import numpy as np
import torch

from morphling.backend import BaseBackend
from morphling.common import get_logger

logger = get_logger()
_backend: BaseBackend = None
_enable_verification = False

HOOK_TYPES = [
    "linear",
    "layer_norm",
    "softmax",
    "add",
    "divide",
    "subtract",
    "matmul",
    "relu",
    "gelu",
    "dropout",
]


def apply_hooks(types: Union[str, List[str]]):
    if isinstance(types, str):
        types = [types]
    for t in types:
        if t not in HOOK_TYPES:
            raise ValueError(f"Unsupported hook type: {t}")

        if t == "linear":
            torch.nn.functional.linear = LinearFunction.apply
            torch.Tensor.__matmul__ = LinearFunction.apply
            torch.bmm = LinearFunction.apply
            # torch.matmul = LinearFunction.apply

            def forward_decorator(func):
                def wrapper(self, input):
                    return LinearFunction.apply(
                        input, self.weight.t(), self.bias
                    )

                return wrapper

            torch.nn.Linear.forward = forward_decorator(torch.nn.Linear.forward)
            print("Linear hook applied")
        else:
            raise NotImplementedError(f"Hook type {t} is not implemented yet")


# custom autograd function for linear layer
class LinearFunction(torch.autograd.Function):
    extra_dict = {"className": "LinearFunction"}
    logger.debug = functools.partial(logger.debug, extra=extra_dict)

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        print("LinearFunction forward", input.shape, weight.shape)
        ctx.save_for_backward(input, weight, bias)
        # output = input.mm(weight.t())
        # logger.debug(f"input shape: {input.shape}")
        # logger.debug(f"weight shape: {weight.shape}")
        # output = torch.as_tensor(np.matmul(input, weight))

        # FIXME: this only applies to mqtt backend
        _backend.async_dispatch_matmul(input, weight.transpose(-2, -1))
        output = _backend.wait_matmul(0)
        # ref = torch.matmul(input.to("cuda:0"), weight.to("cuda:0")).to("cpu")
        # validate output
        # assert torch.allclose(output, ref), f"Output is not close! input shape: {input.shape}, weight shape: {weight.shape}, max diff: {torch.max(torch.abs(output - ref))}"

        if bias is not None:
            output = output + bias.unsqueeze(0).expand_as(output)

        if _enable_verification:
            ref = torch.matmul(input, weight)
            if bias is not None:
                ref = ref + bias.unsqueeze(0).expand_as(ref)
            assert torch.allclose(output, ref, atol=1e-5), (
                f"Output is not close! input shape: {input.shape}, weight shape: {weight.shape}, max diff: {torch.max(torch.abs(output - ref))}"
            )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        print(
            "LinearFunction backward",
            grad_output.shape,
            weight.shape,
            input.shape,
        )
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            # grad_input = grad_output.mm(weight)
            # grad_input = torch.as_tensor(
            #     np.matmul(grad_output, weight.transpose(-2, -1))
            # )
            # grad_input = _backend.sync_dispatch_matmul(
            #     grad_output, weight.transpose(-2, -1)
            # )
            _backend.async_dispatch_matmul(grad_output, weight)
        if ctx.needs_input_grad[1]:
            # grad_weight = grad_output.t().mm(input)
            # grad_weight = torch.as_tensor(
            #     np.matmul(grad_output.transpose(-2, -1), input)
            # ).transpose(-2, -1)
            # grad_weight = _backend.sync_dispatch_matmul(
            #     grad_output.transpose(-2, -1), input
            # ).transpose(-2, -1)
            _backend.async_dispatch_matmul(
                grad_output.transpose(-2, -1), input.transpose(-2, -1)
            )

        dispatch_count = 0
        if ctx.needs_input_grad[0]:
            grad_input = _backend.wait_matmul(dispatch_count)
            dispatch_count += 1

        if ctx.needs_input_grad[1]:
            grad_weight = _backend.wait_matmul(dispatch_count).transpose(-2, -1)
            dispatch_count += 1

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        if _enable_verification:
            if ctx.needs_input_grad[0]:
                ref_grad_input = torch.matmul(grad_output, weight, atol=1e-5)
                assert torch.allclose(grad_input, ref_grad_input), (
                    f"grad_input is not close! max diff: {torch.max(torch.abs(grad_input - ref_grad_input))}"
                )
            if ctx.needs_input_grad[1]:
                ref_grad_weight = torch.matmul(
                    grad_output.transpose(-2, -1), input
                ).transpose(-2, -1)
                assert torch.allclose(
                    grad_weight, ref_grad_weight, atol=1e-5
                ), (
                    f"grad_weight is not close! max diff: {torch.max(torch.abs(grad_weight - ref_grad_weight))}"
                )
            if bias is not None and ctx.needs_input_grad[2]:
                ref_grad_bias = grad_output.sum(0)
                assert torch.allclose(grad_bias, ref_grad_bias, atol=1e-5), (
                    f"grad_bias is not close! max diff: {torch.max(torch.abs(grad_bias - ref_grad_bias))}"
                )

        return grad_input, grad_weight, grad_bias


# custom autograd function for torch.nn.functional.layer_norm
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, normalized_shape, weight, bias, eps):
        logger.debug("LayerNormFunction forward")
        ctx.save_for_backward(input, weight, bias)
        # output = torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps)
        output = input
        return output

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("LayerNormFunction backward")
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output
        if ctx.needs_input_grad[2]:
            grad_weight = grad_output
        if ctx.needs_input_grad[3]:
            grad_bias = grad_output

        return grad_input, None, grad_weight, grad_bias, None


# custom autograd function for softmax
class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.nn.functional.softmax(input, dim=1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        return grad_input


# custom autograd function for torch.Tensor.__add__(self, other) -> Tensor
class AddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, other):
        logger.debug("AddFunction forward")
        output = torch.as_tensor(np.add(input, other))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("AddFunction backward")
        grad_input = grad_other = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output
        if ctx.needs_input_grad[1]:
            grad_other = grad_output

        return grad_input, grad_other


# custom autograd function for divide
class DivideFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, other):
        ctx.save_for_backward(input, other)
        output = input / other
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, other = ctx.saved_tensors
        grad_input = grad_other = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output / other
        if ctx.needs_input_grad[1]:
            grad_other = -grad_output * input / (other**2)

        return grad_input, grad_other


# custom autograd function for subtract
class SubtractFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, other):
        ctx.save_for_backward(input, other)
        output = input - other
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, other = ctx.saved_tensors
        grad_input = grad_other = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output
        if ctx.needs_input_grad[1]:
            grad_other = -grad_output

        return grad_input, grad_other


# custom autograd function for matmul
class MatmulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, other):
        ctx.save_for_backward(input, other)
        output = input @ other
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, other = ctx.saved_tensors
        grad_input = grad_other = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output @ other.t()
        if ctx.needs_input_grad[1]:
            grad_other = input.t() @ grad_output

        return grad_input, grad_other


# custom autograd function for relu
class ReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.nn.functional.relu(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0

        return grad_input


# custom autograd function for gelu
class GeLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.nn.functional.gelu(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output * torch.nn.functional.gelu(input, True)

        return grad_input


# custom autograd function for dropout
class DropoutFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, p, train):
        ctx.save_for_backward(input)
        if train:
            output = torch.nn.functional.dropout(input, p)
        else:
            output = input
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        return grad_input
