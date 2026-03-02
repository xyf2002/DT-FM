import torch

from morphling.common.keywords import pyobj_keywords, pytorch_keywords


def torch_decorator(func, funcname=None):
    def wrapper(*args, **kwargs):
        print("torch_decorator ", funcname)
        return func(*args, **kwargs)

    return wrapper


a = torch.ones((1, 3))
b = torch.ones((1, 3))


# for all functions in torch.Tensor, add decorator
functions = [f for f in dir(torch.Tensor) if callable(getattr(torch.Tensor, f))]
for f in functions:
    if f in pyobj_keywords or f in pytorch_keywords:
        continue
    func = getattr(torch.Tensor, f)
    setattr(torch.Tensor, f, torch_decorator(func, f))

# for all functions in torch, add decorator
functions = [f for f in dir(torch) if callable(getattr(torch, f))]
for f in functions:
    if f in pyobj_keywords or f in pytorch_keywords:
        continue
    func = getattr(torch, f)
    setattr(torch, f, torch_decorator(func, f))

print(dir(a))

c = a @ b.T
c = a + 1
