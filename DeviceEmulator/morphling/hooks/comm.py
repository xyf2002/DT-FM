import torch


def p2p_decorator(func):
    def wrapper(*args, **kwargs):
        print("point to point decorator")
        return func(*args, **kwargs)

    return wrapper


def collective_decorator(func):
    def wrapper(*args, **kwargs):
        print("collective decorator")
        return func(*args, **kwargs)

    return wrapper


torch.nn.Linear = p2p_decorator(torch.nn.Linear)
