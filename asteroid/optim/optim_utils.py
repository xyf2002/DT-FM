import torch
import torch.nn as nn
import math
from ..core.config import AsteroidConfig

def flatten_params(param_set):
    """Flatten model parameters into a single contiguous buffer."""
    params = list(param_set)
    weights = [p.data for p in params]
    grads = [p.grad.data if p.grad is not None else torch.zeros_like(p.data)
             for p in params]
    sizes = [p.numel() for p in params]
    total = sum(sizes)
    flat_w = torch.zeros(total, dtype=weights[0].dtype, device=weights[0].device)
    flat_g = torch.zeros(total, dtype=weights[0].dtype, device=weights[0].device)
    fw_s, fg_s = flat_w.storage(), flat_g.storage()

    def _set_storage(param, ws, gs, off):
        with torch.no_grad():
            z = torch.zeros_like(param.data); z.set_(ws, off, param.shape); param.data = z
            t = torch.zeros_like(param.data); t.set_(gs, off, param.shape); param.grad = t

    offset = 0
    for i, p in enumerate(params):
        flat_w[offset:offset + sizes[i]] = weights[i].reshape(-1)
        flat_g[offset:offset + sizes[i]] = grads[i].reshape(-1)
        _set_storage(p, fw_s, fg_s, offset)
        offset += sizes[i]
    with torch.no_grad():
        flat = nn.Parameter(flat_w, requires_grad=False)
        flat.grad = flat_g
        return flat

def get_lr(it: int, cfg: AsteroidConfig):
    """Cosine LR schedule with warmup — from DT-FM."""
    if it < cfg.warmup_iters:
        return cfg.lr * (it + 1) / cfg.warmup_iters
    if it > cfg.max_iters:
        return cfg.min_lr
    decay = (it - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)