#!/usr/bin/env python3
"""
Asteroid — Resource-Efficient Hybrid Pipeline Parallelism
for Collaborative DNN Training on Heterogeneous Edge Devices
==============================================================
Implements the full Asteroid system (MobiCom '24) using the layered
architecture from DT-FM (L1-L8) and Confident codebases.

Architecture:
  L1: Configuration & State  (from DT-FM/Confident pattern)
  L2: Compute Backend        (reused DT-FM GPTStageBase + custom sharding)
  L3: Dataset                (standard PyTorch loaders)
  L4: Optimizer Backend      (reused DT-FM flatten_params + AdamW)
  L5: Profiler & Planner     (Confident ProfilerBackend + Asteroid DP Algorithm)
  L6: Communication          (DT-FM NCCL send/recv + AllReduce primitives)
  L7: Fault Tolerance        (Confident FT + Asteroid topology-driven replication)
  L8: Orchestration          (Asteroid Coordinator + Worker with 1F1B HPP)

Usage:
    python asteroid_train.py
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════
from datetime import timedelta
import os
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ["NCCL_SOCKET_IFNAME"] = "lo"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "0"
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")

import sys
import time
import json
import math
import copy
import random
import logging
import threading
import struct
import traceback
from abc import ABC, abstractmethod
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Callable, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.checkpoint import checkpoint as torch_checkpoint

try:
    import cupy
    import cupy.cuda.nccl
    CUPY_NCCL_AVAILABLE = True
except (ImportError, AttributeError):
    cupy = None
    CUPY_NCCL_AVAILABLE = False

try:
    import torch.distributed as dist
    TORCH_DIST_AVAILABLE = True
except ImportError:
    dist = None
    TORCH_DIST_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger("Asteroid")


# =============================================================================
# L1: CONFIGURATION & STATE  (DT-FM/Confident pattern)
# =============================================================================

@dataclass
class DeviceSpec:
    """Hardware specification for a single edge device."""
    device_id: int = 0
    device_type: str = "jetson_nano"  # nano, tx2, nx, gpu
    memory_budget_mb: float = 4096.0
    cuda_id: int = 0
    compute_capacity: float = 1.0     # relative throughput factor

@dataclass
class AsteroidConfig:
    """Unified configuration for the Asteroid system."""
    # Model
    model_name: str = "gpt2"
    model_type: str = "gpt2"          # key into MODEL_REGISTRY
    task_type: str = "classification"  # classification | lm
    num_layers: int = 12
    embedding_dim: int = 768
    num_heads: int = 12
    d_ff: int = 3072
    max_seq_len: int = 128
    vocab_size: int = 50257
    num_classes: int = 2
    dropout: float = 0.1
    use_flash_attention: bool = True
    # Training
    global_batch_size: int = 256
    micro_batch_size: int = 4
    num_microbatches: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_iters: int = 500
    warmup_iters: int = 50
    min_lr: float = 1e-5
    grad_clip: float = 1.0
    eval_interval: int = 100
    log_interval: int = 10
    seed: int = 42
    # Parallelism (HPP)
    world_size: int = 4
    num_stages: int = 2       # P in the paper
    # Communication
    dist_url: str = "tcp://127.0.0.1:29600"
    d2d_bandwidth_mbps: float = 100.0  # default edge bandwidth
    # Fault Tolerance
    heartbeat_interval_s: float = 5.0
    heartbeat_timeout_s: float = 15.0
    backward_timeout_ms: float = 30000.0  # passive FT backward timeout (ms)
    replication_mode: str = "topology"  # topology | local | global | none
    replication_interval: int = 50       # replicate weights every N iters
    ft_check_interval: int = 10          # check for failures every N iters
    # I/O
    output_dir: str = "./asteroid_output"
    dataset: str = "sst2"

    def __post_init__(self):
        self.num_microbatches = self.global_batch_size // self.micro_batch_size
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class HPPPlanConfig:
    """Output of the Asteroid Planner — the HPP execution plan."""
    num_stages: int = 2
    # partition_points[i] = first layer index of stage i+1 (stage 0 starts at 0)
    partition_points: List[int] = field(default_factory=list)
    # device_groups[stage_idx] = list of device_ids assigned to that stage
    device_groups: Dict[int, List[int]] = field(default_factory=dict)
    # micro_batch_alloc[stage_idx][device_id] = num samples for that device
    micro_batch_alloc: Dict[int, Dict[int, int]] = field(default_factory=dict)
    # dominant_step index
    dominant_step: int = 0
    # estimated HPP-Round latency (ms)
    estimated_latency_ms: float = float('inf')


class AsteroidStateManager:
    """Runtime state for each worker — reuses DT-FM StateManager pattern."""

    def __init__(self):
        self._global_rank: int = 0
        self._stage_idx: int = 0
        self._device_group: List[int] = []
        self._dp_rank: int = 0     # rank within device group
        self._dp_size: int = 1     # size of device group
        self._device: torch.device = torch.device("cpu")
        self._plan: Optional[HPPPlanConfig] = None
        self._system_status: str = "NORMAL"
        self._lock = threading.Lock()
        # ── Confident FT state tracking ──
        self._received_iter_ids: set = set()
        self._partition_point: List[int] = []
        self._workers: Dict[int, str] = {}  # device_id → address
        self._backward_timeout_ms: float = 30000.0

    # Getters/setters (DT-FM pattern)
    @property
    def global_rank(self): return self._global_rank
    @global_rank.setter
    def global_rank(self, v): self._global_rank = v

    @property
    def stage_idx(self): return self._stage_idx
    @stage_idx.setter
    def stage_idx(self, v): self._stage_idx = v

    @property
    def device(self): return self._device
    @device.setter
    def device(self, v): self._device = v

    @property
    def plan(self): return self._plan
    @plan.setter
    def plan(self, v): self._plan = v

    @property
    def system_status(self):
        with self._lock:
            return self._system_status
    @system_status.setter
    def system_status(self, v):
        with self._lock:
            self._system_status = v

    def record_backward(self, iter_id: int):
        """Record that backward was received for this iteration.
        Maps to Confident's received_iter_ids tracking."""
        with self._lock:
            self._received_iter_ids.add(iter_id)

    def get_received_iter_ids(self) -> set:
        with self._lock:
            return set(self._received_iter_ids)

    def get_partition_point(self) -> List[int]:
        with self._lock:
            return list(self._partition_point)

    def set_partition_point(self, pts: List[int]):
        with self._lock:
            self._partition_point = list(pts)

    def get_workers(self) -> Dict[int, str]:
        with self._lock:
            return dict(self._workers)

    def set_workers(self, w: Dict[int, str]):
        with self._lock:
            self._workers = dict(w)


# =============================================================================
# OBSERVABILITY  (DT-FM EventLogger pattern)
# =============================================================================

@dataclass
class TrainingEvent:
    timestamp: float
    device_id: int
    event_type: str
    iter_id: int
    phase: str
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class AsteroidEventLogger:
    """Thread-safe event logger with Chrome trace export."""

    def __init__(self):
        self._events: List[TrainingEvent] = []
        self._lock = threading.Lock()
        self._epoch_start: float = 0.0

    def set_epoch_start(self, t): self._epoch_start = t

    @contextmanager
    def log_event(self, device_id, event_type, iter_id, phase="", **metadata):
        start = time.time()
        yield
        dur = (time.time() - start) * 1000
        ev = TrainingEvent(start, device_id, event_type, iter_id, phase, dur, metadata)
        with self._lock:
            self._events.append(ev)

    def record_event(self, device_id, event_type, iter_id, phase, duration_ms,
                     timestamp=None, **meta):
        ev = TrainingEvent(timestamp or time.time(), device_id, event_type,
                           iter_id, phase, duration_ms, meta)
        with self._lock:
            self._events.append(ev)

    def to_chrome_trace(self, filepath):
        t0 = self._epoch_start or (
            min(e.timestamp for e in self._events) if self._events else 0)
        trace = [{"name": e.event_type, "ph": "X", "pid": e.device_id,
                  "tid": e.event_type, "ts": (e.timestamp - t0) * 1e6,
                  "dur": e.duration_ms * 1000,
                  "args": {"micro-batch": e.iter_id, **e.metadata}}
                 for e in self._events]
        with open(filepath, 'w') as f:
            json.dump(trace, f, indent=2)

    def get_summary(self) -> Dict:
        stats = defaultdict(lambda: {"count": 0, "total_ms": 0.0})
        for e in self._events:
            k = (e.device_id, e.event_type)
            stats[k]["count"] += 1
            stats[k]["total_ms"] += e.duration_ms
        return dict(stats)


EVENT_LOGGER = AsteroidEventLogger()


# =============================================================================
# L2: COMPUTE BACKEND  (DT-FM GPTStageBase + model sharding)
# =============================================================================

# ── Architecture building blocks ─────────────────────────────────────
# GPT-2 components (reused from DT-FM)

class CausalSelfAttention(nn.Module):
    """GPT-2 style causal attention — reused from DT-FM."""
    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.1,
                 use_flash=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("bias",
            torch.tril(torch.ones(max_seq_len, max_seq_len))
            .view(1, 1, max_seq_len, max_seq_len))
        self.use_flash = use_flash and hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        if self.use_flash:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                dropout_p=self.attn_dropout.p if self.training else 0.0)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class GPT2MLP(nn.Module):
    """GPT-2 MLP — reused from DT-FM (c_fc/c_proj naming for checkpoint compat)."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_ff)
        self.c_proj = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class GPT2Block(nn.Module):
    """GPT-2 transformer block — pre-norm (LN → Attn → LN → MLP)."""
    def __init__(self, d_model, n_heads, d_ff, max_seq_len, dropout=0.1,
                 use_flash=True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout,
                                         use_flash=use_flash)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = GPT2MLP(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# DT-FM generic encoder block (bidirectional attention — BERT-like)

class BidirectionalAttention(nn.Module):
    """Multi-head attention without causal mask — from DT-FM gpt_modules.py."""
    def __init__(self, d_model, n_heads, dropout=0.1, use_flash=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.use_flash = use_flash and hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        if self.use_flash:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False,
                dropout_p=self.attn_dropout.p if self.training else 0.0)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class EncoderBlock(nn.Module):
    """Generic encoder transformer block (BERT-style, no causal mask)."""
    def __init__(self, d_model, n_heads, d_ff, max_seq_len, dropout=0.1,
                 use_flash=True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = BidirectionalAttention(d_model, n_heads, dropout,
                                            use_flash=use_flash)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = GPT2MLP(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ── Task heads ───────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """Sequence classification: pool hidden states → linear.
    Matches DT-FM's SeqClassification but configurable pooling."""
    def __init__(self, d_model, num_classes, pool="mean"):
        super().__init__()
        self.pool = pool  # "mean" | "first" | "last"
        self.ln_f = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, targets=None):
        x = self.ln_f(x)
        if self.pool == "first":
            pooled = x[:, 0]           # CLS-style (BERT)
        elif self.pool == "last":
            pooled = x[:, -1]          # GPT-style last token
        else:
            pooled = x.mean(dim=1)     # mean pooling (default)
        logits = self.classifier(pooled)
        if targets is not None:
            return F.cross_entropy(logits, targets, reduction='mean')
        return logits


class LMHead(nn.Module):
    """Language modelling head: LayerNorm → linear projection to vocab.
    Matches DT-FM's Seq2SeqClassification with causal LM shift."""
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x, targets=None):
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is not None:
            # Causal LM shift: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_targets.view(-1), reduction='mean')
        return logits


# ── Model registry & factory (makes Asteroid LLM-agnostic) ──────────

# Each entry: { "block": BlockClass, "causal": bool }
# To add a new architecture: define its block class above, register here.
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gpt2": {"block": GPT2Block, "causal": True},
    "encoder": {"block": EncoderBlock, "causal": False},
}

TASK_REGISTRY: Dict[str, type] = {
    "classification": ClassificationHead,
    "lm": LMHead,
}


def register_model(name: str, block_cls: type, causal: bool = True):
    """Register a custom transformer block for use with Asteroid.
    Example::
        register_model("llama", LlamaBlock, causal=True)
    """
    MODEL_REGISTRY[name] = {"block": block_cls, "causal": causal}


def register_task(name: str, head_cls: type):
    """Register a custom task head."""
    TASK_REGISTRY[name] = head_cls


def _create_block(cfg: AsteroidConfig) -> nn.Module:
    """Factory: instantiate one transformer block from config.
    DT-FM equivalent: GPTStageBase._create_transformer_layer()."""
    entry = MODEL_REGISTRY.get(cfg.model_type)
    if entry is None:
        raise ValueError(
            f"Unknown model_type '{cfg.model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}. "
            f"Use register_model() to add custom architectures.")
    return entry["block"](cfg.embedding_dim, cfg.num_heads, cfg.d_ff,
                          cfg.max_seq_len, cfg.dropout,
                          use_flash=cfg.use_flash_attention)


def _create_head(cfg: AsteroidConfig) -> nn.Module:
    """Factory: instantiate a task head from config.
    DT-FM equivalent: GPTStageBase._create_last_layer()."""
    head_cls = TASK_REGISTRY.get(cfg.task_type)
    if head_cls is None:
        raise ValueError(
            f"Unknown task_type '{cfg.task_type}'. "
            f"Available: {list(TASK_REGISTRY.keys())}. "
            f"Use register_task() to add custom task heads.")
    if cfg.task_type == "lm":
        return head_cls(cfg.embedding_dim, cfg.vocab_size)
    return head_cls(cfg.embedding_dim, cfg.num_classes)


# ── AsteroidStage (architecture-agnostic via factory) ────────────────

class AsteroidStage(nn.Module):
    """A pipeline stage holding a contiguous slice of transformer layers.

    Follows DT-FM's GPTStageFirst/Middle/Last factory-method pattern but
    uses MODEL_REGISTRY and TASK_REGISTRY for LLM-agnostic instantiation.
    Layer range [start_layer, end_layer) for flexible Asteroid partitioning.
    """
    def __init__(self, cfg: AsteroidConfig, start_layer: int, end_layer: int,
                 is_first: bool = False, is_last: bool = False):
        super().__init__()
        self.is_first = is_first
        self.is_last = is_last
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.task_type = cfg.task_type

        # ── Embedding (DT-FM GPTStageFirst._create_first_layer) ──
        if is_first:
            self.embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
            self.pos_embedding = nn.Embedding(cfg.max_seq_len, cfg.embedding_dim)
            self.drop = nn.Dropout(cfg.dropout)

        # ── Transformer blocks (via factory) ──
        modules = []
        for _ in range(start_layer, end_layer):
            modules.append(_create_block(cfg))
        self.blocks = nn.ModuleList(modules)

        # ── Task head (via factory) ──
        if is_last:
            self.head = _create_head(cfg)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        if self.is_first:
            B, T = x.shape[:2]
            if x.dtype == torch.long:
                pos = torch.arange(T, device=x.device).unsqueeze(0)
                x = self.drop(self.embedding(x) + self.pos_embedding(pos))
            # else: pre-embedded input (float), skip embedding
        for block in self.blocks:
            x = block(x)
        if self.is_last:
            return self.head(x, targets)
        return x

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def weight_size_bytes(self):
        return sum(p.numel() * p.element_size() for p in self.parameters())

    def activation_size_bytes(self, batch_size, seq_len, d_model):
        """Estimate activation memory for this stage."""
        n_blocks = len(self.blocks)
        # Each block stores input for backward: batch * seq * d_model * 4 bytes
        return n_blocks * batch_size * seq_len * d_model * 4


# =============================================================================
# L4: OPTIMIZER BACKEND  (DT-FM flatten_params + AdamW)
# =============================================================================

def flatten_params(param_set):
    """Flatten model parameters into a single contiguous buffer.
    Reused directly from DT-FM for efficient single-AllReduce."""
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


# =============================================================================
# L5: PROFILER & PLANNER  (Confident ProfilerBackend + Asteroid DP Algorithm)
# =============================================================================

class AsteroidProfiler:
    """Device profiler — adapted from Confident's ConfidantProfiler.

    Measures per-layer forward/backward time at multiple batch sizes,
    memory footprint, and D2D bandwidth. Implements Section 3.3 of
    the Asteroid paper (non-linear batch-size profiling).
    """

    def __init__(self, cfg: AsteroidConfig, devices: List[DeviceSpec]):
        self.cfg = cfg
        self.devices = {d.device_id: d for d in devices}
        # Profiled data: device_id → layer_idx → batch_size → (fwd_ms, bwd_ms)
        self.exec_times: Dict[int, Dict[int, Dict[int, Tuple[float, float]]]] = \
            defaultdict(lambda: defaultdict(dict))
        # Per-layer output activation size in bytes (for a single sample)
        self.activation_sizes: List[float] = []
        # Per-layer weight size in bytes
        self.weight_sizes: List[float] = []
        # D2D bandwidth matrix: (src, dst) → MB/s
        self.bandwidths: Dict[Tuple[int, int], float] = {}

    def profile_model(self, model_layers: nn.ModuleList,
                      batch_sizes: List[int],
                      device: torch.device, device_id: int,
                      seq_len: int, d_model: int):
        """Profile all layers at multiple batch sizes on a device.
        Adapted from Confident's profile_layer() with CUDA events."""
        torch.cuda.set_device(device)
        num_iters = 20  # more iterations for robust median

        for layer_idx, layer in enumerate(model_layers):
            layer = layer.to(device)
            for bs in batch_sizes:
                x = torch.randn(bs, seq_len, d_model, device=device)
                x.requires_grad_(True)

                # Warmup
                for _ in range(3):
                    with torch.no_grad():
                        _ = layer(x)
                torch.cuda.synchronize(device)

                # Forward timing (record on current_stream per Confident)
                cur_stream = torch.cuda.current_stream(device)
                fwd_times = []
                for _ in range(num_iters):
                    torch.cuda.synchronize(device)
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record(cur_stream)
                    with torch.no_grad():
                        _ = layer(x)
                    e.record(cur_stream)
                    torch.cuda.synchronize(device)
                    fwd_times.append(s.elapsed_time(e))

                # Backward timing
                bwd_times = []
                for _ in range(num_iters):
                    torch.cuda.synchronize(device)
                    layer.zero_grad()
                    if x.grad is not None:
                        x.grad.zero_()
                    out = layer(x)
                    grad_out = torch.ones_like(out)
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record(cur_stream)
                    out.backward(gradient=grad_out)
                    e.record(cur_stream)
                    torch.cuda.synchronize(device)
                    bwd_times.append(s.elapsed_time(e))

                # Skip first 3 warmup measurements, take median of rest
                fwd_ms = float(np.median(fwd_times[3:]))
                bwd_ms = float(np.median(bwd_times[3:]))
                self.exec_times[device_id][layer_idx][bs] = (fwd_ms, bwd_ms)

            # Peak memory profiling (from Confident's profile_layer)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            layer.zero_grad()
            x_mem = torch.randn(batch_sizes[-1], seq_len, d_model,
                                device=device, requires_grad=True)
            out = layer(x_mem)
            out.backward(gradient=torch.ones_like(out))
            peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            logger.debug(f"Profile: layer {layer_idx} peak_mem={peak_mem_mb:.1f}MB")

            layer.cpu()
            torch.cuda.empty_cache()

        # Compute activation and weight sizes
        if not self.activation_sizes:
            for layer in model_layers:
                # activation size per sample = seq_len * d_model * 4 bytes
                self.activation_sizes.append(seq_len * d_model * 4)
                self.weight_sizes.append(
                    sum(p.numel() * p.element_size() for p in layer.parameters()))

    def profile_bandwidth(self, src_device: torch.device, dst_device: torch.device,
                          src_id: int, dst_id: int, data_size_mb: float = 50.0):
        """Measure P2P bandwidth — from Confident's profile_bandwidth()."""
        n = int(data_size_mb * 1024 * 1024 / 4)
        src_t = torch.empty(n, dtype=torch.float32, device=src_device)
        for _ in range(3):
            _ = src_t.to(dst_device)
        torch.cuda.synchronize()
        times = []
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.time()
            _ = src_t.to(dst_device)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        avg = np.median(times[3:])
        bw = data_size_mb / avg if avg > 0 else float('inf')
        self.bandwidths[(src_id, dst_id)] = bw

    def set_synthetic_profiles(self, num_layers: int, num_devices: int,
                               batch_sizes: List[int]):
        """Generate synthetic profiles for testing without real hardware."""
        for d in range(num_devices):
            cap = self.devices.get(d, DeviceSpec(device_id=d)).compute_capacity
            for l in range(num_layers):
                for bs in batch_sizes:
                    # Non-linear scaling: smaller BS underutilizes GPU
                    fwd = (0.5 + 0.1 * l) * (bs ** 0.85) / cap
                    bwd = fwd * 2.0
                    self.exec_times[d][l][bs] = (fwd, bwd)
        self.activation_sizes = [128 * 768 * 4] * num_layers  # approx
        self.weight_sizes = [768 * 768 * 4 * 4] * num_layers  # approx
        for d1 in range(num_devices):
            for d2 in range(num_devices):
                if d1 != d2:
                    self.bandwidths[(d1, d2)] = 100.0  # 100 MB/s default

    def get_exec_time(self, device_id: int, layer_idx: int, batch_size: int) \
            -> Tuple[float, float]:
        """Get (fwd_ms, bwd_ms) for a layer on a device at given batch size."""
        times = self.exec_times.get(device_id, {}).get(layer_idx, {})
        if batch_size in times:
            return times[batch_size]
        # Interpolate from nearest profiled batch size
        if not times:
            return (1.0, 2.0)  # fallback
        keys = sorted(times.keys())
        if batch_size <= keys[0]:
            return times[keys[0]]
        if batch_size >= keys[-1]:
            # Extrapolate with power law
            ref_bs, (ref_f, ref_b) = keys[-1], times[keys[-1]]
            ratio = (batch_size / ref_bs) ** 0.85
            return (ref_f * ratio, ref_b * ratio)
        # Linear interpolation
        for i in range(len(keys) - 1):
            if keys[i] <= batch_size <= keys[i + 1]:
                lo, hi = keys[i], keys[i + 1]
                alpha = (batch_size - lo) / (hi - lo)
                f = times[lo][0] * (1 - alpha) + times[hi][0] * alpha
                b = times[lo][1] * (1 - alpha) + times[hi][1] * alpha
                return (f, b)
        return (1.0, 2.0)


class AsteroidPlanner:
    """Dynamic Programming HPP Planner — implements Section 3.3 of the paper.

    Finds optimal: model partition, device grouping, micro-batch allocation
    to minimize HPP-Round Latency under memory constraints.
    """

    def __init__(self, profiler: AsteroidProfiler, cfg: AsteroidConfig,
                 devices: List[DeviceSpec]):
        self.profiler = profiler
        self.cfg = cfg
        self.devices = sorted(devices, key=lambda d: d.memory_budget_mb, reverse=True)
        self.N = len(devices)
        self.L = cfg.num_layers
        self.M = cfg.num_microbatches
        self.B = cfg.micro_batch_size

    def _memory_footprint(self, stage_idx: int, num_stages: int,
                          start_l: int, end_l: int, batch_size: int) -> float:
        """Compute memory footprint for a stage (Eq. 3 in paper).
        Mem_p(β) = Mem_MOD + Mem_OPT + K_p × Mem_ACT(β)
        K_p = 2*(P-p) - 1 for optimal 1F1B (Section 3.2)
        """
        P = num_stages
        K_p = max(1, 2 * (P - stage_idx) - 1)

        # Model weights + optimizer (Adam: 2x model for m,v)
        weight_bytes = sum(self.profiler.weight_sizes[l]
                          for l in range(start_l, end_l)
                          if l < len(self.profiler.weight_sizes))
        mem_mod = weight_bytes
        mem_opt = weight_bytes * 2  # Adam states

        # Activation memory per micro-batch
        mem_act = sum(self.profiler.activation_sizes[l]
                     for l in range(start_l, end_l)
                     if l < len(self.profiler.activation_sizes)) * batch_size

        total = mem_mod + mem_opt + K_p * mem_act
        return total / (1024 * 1024)  # Convert to MB

    def _alloc_microbatch(self, device_ids: List[int], start_l: int, end_l: int,
                          micro_bs: int) -> Tuple[Dict[int, int], float]:
        """Algorithm 1: Memory-aware micro-batch allocation within a device group.

        Phase 1: Distribute proportional to compute capacity under memory budget.
        Phase 2: Offload straggler work to fastest device with spare memory.

        Returns: (allocation dict, execution time of slowest device)
        """
        if not device_ids:
            return {}, float('inf')

        devices_here = [self.devices[i] if i < len(self.devices)
                        else DeviceSpec(device_id=i) for i in device_ids]

        # Phase 1: Memory-aware balancing
        alloc = {d.device_id: 0 for d in devices_here}
        remaining = micro_bs
        active = list(devices_here)

        while remaining > 0 and active:
            total_cap = sum(d.compute_capacity for d in active)
            if total_cap <= 0:
                break
            new_active = []
            for d in active:
                share = max(1, int(round(d.compute_capacity / total_cap * remaining)))
                # Check memory budget
                mem_needed = self._memory_footprint(0, 1, start_l, end_l, share)
                max_bs = share
                while mem_needed > d.memory_budget_mb and max_bs > 1:
                    max_bs -= 1
                    mem_needed = self._memory_footprint(0, 1, start_l, end_l, max_bs)
                actual = min(share, max_bs, remaining)
                alloc[d.device_id] += actual
                remaining -= actual
                if mem_needed < d.memory_budget_mb * 0.95:
                    new_active.append(d)
            active = new_active

        # Phase 2: Straggler offloading
        def exec_time(did, bs):
            if bs <= 0:
                return 0.0
            total = 0.0
            for l in range(start_l, end_l):
                f, b = self.profiler.get_exec_time(did, l, bs)
                total += f + b
            return total

        for _ in range(5):  # max offload iterations
            times = {did: exec_time(did, bs) for did, bs in alloc.items() if bs > 0}
            if not times:
                break
            slowest = max(times, key=times.get)
            fastest = min(times, key=times.get)
            if slowest == fastest or alloc[slowest] <= 1:
                break
            # Try moving 1 sample
            old_time = times[slowest]
            alloc[slowest] -= 1
            alloc[fastest] += 1
            new_time = max(exec_time(slowest, alloc[slowest]),
                          exec_time(fastest, alloc[fastest]))
            if new_time >= old_time:
                alloc[slowest] += 1
                alloc[fastest] -= 1
                break

        straggler_time = max(exec_time(did, bs) for did, bs in alloc.items()) \
            if any(bs > 0 for bs in alloc.values()) else float('inf')
        return alloc, straggler_time

    def _comm_time_inter_stage(self, layer_idx: int, src_group: List[int],
                                dst_group: List[int],
                                batch_size: int = 0) -> float:
        """Communication time for inter-stage activation transfer."""
        if not src_group or not dst_group:
            return 0.0
        bs = batch_size if batch_size > 0 else self.B
        act_size = self.profiler.activation_sizes[min(layer_idx,
            len(self.profiler.activation_sizes) - 1)] * bs
        act_size_mb = act_size / (1024 * 1024)

        # Use minimum bandwidth pair (bottleneck) — paper Eq. 5
        min_bw = float('inf')
        for s in src_group:
            for d in dst_group:
                bw = self.profiler.bandwidths.get((s, d), self.cfg.d2d_bandwidth_mbps)
                min_bw = min(min_bw, bw)
        if min_bw <= 0:
            return float('inf')
        return 2 * act_size_mb / min_bw * 1000  # ms (fwd + bwd)

    def _allreduce_time(self, device_group: List[int], start_l: int,
                         end_l: int) -> float:
        """AllReduce time for gradient sync within a device group — paper Eq. 5."""
        g_size = len(device_group)
        if g_size <= 1:
            return 0.0
        weight_bytes = sum(self.profiler.weight_sizes[l]
                          for l in range(start_l, end_l)
                          if l < len(self.profiler.weight_sizes))
        # Ring AllReduce: 2*(|G|-1)/|G| * total_bytes / min_bandwidth
        min_bw = float('inf')
        for i, d1 in enumerate(device_group):
            for d2 in device_group:
                if d1 != d2:
                    bw = self.profiler.bandwidths.get((d1, d2),
                        self.cfg.d2d_bandwidth_mbps)
                    min_bw = min(min_bw, bw)
        if min_bw <= 0:
            return float('inf')
        vol_mb = 2 * (g_size - 1) / g_size * weight_bytes / (1024 * 1024)
        return vol_mb / min_bw * 1000  # ms

    def plan(self) -> HPPPlanConfig:
        """Dynamic Programming HPP Planning — Algorithm 2 from the paper.

        Searches over: number of stages P, layer partitions, device groupings.
        Minimizes HPP-Round Latency (Eq. 4).

        Q(l, n, p) = min HPP-Round Latency with last l layers on last n devices
                      split into p stages.
        """
        L, N = self.L, self.N
        device_ids = [d.device_id for d in self.devices]
        best_plan = HPPPlanConfig()
        best_latency = float('inf')

        # Search over number of stages
        max_stages = min(L, N, self.cfg.num_stages + 2)  # search neighborhood
        for P in range(1, max_stages + 1):
            result = self._dp_plan(L, N, P, device_ids)
            if result is not None and result.estimated_latency_ms < best_latency:
                best_latency = result.estimated_latency_ms
                best_plan = result

        if best_plan.estimated_latency_ms == float('inf'):
            # Fallback: equal partition
            best_plan = self._fallback_plan(device_ids)

        logger.info(f"Planner: {best_plan.num_stages} stages, "
                    f"partition={best_plan.partition_points}, "
                    f"latency={best_plan.estimated_latency_ms:.1f}ms")
        return best_plan

    def _dp_plan(self, L: int, N: int, P: int, device_ids: List[int]) \
            -> Optional[HPPPlanConfig]:
        """Core DP: find optimal partition of L layers into P stages across N devices."""
        if P > L or P > N:
            return None

        INF = float('inf')
        # Q[l][n][p] = (latency, config)
        Q = [[[INF for _ in range(P + 1)] for _ in range(N + 1)] for _ in range(L + 1)]
        Config = [[[None for _ in range(P + 1)] for _ in range(N + 1)] for _ in range(L + 1)]

        # Base: 1 stage using last l layers on last n devices
        # This is the LAST stage (stage P-1), so global_stage_idx = P - 1
        for l in range(1, L + 1):
            for n in range(1, N + 1):
                group = device_ids[N - n:]
                alloc, exec_t = self._alloc_microbatch(group, L - l, L, self.B)
                ar_t = self._allreduce_time(group, L - l, L)
                # For single stage: latency = M * exec_t + ar_t
                lat = self.M * exec_t + ar_t
                if lat < Q[l][n][1]:
                    Q[l][n][1] = lat
                    Config[l][n][1] = {
                        'partition': [L - l],
                        'groups': {0: group},
                        'allocs': {0: alloc}
                    }

        # Fill DP table
        for p in range(2, P + 1):
            for l in range(p, L + 1):
                for n in range(p, N + 1):
                    for l_prime in range(p - 1, l):
                        for n_prime in range(p - 1, n):
                            if Q[l_prime][n_prime][p - 1] >= INF:
                                continue
                            # New stage: layers [L-l, L-l_prime) on devices [N-n, N-n_prime)
                            new_group = device_ids[N - n:N - n_prime]
                            if not new_group:
                                continue
                            start_l = L - l
                            end_l = L - l_prime

                            alloc, exec_t = self._alloc_microbatch(
                                new_group, start_l, end_l, self.B)

                            # Memory check — use global stage index
                            # The new stage being added is the FIRST stage
                            # (stage 0 in the final plan) since we build
                            # from the last stage backward. Its global index
                            # in the P-stage pipeline is: P - p (0-indexed).
                            global_stage_idx = P - p
                            mem_ok = True
                            for did, bs in alloc.items():
                                if bs > 0:
                                    mem = self._memory_footprint(
                                        global_stage_idx, P, start_l, end_l, bs)
                                    dev = self.devices[did] if did < len(self.devices) \
                                        else DeviceSpec(device_id=did)
                                    if mem > dev.memory_budget_mb:
                                        mem_ok = False
                                        break
                            if not mem_ok:
                                continue

                            # Inter-stage communication
                            prev_cfg = Config[l_prime][n_prime][p - 1]
                            prev_last_group = prev_cfg['groups'][p - 2] \
                                if prev_cfg and p - 2 in prev_cfg['groups'] else []
                            # Use total allocated batch size for comm time
                            total_alloc_bs = sum(alloc.values())
                            comm_t = self._comm_time_inter_stage(
                                end_l - 1, new_group, prev_last_group,
                                batch_size=total_alloc_bs) \
                                if prev_last_group else 0.0

                            ar_t = self._allreduce_time(new_group, start_l, end_l)

                            # HPP-Round Latency estimation (Eq. 4, 6)
                            sub_lat = Q[l_prime][n_prime][p - 1]
                            # Dominant step approximation
                            new_step_lat = self.M * exec_t
                            total_lat = max(sub_lat, new_step_lat + comm_t) + ar_t

                            if total_lat < Q[l][n][p]:
                                Q[l][n][p] = total_lat
                                new_config = copy.deepcopy(prev_cfg) if prev_cfg else {
                                    'partition': [], 'groups': {}, 'allocs': {}}
                                new_config['partition'] = [start_l] + \
                                    new_config.get('partition', [])
                                new_config['groups'][p - 1] = new_group  # stage index
                                # Renumber stages
                                groups_new = {}
                                groups_new[0] = new_group
                                for si, g in prev_cfg['groups'].items():
                                    groups_new[si + 1] = g
                                allocs_new = {}
                                allocs_new[0] = alloc
                                for si, a in prev_cfg.get('allocs', {}).items():
                                    allocs_new[si + 1] = a
                                Config[l][n][p] = {
                                    'partition': [start_l] + prev_cfg.get('partition', []),
                                    'groups': groups_new,
                                    'allocs': allocs_new
                                }

        best_lat = Q[L][N][P]
        best_cfg = Config[L][N][P]
        if best_lat >= INF or best_cfg is None:
            return None

        plan = HPPPlanConfig(
            num_stages=P,
            partition_points=best_cfg['partition'],
            device_groups=best_cfg['groups'],
            micro_batch_alloc=best_cfg['allocs'],
            estimated_latency_ms=best_lat
        )
        return plan

    def _fallback_plan(self, device_ids: List[int]) -> HPPPlanConfig:
        """Simple equal partition fallback."""
        P = min(self.cfg.num_stages, len(device_ids), self.L)
        layers_per = self.L // P
        partition = [i * layers_per for i in range(1, P)]
        groups = {}
        devices_per = max(1, len(device_ids) // P)
        for s in range(P):
            start_d = s * devices_per
            end_d = start_d + devices_per if s < P - 1 else len(device_ids)
            groups[s] = device_ids[start_d:end_d]
        allocs = {s: {d: self.B // max(1, len(groups[s]))
                      for d in groups[s]} for s in range(P)}
        return HPPPlanConfig(
            num_stages=P, partition_points=partition,
            device_groups=groups, micro_batch_alloc=allocs,
            estimated_latency_ms=float('inf'))


# =============================================================================
# L6: COMMUNICATION  (DT-FM NCCL send/recv + AllReduce)
# =============================================================================

def _type_torch_to_nccl(torch_dtype):
    """Map torch dtype to CuPy NCCL dtype — from DT-FM."""
    import cupy.cuda.nccl as nccl
    return {
        torch.float32: nccl.NCCL_FLOAT32,
        torch.float:   nccl.NCCL_FLOAT32,
        torch.float16: nccl.NCCL_FLOAT16,
        torch.float64: nccl.NCCL_FLOAT64,
        torch.int32:   nccl.NCCL_INT32,
        torch.int:     nccl.NCCL_INT,
        torch.uint8:   nccl.NCCL_UINT8,
    }[torch_dtype]


def _nccl_send(tensor: torch.Tensor, dst_rank: int,
               nccl_comm, stream: torch.cuda.Stream):
    """Point-to-point send using CuPy NCCL — from DT-FM."""
    assert tensor.is_contiguous() and tensor.is_cuda
    cupy_stream = cupy.cuda.ExternalStream(stream.cuda_stream)
    nccl_comm.send(tensor.data_ptr(), tensor.numel(),
                   _type_torch_to_nccl(tensor.dtype), dst_rank,
                   cupy_stream.ptr)


def _nccl_recv(tensor: torch.Tensor, src_rank: int,
               nccl_comm, stream: torch.cuda.Stream):
    """Point-to-point recv using CuPy NCCL — from DT-FM."""
    assert tensor.is_contiguous() and tensor.is_cuda
    cupy_stream = cupy.cuda.ExternalStream(stream.cuda_stream)
    nccl_comm.recv(tensor.data_ptr(), tensor.numel(),
                   _type_torch_to_nccl(tensor.dtype), src_rank,
                   cupy_stream.ptr)


def _nccl_allreduce(tensor: torch.Tensor, nccl_comm,
                    stream: torch.cuda.Stream):
    """In-place AllReduce sum using CuPy NCCL — from DT-FM."""
    assert tensor.is_contiguous() and tensor.is_cuda
    cupy_stream = cupy.cuda.ExternalStream(stream.cuda_stream)
    nccl_comm.allReduce(tensor.data_ptr(), tensor.data_ptr(),
                        tensor.numel(), _type_torch_to_nccl(tensor.dtype),
                        cupy.cuda.nccl.NCCL_SUM,
                        cupy_stream.ptr)


def setup_nccl_communicators(rank, cfg, pp_rank, dp_rank, cuda_id, dist_store):
    """Create PP and DP NCCL communicators — from DT-FM worker setup."""
    cupy.cuda.Device(cuda_id).use()

    # PP communicator
    pp_group_id = dp_rank
    pp_comm_name = f"asteroid_pp_{pp_group_id}"
    if pp_rank == 0:
        uid = cupy.cuda.nccl.get_unique_id()
        uid_bytes = uid if isinstance(uid, bytes) else np.array(uid).tobytes()
        dist_store.set(f'group-{pp_comm_name}-uid', uid_bytes)
    torch.distributed.barrier()
    if pp_rank != 0:
        uid_bytes = dist_store.get(f'group-{pp_comm_name}-uid')
    pp_nccl_id = uid_bytes if isinstance(uid_bytes, bytes) else bytes(uid_bytes)

    pp_size = cfg.num_stages  # one device per stage in the PP dimension
    pp_nccl = cupy.cuda.nccl.NcclCommunicator(pp_size, pp_nccl_id, pp_rank)

    # DP communicator (within a stage's device group)
    dp_nccl = None
    dp_size = cfg.world_size // cfg.num_stages
    if dp_size > 1:
        dp_comm_name = f"asteroid_dp_{pp_rank}"
        if dp_rank == 0:
            uid_dp = cupy.cuda.nccl.get_unique_id()
            uid_bytes_dp = uid_dp if isinstance(uid_dp, bytes) \
                else np.array(uid_dp).tobytes()
            dist_store.set(f'group-{dp_comm_name}-uid', uid_bytes_dp)
        torch.distributed.barrier()
        if dp_rank != 0:
            uid_bytes_dp = dist_store.get(f'group-{dp_comm_name}-uid')
        dp_nccl_id = uid_bytes_dp if isinstance(uid_bytes_dp, bytes) \
            else bytes(uid_bytes_dp)
        dp_nccl = cupy.cuda.nccl.NcclCommunicator(dp_size, dp_nccl_id, dp_rank)

    return pp_nccl, dp_nccl


# =============================================================================
# L7: FAULT TOLERANCE  (Confident FT + Asteroid topology-driven replication)
# =============================================================================

class AsteroidFaultTolerance:
    """Fault tolerance with topology-driven model replication.

    Implements Section 3.4 of the paper + Confident FT patterns:
    1. Heartbeat-guided failure detection (active heartbeat)
    2. Passive backward-timeout detection (from Confident PassiveFTHandler)
    3. Topology-driven model replication (Asteroid-specific)
    4. LOCAL/GLOBAL weight replication (from Confident ReplicationUtils)
    5. Weight redistribution after failure
    6. State sync + fault commit (from Confident RestartSyncState/CommitFaultSync)
    7. Layer-wise lightweight pipeline re-planning
    """

    def __init__(self, state: AsteroidStateManager, checkpoint_dir: str = "./checkpoints"):
        self.state = state
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Topology replication: backup weights stored on neighboring stage
        self.backup_weights: Dict[int, Dict[str, torch.Tensor]] = {}
        # Confident-style LOCAL/GLOBAL replicas
        self.local_replicas: Dict[int, Dict[str, torch.Tensor]] = {}
        self.global_replicas: Dict[int, Dict[str, torch.Tensor]] = {}
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        # Cross-process heartbeat via torch distributed store
        self._dist_store: Optional[torch.distributed.Store] = None
        self._own_rank: int = -1

    # ── Heartbeat (active failure detection via distributed store) ─

    def start_heartbeat(self, device_id: int, interval_s: float = 5.0):
        """Start heartbeat sender — writes timestamp to the distributed
        store so that other processes (ranks) can read it."""
        self._own_rank = device_id
        try:
            self._dist_store = torch.distributed.distributed_c10d._get_default_store()
        except Exception:
            self._dist_store = None
            logger.debug("FT: No distributed store available, "
                         "heartbeat will be local-only")

        def _beat():
            while not self._stop_event.is_set():
                ts_bytes = str(time.time()).encode('utf-8')
                if self._dist_store is not None:
                    self._dist_store.set(f"hb_{device_id}", ts_bytes)
                self._stop_event.wait(interval_s)
        self._heartbeat_thread = threading.Thread(target=_beat, daemon=True)
        self._heartbeat_thread.start()

    def stop_heartbeat(self):
        self._stop_event.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2.0)

    def check_device_alive(self, device_id: int, timeout_s: float = 15.0) -> bool:
        """Check if a device is alive by reading its heartbeat from the
        distributed store (cross-process safe)."""
        if self._dist_store is None:
            return True  # can't check, assume alive
        try:
            ts_bytes = self._dist_store.get(f"hb_{device_id}")
            last = float(ts_bytes.decode('utf-8'))
            return (time.time() - last) < timeout_s
        except Exception:
            return True  # key not yet written, assume alive

    # ── Passive backward-timeout detection (from Confident) ──────

    def detect_failure(self, iter_id: int, timeout_ms: float) -> bool:
        """Check if backward was received within timeout.
        Maps to setPassiveTimeout() in Confident's TrainCentral.java /
        PassiveFTHandler.backwardTimeoutHandler().
        """
        received = self.state.get_received_iter_ids()
        if iter_id not in received and self.state.system_status == "NORMAL":
            logger.warning(f"FT: Backward not received for iter {iter_id} "
                           f"within {timeout_ms}ms")
            return True  # failure detected
        return False

    def handle_passive_timeout(self, iter_id: int,
                               stage_models: Optional[Dict[int, nn.Module]] = None,
                               planner: Optional['AsteroidPlanner'] = None):
        """3-phase passive FT recovery — from Confident PassiveFTHandler.
        Phase 1: Mark system as RECOVERING
        Phase 2: Identify failed device, redistribute weights via re-partition
        Phase 3: Sync state and resume from last good iteration
        """
        logger.warning(f"FT: Passive timeout triggered for iter {iter_id}")
        self.state.system_status = "RECOVERING"

        # Phase 1: Identify failed devices via heartbeat (distributed store)
        failed_devices = []
        world_size = self.state.plan.num_stages if self.state.plan else 4
        for did in range(world_size):
            if did == self._own_rank:
                continue  # skip self
            if not self.check_device_alive(did):
                failed_devices.append(did)
                logger.warning(f"FT: Device {did} detected as failed")

        if not failed_devices:
            logger.info("FT: No failed devices found, resuming")
            self.state.system_status = "NORMAL"
            return

        # Phase 2: Redistribute weights
        if stage_models:
            surviving_models = {did: m for did, m in stage_models.items()
                                if did not in failed_devices}
            self.redistribute_weights(failed_devices,
                                      self.state.get_partition_point(),
                                      surviving_models)

        # Phase 3: Re-plan if planner available
        current_plan = self.state.plan
        if planner and current_plan:
            for fd in failed_devices:
                current_plan = self.lightweight_replan(current_plan, fd, planner)
            self.state.plan = current_plan
            new_partition = current_plan.partition_points
            self.commit_fault_sync(iter_id, new_partition)

        self.state.system_status = "NORMAL"
        logger.info(f"FT: Recovery complete, resuming from iter {iter_id}")

    # ── Weight replication (from Confident ReplicationUtils) ──────

    def replicate_to_neighbor(self, stage_model: nn.Module, stage_idx: int,
                               num_stages: int):
        """Topology-driven model replication: backup to next stage's device.
        Last stage backs up to first stage (Section 3.4)."""
        backup_stage = (stage_idx + 1) % num_stages
        state_dict = {k: v.cpu().clone() for k, v in stage_model.state_dict().items()}
        self.backup_weights[stage_idx] = state_dict
        logger.debug(f"FT: Stage {stage_idx} backed up to stage {backup_stage}")

    def replicate_weights(self, replication_type: str,
                          stage_models: Dict[int, nn.Module]):
        """Replicate weights to other devices — from Confident ReplicationUtils.

        LOCAL: save to adjacent device's replica store
        GLOBAL: save to all devices' replica store
        TOPOLOGY: alias for replicate_to_neighbor (Asteroid-specific)
        """
        for device_id, model in stage_models.items():
            if model is not None:
                state_dict = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                if replication_type == "local":
                    self.local_replicas[device_id] = state_dict
                elif replication_type == "global":
                    self.global_replicas[device_id] = state_dict
                elif replication_type == "topology":
                    self.backup_weights[device_id] = state_dict
        logger.info(f"FT: {replication_type.upper()} replication complete "
                    f"for {len(stage_models)} devices")

    # ── Weight restoration & redistribution ───────────────────────

    def restore_from_backup(self, failed_stage: int) -> Optional[Dict[str, torch.Tensor]]:
        """Restore weights from backup after device failure."""
        if failed_stage in self.backup_weights:
            logger.info(f"FT: Restoring stage {failed_stage} from backup")
            return self.backup_weights[failed_stage]
        # Fallback: check local then global replicas
        if failed_stage in self.local_replicas:
            logger.info(f"FT: Restoring stage {failed_stage} from local replica")
            return self.local_replicas[failed_stage]
        if failed_stage in self.global_replicas:
            logger.info(f"FT: Restoring stage {failed_stage} from global replica")
            return self.global_replicas[failed_stage]
        logger.warning(f"FT: No backup found for stage {failed_stage}")
        return None

    def redistribute_weights(self, failed_devices: List[int],
                             current_partition: List[int],
                             surviving_models: Dict[int, nn.Module]):
        """Redistribute weights after re-partitioning.
        Maps to Confident's RedistributionUtils.

        1. Collect all weight tensors from surviving devices
        2. Attempt to restore failed devices from backups
        3. Reassign layers according to surviving topology
        """
        logger.info(f"FT: Redistributing weights. "
                    f"Failed: {failed_devices}, "
                    f"Surviving: {list(surviving_models.keys())}")

        # Collect weights from surviving devices
        all_weights: Dict[int, Dict[str, torch.Tensor]] = {}
        for did, model in surviving_models.items():
            all_weights[did] = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}

        # Try to restore failed devices from backups
        for fd in failed_devices:
            restored = self.restore_from_backup(fd)
            if restored:
                all_weights[fd] = restored
                logger.info(f"FT: Restored stage {fd} weights from backup")
            else:
                logger.warning(f"FT: Could not restore stage {fd}, "
                               f"will need retraining")

        # Store collected weights for later loading by restart_sync_state
        self._redistributed_weights = all_weights
        return all_weights

    # ── State sync & fault commit (from Confident) ────────────────

    def restart_sync_state(self, device_idx: int, workers: Dict[int, str],
                           partition: List[int]):
        """Sync state after restart — maps to Confident's RestartSyncState RPC.
        Updates partition, worker registry, and loads any redistributed weights."""
        self.state.set_partition_point(partition)
        self.state.set_workers(workers)
        logger.info(f"FT: State synced for device {device_idx}, "
                    f"partition={partition}")

    def commit_fault_sync(self, iter_id: int, partition: List[int]):
        """Commit fault sync — maps to Confident's CommitFaultSync RPC.
        Finalizes the new partition and clears recovery state."""
        self.state.set_partition_point(partition)
        # Clear the recorded backward IDs up to the commit point
        # so the next iteration starts fresh
        with self.state._lock:
            self.state._received_iter_ids = {
                iid for iid in self.state._received_iter_ids if iid > iter_id
            }
        logger.info(f"FT: Fault sync committed at iter {iter_id}, "
                    f"partition={partition}")

    # ── Re-planning ───────────────────────────────────────────────

    def lightweight_replan(self, current_plan: HPPPlanConfig,
                           failed_device: int,
                           planner: AsteroidPlanner) -> HPPPlanConfig:
        """Layer-wise lightweight re-planning (Section 3.4).
        Redistributes failed device's workload via FLOPs-proportional migration."""
        new_groups = {}
        for stage_idx, devices in current_plan.device_groups.items():
            new_groups[stage_idx] = [d for d in devices if d != failed_device]

        # Identify empty stage
        empty_stages = [s for s, devs in new_groups.items() if not devs]
        if not empty_stages:
            # Device was in a multi-device group, just remove it
            new_plan = copy.deepcopy(current_plan)
            new_plan.device_groups = new_groups
            return new_plan

        # Full re-plan needed if a stage lost all devices
        logger.info(f"FT: Full re-plan needed, stage {empty_stages} "
                    f"lost all devices")
        return planner.plan()

    # ── Checkpointing ─────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, iter_id: int,
                        model: nn.Module, optimizer: optim.Optimizer):
        path = self.checkpoint_dir / f"ckpt_e{epoch}_i{iter_id}_r{self.state.global_rank}.pt"
        torch.save({
            'epoch': epoch, 'iter_id': iter_id,
            'stage_idx': self.state.stage_idx,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, path)
        return path

    def load_checkpoint(self, path: str) -> Dict:
        return torch.load(path, map_location='cpu', weights_only=False)


# =============================================================================
# L8: ORCHESTRATION — Asteroid Coordinator + Worker with 1F1B HPP
# =============================================================================

def get_lr(it, cfg: AsteroidConfig):
    """Cosine LR schedule with warmup — from DT-FM."""
    if it < cfg.warmup_iters:
        return cfg.lr * (it + 1) / cfg.warmup_iters
    if it > cfg.max_iters:
        return cfg.min_lr
    decay = (it - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


def worker(rank: int, cfg: AsteroidConfig,
           train_data: Tuple[torch.Tensor, torch.Tensor],
           val_data: Tuple[torch.Tensor, torch.Tensor]):
    """Asteroid Worker — deployed on each device.

    Implements:
    - Model stage loading (DT-FM pattern)
    - 1F1B micro-batch scheduling (Asteroid Section 3.2)
    - NCCL P2P communication for pipeline (DT-FM)
    - NCCL AllReduce for intra-stage DP (DT-FM)
    - Topology-driven model replication (Asteroid Section 3.4)
    """
    # ── Device Setup ──────────────────────────────────────────────────
    cuda_id = rank % torch.cuda.device_count()
    device = torch.device(f'cuda:{cuda_id}')
    torch.cuda.set_device(device)
    torch.manual_seed(cfg.seed + rank)

    # Derive PP/DP ranks from the plan
    # Simple mapping: rank = dp_rank * num_stages + pp_rank
    pp_size = cfg.num_stages
    dp_size = cfg.world_size // pp_size
    pp_rank = rank % pp_size
    dp_rank = rank // pp_size
    is_first = (pp_rank == 0)
    is_last = (pp_rank == pp_size - 1)

    # ── State Manager ─────────────────────────────────────────────────
    state = AsteroidStateManager()
    state.global_rank = rank
    state.stage_idx = pp_rank
    state.device = device
    state._dp_rank = dp_rank
    state._dp_size = dp_size

    # ── Initialize torch.distributed (gloo) ───────────────────────────
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend='gloo', init_method=cfg.dist_url,
            world_size=cfg.world_size, rank=rank,
            timeout=timedelta(seconds=120))
    print(f"[RANK {rank}] pp_rank={pp_rank}/{pp_size} dp_rank={dp_rank}/{dp_size} "
          f"device=cuda:{cuda_id}", flush=True)

    # ── Create process groups (DT-FM pattern) ─────────────────────────
    pp_process_group = None
    pp_ranks_in_group = None
    for d in range(dp_size):
        pp_ranks = [d * pp_size + s for s in range(pp_size)]
        grp = torch.distributed.new_group(ranks=pp_ranks)
        if rank in pp_ranks:
            pp_process_group = grp
            pp_ranks_in_group = pp_ranks

    dp_process_group = None
    dp_ranks_in_group = None
    for s in range(pp_size):
        dp_ranks = [d * pp_size + s for d in range(dp_size)]
        grp = torch.distributed.new_group(ranks=dp_ranks)
        if rank in dp_ranks:
            dp_process_group = grp
            dp_ranks_in_group = dp_ranks

    # ── NCCL Communicators (DT-FM pattern) ────────────────────────────
    dist_store = torch.distributed.distributed_c10d._get_default_store()
    pp_nccl, dp_nccl = setup_nccl_communicators(
        rank, cfg, pp_rank, dp_rank, cuda_id, dist_store)

    # ── CUDA Streams (DT-FM pattern) ──────────────────────────────────
    comp_stream = torch.cuda.default_stream(device=device)
    recv_stream = torch.cuda.Stream(device=device, priority=-1)
    send_stream = torch.cuda.Stream(device=device, priority=-1)
    dp_stream = torch.cuda.Stream(device=device, priority=-1) if dp_size > 1 else None

    # ── Model: Create stage (DT-FM GPTStageBase pattern) ──────────────
    layers_per_stage = cfg.num_layers // pp_size
    start_layer = pp_rank * layers_per_stage
    end_layer = start_layer + layers_per_stage if pp_rank < pp_size - 1 \
        else cfg.num_layers

    model = AsteroidStage(
        cfg, start_layer, end_layer,
        is_first=is_first, is_last=is_last).to(device)

    # Broadcast weights within DP group for exact sync
    for param in model.parameters():
        p_cpu = param.data.cpu()
        torch.distributed.broadcast(p_cpu, src=dp_ranks_in_group[0],
                                     group=dp_process_group)
        param.data.copy_(p_cpu.to(device))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[RANK {rank}] Stage [{start_layer},{end_layer}) "
          f"{n_params:,} params", flush=True)

    # ── Flatten params for DP AllReduce (DT-FM pattern) ───────────────
    flat_param = flatten_params(model.parameters()) if dp_size > 1 else None

    # ── Optimizer (DT-FM pattern) ─────────────────────────────────────
    decay_p = [p for p in model.parameters() if p.dim() >= 2]
    nodecay_p = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_p, "weight_decay": cfg.weight_decay},
        {"params": nodecay_p, "weight_decay": 0.0},
    ], lr=cfg.lr, betas=(0.9, 0.95))

    # ── Fault Tolerance ───────────────────────────────────────────────
    ft = AsteroidFaultTolerance(state, str(Path(cfg.output_dir) / "checkpoints"))
    ft.start_heartbeat(rank, cfg.heartbeat_interval_s)

    # ── Data ──────────────────────────────────────────────────────────
    train_embeds, train_labels = train_data
    val_embeds, val_labels = val_data

    # All PP ranks in the same DP column must draw the SAME batch indices
    # for each (iter, micro-batch) so first-stage inputs match last-stage labels.
    # Use deterministic hashing instead of a shared RNG (since different
    # PP stages call sample_batch different numbers of times, diverging RNG state).
    def sample_batch_for(embeds, labels, bs, iter_num, micro_idx):
        """Deterministic batch sampling keyed by (dp_rank, iter, micro_idx)."""
        seed = cfg.seed * 1000003 + dp_rank * 100003 + iter_num * 997 + micro_idx
        g = torch.Generator()
        g.manual_seed(seed)
        ix = torch.randint(len(embeds), (bs,), generator=g)
        return embeds[ix].float().to(device), labels[ix].long().to(device)

    # ── Activation buffers ────────────────────────────────────────────
    num_micro = cfg.num_microbatches
    act_shape = (cfg.micro_batch_size, cfg.max_seq_len, cfg.embedding_dim)

    pp_prev = pp_rank - 1 if pp_rank > 0 else None
    pp_next = pp_rank + 1 if pp_rank < pp_size - 1 else None

    input_bufs = [torch.zeros(act_shape, requires_grad=True, device=device)
                  for _ in range(num_micro)] if not is_first else None
    grad_bufs = [torch.zeros(act_shape, device=device)
                 for _ in range(num_micro)] if not is_last else None

    fwd_recv_ready = [torch.cuda.Event() for _ in range(num_micro)]
    fwd_comp_ready = [torch.cuda.Event() for _ in range(num_micro)]
    bwd_recv_ready = [torch.cuda.Event() for _ in range(num_micro)]
    bwd_comp_ready = [torch.cuda.Event() for _ in range(num_micro)]

    # ── 1F1B Schedule Parameters (Asteroid Section 3.2) ───────────────
    # K_p = 2*(P-p)-1: number of FWD passes before strict 1F1B
    P = pp_size
    K_p = max(1, 2 * (P - pp_rank) - 1)
    # In practice, we use standard 1F1B with warmup = P - pp_rank - 1
    warmup_microbatches = min(num_micro, P - pp_rank - 1) if pp_rank < P - 1 else 0
    steady_microbatches = num_micro - warmup_microbatches
    print(f"[RANK {rank}] 1F1B: K_p={K_p} warmup={warmup_microbatches} "
          f"steady={steady_microbatches}", flush=True)

    # ── Training Loop ─────────────────────────────────────────────────
    EVENT_LOGGER.set_epoch_start(time.time())
    best_val_loss = float('inf')
    t0 = time.time()

    for iter_num in range(cfg.max_iters):
        model.train()
        optimizer.zero_grad()

        lr = get_lr(iter_num, cfg)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Zero input grads
        if input_bufs:
            for buf in input_bufs:
                if buf.grad is not None:
                    buf.grad.zero_()

        micro_losses = []
        cached_outputs = [None] * num_micro
        loss_scale = 1.0 / num_micro  # Scale loss by 1/M for correct gradient averaging

        # ══════════════════════════════════════════════════════════════
        #  1F1B PIPELINE SCHEDULE (Asteroid Section 3.2)
        #
        #  Phase 1: Warmup forward passes (warmup_microbatches)
        #  Phase 2: Steady state 1F1B (forward + backward alternating)
        #  Phase 3: Cooldown backward passes
        # ══════════════════════════════════════════════════════════════

        # ── Phase 1: Warmup FWD ───────────────────────────────────
        for m in range(warmup_microbatches):
            if is_first:
                x, y = sample_batch_for(train_embeds, train_labels,
                                         cfg.micro_batch_size, iter_num, m)
                with torch.cuda.stream(comp_stream):
                    out = model(x)
                    cached_outputs[m] = out
                    comp_stream.record_event(fwd_comp_ready[m])
                torch.cuda.synchronize()
                if pp_next is not None:
                    with torch.cuda.stream(send_stream):
                        send_stream.wait_event(fwd_comp_ready[m])
                        _nccl_send(out.detach().contiguous(), pp_next,
                                   pp_nccl, send_stream)
                    torch.cuda.synchronize()
            else:
                with torch.cuda.stream(recv_stream):
                    _nccl_recv(input_bufs[m], pp_prev, pp_nccl, recv_stream)
                    recv_stream.record_event(fwd_recv_ready[m])
                torch.cuda.synchronize()
                with torch.cuda.stream(comp_stream):
                    comp_stream.wait_event(fwd_recv_ready[m])
                    inp = input_bufs[m]
                    if not inp.requires_grad:
                        inp = inp.requires_grad_(True)
                        input_bufs[m] = inp
                    if is_last:
                        _, y = sample_batch_for(train_embeds, train_labels,
                                                 cfg.micro_batch_size, iter_num, m)
                        loss = model(inp, y) * loss_scale
                        micro_losses.append(loss.item() / loss_scale)
                        cached_outputs[m] = loss
                    else:
                        out = model(inp)
                        cached_outputs[m] = out
                    comp_stream.record_event(fwd_comp_ready[m])
                torch.cuda.synchronize()
                if pp_next is not None:
                    with torch.cuda.stream(send_stream):
                        send_stream.wait_event(fwd_comp_ready[m])
                        _nccl_send(cached_outputs[m].detach().contiguous(),
                                   pp_next, pp_nccl, send_stream)
                    torch.cuda.synchronize()

        # ── Phase 2: Steady-state 1F1B ────────────────────────────
        for m_idx in range(steady_microbatches):
            fwd_m = warmup_microbatches + m_idx
            bwd_m = m_idx  # backward for earlier micro-batch

            # Forward for micro-batch fwd_m
            if is_first:
                x, y = sample_batch_for(train_embeds, train_labels,
                                         cfg.micro_batch_size, iter_num, fwd_m)
                with torch.cuda.stream(comp_stream):
                    out = model(x)
                    cached_outputs[fwd_m] = out
                    comp_stream.record_event(fwd_comp_ready[fwd_m])
                torch.cuda.synchronize()
                if pp_next is not None:
                    with torch.cuda.stream(send_stream):
                        send_stream.wait_event(fwd_comp_ready[fwd_m])
                        _nccl_send(out.detach().contiguous(), pp_next,
                                   pp_nccl, send_stream)
                    torch.cuda.synchronize()
            else:
                with torch.cuda.stream(recv_stream):
                    _nccl_recv(input_bufs[fwd_m], pp_prev, pp_nccl, recv_stream)
                    recv_stream.record_event(fwd_recv_ready[fwd_m])
                torch.cuda.synchronize()
                with torch.cuda.stream(comp_stream):
                    comp_stream.wait_event(fwd_recv_ready[fwd_m])
                    inp = input_bufs[fwd_m]
                    if not inp.requires_grad:
                        inp = inp.requires_grad_(True)
                        input_bufs[fwd_m] = inp
                    if is_last:
                        _, y = sample_batch_for(train_embeds, train_labels,
                                                 cfg.micro_batch_size, iter_num, fwd_m)
                        loss = model(inp, y) * loss_scale
                        micro_losses.append(loss.item() / loss_scale)
                        cached_outputs[fwd_m] = loss
                    else:
                        out = model(inp)
                        cached_outputs[fwd_m] = out
                    comp_stream.record_event(fwd_comp_ready[fwd_m])
                torch.cuda.synchronize()
                if pp_next is not None:
                    with torch.cuda.stream(send_stream):
                        send_stream.wait_event(fwd_comp_ready[fwd_m])
                        _nccl_send(cached_outputs[fwd_m].detach().contiguous(),
                                   pp_next, pp_nccl, send_stream)
                    torch.cuda.synchronize()

            # Backward for micro-batch bwd_m
            if is_last:
                with torch.cuda.stream(comp_stream):
                    out = cached_outputs[bwd_m]
                    if out is not None:
                        if out.dim() == 0:  # loss scalar
                            out.backward()
                        else:
                            out.backward(torch.ones_like(out))
                        comp_stream.record_event(bwd_comp_ready[bwd_m])
                torch.cuda.synchronize()
                if pp_prev is not None and input_bufs and input_bufs[bwd_m].grad is not None:
                    with torch.cuda.stream(send_stream):
                        send_stream.wait_event(bwd_comp_ready[bwd_m])
                        _nccl_send(input_bufs[bwd_m].grad.contiguous(),
                                   pp_prev, pp_nccl, send_stream)
                    torch.cuda.synchronize()
            else:
                with torch.cuda.stream(recv_stream):
                    _nccl_recv(grad_bufs[bwd_m], pp_next, pp_nccl, recv_stream)
                    recv_stream.record_event(bwd_recv_ready[bwd_m])
                torch.cuda.synchronize()
                with torch.cuda.stream(comp_stream):
                    comp_stream.wait_event(bwd_recv_ready[bwd_m])
                    out = cached_outputs[bwd_m]
                    if out is not None:
                        out.backward(gradient=grad_bufs[bwd_m])
                    comp_stream.record_event(bwd_comp_ready[bwd_m])
                torch.cuda.synchronize()
                if pp_prev is not None and input_bufs and input_bufs[bwd_m].grad is not None:
                    with torch.cuda.stream(send_stream):
                        send_stream.wait_event(bwd_comp_ready[bwd_m])
                        _nccl_send(input_bufs[bwd_m].grad.contiguous(),
                                   pp_prev, pp_nccl, send_stream)
                    torch.cuda.synchronize()
            cached_outputs[bwd_m] = None

        # ── Phase 3: Cooldown BWD ─────────────────────────────────
        for m in range(steady_microbatches, num_micro):
            if is_last:
                with torch.cuda.stream(comp_stream):
                    out = cached_outputs[m]
                    if out is not None:
                        if out.dim() == 0:
                            out.backward()
                        else:
                            out.backward(torch.ones_like(out))
                        comp_stream.record_event(bwd_comp_ready[m])
                torch.cuda.synchronize()
                if pp_prev is not None and input_bufs and input_bufs[m].grad is not None:
                    with torch.cuda.stream(send_stream):
                        send_stream.wait_event(bwd_comp_ready[m])
                        _nccl_send(input_bufs[m].grad.contiguous(),
                                   pp_prev, pp_nccl, send_stream)
                    torch.cuda.synchronize()
            else:
                with torch.cuda.stream(recv_stream):
                    _nccl_recv(grad_bufs[m], pp_next, pp_nccl, recv_stream)
                    recv_stream.record_event(bwd_recv_ready[m])
                torch.cuda.synchronize()
                with torch.cuda.stream(comp_stream):
                    comp_stream.wait_event(bwd_recv_ready[m])
                    out = cached_outputs[m]
                    if out is not None:
                        out.backward(gradient=grad_bufs[m])
                    comp_stream.record_event(bwd_comp_ready[m])
                torch.cuda.synchronize()
                if pp_prev is not None and input_bufs and input_bufs[m].grad is not None:
                    with torch.cuda.stream(send_stream):
                        send_stream.wait_event(bwd_comp_ready[m])
                        _nccl_send(input_bufs[m].grad.contiguous(),
                                   pp_prev, pp_nccl, send_stream)
                    torch.cuda.synchronize()
            cached_outputs[m] = None

        # ── DP AllReduce (DT-FM pattern) ──────────────────────────
        if dp_size > 1 and flat_param is not None and dp_nccl is not None:
            bwd_ready = torch.cuda.Event()
            comp_stream.record_event(bwd_ready)
            with torch.cuda.stream(dp_stream):
                dp_stream.wait_event(bwd_ready)
                _nccl_allreduce(flat_param.grad.data, dp_nccl, dp_stream)
            torch.cuda.synchronize()
            flat_param.grad.data.div_(dp_size)

        # ── Gradient clip + optimizer step ────────────────────────
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        torch.cuda.synchronize()
        torch.distributed.barrier()

        # ── Fault tolerance checks (periodic) ─────────────────────
        if iter_num % cfg.ft_check_interval == 0 and iter_num > 0:
            # Record that this iteration's backward completed
            state.record_backward(iter_num)

            # Passive timeout: check if previous iteration was acknowledged
            if iter_num > cfg.ft_check_interval:
                prev_iter = iter_num - cfg.ft_check_interval
                if ft.detect_failure(prev_iter, cfg.backward_timeout_ms):
                    logger.warning(f"[RANK {rank}] FT: Passive timeout "
                                   f"at iter {prev_iter}")
                    ft.handle_passive_timeout(prev_iter)

            # Active heartbeat: verify neighbor stages are alive
            if pp_prev is not None:
                prev_rank = rank - 1
                if not ft.check_device_alive(prev_rank,
                                             cfg.heartbeat_timeout_s):
                    logger.warning(f"[RANK {rank}] FT: Upstream stage "
                                   f"(rank {prev_rank}) heartbeat lost")
            if pp_next is not None:
                next_rank = rank + 1
                if not ft.check_device_alive(next_rank,
                                             cfg.heartbeat_timeout_s):
                    logger.warning(f"[RANK {rank}] FT: Downstream stage "
                                   f"(rank {next_rank}) heartbeat lost")

        # ── Weight replication (periodic) ──────────────────────────
        if cfg.replication_mode != "none" and iter_num % cfg.replication_interval == 0 \
                and iter_num > 0:
            if cfg.replication_mode == "topology":
                ft.replicate_to_neighbor(model, pp_rank, pp_size)
            elif cfg.replication_mode in ("local", "global"):
                ft.replicate_weights(cfg.replication_mode,
                                     {pp_rank: model})

        # ── Logging ───────────────────────────────────────────────
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if is_last and iter_num % cfg.log_interval == 0:
            avg_loss = sum(micro_losses) / len(micro_losses) if micro_losses else 0
            tps = cfg.global_batch_size * cfg.max_seq_len / dt if dt > 0 else 0
            print(f"  iter {iter_num:>5d} | loss={avg_loss:.4f} | lr={lr:.2e} | "
                  f"{tps:,.0f} tok/s | dt={dt*1000:.1f}ms", flush=True)

        # ── Evaluation (all stages participate via pipeline) ────
        if iter_num > 0 and iter_num % cfg.eval_interval == 0:
            model.eval()
            val_losses = []
            n_eval = 10
            with torch.no_grad():
                for eval_i in range(n_eval):
                    if is_first:
                        vx, vy = sample_batch_for(val_embeds, val_labels,
                                                    cfg.micro_batch_size,
                                                    cfg.max_iters + iter_num, eval_i)
                        out = model(vx)
                        if is_last:
                            # Single-stage: model returns loss directly
                            vl = model(vx, vy)
                            val_losses.append(vl.item())
                        else:
                            # Send activation to next stage
                            with torch.cuda.stream(send_stream):
                                _nccl_send(out.detach().contiguous(),
                                           pp_next, pp_nccl, send_stream)
                            torch.cuda.synchronize()
                    elif is_last:
                        # Receive from previous stage
                        eval_buf = torch.zeros(act_shape, device=device)
                        with torch.cuda.stream(recv_stream):
                            _nccl_recv(eval_buf, pp_prev, pp_nccl, recv_stream)
                        torch.cuda.synchronize()
                        _, vy = sample_batch_for(val_embeds, val_labels,
                                                    cfg.micro_batch_size,
                                                    cfg.max_iters + iter_num, eval_i)
                        vl = model(eval_buf, vy)
                        val_losses.append(vl.item())
                    else:
                        # Middle stage: recv → fwd → send
                        eval_buf = torch.zeros(act_shape, device=device)
                        with torch.cuda.stream(recv_stream):
                            _nccl_recv(eval_buf, pp_prev, pp_nccl, recv_stream)
                        torch.cuda.synchronize()
                        out = model(eval_buf)
                        with torch.cuda.stream(send_stream):
                            _nccl_send(out.detach().contiguous(),
                                       pp_next, pp_nccl, send_stream)
                        torch.cuda.synchronize()
            if is_last and val_losses:
                vl_avg = sum(val_losses) / len(val_losses)
                print(f"\n  [EVAL] iter {iter_num} | val_loss={vl_avg:.4f}", flush=True)
                if vl_avg < best_val_loss:
                    best_val_loss = vl_avg
            # All stages save their own checkpoint for merge
            ft.save_checkpoint(0, iter_num, model, optimizer)

    # ── Cleanup ───────────────────────────────────────────────────
    trace_path = os.path.join(cfg.output_dir, f"rank{rank}_trace.json")
    EVENT_LOGGER.to_chrome_trace(trace_path)
    ft.stop_heartbeat()
    torch.distributed.barrier()
    print(f"[RANK {rank}] Training complete.", flush=True)
    torch.distributed.destroy_process_group()


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_sst2(cfg: AsteroidConfig):
    """Load SST-2 and pre-embed — reused from DT-FM."""
    try:
        from datasets import load_dataset
        from transformers import GPT2Tokenizer, GPT2Model
    except ImportError:
        print("Install: pip install datasets transformers")
        raise

    sst2 = load_dataset("stanfordnlp/sst2")
    train_raw, val_raw = sst2["train"], sst2["validation"]
    num_train = min(4096, len(train_raw))
    num_val = min(872, len(val_raw))

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    hf_gpt2 = GPT2Model.from_pretrained("gpt2")

    embed_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tok_emb = hf_gpt2.wte.to(embed_dev)
    pos_emb = hf_gpt2.wpe.to(embed_dev)

    def embed(texts, labels, max_len=cfg.max_seq_len, bs=256):
        all_e, all_l = [], []
        for i in range(0, len(texts), bs):
            enc = tokenizer(texts[i:i+bs], padding="max_length", truncation=True,
                           max_length=max_len, return_tensors="pt")
            ids = enc["input_ids"].to(embed_dev)
            with torch.no_grad():
                pos = torch.arange(max_len, device=embed_dev).unsqueeze(0)
                e = tok_emb(ids) + pos_emb(pos)
            all_e.append(e.cpu())
            all_l.append(torch.tensor(labels[i:i+bs], dtype=torch.long))
        return torch.cat(all_e), torch.cat(all_l)

    idx = torch.randperm(len(train_raw))[:num_train].tolist()
    tr_e, tr_l = embed([train_raw[i]["sentence"] for i in idx],
                       [train_raw[i]["label"] for i in idx])
    va_e, va_l = embed([val_raw[i]["sentence"] for i in range(num_val)],
                       [val_raw[i]["label"] for i in range(num_val)])

    del hf_gpt2, tok_emb, pos_emb
    torch.cuda.empty_cache()
    return (tr_e, tr_l), (va_e, va_l)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    cfg = AsteroidConfig(
        # Model
        embedding_dim=768, num_heads=12, num_layers=12, d_ff=3072,
        max_seq_len=128, vocab_size=50257, num_classes=2,
        # Training
        global_batch_size=16, micro_batch_size=4,
        num_microbatches=4,
        lr=3e-4, max_iters=500, warmup_iters=50,
        eval_interval=100, log_interval=10,
        # HPP
        world_size=4, num_stages=2,
        # Communication
        dist_url="tcp://127.0.0.1:29600",
        d2d_bandwidth_mbps=100.0,
        # Fault Tolerance
        replication_mode="topology",
        heartbeat_interval_s=5.0,
        # Output
        output_dir="./asteroid_output",
        dataset="sst2",
    )

    print(f"\n{'='*60}")
    print(f"  Asteroid — Hybrid Pipeline Parallelism for Edge DNN Training")
    print(f"  world_size={cfg.world_size}  stages={cfg.num_stages}  "
          f"DP={cfg.world_size // cfg.num_stages}")
    print(f"  batch={cfg.global_batch_size}  micro={cfg.micro_batch_size}  "
          f"num_micro={cfg.num_microbatches}")
    print(f"  Layers: {cfg.num_layers} total, "
          f"{cfg.num_layers // cfg.num_stages} per stage")
    print(f"{'='*60}\n")

    # ── Profiling Phase (offline, Section 3.1 Step 1) ─────────────────
    print("Phase 1: Profiling (synthetic for demo)...")
    devices = [DeviceSpec(device_id=i, memory_budget_mb=4096.0,
                          compute_capacity=1.0 + 0.5 * (i % 2))
               for i in range(cfg.world_size)]
    profiler = AsteroidProfiler(cfg, devices)
    profiler.set_synthetic_profiles(cfg.num_layers, cfg.world_size,
                                    batch_sizes=[1, 2, 4, 8, 16])

    # ── Planning Phase (offline, Section 3.1 Step 2) ──────────────────
    print("Phase 2: HPP Planning (DP algorithm)...")
    planner = AsteroidPlanner(profiler, cfg, devices)
    plan = planner.plan()
    print(f"  Plan: {plan.num_stages} stages, "
          f"partition={plan.partition_points}, "
          f"groups={plan.device_groups}")

    # ── Execution Phase (Section 3.1 Step 3) ──────────────────────────
    print("\nPhase 3: Execution — Preparing data...")
    train_data, val_data = prepare_sst2(cfg)
    print(f"  Train: {train_data[0].shape}, Val: {val_data[0].shape}")

    print("\nSpawning workers...")
    mp.set_start_method('fork', force=True)
    mp.spawn(worker, args=(cfg, train_data, val_data),
             nprocs=cfg.world_size, join=True)

    print("\nAll workers finished.")

    # ── Merge checkpoints (generic for any PP/DP size) ─────────────────
    import glob
    ckpt_dir = os.path.join(cfg.output_dir, "checkpoints")
    pp_size_final = cfg.num_stages
    dp_size_final = cfg.world_size // pp_size_final
    if os.path.exists(ckpt_dir):
        merged = {}
        all_found = True
        for stage_idx in range(pp_size_final):
            stage_found = False
            # Try each DP replica for this stage
            for dp_idx in range(dp_size_final):
                global_rank = dp_idx * pp_size_final + stage_idx
                pattern = os.path.join(ckpt_dir, f"ckpt_*.pt")
                files = glob.glob(pattern)
                if files:
                    latest = max(files, key=os.path.getmtime)
                    ckpt = torch.load(latest, map_location='cpu',
                                     weights_only=False)
                    for k, v in ckpt['model'].items():
                        merged[f"stage{stage_idx}.{k}"] = v
                    stage_found = True
                    break
            if not stage_found:
                print(f"Warning: No checkpoint found for stage {stage_idx}")
                all_found = False
        if merged:
            merged_path = os.path.join(ckpt_dir, "merged_model.pt")
            torch.save({'model': merged,
                        'config': vars(cfg),
                        'pp_size': pp_size_final}, merged_path)
            n_params = sum(v.numel() for v in merged.values())
            status = "complete" if all_found else "partial"
            print(f"Merged model ({status}): {merged_path} "
                  f"({n_params:,} params, {len(merged)} keys)")