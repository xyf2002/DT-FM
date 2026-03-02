#!/usr/bin/env python3
"""
DT-FM Refactored — Single-Script GPT-2 Distributed Training
============================================================
Combines the full 8-layer DT-FM stack with a GPT-2 training pipeline.
Launches 4 GPU workers with Pipeline Parallelism=2, Data Parallelism=2.

Usage:
    CUDA_VISIBLE_DEVICES=1,2,3,6 python dtfm_gpt2_train.py
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT (must be BEFORE any torch import)
# ═══════════════════════════════════════════════════════════════════════════════
from datetime import timedelta
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1,2,3,6")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ["NCCL_SOCKET_IFNAME"] = "lo"           # force NCCL to use loopback
os.environ["NCCL_P2P_DISABLE"] = "1"              # disable P2P (fixes cross-bus hangs)
os.environ["NCCL_IB_DISABLE"] = "1"               # disable InfiniBand
os.environ["NCCL_SHM_DISABLE"] = "0"              # enable shared memory transport
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
import itertools
import re
import pickle
import signal
import traceback
from abc import ABC, abstractmethod
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, Future
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
from torch.utils.data import (
    Dataset as TorchDataset, IterableDataset,
    DataLoader, TensorDataset
)
from torch.utils.checkpoint import checkpoint as torch_checkpoint

# CuPy + NCCL (optional)
try:
    import cupy
    import cupy.cuda.nccl
    CUPY_NCCL_AVAILABLE = True
    CUPY_AVAILABLE = True
except (ImportError, AttributeError):
    cupy = None
    CUPY_NCCL_AVAILABLE = False
    CUPY_AVAILABLE = False

# PyTorch distributed
try:
    import torch.distributed as dist
    TORCH_DIST_AVAILABLE = True
except ImportError:
    dist = None
    TORCH_DIST_AVAILABLE = False

# scipy (for scheduler)
try:
    from scipy.optimize import linear_sum_assignment
    import scipy.linalg
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# HuggingFace datasets (optional)
try:
    from datasets import load_dataset as hf_load_dataset, load_from_disk
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

# Rich
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich.live import Live
from rich.layout import Layout
from rich import box

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger("DT-FM")
CONSOLE = Console()


# =============================================================================
# L1: CONFIGURATION & STATE
# =============================================================================

@dataclass
class DeviceConfig:
    use_cuda: bool = True
    cuda_id: int = 0
    cuda_num: int = 4
    debug_mem: bool = True

@dataclass
class DistributedConfig:
    dist_backend: str = "cupy_nccl"
    dist_url: str = "tcp://127.0.0.1:9000"
    world_size: int = 4
    pipeline_group_size: int = 4
    data_group_size: int = 1
    rank: int = 0

@dataclass
class ModelConfig:
    seq_length: int = 2048
    embedding_dim: int = 768
    num_layers: int = 4
    num_heads: int = 16
    task: str = "SeqClassification"
    vocab_size: int = 30522
    num_classes: int = 2

@dataclass
class TrainingConfig:
    batch_size: int = 16
    micro_batch_size: int = 4
    lr: float = 0.01
    num_iters: int = 5
    num_epochs: int = 3
    gradient_accumulate_step: int = 1
    seed: int = 1
    weight_decay: float = 0.01

@dataclass
class MixedPrecisionConfig:
    fp16: bool = False
    loss_scale: float = 64.0
    initial_loss_scale: float = 2 ** 32
    min_loss_scale: float = 1.0
    loss_scale_window: float = 1000.0
    hysteresis: int = 2
    use_offload: bool = True

@dataclass
class ParallelConfig:
    pp_mode: str = "gpipe"
    dp_mode: str = "allreduce"
    gradient_accumulate_step: int = 1

@dataclass
class QQPTaskConfig:
    train_data: List[str] = field(default_factory=lambda: ["./task_datasets/data/QQP/train.tsv"])
    valid_data: List[str] = field(default_factory=lambda: ["./task_datasets/data/QQP/test.tsv"])
    tokenizer_type: str = "BertWordPieceLowerCase"
    vocab_file: str = "./task_datasets/data/bert-large-cased-vocab.txt"
    vocab_extra_ids: int = 0
    make_vocab_size_divisible_by: int = 128

@dataclass
class ProfilingConfig:
    profiling: str = "tidy_profiling"
    trace_postfix: str = "default"


class ConfigManager(ABC):
    @abstractmethod
    def get_device_config(self) -> DeviceConfig: ...
    @abstractmethod
    def get_distributed_config(self) -> DistributedConfig: ...
    @abstractmethod
    def get_model_config(self) -> ModelConfig: ...
    @abstractmethod
    def get_training_config(self) -> TrainingConfig: ...
    @abstractmethod
    def get_mixed_precision_config(self) -> MixedPrecisionConfig: ...
    @abstractmethod
    def get_parallel_config(self) -> ParallelConfig: ...
    @abstractmethod
    def get_qqp_task_config(self) -> QQPTaskConfig: ...
    @abstractmethod
    def get_profiling_config(self) -> ProfilingConfig: ...


class DTFMConfigManager(ConfigManager):
    def __init__(self, device=None, distributed=None, model=None, training=None,
                 mixed_precision=None, parallel=None, qqp_task=None, profiling=None):
        self._device = device or DeviceConfig()
        self._distributed = distributed or DistributedConfig()
        self._model = model or ModelConfig()
        self._training = training or TrainingConfig()
        self._mixed_precision = mixed_precision or MixedPrecisionConfig()
        self._parallel = parallel or ParallelConfig()
        self._qqp_task = qqp_task or QQPTaskConfig()
        self._profiling = profiling or ProfilingConfig()

    def get_device_config(self): return self._device
    def get_distributed_config(self): return self._distributed
    def get_model_config(self): return self._model
    def get_training_config(self): return self._training
    def get_mixed_precision_config(self): return self._mixed_precision
    def get_parallel_config(self): return self._parallel
    def get_qqp_task_config(self): return self._qqp_task
    def get_profiling_config(self): return self._profiling


class StateManager(ABC):
    @abstractmethod
    def get_pipeline_parallel_rank(self) -> int: ...
    @abstractmethod
    def set_pipeline_parallel_rank(self, rank: int): ...
    @abstractmethod
    def get_pipeline_parallel_world_size(self) -> int: ...
    @abstractmethod
    def set_pipeline_parallel_world_size(self, size: int): ...
    @abstractmethod
    def get_data_parallel_rank(self) -> int: ...
    @abstractmethod
    def set_data_parallel_rank(self, rank: int): ...
    @abstractmethod
    def get_data_parallel_world_size(self) -> int: ...
    @abstractmethod
    def set_data_parallel_world_size(self, size: int): ...
    @abstractmethod
    def get_device(self) -> torch.device: ...
    @abstractmethod
    def set_device(self, device: torch.device): ...
    @abstractmethod
    def get_global_rank(self) -> int: ...
    @abstractmethod
    def set_global_rank(self, rank: int): ...
    @abstractmethod
    def get_pipeline_comm(self) -> Any: ...
    @abstractmethod
    def set_pipeline_comm(self, comm: Any): ...
    @abstractmethod
    def get_data_parallel_comm(self) -> Any: ...
    @abstractmethod
    def set_data_parallel_comm(self, comm: Any): ...
    @abstractmethod
    def get_current_epoch(self) -> int: ...
    @abstractmethod
    def set_current_epoch(self, epoch: int): ...
    @abstractmethod
    def get_current_iter(self) -> int: ...
    @abstractmethod
    def set_current_iter(self, it: int): ...


class DTFMStateManager(StateManager):
    def __init__(self):
        self._pp_rank = 0
        self._pp_world_size = 1
        self._pp_comm = None
        self._dp_rank = 0
        self._dp_world_size = 1
        self._dp_comm = None
        self._device = torch.device("cpu")
        self._global_rank = 0
        self._current_epoch = 0
        self._current_iter = 0

    def get_pipeline_parallel_rank(self): return self._pp_rank
    def set_pipeline_parallel_rank(self, rank): self._pp_rank = rank
    def get_pipeline_parallel_world_size(self): return self._pp_world_size
    def set_pipeline_parallel_world_size(self, size): self._pp_world_size = size
    def get_data_parallel_rank(self): return self._dp_rank
    def set_data_parallel_rank(self, rank): self._dp_rank = rank
    def get_data_parallel_world_size(self): return self._dp_world_size
    def set_data_parallel_world_size(self, size): self._dp_world_size = size
    def get_device(self): return self._device
    def set_device(self, device): self._device = device
    def get_global_rank(self): return self._global_rank
    def set_global_rank(self, rank): self._global_rank = rank
    def get_pipeline_comm(self): return self._pp_comm
    def set_pipeline_comm(self, comm): self._pp_comm = comm
    def get_data_parallel_comm(self): return self._dp_comm
    def set_data_parallel_comm(self, comm): self._dp_comm = comm
    def get_current_epoch(self): return self._current_epoch
    def set_current_epoch(self, epoch): self._current_epoch = epoch
    def get_current_iter(self): return self._current_iter
    def set_current_iter(self, it): self._current_iter = it

    def init_from_config(self, cfg: DTFMConfigManager):
        dist_cfg = cfg.get_distributed_config()
        dev_cfg = cfg.get_device_config()
        self._global_rank = dist_cfg.rank
        self._pp_world_size = dist_cfg.pipeline_group_size
        self._pp_rank = dist_cfg.rank % dist_cfg.pipeline_group_size
        self._dp_world_size = dist_cfg.data_group_size
        self._dp_rank = dist_cfg.rank // dist_cfg.pipeline_group_size
        if dev_cfg.use_cuda:
            self._device = torch.device('cuda', dev_cfg.cuda_id)
        else:
            self._device = torch.device('cpu')


# =============================================================================
# OBSERVABILITY
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


class DTFMEventLogger:
    def __init__(self):
        self._events: List[TrainingEvent] = []
        self._lock = threading.Lock()
        self._epoch_start_time: float = 0.0
        self._enabled: bool = True

    def set_epoch_start(self, t): self._epoch_start_time = t
    def enable(self): self._enabled = True
    def disable(self): self._enabled = False

    @contextmanager
    def log_event(self, device_id, event_type, iter_id, phase="", **metadata):
        if not self._enabled:
            yield; return
        start = time.time()
        yield
        duration_ms = (time.time() - start) * 1000
        event = TrainingEvent(
            timestamp=start, device_id=device_id, event_type=event_type,
            iter_id=iter_id, phase=phase, duration_ms=duration_ms, metadata=metadata)
        with self._lock:
            self._events.append(event)

    def record_event(self, device_id, event_type, iter_id, phase, duration_ms,
                     timestamp=None, **metadata):
        if not self._enabled: return
        event = TrainingEvent(
            timestamp=timestamp or time.time(), device_id=device_id,
            event_type=event_type, iter_id=iter_id, phase=phase,
            duration_ms=duration_ms, metadata=metadata)
        with self._lock:
            self._events.append(event)

    def get_events(self):
        with self._lock: return list(self._events)

    def clear(self):
        with self._lock: self._events.clear()

    def to_chrome_trace(self, filepath="dtfm_trace.json"):
        trace_events = []
        t0 = self._epoch_start_time or (
            min(e.timestamp for e in self._events) if self._events else 0)
        for ev in self._events:
            trace_events.append({
                "name": ev.event_type, "ph": "X", "pid": ev.device_id,
                "tid": ev.event_type,
                "ts": (ev.timestamp - t0) * 1e6,
                "dur": ev.duration_ms * 1000,
                "args": {"micro-batch": ev.iter_id, **ev.metadata},
            })
        with open(filepath, 'w') as f:
            json.dump(trace_events, f, indent=2)

    def print_rich_summary(self):
        if not self._events:
            CONSOLE.print("[dim]No events recorded.[/]"); return
        stats = defaultdict(lambda: {"count": 0, "total_ms": 0.0, "min_ms": float('inf'), "max_ms": 0.0})
        for ev in self._events:
            key = (ev.device_id, ev.event_type)
            s = stats[key]
            s["count"] += 1
            s["total_ms"] += ev.duration_ms
            s["min_ms"] = min(s["min_ms"], ev.duration_ms)
            s["max_ms"] = max(s["max_ms"], ev.duration_ms)
        table = Table(title="DT-FM Event Summary", box=box.DOUBLE_EDGE, show_lines=True)
        table.add_column("Stage", justify="center", width=6)
        table.add_column("Event", width=22)
        table.add_column("Count", justify="right", width=6)
        table.add_column("Total ms", justify="right", width=11)
        table.add_column("Avg ms", justify="right", width=10)
        for (dev, etype), s in sorted(stats.items()):
            avg = s["total_ms"] / s["count"] if s["count"] else 0
            table.add_row(str(dev), etype, str(s["count"]),
                          f"{s['total_ms']:.1f}", f"{avg:.2f}")
        CONSOLE.print(table)


EVENT_LOGGER = DTFMEventLogger()


# =============================================================================
# L2: COMPUTE BACKEND
# =============================================================================

class ModelArchitecture(ABC):
    @abstractmethod
    def create_embedding_layer(self, vocab_size, embedding_dim, seq_length) -> nn.Module: ...
    @abstractmethod
    def create_transformer_layer(self, embedding_dim, num_heads, feedforward_dim,
                                  use_checkpoint) -> nn.Module: ...
    @abstractmethod
    def create_task_head(self, task, embedding_dim, num_classes, vocab_size) -> nn.Module: ...
    @abstractmethod
    def create_stage(self, stage_type, model_config, vocab_size, num_classes,
                     device) -> nn.Module: ...


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, head_num):
        super().__init__()
        assert model_dim % head_num == 0
        self.model_dim = model_dim
        self.head_num = head_num
        self.split_size = model_dim // head_num
        self.q_linear = nn.Linear(model_dim, model_dim)
        self.v_linear = nn.Linear(model_dim, model_dim)
        self.k_linear = nn.Linear(model_dim, model_dim)
        self.scale = math.sqrt(self.split_size)
        self.out = nn.Linear(model_dim, model_dim)

    def forward(self, x):
        bs = x.size(0)
        k = self.k_linear(x).view(bs, -1, self.head_num, self.split_size)
        q = self.q_linear(x).view(bs, -1, self.head_num, self.split_size)
        v = self.v_linear(x).view(bs, -1, self.head_num, self.split_size)
        k, q, v = k.transpose(1, 2), q.transpose(1, 2), v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        scores = F.softmax(scores, dim=-1)
        scores = torch.matmul(scores, v)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.model_dim)
        return self.out(concat) + x


class TwoLayerMLP(nn.Module):
    def __init__(self, model_dim, feedforward_dim):
        super().__init__()
        self.linear1 = nn.Linear(model_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, model_dim)

    def forward(self, x):
        return x + self.linear2(F.relu(self.linear1(x)))


class GPTTransformerLayer(nn.Module):
    def __init__(self, model_dim, head_num, feedforward_dim=2048,
                 layer_norm_eps=1e-5, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.attn = MultiHeadAttention(model_dim, head_num)
        self.mlp = TwoLayerMLP(model_dim, feedforward_dim)
        self.norm1 = nn.LayerNorm(model_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(model_dim, eps=layer_norm_eps)

    def forward(self, x):
        x = self.norm1(x)
        x = torch_checkpoint(self.attn, x, use_reentrant=False) if self.use_checkpoint else self.attn(x)
        x = self.norm2(x)
        x = torch_checkpoint(self.mlp, x, use_reentrant=False) if self.use_checkpoint else self.mlp(x)
        return x


def _get_position_ids(seq_length, batch_size, device):
    return torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, seq_length)


class GPTEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, seq_length, num_token_types=0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        self.vocab_embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.xavier_normal_(self.vocab_embedding.weight)
        self.position_embedding = nn.Embedding(seq_length, embedding_dim)
        nn.init.xavier_normal_(self.position_embedding.weight)
        self.token_type_embedding = (
            nn.Embedding(num_token_types, embedding_dim) if num_token_types > 0 else None)

    def forward(self, input_ids, position_ids=None, tokentype_ids=None):
        word_emb = self.vocab_embedding(input_ids)
        if position_ids is None:
            position_ids = _get_position_ids(self.seq_length, word_emb.shape[0], word_emb.device)
        pos_emb = self.position_embedding(position_ids)
        embeddings = word_emb + pos_emb
        if tokentype_ids is not None and self.token_type_embedding is not None:
            embeddings = embeddings + self.token_type_embedding(tokentype_ids)
        return embeddings


class SeqClassification(nn.Module):
    def __init__(self, model_dim, num_classes):
        super().__init__()
        self.pooler_layer = nn.Linear(model_dim, model_dim)
        self.fc_layer = nn.Linear(model_dim, num_classes)

    def forward(self, hidden_states, pooler_index=0):
        pooled = torch.tanh(self.pooler_layer(hidden_states[:, pooler_index, :]))
        return self.fc_layer(pooled)


class Seq2SeqClassification(nn.Module):
    def __init__(self, vocab_size, model_dim, layer_norm_eps=1e-5, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.ln_f = nn.LayerNorm(model_dim, eps=layer_norm_eps)
        self.lm_head = nn.AdaptiveLogSoftmaxWithLoss(model_dim, vocab_size, [1000, 2000, 5000])

    def forward(self, x, targets):
        x = self.ln_f(x)
        shift_logits = x[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        if self.use_checkpoint:
            out = torch_checkpoint(self.lm_head, shift_logits.view(-1, self.lm_head.in_features),
                                   shift_labels.view(-1), use_reentrant=False)
            return out[1]
        else:
            out = self.lm_head(shift_logits.view(-1, self.lm_head.in_features), shift_labels.view(-1))
            return out.loss


class GPTStageBase(nn.Module):
    def __init__(self, model_cfg, vocab_size, num_classes):
        super().__init__()
        self._to_cpu = False
        self.task = model_cfg.task
        self._vocab_size = vocab_size
        self._embedding_dim = model_cfg.embedding_dim
        self._seq_length = model_cfg.seq_length
        self._num_classes = num_classes
        self._feedforward_dim = model_cfg.embedding_dim * 4
        self._num_heads = model_cfg.num_heads
        self._num_layers = model_cfg.num_layers

    def _create_first_layer(self):
        return GPTEmbedding(self._vocab_size, self._embedding_dim, self._seq_length)

    def _create_last_layer(self):
        if self.task == 'SeqClassification':
            return SeqClassification(self._embedding_dim, self._num_classes)
        elif self.task == 'Seq2SeqClassification':
            return Seq2SeqClassification(self._vocab_size, self._embedding_dim)
        raise ValueError(f"Unknown task: {self.task}")

    def _create_transformer_layer(self):
        return GPTTransformerLayer(self._embedding_dim, self._num_heads,
                                   self._feedforward_dim, use_checkpoint=True)


class GPTStageFirst(GPTStageBase):
    def __init__(self, model_cfg, vocab_size, num_classes, device):
        super().__init__(model_cfg, vocab_size, num_classes)
        self.device = device
        module_list = [self._create_first_layer()]
        for _ in range(self._num_layers):
            module_list.append(self._create_transformer_layer())
        self.model = nn.Sequential(*module_list).to(device)

    def forward(self, x):
        out = self.model(x.to(self.device))
        return out.cpu() if self._to_cpu else out


class GPTStageMiddle(GPTStageBase):
    def __init__(self, model_cfg, vocab_size, num_classes, device):
        super().__init__(model_cfg, vocab_size, num_classes)
        self.device = device
        module_list = [self._create_transformer_layer() for _ in range(self._num_layers)]
        self.model = nn.Sequential(*module_list).to(device)

    def forward(self, x):
        out = self.model(x.to(self.device)) if self._to_cpu else self.model(x)
        return out.cpu() if self._to_cpu else out


class GPTStageLast(GPTStageBase):
    def __init__(self, model_cfg, vocab_size, num_classes, device):
        super().__init__(model_cfg, vocab_size, num_classes)
        self.device = device
        module_list = [self._create_transformer_layer() for _ in range(self._num_layers)]
        self.model = nn.Sequential(*module_list).to(device)
        self.task_layer = self._create_last_layer().to(device)

    def forward(self, x, target=None):
        if self.task == 'SeqClassification':
            x = self.model(x.to(self.device)) if self._to_cpu else self.model(x)
            out = self.task_layer(x)
            return out.cpu() if self._to_cpu else out
        elif self.task == 'Seq2SeqClassification':
            assert target is not None
            return self.task_layer(self.model(x), target)


class GPTStageSingle(GPTStageBase):
    def __init__(self, model_cfg, vocab_size, num_classes, device):
        super().__init__(model_cfg, vocab_size, num_classes)
        self.device = device
        module_list = [self._create_first_layer()]
        for _ in range(self._num_layers):
            module_list.append(self._create_transformer_layer())
        self.model = nn.Sequential(*module_list).to(device)
        self.task_layer = self._create_last_layer().to(device)

    def forward(self, x, target=None):
        x = self.model(x.to(self.device)) if self._to_cpu else self.model(x)
        if self.task == 'SeqClassification':
            out = self.task_layer(x)
            return out.cpu() if self._to_cpu else out
        elif self.task == 'Seq2SeqClassification':
            assert target is not None
            return self.task_layer(x, target)
        return x.cpu() if self._to_cpu else x


class GPTArchitecture(ModelArchitecture):
    def create_embedding_layer(self, vocab_size, embedding_dim, seq_length):
        return GPTEmbedding(vocab_size, embedding_dim, seq_length)

    def create_transformer_layer(self, embedding_dim, num_heads, feedforward_dim, use_checkpoint=True):
        return GPTTransformerLayer(embedding_dim, num_heads, feedforward_dim, use_checkpoint=use_checkpoint)

    def create_task_head(self, task, embedding_dim, num_classes, vocab_size):
        if task == 'SeqClassification':
            return SeqClassification(embedding_dim, num_classes)
        elif task == 'Seq2SeqClassification':
            return Seq2SeqClassification(vocab_size, embedding_dim)
        raise ValueError(f"Unknown task: {task}")

    def create_stage(self, stage_type, model_config, vocab_size, num_classes, device):
        if stage_type == 'first':
            return GPTStageFirst(model_config, vocab_size, num_classes, device)
        elif stage_type == 'middle':
            return GPTStageMiddle(model_config, vocab_size, num_classes, device)
        elif stage_type == 'last':
            return GPTStageLast(model_config, vocab_size, num_classes, device)
        elif stage_type == 'single':
            return GPTStageSingle(model_config, vocab_size, num_classes, device)
        raise ValueError(f"Unknown stage type: {stage_type}")


class ComputeBackend(ABC):
    @abstractmethod
    def get_model(self) -> nn.Module: ...
    @abstractmethod
    def get_parameters(self) -> List[torch.nn.Parameter]: ...
    @abstractmethod
    def forward(self, input_data, micro_batch_id, target=None) -> torch.Tensor: ...
    @abstractmethod
    def backward(self, micro_batch_id, grad=None, target=None): ...
    @abstractmethod
    def zero_input_grad(self): ...
    @abstractmethod
    def half(self): ...


class DTFMComputeBackend(ComputeBackend):
    def __init__(self, model=None, stage_type='first', device=None, use_fp16=False):
        self._model = model
        self._stage_type = stage_type
        self._device = device or torch.device('cpu')
        self._use_fp16 = use_fp16
        self._dtype = torch.float16 if use_fp16 else torch.float32
        self._cached_outputs = {}
        self._cached_inputs = {}
        if use_fp16 and model:
            self._model.half()

    def get_model(self): return self._model
    def get_parameters(self): return list(self._model.parameters()) if self._model else []

    def forward(self, input_data, micro_batch_id, target=None):
        input_data = input_data.to(self._device, dtype=self._dtype)
        if not input_data.requires_grad and self._stage_type != 'first':
            input_data = input_data.requires_grad_(True)
        self._cached_inputs[micro_batch_id] = input_data
        if self._stage_type in ('last', 'single') and target is not None:
            output = self._model(input_data, target)
        else:
            output = self._model(input_data)
        self._cached_outputs[micro_batch_id] = output
        return output

    def backward(self, micro_batch_id, grad=None, target=None):
        output = self._cached_outputs.get(micro_batch_id)
        input_data = self._cached_inputs.get(micro_batch_id)
        if output is None: return None
        if self._stage_type in ('last', 'single'):
            if target is not None and output.dim() > 1:
                loss = F.cross_entropy(output, target.to(self._device))
                loss.backward()
            else:
                output.backward()
        else:
            if grad is not None:
                output.backward(gradient=grad.to(self._device, dtype=self._dtype))
            else:
                output.backward()
        input_grad = input_data.grad.clone() if input_data is not None and input_data.grad is not None else None
        del self._cached_outputs[micro_batch_id]
        del self._cached_inputs[micro_batch_id]
        return input_grad

    def zero_input_grad(self):
        for inp in self._cached_inputs.values():
            if inp.grad is not None:
                inp.grad.zero_()

    def half(self):
        if self._model:
            self._model.half()
        self._use_fp16 = True
        self._dtype = torch.float16


def get_stage_model(model_cfg, vocab_size, num_classes, device, pp_rank, pp_world_size,
                    architecture=None):
    if architecture is None:
        architecture = GPTArchitecture()
    if pp_world_size == 1:
        stage_type = 'single'
    elif pp_rank == 0:
        stage_type = 'first'
    elif pp_rank == pp_world_size - 1:
        stage_type = 'last'
    else:
        stage_type = 'middle'
    model = architecture.create_stage(stage_type, model_cfg, vocab_size, num_classes, device)
    return model, stage_type


# =============================================================================
# L4: OPTIMIZER BACKEND (Fp16Optimizer, GradScaler, flatten)
# =============================================================================

class _GradScalerBase(ABC):
    def __init__(self, initial_scale, offload=False):
        self._scale = (torch.FloatTensor([initial_scale]) if offload
                       else torch.cuda.FloatTensor([initial_scale]))
    @property
    def scale(self): return self._scale
    @property
    def inv_scale(self): return self._scale.double().reciprocal().float()
    @abstractmethod
    def update(self, found_inf): ...
    @abstractmethod
    def state_dict(self): ...
    @abstractmethod
    def load_state_dict(self, state_dict): ...

class ConstantGradScaler(_GradScalerBase):
    def update(self, found_inf): pass
    def state_dict(self): return {}
    def load_state_dict(self, state_dict): pass

class DynamicGradScaler(_GradScalerBase):
    def __init__(self, initial_scale, offload, min_scale, growth_factor,
                 backoff_factor, growth_interval, hysteresis):
        super().__init__(initial_scale, offload)
        _ft = torch.FloatTensor if offload else torch.cuda.FloatTensor
        self.min_scale = _ft([min_scale])
        self.growth_factor = _ft([growth_factor])
        self.backoff_factor = _ft([backoff_factor])
        self.growth_interval = growth_interval
        self.hysteresis = hysteresis
        self._growth_tracker = 0
        self._hysteresis_tracker = self.hysteresis

    def update(self, found_inf):
        if found_inf:
            self._growth_tracker = 0
            self._hysteresis_tracker -= 1
            if self._hysteresis_tracker <= 0:
                self._scale = torch.max(self._scale * self.backoff_factor, self.min_scale)
        else:
            self._growth_tracker += 1
            if self._growth_tracker == self.growth_interval:
                self._growth_tracker = 0
                self._hysteresis_tracker = self.hysteresis
                self._scale = self._scale * self.growth_factor

    def state_dict(self):
        return {'scale': self._scale, 'growth_tracker': self._growth_tracker,
                'hysteresis_tracker': self._hysteresis_tracker}
    def load_state_dict(self, state_dict):
        self._scale = state_dict['scale'].cuda(torch.cuda.current_device())
        self._growth_tracker = state_dict['growth_tracker']
        self._hysteresis_tracker = state_dict['hysteresis_tracker']


def flatten_params(param_set, chunk=None):
    params = list(param_set)
    weights = [p.data for p in params]
    grads = [p.grad.data if p.grad is not None else torch.zeros_like(p.data) for p in params]
    sizes = [p.numel() for p in params]
    total_size = sum(sizes)
    if chunk:
        total_size = ((total_size + chunk - 1) // chunk) * chunk
    flatten_weights = torch.zeros(total_size, dtype=weights[0].dtype, device=weights[0].device)
    flatten_grads = torch.zeros(total_size, dtype=weights[0].dtype, device=weights[0].device)
    fw_storage = flatten_weights.storage()
    fg_storage = flatten_grads.storage()

    def _set_storage(param, w_storage, g_storage, offset):
        with torch.no_grad():
            z = torch.zeros_like(param.data); z.set_(w_storage, offset, param.shape); param.data = z
            t = torch.zeros_like(param.data); t.set_(g_storage, offset, param.shape); param.grad = t

    offset = 0
    for i, p in enumerate(params):
        flatten_weights[offset:offset + sizes[i]] = weights[i].reshape(-1)
        flatten_grads[offset:offset + sizes[i]] = grads[i].reshape(-1)
        _set_storage(p, fw_storage, fg_storage, offset)
        offset += sizes[i]
    with torch.no_grad():
        flat_param = torch.nn.Parameter(flatten_weights, requires_grad=False)
        flat_param.grad = flatten_grads
        return flat_param


class OptimizerBackend(ABC):
    @abstractmethod
    def create_optimizer(self, model, **kwargs): ...
    @abstractmethod
    def zero_grad(self): ...
    @abstractmethod
    def step(self) -> bool: ...


class DTFMOptimizerBackend(OptimizerBackend):
    def __init__(self, config):
        self._config = config
        self._optimizer = None

    def create_optimizer(self, model, **kwargs):
        train_cfg = self._config.get_training_config()
        lr = kwargs.get('lr', train_cfg.lr)
        self._optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                           weight_decay=train_cfg.weight_decay)
        return self._optimizer

    def zero_grad(self):
        if self._optimizer: self._optimizer.zero_grad()

    def step(self):
        if self._optimizer: self._optimizer.step(); return True
        return False

    def get_loss_scale(self):
        return torch.tensor(1.0)


# =============================================================================
# L7: FAULT TOLERANCE
# =============================================================================

class FaultToleranceBackend(ABC):
    @abstractmethod
    def save_checkpoint(self, state, path): ...
    @abstractmethod
    def load_checkpoint(self, path): ...
    @abstractmethod
    def health_check(self) -> bool: ...
    @abstractmethod
    def on_failure(self, error) -> bool: ...

class DTFMFaultTolerance(FaultToleranceBackend):
    def __init__(self, checkpoint_dir="./checkpoints"):
        self._checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, state, path):
        full_path = os.path.join(self._checkpoint_dir, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        torch.save(state, full_path)

    def load_checkpoint(self, path):
        full_path = os.path.join(self._checkpoint_dir, path)
        if os.path.exists(full_path):
            return torch.load(full_path, map_location='cpu')
        return None

    def health_check(self): return True
    def on_failure(self, error): return False


# =============================================================================
# L5: PROFILING & SCHEDULING
# =============================================================================
# Abstract: ProfilerBackend + SchedulerBackend
# Concrete: DTFMProfiler + DTFMScheduler
# Maps to: OfflineProfiler.java + DynamicScheduler.java + profiler.cpp
#   + DT-FM/pipeline_parallel profiling (CUDA event timing for fwd/bwd/send/recv)
#   + DT-FM/data_parallel profiling  (CUDA event timing for reduce/broadcast/optim)
#   + DT-FM/scheduler (GCMA: peer_delay + peer_bandwidth + gradient/activation sizes)
# =============================================================================


class ProfilerBackend(ABC):
    """Abstract GPU profiler — defines the interface the scheduler consumes."""
    @abstractmethod
    def profile_layer(self, model: nn.Module, input_data: torch.Tensor,
                      num_iterations: int) -> Tuple[float, float, float]: ...
    @abstractmethod
    def profile_bandwidth(self, src_device: torch.device, dst_device: torch.device,
                          data_size_mb: float) -> float: ...
    @abstractmethod
    def get_memory_info(self, device: torch.device) -> Tuple[float, float]: ...
    @abstractmethod
    def get_time_interval(self, device_id: int, start: int, end: int, phase: int) -> float: ...
    @abstractmethod
    def get_output_size(self, layer_idx: int) -> float: ...
    @abstractmethod
    def get_bandwidth(self, device_id: int) -> float: ...
    @abstractmethod
    def get_computing_capacity(self, device_id: int) -> float: ...
    @abstractmethod
    def get_available_memory(self, device_id: int) -> float: ...


class DTFMProfiler(ProfilerBackend):
    """GPU profiler + offline data store.

    Two modes:
    1. Active profiling: ``profile_layer()`` measures real GPU times using
       CUDA events (matching DT-FM's ``tidy_profiling`` mode that records
       forward-compute, backward-compute, send, recv, reduce, broadcast,
       and optimizer-step durations via ``torch.cuda.Event(enable_timing=True)``).
    2. Offline data: stores aggregated results that the scheduler queries
       via the abstract getters (``get_time_interval``, ``get_bandwidth``, etc.).

    The DT-FM scheduler (GCMA in ``heuristic_evolutionary_solver/scheduler.py``)
    consumes:
      - ``peer_delay``  (ms)       — captured by ``profile_bandwidth``
      - ``peer_bandwidth`` (Gbps)  — captured by ``profile_bandwidth``
      - ``send_gradient_size``     — derived from model param count
      - ``send_activation_size``   — derived from d_model × seq_len × batch × dtype
    """

    def __init__(self, num_devices: int = 2):
        self.num_devices = num_devices

        # Per-layer profiles: device → list of {forward_ms, backward_ms, memory_mb}
        self.layer_profiles: Dict[int, List[Dict[str, float]]] = {
            d: [] for d in range(num_devices)
        }

        # Aggregated time intervals: device_id → {(start, end, phase) → time_ms}
        #   phase 0 = forward, phase 1 = backward
        self.time_intervals: Dict[int, Dict[Tuple[int, int, int], float]] = {
            d: {} for d in range(num_devices)
        }

        # Per-layer output sizes (MB) — for comm cost estimation
        self.output_sizes: List[float] = []

        # Per-device P2P bandwidth (MB/s)
        self.bandwidths: List[float] = [0.0] * num_devices

        # Per-device peer delay (ms) — maps to DT-FM scheduler peer_delay
        self.peer_delays: List[float] = [0.0] * num_devices

        # Per-device computing capacity ratio (1.0 = reference speed)
        self.computing_capacities: List[float] = [1.0] * num_devices

        # Per-device available memory (GB)
        self.available_memory: List[float] = [0.0] * num_devices

    # ── Active profiling methods ─────────────────────────────────────

    def profile_layer(self, model: nn.Module, input_data: torch.Tensor,
                      num_iterations: int = 10) -> Tuple[float, float, float]:
        """Profile a single layer/sub-model using CUDA events.

        Matches DT-FM ``tidy_profiling`` pattern:
          - ``forward_comp_start_events`` / ``forward_comp_ready_events``
          - ``backward_comp_start_events`` / ``backward_comp_ready_events``

        Returns ``(fwd_ms, bwd_ms, peak_mem_mb)``.
        """
        device = next(model.parameters()).device
        torch.cuda.set_device(device)
        model.train()

        input_data = input_data.to(device)
        if not input_data.requires_grad:
            input_data = input_data.clone().detach().requires_grad_(True)

        # Warmup (3 iters, matching DT-FM convention)
        for _ in range(3):
            with torch.no_grad():
                _ = model(input_data)
        torch.cuda.synchronize(device)

        # Forward timing via CUDA events (matches dist_gpipe_pipeline_async.py)
        fwd_times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize(device)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(torch.cuda.current_stream(device))
            with torch.no_grad():
                _ = model(input_data)
            e.record(torch.cuda.current_stream(device))
            torch.cuda.synchronize(device)
            fwd_times.append(s.elapsed_time(e))

        # Backward timing via CUDA events
        bwd_times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize(device)
            model.zero_grad()
            if input_data.grad is not None:
                input_data.grad.zero_()
            out = model(input_data)
            grad_out = torch.ones_like(out)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(torch.cuda.current_stream(device))
            out.backward(gradient=grad_out)
            e.record(torch.cuda.current_stream(device))
            torch.cuda.synchronize(device)
            bwd_times.append(s.elapsed_time(e))

        # Peak memory (matches DT-FM's memory tracking via torch.cuda)
        dev_idx = device.index if device.index is not None else 0
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev_idx)
        model.zero_grad()
        out = model(input_data)
        out.backward(gradient=torch.ones_like(out))
        peak_mem_mb = torch.cuda.max_memory_allocated(dev_idx) / (1024 ** 2)

        # Use median of non-warmup samples (skip first 3)
        fwd_ms = float(np.median(fwd_times[3:])) if len(fwd_times) > 3 else float(np.median(fwd_times))
        bwd_ms = float(np.median(bwd_times[3:])) if len(bwd_times) > 3 else float(np.median(bwd_times))

        model.zero_grad()
        torch.cuda.empty_cache()
        return fwd_ms, bwd_ms, peak_mem_mb

    def profile_bandwidth(self, src_device: torch.device, dst_device: torch.device,
                          data_size_mb: float = 100.0) -> float:
        """Measure P2P GPU bandwidth (MB/s) using CUDA events.

        Uses CUDA events on the **destination** device to accurately time the
        D2D copy.  ``torch.cuda.synchronize()`` is called on **both** devices
        before each trial to drain all prior work.

        Maps to DT-FM ``peer_bandwidth[i,j]``.
        """
        num_elements = int(data_size_mb * 1024 * 1024) // 4
        src_tensor = torch.empty(num_elements, dtype=torch.float32, device=src_device)

        # Warmup (sync both devices to flush caches / establish peer mappings)
        for _ in range(5):
            dst_tensor = src_tensor.to(dst_device)
            del dst_tensor
        torch.cuda.synchronize(src_device)
        torch.cuda.synchronize(dst_device)

        times_ms: List[float] = []
        for _ in range(15):
            torch.cuda.synchronize(src_device)
            torch.cuda.synchronize(dst_device)

            # Record on dst device stream — brackets the actual copy
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt   = torch.cuda.Event(enable_timing=True)

            # Set dst device as current so events record on its default stream
            torch.cuda.set_device(dst_device)
            start_evt.record()
            dst_tensor = src_tensor.to(dst_device)
            end_evt.record()
            torch.cuda.synchronize(dst_device)

            elapsed_ms = start_evt.elapsed_time(end_evt)
            times_ms.append(elapsed_ms)
            del dst_tensor

        # Restore device & use median of non-warmup samples
        torch.cuda.set_device(src_device)
        med_ms = float(np.median(times_ms[5:]))
        bandwidth_mbs = (data_size_mb / (med_ms / 1000.0)) if med_ms > 0 else float('inf')
        return bandwidth_mbs

    def profile_latency(self, src_device: torch.device, dst_device: torch.device) -> float:
        """Measure P2P GPU latency in milliseconds using CUDA events.

        Transfers a tiny tensor (4 bytes) to isolate per-transfer overhead
        from bulk bandwidth.  Maps to DT-FM ``peer_delay[i,j]`` in ms.
        """
        tiny = torch.tensor([1.0], device=src_device)

        # Warmup
        for _ in range(5):
            t = tiny.to(dst_device)
            del t
        torch.cuda.synchronize(src_device)
        torch.cuda.synchronize(dst_device)

        times_ms: List[float] = []
        for _ in range(25):
            torch.cuda.synchronize(src_device)
            torch.cuda.synchronize(dst_device)

            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt   = torch.cuda.Event(enable_timing=True)

            torch.cuda.set_device(dst_device)
            start_evt.record()
            t = tiny.to(dst_device)
            end_evt.record()
            torch.cuda.synchronize(dst_device)

            times_ms.append(start_evt.elapsed_time(end_evt))
            del t

        torch.cuda.set_device(src_device)
        return float(np.median(times_ms[5:]))

    def get_memory_info(self, device: torch.device) -> Tuple[float, float]:
        """Returns ``(available_gb, total_gb)``."""
        if not torch.cuda.is_available():
            return (0.0, 0.0)
        props = torch.cuda.get_device_properties(device)
        total_gb = props.total_memory / (1024 ** 3)
        allocated_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
        return (total_gb - allocated_gb, total_gb)

    def build_time_intervals(self, device_id: int, profiles: List[Dict[str, float]]):
        """Build all ``(start, end, phase)`` intervals from per-layer profiles.

        This pre-computes the cumulative forward/backward time for every
        contiguous layer range ``[s, e]`` so the scheduler's DP can query
        ``get_time_interval(dev, s, e, phase)`` in O(1).
        """
        n = len(profiles)
        for s in range(n):
            for e in range(s, n):
                fwd = sum(profiles[i]['forward_ms'] for i in range(s, e + 1))
                bwd = sum(profiles[i]['backward_ms'] for i in range(s, e + 1))
                self.time_intervals[device_id][(s, e, 0)] = fwd
                self.time_intervals[device_id][(s, e, 1)] = bwd

    # ── Offline data getters (read by SchedulerBackend) ──────────────

    def get_time_interval(self, device_id: int, start: int, end: int, phase: int) -> float:
        return self.time_intervals.get(device_id, {}).get((start, end, phase), 0.0)

    def get_output_size(self, layer_idx: int) -> float:
        return self.output_sizes[layer_idx] if layer_idx < len(self.output_sizes) else 0.0

    def get_bandwidth(self, device_id: int) -> float:
        return self.bandwidths[device_id] if device_id < len(self.bandwidths) else 1.0

    def get_computing_capacity(self, device_id: int) -> float:
        return self.computing_capacities[device_id] if device_id < len(self.computing_capacities) else 1.0

    def get_available_memory(self, device_id: int) -> float:
        return self.available_memory[device_id] if device_id < len(self.available_memory) else 0.0


class SchedulerBackend(ABC):
    """Abstract scheduler — determines how layers map to pipeline stages."""
    @abstractmethod
    def calculate_partition_point(self, is_average: bool) -> List[int]: ...
    @abstractmethod
    def calculate_partition_point_memory(self, is_average: bool) -> List[int]: ...


class DTFMScheduler(SchedulerBackend):
    """DP-based dynamic scheduler.

    Uses profiled per-layer times and inter-device bandwidths to find
    the partition of ``num_layers`` across ``num_devices`` pipeline stages
    that minimises the maximum (bottleneck) stage time.

    Algorithm:
      ``dp[i][j]`` = min bottleneck to assign layers ``0..i`` across stages ``0..j``
      Backtrack to recover split points.

    This is the same DP formulation as ``ConfidantScheduler`` / ``DynamicScheduler.java``
    and complements DT-FM's ``GCMA`` solver (which optimises across *multiple*
    pipeline groups under heterogeneous networking).
    """

    def __init__(self, profiler: ProfilerBackend, num_layers: int, num_devices: int):
        self.profiler = profiler
        self.num_layers = num_layers
        self.num_devices = num_devices

    def _get_time(self, device_id: int, start: int, end: int) -> float:
        """Total fwd + bwd time for a layer range on a device."""
        fwd = self.profiler.get_time_interval(device_id, start, end, 0)
        bwd = self.profiler.get_time_interval(device_id, start, end, 1)
        return fwd + bwd

    def _get_comm_time(self, layer_idx: int, device_id: int) -> float:
        """Communication cost for transferring output of ``layer_idx``
        from ``device_id`` to the next stage.
        Maps to DT-FM: ``send_activation_size / peer_bandwidth``.
        """
        output_size = self.profiler.get_output_size(layer_idx)
        bandwidth = self.profiler.get_bandwidth(device_id)
        return output_size / bandwidth if bandwidth > 0 else 0.0

    def calculate_partition_point(self, is_average: bool = True) -> List[int]:
        """DP partition — returns a list of split-point layer indices.

        For ``pp_size`` stages the returned list has ``pp_size - 1`` elements.
        ``points[0]`` means layers ``0..points[0]`` go to stage 0, etc.
        """
        n = self.num_layers
        k = self.num_devices

        if is_average:
            capacities = [self.profiler.get_computing_capacity(d) for d in range(k)]
        else:
            capacities = [1.0] * k

        INF = float('inf')
        dp = [[INF] * k for _ in range(n)]
        split = [[-1] * k for _ in range(n)]

        # Base: all layers 0..i on device 0
        for i in range(n):
            dp[i][0] = self._get_time(0, 0, i) / capacities[0]

        # Fill
        for j in range(1, k):
            for i in range(j, n):
                for m in range(j - 1, i):
                    compute_time = self._get_time(j, m + 1, i) / capacities[j]
                    comm_time = self._get_comm_time(m, j - 1)
                    cost = max(dp[m][j - 1], compute_time + comm_time)
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        split[i][j] = m

        # Backtrack
        points = []
        i, j = n - 1, k - 1
        while j > 0:
            m = split[i][j]
            points.append(m)
            i = m
            j -= 1
        points.reverse()

        bottleneck = dp[n - 1][k - 1]
        print(f"[SCHEDULER] Partition points: {points}, "
              f"Bottleneck: {bottleneck:.2f}ms")
        return points

    def calculate_partition_point_memory(self, is_average: bool = True) -> List[int]:
        """Memory-constrained variant (future). Currently delegates."""
        return self.calculate_partition_point(is_average)


# ═══════════════════════════════════════════════════════════════════════════════
# GCMA Scheduler — Full DT-FM 2-stage optimisation
# ═══════════════════════════════════════════════════════════════════════════════
# Implements the complete algorithm from
#   DT-FM/scheduler/heuristic_evolutionary_solver/scheduler.py
#
# Stage 1 — GCMA (Genetic Crossover + Multi-cycle Assignment):
#   Decides *which GPUs* form each pipeline group and DP replica group.
#   Combines:
#     • Bipartite matching (Hungarian algorithm via scipy.linear_sum_assignment)
#       → optimal P2P pairing between GPU groups for activation transfer
#     • Open-loop TSP (DP-based)
#       → optimal pipeline-stage ordering to minimise total comm cost
#     • Evolutionary solver (GCMA)
#       → population-based search over partition assignments
#
# Stage 2 — DP partition (existing DTFMScheduler):
#   Decides *how many layers* per pipeline stage (bottleneck minimisation).
#
# The GCMA output replaces the hardcoded gpu_map with an optimal one.
# ═══════════════════════════════════════════════════════════════════════════════


class DTFMGCMAScheduler:
    """Full DT-FM GCMA scheduler for heterogeneous GPU topology.

    Given profiled ``peer_delay[i,j]`` and ``peer_bandwidth[i,j]`` matrices
    (all-pairs), determines the optimal assignment of GPUs to pipeline stages
    and DP replicas, minimising ``data_parallel_cost + 2 * pipeline_parallel_cost``.

    Args:
        num_devices:   Total number of GPUs (world_size).
        pp_size:       Number of pipeline stages (called ``way`` in DT-FM).
        dp_size:       Number of DP replicas per stage (called ``partition_size``).
        peer_delay:    ``np.ndarray`` shape ``(num_devices, num_devices)`` — latency in ms.
        peer_bandwidth: ``np.ndarray`` shape ``(num_devices, num_devices)`` — BW in Gbps.
        send_gradient_size:  Gradient size per stage in GB.
        send_activation_size: Activation size per stage in GB.
    """

    def __init__(
        self,
        num_devices: int,
        pp_size: int,
        dp_size: int,
        peer_delay: np.ndarray,
        peer_bandwidth: np.ndarray,
        send_gradient_size: float,
        send_activation_size: float,
    ):
        self.num_devices = num_devices
        self.way = pp_size          # DT-FM naming: ``way`` = num pipeline stages
        self.partition_size = dp_size  # DT-FM naming: ``partition_size`` = DP replicas
        self.peer_delay = peer_delay
        self.peer_bandwidth = peer_bandwidth
        self.send_gradient_size = send_gradient_size
        self.send_activation_size = send_activation_size

        assert num_devices == pp_size * dp_size, (
            f"num_devices ({num_devices}) != pp_size ({pp_size}) * dp_size ({dp_size})"
        )

    # ── Cost functions ────────────────────────────────────────────────────

    def compute_data_parallel_cost(self, candidate_partition: List[tuple]) -> float:
        """Max all-reduce communication cost across all DP groups.

        For each partition (DP group), computes the ring-allreduce cost for
        every pair of GPUs in that group:
            ``2 * (delay_ij / 1e3 + gradient_size * 8 / (bandwidth_ij * dp_size))``
        Returns the maximum across all groups.
        """
        data_parallel_cost = float('-inf')
        for partition in candidate_partition:
            within_cost = [0.0] * self.partition_size
            for i in range(self.partition_size):
                for j in range(self.partition_size):
                    if i != j:
                        within_cost[i] += 2 * (
                            self.peer_delay[partition[i], partition[j]] / 1e3
                            + self.send_gradient_size * 8
                            / (self.peer_bandwidth[partition[i], partition[j]]
                               * self.partition_size)
                        )
            if data_parallel_cost < np.max(within_cost):
                data_parallel_cost = np.max(within_cost)
        return data_parallel_cost

    def compute_pipeline_parallel_cost(
        self, candidate_partition: List[tuple]
    ) -> Tuple[float, List[int], List[List]]:
        """Pipeline comm cost using bipartite matching + open-loop TSP.

        1. For every pair of partition groups (i, j), find the optimal
           bipartite matching using the Hungarian algorithm — this pairs
           GPUs across groups to minimise the max activation-transfer time.
        2. Build a cross-partition cost matrix from the matchings.
        3. Solve an open-loop TSP to find the optimal pipeline-stage ordering
           that minimises total inter-stage communication cost.

        Returns:
            (cost, path, match_matrix) where:
              - cost: total pipeline parallel communication cost
              - path: optimal ordering of partition groups as pipeline stages
              - match_matrix: ``match_matrix[i][j]`` = list of (row, col) pairs
        """
        way = self.way
        psz = self.partition_size

        def bipartite_matching(part_0: tuple, part_1: tuple):
            """Hungarian algorithm for optimal P2P pairing between two GPU groups."""
            cost_mat = np.zeros((psz, psz))
            for i in range(psz):
                for j in range(psz):
                    cost_mat[i, j] = (
                        self.peer_delay[part_0[i], part_1[j]] / 1e3
                        + self.send_activation_size * 8
                        / self.peer_bandwidth[part_0[i], part_1[j]]
                    )
            # Descending-order trick: inflate costs one-by-one until the
            # bottleneck matching cost is identified.
            descending = np.argsort(cost_mat.flatten())[::-1]
            inf_weight = 1e6
            for idx in descending:
                r, c = idx // psz, idx % psz
                cur_max = cost_mat[r, c]
                cost_mat[r, c] = inf_weight
                row_ind, col_ind = linear_sum_assignment(cost_mat)
                if cost_mat[row_ind, col_ind].sum() >= inf_weight:
                    return cur_max, list(zip(row_ind, col_ind))

        # Build cross-partition cost matrix and bipartite matches
        cross_cost = np.zeros((way, way))
        match_matrix = [[None] * way for _ in range(way)]

        for i in range(way):
            for j in range(i + 1, way):
                cost, match = bipartite_matching(
                    candidate_partition[i], candidate_partition[j]
                )
                cross_cost[i, j] = cost
                cross_cost[j, i] = cost
                match_matrix[i][j] = match
                match_matrix[j][i] = [(c, r) for r, c in match]

        # Open-loop TSP (DP-based) for optimal pipeline stage ordering
        # Reference: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5388488
        best_cost = float('inf')
        best_path = None

        for start in range(way):
            dp_table = np.full((way, 1 << way), np.inf)
            trace = np.zeros((way, 1 << way), dtype=int)

            def _bitmask(nodes):
                b = 0
                for n in nodes:
                    b += (1 << n)
                return b

            def _solve(node, future):
                if not future:
                    return 0.0
                bm = _bitmask(future)
                if dp_table[node][bm] < np.inf:
                    return dp_table[node][bm]
                best_d = np.inf
                best_next = future[0]
                for nxt in future:
                    nxt_future = [f for f in future if f != nxt]
                    nxt_bm = _bitmask(nxt_future)
                    if dp_table[nxt][nxt_bm] == np.inf:
                        d = cross_cost[node][nxt] + _solve(nxt, nxt_future)
                    else:
                        d = cross_cost[node][nxt] + dp_table[nxt][nxt_bm]
                    if d < best_d:
                        best_d = d
                        best_next = nxt
                dp_table[node][bm] = best_d
                trace[node][bm] = best_next
                return best_d

            future = [n for n in range(way) if n != start]
            cost = _solve(start, future)
            if cost < best_cost:
                best_cost = cost
                # Reconstruct path
                path = [start]
                cur = start
                remaining = list(future)
                while remaining:
                    bm = _bitmask(remaining)
                    nxt = int(trace[cur][bm])
                    path.append(nxt)
                    remaining.remove(nxt)
                    cur = nxt
                best_path = path

        return best_cost, best_path, match_matrix

    # ── GCMA evolutionary solver ──────────────────────────────────────────

    def gcma(
        self,
        population_size: int = 100,
        trails: int = 4900,
        mode: str = "default",
    ) -> Tuple[List[List[int]], List[float], List[float]]:
        """Genetic Crossover + Multi-cycle Assignment (GCMA).

        Evolves a population of GPU-to-partition assignments to minimise
        ``data_parallel_cost + 2 * pipeline_parallel_cost``.

        Returns:
            (partitions, scores, min_scores)
        """
        nd = self.num_devices
        way = self.way
        psz = self.partition_size

        def _to_partition_list(flat: List[int]) -> List[tuple]:
            """Convert flat device list → list of tuples (one per stage)."""
            return [tuple(flat[i:i + psz]) for i in range(0, nd, psz)]

        # ── Genetic operators ─────────────────────────────────────────────

        def five_point_crossover(parent1: list, parent2: list) -> list:
            """5-point crossover between two partition assignments."""
            p1_str = [0] * nd
            p2_str = [0] * nd
            for i in range(nd):
                p1_str[parent1[i]] = i // psz
                p2_str[parent2[i]] = i // psz

            points = list(range(nd))
            random.shuffle(points)
            points = points[:5]

            for pt in points:
                p2_str[pt] = p1_str[pt]

            # Repair: ensure each partition has exactly ``psz`` members
            sizes = [0] * way
            for pidx in p2_str:
                sizes[pidx] += 1
            for i in range(nd):
                if sizes[p2_str[i]] > psz:
                    for j in range(way):
                        if sizes[j] < psz:
                            sizes[j] += 1
                            break
                    sizes[p2_str[i]] -= 1
                    p2_str[i] = j
            return p2_str

        # ── Cyclic partitioning (multi-cycle local search) ────────────────

        def cyclic_partitioning(offspring_str: list) -> list:

            def calculate_gain_default(cur_off, locked_v):
                sizes = [0] * way
                for pidx in cur_off:
                    sizes[pidx] += 1

                gain = np.zeros((nd, way))
                for v, pidx in enumerate(cur_off):
                    if locked_v[v] == 0:
                        gain[v][pidx] = np.inf
                        for t, tpidx in enumerate(cur_off):
                            pp_cost = (
                                self.peer_delay[v, t] / 1e3
                                + self.send_activation_size * 8
                                / self.peer_bandwidth[v, t]
                            )
                            if pidx != tpidx:
                                gain[v][tpidx] += pp_cost / sizes[tpidx]
                            elif v != t:
                                if gain[v][tpidx] > pp_cost:
                                    gain[v][tpidx] = pp_cost

                G_i = np.full(way, np.inf)
                G_i_trace = [[None, None] for _ in range(way)]
                for v, pidx in enumerate(cur_off):
                    if locked_v[v] == 0:
                        if gain[v][pidx] < G_i[pidx]:
                            G_i[pidx] = gain[v][pidx]
                            G_i_trace[pidx][0] = v

                G_i = np.full(way, -np.inf)
                G_ij = np.full((way, way), -np.inf)
                for pidx, trace in enumerate(G_i_trace):
                    v = trace[0]
                    if v is not None:
                        for tpidx, tgain in enumerate(gain[v]):
                            if tpidx != pidx:
                                tgain_net = tgain - gain[v][pidx]
                                if tgain_net > G_ij[pidx, tpidx]:
                                    G_ij[pidx, tpidx] = tgain_net
                                if tgain_net > G_i[pidx]:
                                    G_i[pidx] = tgain_net
                                    G_i_trace[pidx] = [v, tpidx]
                return G_ij, G_i, G_i_trace

            def calculate_gain_baseline(cur_off, locked_v):
                gain = np.zeros((nd, way))
                for v, pidx in enumerate(cur_off):
                    if locked_v[v] == 0:
                        for t, tpidx in enumerate(cur_off):
                            pp_cost = (
                                self.peer_delay[v, t] / 1e3
                                + self.send_activation_size * 8
                                / self.peer_bandwidth[v, t]
                            )
                            dp_cost = (
                                self.peer_delay[v, t] / 1e3
                                + self.send_gradient_size * 8
                                / self.peer_bandwidth[v, t]
                            )
                            if v != t:
                                gain[v][tpidx] += pp_cost
                                gain[v][tpidx] -= dp_cost

                G_i_trace = [[None, None] for _ in range(way)]
                G_i = np.full(way, -np.inf)
                G_ij = np.full((way, way), -np.inf)
                for v, pidx in enumerate(cur_off):
                    if locked_v[v] == 0:
                        for tpidx, tgain in enumerate(gain[v]):
                            if tpidx != pidx:
                                tgain_net = tgain - gain[v][pidx]
                                if tgain_net > G_ij[pidx, tpidx]:
                                    G_ij[pidx, tpidx] = tgain_net
                                if tgain_net > G_i[pidx]:
                                    G_i[pidx] = tgain_net
                                    G_i_trace[pidx] = [v, tpidx]
                return G_ij, G_i, G_i_trace

            def move_cycles(off_str):
                sums = [0]
                locked_part = [0] * way
                locked_v = [0] * nd
                offsprings = [off_str]

                for _ in range(way):
                    cur = offsprings[-1].copy()
                    movements = []
                    epsilon = []
                    tau = []

                    if mode == "default":
                        G_ij, G_i, G_i_trace = calculate_gain_default(cur, locked_v)
                    else:
                        G_ij, G_i, G_i_trace = calculate_gain_baseline(cur, locked_v)

                    S0 = Si = int(np.argmax(G_i))

                    for _ in range(nd):
                        v, Pv = G_i_trace[Si]
                        if v is None:
                            v = movements[-1][0]
                            Pv = S0
                        cur[v] = Pv
                        locked_v[v] = 1
                        locked_part[Pv] = 1
                        movements.append((v, Si, Pv))
                        epsilon.append(G_i[Si])
                        tau.append(G_ij[Si, S0])
                        Si = Pv
                        if Si == S0:
                            break
                        if mode == "default":
                            G_ij, G_i, G_i_trace = calculate_gain_default(cur, locked_v)
                        else:
                            G_ij, G_i, G_i_trace = calculate_gain_baseline(cur, locked_v)

                    # Find best prefix
                    max_sum = 0
                    best_l = 0
                    for i in range(1, len(epsilon)):
                        val = np.sum(epsilon[:i]) + tau[i]
                        if val > max_sum:
                            max_sum = val
                            best_l = i

                    # Undo excess moves
                    for i in range(len(epsilon) - 1, best_l, -1):
                        cur[movements[i][0]] = movements[i][1]
                    cur[movements[best_l][0]] = S0
                    offsprings.append(cur)
                    sums.append(max_sum)

                    if sum(locked_part) == len(locked_part):
                        break

                # Pick best cycle prefix
                max_sum = 0
                best_m = 0
                for i in range(1, len(sums)):
                    if np.sum(sums[:i]) > max_sum:
                        max_sum = np.sum(sums[:i])
                        best_m = i - 1
                return offsprings[best_m]

            for _ in range(1):
                offspring_str = move_cycles(offspring_str)
            return offspring_str

        # ── Population initialisation ─────────────────────────────────────
        nodes = list(range(nd))
        partitions = []
        scores = []
        min_scores = []

        for i in range(population_size):
            cur = nodes.copy()
            random.seed(i)
            random.shuffle(cur)
            partitions.append(cur)

        for part in partitions:
            cp = _to_partition_list(part)
            dp_cost = self.compute_data_parallel_cost(cp)
            pp_cost, _, _ = self.compute_pipeline_parallel_cost(cp)
            scores.append(dp_cost + 2 * pp_cost)
            min_scores.append(np.min(scores))

        # ── Evolution loop ────────────────────────────────────────────────
        for i in range(trails):
            np.random.seed = i
            p1_idx, p2_idx = np.random.randint(population_size, size=2).tolist()
            ga_off = five_point_crossover(partitions[p1_idx], partitions[p2_idx])
            off_str = cyclic_partitioning(ga_off)

            # Convert offspring string back to flat list
            off_flat = [[] for _ in range(way)]
            for v_idx, pidx in enumerate(off_str):
                off_flat[pidx].append(v_idx)
            off_cp = [tuple(g) for g in off_flat]
            off_dp_cost = self.compute_data_parallel_cost(off_cp)
            off_pp_cost, _, _ = self.compute_pipeline_parallel_cost(off_cp)
            off_score = off_dp_cost + 2 * off_pp_cost
            off_list = list(itertools.chain.from_iterable(off_flat))

            if off_score > max(scores[p1_idx], scores[p2_idx]):
                partitions.append(off_list)
                scores.append(off_score)
            else:
                # Replace the worse parent
                replaced = p1_idx if scores[p1_idx] > scores[p2_idx] else p2_idx
                old_part = partitions[replaced]
                partitions[replaced] = off_list
                partitions.append(old_part)
                old_score = scores[replaced]
                scores[replaced] = off_score
                scores.append(old_score)
            min_scores.append(np.min(scores))

        return partitions, scores, min_scores

    # ── Convert GCMA result to gpu_map ────────────────────────────────────

    def get_pipelines(
        self,
        candidate_partition: List[tuple],
        path: List[int],
        match_matrix: List[List],
    ) -> np.ndarray:
        """Convert GCMA output to pipeline assignment matrix.

        Returns ``pipeline_matrix[stage_idx, pipeline_idx]`` = device_id.
        Shape: ``(pp_size, dp_size)``.
        This defines the gpu_map: for pipeline ``p``, stage ``s``, the
        GPU is ``pipeline_matrix[s, p]``.
        """
        way = self.way
        psz = self.partition_size

        pipeline = np.zeros((way, psz), dtype=int)

        for stage_idx, part_idx in enumerate(path):
            if stage_idx > 0:
                last_part_idx = path[stage_idx - 1]
                bm = match_matrix[last_part_idx][part_idx]
                for match in bm:
                    for i in range(psz):
                        if pipeline[stage_idx - 1][i] == match[0]:
                            pipeline[stage_idx][i] = match[1]
            else:
                next_part_idx = path[1] if way > 1 else 0
                bm = match_matrix[part_idx][next_part_idx]
                for i, match in enumerate(bm):
                    pipeline[0][i] = match[0]

        # Map local indices → actual device IDs
        for stage_idx, part_idx in enumerate(path):
            for i in range(psz):
                pipeline[stage_idx][i] = candidate_partition[part_idx][pipeline[stage_idx][i]]

        return pipeline

    def build_gpu_map(self, pipeline_matrix: np.ndarray) -> Dict[int, int]:
        """Convert the pipeline assignment matrix to a ``gpu_map`` dict.

        DT-FM rank layout: ``global_rank = dp_rank * pp_size + pp_rank``
        So for pipeline ``p`` (dp_rank=p), stage ``s`` (pp_rank=s):
            ``global_rank = p * pp_size + s``
            ``cuda_id = pipeline_matrix[s, p]``
        """
        gpu_map = {}
        for p in range(self.partition_size):
            for s in range(self.way):
                global_rank = p * self.way + s
                gpu_map[global_rank] = int(pipeline_matrix[s, p])
        return gpu_map

    def run(
        self,
        population_size: int = 100,
        trails: int = 4900,
        mode: str = "default",
    ) -> Tuple[Dict[int, int], float]:
        """Full GCMA pipeline: evolve → best partition → get_pipelines → gpu_map.

        Returns:
            (gpu_map, total_cost) where gpu_map maps global_rank → cuda_id.
        """
        partitions, scores, _ = self.gcma(population_size, trails, mode)

        # Best partition
        best_idx = int(np.argmin(scores))
        best_flat = partitions[best_idx]
        best_cp = [tuple(best_flat[i:i + self.partition_size])
                    for i in range(0, self.num_devices, self.partition_size)]

        dp_cost = self.compute_data_parallel_cost(best_cp)
        pp_cost, pp_path, pp_match = self.compute_pipeline_parallel_cost(best_cp)
        total_cost = dp_cost + 2 * pp_cost

        pipeline_matrix = self.get_pipelines(best_cp, pp_path, pp_match)
        gpu_map = self.build_gpu_map(pipeline_matrix)

        print(f"[GCMA] Best partition: {best_cp}")
        print(f"[GCMA] Pipeline path: {pp_path}")
        print(f"[GCMA] Pipeline matrix:\n{pipeline_matrix}")
        print(f"[GCMA] DP cost: {dp_cost:.4f}, PP cost: {2*pp_cost:.4f}, "
              f"Total: {total_cost:.4f}")
        print(f"[GCMA] Optimal gpu_map: {gpu_map}")

        return gpu_map, total_cost


def partition_points_to_layers_per_stage(
    partition_points: List[int], num_layers: int, pp_size: int
) -> List[int]:
    """Convert scheduler output to per-stage layer counts.

    ``partition_points`` has ``pp_size - 1`` elements.
    ``partition_points[i]`` = last layer index assigned to stage ``i``.

    Returns list of length ``pp_size`` where element ``s`` is the number
    of layers assigned to pipeline stage ``s``.
    """
    boundaries = partition_points + [num_layers - 1]
    counts = []
    prev = -1
    for b in boundaries:
        counts.append(b - prev)
        prev = b
    assert sum(counts) == num_layers, (
        f"Layer count mismatch: {counts} sums to {sum(counts)} != {num_layers}"
    )
    assert len(counts) == pp_size, (
        f"Stage count mismatch: {len(counts)} != {pp_size}"
    )
    return counts


def run_profiling(
    cfg: 'GPT2DTFMConfig',
    initial_gpu_map: Dict[int, int],
) -> Tuple[Dict[int, int], List[int]]:
    """Pre-training profiling + GCMA scheduling — runs on the main process.

    Two-stage optimisation matching DT-FM/scheduler algorithm:

    **Stage 1 — All-pairs GPU profiling + GCMA:**
      Profiles every GPU (not just pp_size) for per-layer compute time.
      Measures pairwise bandwidth and latency between ALL GPU pairs.
      Runs the GCMA evolutionary solver to determine the *optimal gpu_map*
      (which GPUs form which pipeline group and in what order).

    **Stage 2 — DP layer partitioner:**
      Uses profiled per-layer times on the chosen stage devices to determine
      *how many layers per pipeline stage* (bottleneck minimisation).

    Args:
        cfg: Training config.
        initial_gpu_map: Fallback gpu_map (used if GCMA is unavailable).

    Returns:
        ``(gpu_map, layers_per_stage)`` where:
          - ``gpu_map``: ``Dict[global_rank → cuda_id]``  (from GCMA)
          - ``layers_per_stage``: ``List[int]`` of length ``pp_size``
    """
    pp_size = cfg.pp_size
    dp_size = cfg.dp_size
    world_size = cfg.world_size
    n_layers = cfg.n_layers

    # Discover all unique CUDA device IDs from the initial map
    all_cuda_ids = sorted(set(initial_gpu_map.values()))
    num_devices = len(all_cuda_ids)
    # Map cuda_id → index in our profiling arrays (0..num_devices-1)
    id_to_idx = {cid: idx for idx, cid in enumerate(all_cuda_ids)}
    all_devices = [torch.device('cuda', cid) for cid in all_cuda_ids]

    print(f"\n{'='*60}")
    print(f"  STAGE 1: Full GPU Profiling + GCMA Scheduling")
    print(f"  Profiling {num_devices} GPUs: {all_cuda_ids}")
    print(f"  PP={pp_size}, DP={dp_size}, world_size={world_size}")
    print(f"{'='*60}")

    # ── 1a. Profile per-layer compute time on EVERY GPU ───────────────
    profiler = DTFMProfiler(num_devices=num_devices)

    print(f"\n[PROFILER] Profiling {n_layers} layers on {num_devices} devices...")
    for dev_idx, device in enumerate(all_devices):
        torch.cuda.set_device(device)
        profiles = []

        for layer_idx in range(n_layers):
            block = GPT2Block(cfg).to(device)
            dummy_input = torch.randn(
                cfg.micro_batch_size, cfg.max_seq_len, cfg.d_model,
                device=device, requires_grad=True,
            )
            fwd_ms, bwd_ms, peak_mem = profiler.profile_layer(
                block, dummy_input, num_iterations=10,
            )
            profiles.append({
                'forward_ms': fwd_ms,
                'backward_ms': bwd_ms,
                'memory_mb': peak_mem,
            })
            print(f"  [GPU {all_cuda_ids[dev_idx]}] Layer {layer_idx}: "
                  f"fwd={fwd_ms:.2f}ms, bwd={bwd_ms:.2f}ms, mem={peak_mem:.1f}MB")
            del block, dummy_input
            torch.cuda.empty_cache()

        profiler.layer_profiles[dev_idx] = profiles
        profiler.build_time_intervals(dev_idx, profiles)
        avail_gb, total_gb = profiler.get_memory_info(device)
        profiler.available_memory[dev_idx] = avail_gb
        print(f"  [GPU {all_cuda_ids[dev_idx]}] Memory: {avail_gb:.1f}/{total_gb:.1f} GB available")

    # ── 1b. All-pairs bandwidth and latency ───────────────────────────
    print(f"\n[PROFILER] Measuring all-pairs bandwidth and latency...")
    peer_delay_ms = np.zeros((num_devices, num_devices))
    peer_bandwidth_gbps = np.zeros((num_devices, num_devices))

    for i in range(num_devices):
        for j in range(num_devices):
            if i == j:
                # Same GPU: effectively zero delay, infinite bandwidth
                peer_delay_ms[i, j] = 0.001  # sub-microsecond
                peer_bandwidth_gbps[i, j] = 1000.0  # very high
            else:
                bw_mbs = profiler.profile_bandwidth(all_devices[i], all_devices[j])
                lat_ms = profiler.profile_latency(all_devices[i], all_devices[j])
                peer_bandwidth_gbps[i, j] = bw_mbs * 8 / 1000.0  # MB/s → Gbps
                peer_delay_ms[i, j] = lat_ms
                print(f"  [GPU {all_cuda_ids[i]} → GPU {all_cuda_ids[j]}] "
                      f"BW={bw_mbs:.1f} MB/s ({peer_bandwidth_gbps[i,j]:.2f} Gbps), "
                      f"Latency={lat_ms:.3f} ms")

    # ── 1c. Compute gradient and activation sizes (in GB, matching DT-FM) ──
    # Gradient size per stage: total model params × 4 bytes / pp_size
    # Rough estimate: each transformer layer has ~12 * d_model^2 params
    params_per_layer = 12 * cfg.d_model * cfg.d_model
    total_params = params_per_layer * n_layers
    send_gradient_size_gb = (
        total_params * np.dtype(np.float32).itemsize / pp_size
    ) / (1024 ** 3)

    # Activation size: (micro_batch * seq_len * d_model) × fp32
    send_activation_size_gb = (
        cfg.micro_batch_size * cfg.max_seq_len * cfg.d_model
        * np.dtype(np.float32).itemsize
    ) / (1024 ** 3)

    print(f"\n[GCMA] send_gradient_size = {send_gradient_size_gb:.6f} GB")
    print(f"[GCMA] send_activation_size = {send_activation_size_gb:.6f} GB")

    # ── 1d. Run GCMA ──────────────────────────────────────────────────
    if SCIPY_AVAILABLE and num_devices >= pp_size * dp_size:
        gcma_scheduler = DTFMGCMAScheduler(
            num_devices=num_devices,
            pp_size=pp_size,
            dp_size=dp_size,
            peer_delay=peer_delay_ms,
            peer_bandwidth=peer_bandwidth_gbps,
            send_gradient_size=send_gradient_size_gb,
            send_activation_size=send_activation_size_gb,
        )
        # Use reduced population for faster convergence on small GPU counts
        pop = min(100, max(20, num_devices * 10))
        trails = min(4900, max(500, pop * 50))
        print(f"\n[GCMA] Running GCMA solver (pop={pop}, trails={trails})...")
        gcma_gpu_map, total_cost = gcma_scheduler.run(
            population_size=pop, trails=trails, mode="default",
        )
        # Remap: GCMA gpu_map uses device indices (0..num_devices-1),
        # convert back to actual CUDA IDs
        gpu_map = {rank: all_cuda_ids[cid] for rank, cid in gcma_gpu_map.items()}
        print(f"[GCMA] Final gpu_map (CUDA IDs): {gpu_map}")
    else:
        if not SCIPY_AVAILABLE:
            print("[GCMA] WARNING: scipy not available, using initial gpu_map")
        gpu_map = dict(initial_gpu_map)
        print(f"[GCMA] Using initial gpu_map: {gpu_map}")

    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  STAGE 2: DP Layer Partitioner")
    print(f"{'='*60}")

    # ── 2a. Build profiler for the chosen stage devices ───────────────
    # After GCMA, we know which GPUs are in the first pipeline group
    # (dp_rank=0). Use those for layer-count optimisation.
    stage_profiler = DTFMProfiler(num_devices=pp_size)

    for s in range(pp_size):
        global_rank = s  # dp_rank=0 → pp_rank=s
        cuda_id = gpu_map[global_rank]
        dev_idx = id_to_idx[cuda_id]

        # Copy profiling data from the full profiler
        stage_profiler.layer_profiles[s] = profiler.layer_profiles[dev_idx]
        stage_profiler.time_intervals[s] = {}
        # Remap time_intervals from dev_idx to s
        for key, val in profiler.time_intervals.get(dev_idx, {}).items():
            stage_profiler.time_intervals[s][key] = val
        stage_profiler.available_memory[s] = (
            profiler.available_memory[dev_idx]
            if dev_idx < len(profiler.available_memory) else 0.0
        )

    # Per-layer output sizes for comm cost
    act_size_mb = (cfg.micro_batch_size * cfg.max_seq_len * cfg.d_model * 4) / (1024 * 1024)
    stage_profiler.output_sizes = [act_size_mb] * n_layers

    # P2P bandwidth between adjacent stages (using profiled data)
    for s in range(pp_size - 1):
        r0 = gpu_map[s]
        r1 = gpu_map[s + 1]
        i0, i1 = id_to_idx[r0], id_to_idx[r1]
        bw_mbs = peer_bandwidth_gbps[i0, i1] * 1000.0 / 8  # Gbps → MB/s
        stage_profiler.bandwidths[s] = bw_mbs
        print(f"  [BANDWIDTH] Stage {s} → {s+1} (GPU {r0} → GPU {r1}): {bw_mbs:.1f} MB/s")
    stage_profiler.bandwidths[pp_size - 1] = (
        stage_profiler.bandwidths[pp_size - 2] if pp_size > 1 else 1.0
    )

    # Computing capacity (relative speed)
    if pp_size > 1:
        ref_time = stage_profiler.layer_profiles[0][0]['forward_ms']
        for d in range(pp_size):
            dev_time = stage_profiler.layer_profiles[d][0]['forward_ms']
            stage_profiler.computing_capacities[d] = (
                ref_time / dev_time if dev_time > 0 else 1.0
            )

    # ── 2b. Run DP scheduler ─────────────────────────────────────────
    scheduler = DTFMScheduler(stage_profiler, n_layers, pp_size)
    partition_points = scheduler.calculate_partition_point(is_average=True)
    layers_per_stage = partition_points_to_layers_per_stage(
        partition_points, n_layers, pp_size,
    )

    print(f"\n[PROFILER] gpu_map = {gpu_map}")
    print(f"[PROFILER] layers_per_stage = {layers_per_stage}")
    return gpu_map, layers_per_stage


print("L5: ProfilerBackend + SchedulerBackend loaded")


# =============================================================================
# GPT-2 TRAINING: Config, Data, Model, Worker
# =============================================================================

@dataclass
class GPT2DTFMConfig:
    """Unified config for GPT-2 training on DT-FM."""
    # Model
    vocab_size: int = 50257
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    d_ff: int = 1536
    max_seq_len: int = 256
    dropout: float = 0.1
    activation: str = "gelu"
    tie_weights: bool = False
    use_flash_attention: bool = True
    # Training
    batch_size: int = 16
    micro_batch_size: int = 4
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    max_iters: int = 2000
    warmup_iters: int = 100
    min_lr: float = 1e-5
    grad_clip: float = 1.0
    eval_interval: int = 200
    eval_iters: int = 10
    log_interval: int = 10
    checkpoint_interval: int = 500
    gradient_accumulate_step: int = 1
    # Parallelism
    world_size: int = 4
    pp_size: int = 2
    dp_size: int = 2
    dp_mode: str = "allreduce"
    num_microbatches: int = 4
    # Communication
    dist_url: str = "tcp://127.0.0.1:29510"
    dist_backend: str = "cupy_nccl"
    # I/O
    output_dir: str = "./dtfm_gpt2_output"
    data_dir: str = "./data"
    dataset: str = "shakespeare"
    seed: int = 42
    # Profiling & Scheduling (L5)
    enable_profiling: bool = True   # Run GPU profiling before training to optimize layer partition
    layers_per_stage_list: Optional[List[int]] = None  # Override: set manually to skip profiling
    gpu_map: Optional[Dict[int, int]] = None  # GCMA-determined gpu_map (global_rank → cuda_id)

    def __post_init__(self):
        self.d_ff = self.d_ff or 4 * self.d_model
        assert self.world_size == self.pp_size * self.dp_size
        self.num_microbatches = self.batch_size // self.micro_batch_size
        os.makedirs(self.output_dir, exist_ok=True)


# ── Data ──────────────────────────────────────────────────────────────────

def prepare_sst2(cfg: GPT2DTFMConfig):
    """Load SST-2 dataset and pre-embed with pretrained GPT-2 embeddings."""
    try:
        from datasets import load_dataset
        from transformers import GPT2Tokenizer, GPT2Model
    except ImportError:
        print("Error: datasets and transformers are required. Install with:")
        print("  pip install datasets transformers")
        raise
    
    print("Loading SST-2 dataset...")
    sst2 = load_dataset("stanfordnlp/sst2")
    train_raw = sst2["train"]
    val_raw = sst2["validation"]
    print(f"SST-2 loaded: {len(train_raw)} train, {len(val_raw)} validation")
    
    num_train = min(4096, len(train_raw))
    num_val = min(872, len(val_raw))
    
    print("Loading GPT-2 tokenizer and embeddings...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    hf_gpt2 = GPT2Model.from_pretrained("gpt2")
    
    embed_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    token_emb = hf_gpt2.wte.to(embed_device)
    pos_emb = hf_gpt2.wpe.to(embed_device)
    
    def embed_batch(texts, labels, max_len=cfg.max_seq_len, batch_size=256):
        all_embeds, all_labels = [], []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            encoded = tokenizer(batch_texts, padding="max_length", truncation=True,
                              max_length=max_len, return_tensors="pt")
            input_ids = encoded["input_ids"].to(embed_device)
            with torch.no_grad():
                pos_ids = torch.arange(max_len, device=embed_device).unsqueeze(0)
                embeds = token_emb(input_ids) + pos_emb(pos_ids)
            all_embeds.append(embeds.cpu())
            all_labels.append(torch.tensor(batch_labels, dtype=torch.long))
            if i % 1000 == 0 and i > 0:
                print(f"  Embedded {i}/{len(texts)} samples")
        return torch.cat(all_embeds, dim=0), torch.cat(all_labels, dim=0)
    
    train_indices = torch.randperm(len(train_raw))[:num_train].tolist()
    train_texts = [train_raw[i]["sentence"] for i in train_indices]
    train_labels = [train_raw[i]["label"] for i in train_indices]
    print(f"Embedding {num_train} training samples...")
    train_embeds, train_labels_t = embed_batch(train_texts, train_labels)
    
    val_texts = [val_raw[i]["sentence"] for i in range(num_val)]
    val_labels = [val_raw[i]["label"] for i in range(num_val)]
    print(f"Embedding {num_val} validation samples...")
    val_embeds, val_labels_t = embed_batch(val_texts, val_labels)
    
    del hf_gpt2, token_emb, pos_emb
    torch.cuda.empty_cache()
    
    cfg.vocab_size = 2
    return (train_embeds, train_labels_t), (val_embeds, val_labels_t)


class SST2DataLoader:
    """Wrapper to sample batches from pre-embedded SST-2 data."""
    def __init__(self, embeddings, labels, batch_size, generator=None):
        self.embeddings = embeddings
        self.labels = labels
        self.batch_size = batch_size
        self.generator = generator
        self.n = len(embeddings)
    
    def get_batch(self):
        ix = torch.randint(self.n - 1, (self.batch_size,), generator=self.generator)
        x = self.embeddings[ix].float()
        y = self.labels[ix].long()
        return x, y


def get_batch(cfg: GPT2DTFMConfig, split: str, device, loader=None, generator=None):
    """Sample a batch from SST-2 data."""
    if loader is None:
        raise ValueError("SST-2 loader must be provided")
    x, y = loader.get_batch()
    return x.to(device), y.to(device)


# ── Model (GPT-2 pipeline stages) ────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPT2DTFMConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        self.head_dim = cfg.d_model // cfg.n_heads
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.register_buffer("bias", torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len))
                             .view(1, 1, cfg.max_seq_len, cfg.max_seq_len))
        self.use_flash = cfg.use_flash_attention and hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
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
    def __init__(self, cfg: GPT2DTFMConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.d_model, cfg.d_ff)
        self.c_proj = nn.Linear(cfg.d_ff, cfg.d_model)
        self.act = nn.GELU() if cfg.activation == "gelu" else nn.ReLU()
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class GPT2Block(nn.Module):
    def __init__(self, cfg: GPT2DTFMConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.d_model)
        self.mlp = GPT2MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class SST2StageFirst(nn.Module):
    """First stage: transformer blocks (input already embedded by pretrained GPT-2)."""
    def __init__(self, cfg: GPT2DTFMConfig, layers_per_stage):
        super().__init__()
        self.cfg = cfg
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([GPT2Block(cfg) for _ in range(layers_per_stage)])
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        """x: (batch, seq_len, d_model) already embedded."""
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        return x


class SST2StageMiddle(nn.Module):
    """Middle stage: transformer blocks only (receives hidden, outputs hidden)."""
    def __init__(self, cfg: GPT2DTFMConfig, layers_per_stage):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([GPT2Block(cfg) for _ in range(layers_per_stage)])
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        """x: (batch, seq_len, d_model) hidden from previous stage."""
        for block in self.blocks:
            x = block(x)
        return x


class SST2StageLast(nn.Module):
    """Last stage: transformer blocks + classification head."""
    def __init__(self, cfg: GPT2DTFMConfig, layers_per_stage, num_classes=2):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.blocks = nn.ModuleList([GPT2Block(cfg) for _ in range(layers_per_stage)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.classifier = nn.Linear(cfg.d_model, num_classes, bias=True)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(self, x, targets=None):
        """x: (batch, seq_len, d_model)"""
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        if targets is not None:
            return F.cross_entropy(logits, targets)
        return logits


class SST2StageSingle(nn.Module):
    """Single-GPU stage: embedding dropout + all transformer blocks + classification head."""
    def __init__(self, cfg: GPT2DTFMConfig, num_classes=2):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([GPT2Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.classifier = nn.Linear(cfg.d_model, num_classes, bias=True)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(self, x, targets=None):
        """x: (batch, seq_len, d_model) already embedded."""
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        if targets is not None:
            return F.cross_entropy(logits, targets)
        return logits


# ── Worker ────────────────────────────────────────────────────────────────

def worker(rank: int, cfg: GPT2DTFMConfig, train_data, val_data):
    try:
        _worker_impl(rank, cfg, train_data, val_data)
    except Exception as e:
        print(f"[RANK {rank}] FATAL: {e}", flush=True)
        traceback.print_exc()
        raise


def _worker_impl(rank: int, cfg: GPT2DTFMConfig, train_data, val_data):
    sys.stdout.flush(); sys.stderr.flush()
    print(f"[RANK {rank}] Worker started, CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}", flush=True)
    print(f"[RANK {rank}] Available GPUs: {torch.cuda.device_count()}", flush=True)
    # Rank geometry (computed BEFORE seeding so we can seed by pp_rank)
    pp_rank = rank % cfg.pp_size
    dp_rank = rank // cfg.pp_size

    # Seed by pp_rank so DP replicas get IDENTICAL initial weights
    # (RANK 0 & RANK 2 are both 'first stage' → same seed → same weights)
    torch.manual_seed(cfg.seed + pp_rank)
    np.random.seed(cfg.seed + pp_rank)

    # GPU assignment: determined by GCMA scheduler (or default fallback).
    # cfg.gpu_map is set by main() after running GCMA profiling.
    if cfg.gpu_map is not None:
        gpu_map = cfg.gpu_map
    else:
        # Fallback: default layout
        gpu_map = {
            0: 1, 1: 2, 2: 0, 3: 3,
        }
    cuda_id = gpu_map[rank]
    device = torch.device('cuda', cuda_id)
    torch.cuda.set_device(device)

    is_first = (pp_rank == 0)
    is_last = (pp_rank == cfg.pp_size - 1)
    is_middle = (not is_first and not is_last)
    is_single = (cfg.pp_size == 1)

    # L5: Use profiler-determined per-stage layer count (set by main() via run_profiling)
    if cfg.layers_per_stage_list is not None and len(cfg.layers_per_stage_list) == cfg.pp_size:
        layers_per_stage = cfg.layers_per_stage_list[pp_rank]
    else:
        layers_per_stage = cfg.n_layers // cfg.pp_size

    print(f"[RANK {rank}] pp_rank={pp_rank}, dp_rank={dp_rank}, cuda:{cuda_id}, "
          f"{'FIRST' if is_first else 'LAST'} stage, {layers_per_stage} layers", flush=True)

    # L1: DT-FM Config & State
    dtfm_config = DTFMConfigManager(
        device=DeviceConfig(use_cuda=True, cuda_id=cuda_id, cuda_num=cfg.world_size),
        distributed=DistributedConfig(dist_backend="nccl", dist_url=cfg.dist_url,
                                      world_size=cfg.world_size,
                                      pipeline_group_size=cfg.pp_size,
                                      data_group_size=cfg.dp_size, rank=rank),
        model=ModelConfig(seq_length=cfg.max_seq_len, embedding_dim=cfg.d_model,
                          num_layers=layers_per_stage, num_heads=cfg.n_heads,
                          task="Seq2SeqClassification", vocab_size=cfg.vocab_size,
                          num_classes=cfg.vocab_size),
        training=TrainingConfig(batch_size=cfg.batch_size, micro_batch_size=cfg.micro_batch_size,
                                lr=cfg.lr, num_iters=cfg.max_iters,
                                gradient_accumulate_step=cfg.gradient_accumulate_step,
                                seed=cfg.seed, weight_decay=cfg.weight_decay),
        parallel=ParallelConfig(pp_mode="gpipe", dp_mode=cfg.dp_mode,
                                gradient_accumulate_step=cfg.gradient_accumulate_step),
        mixed_precision=MixedPrecisionConfig(fp16=False),
        profiling=ProfilingConfig(profiling="no_profiling"),
    )
    state = DTFMStateManager()
    state.init_from_config(dtfm_config)
    # L6: Communication (torch.distributed)
    # Use GLOO as default backend (reliable for send/recv on CPU tensors)
    # Then create a separate NCCL group for DP all_reduce (GPU tensors)
    if not torch.distributed.is_initialized():
        print(f"[RANK {rank}] Initializing torch.distributed (gloo)...", flush=True)
        try:
            torch.distributed.init_process_group(
                backend='gloo',
                init_method=cfg.dist_url,
                world_size=cfg.world_size,
                rank=rank,
                timeout=timedelta(seconds=120),
            )
        except Exception as e:
            print(f"[RANK {rank}] FATAL: dist init failed: {e}", flush=True)
            raise
    print(f"[RANK {rank}] torch.distributed initialized (gloo)", flush=True)
    sys.stdout.flush()

    # new_group() is COLLECTIVE — ALL ranks must call it for EVERY group.
    # Create ALL PP groups and ALL DP groups on every rank, then select ours.

    # PP groups: [0,1] and [2,3]
    pp_process_group = None
    pp_ranks_in_group = None
    for d in range(cfg.dp_size):
        pp_ranks = [d * cfg.pp_size + s for s in range(cfg.pp_size)]
        grp = torch.distributed.new_group(ranks=pp_ranks)
        if rank in pp_ranks:
            pp_process_group = grp
            pp_ranks_in_group = pp_ranks

    # DP groups: [0,2] and [1,3]
    dp_process_group = None
    dp_ranks_in_group = None
    for s in range(cfg.pp_size):
        dp_ranks = [d * cfg.pp_size + s for d in range(cfg.dp_size)]
        grp = torch.distributed.new_group(ranks=dp_ranks)
        if rank in dp_ranks:
            dp_process_group = grp
            dp_ranks_in_group = dp_ranks

    print(f"[RANK {rank}] PP group: {pp_ranks_in_group}, DP group: {dp_ranks_in_group}", flush=True)

    # ── CuPy NCCL Communicators (GPU-direct, matches original DT-FM) ──────
    cupy.cuda.Device(cuda_id).use()
    dist_store = torch.distributed.distributed_c10d._get_default_store()

    # PP NCCL communicator: one per pipeline group
    pp_group_id = dp_rank  # pipeline_group_0, pipeline_group_1, ...
    pp_comm_name = f"pipeline_group_{pp_group_id}"
    if pp_rank == 0:
        nccl_uid = cupy.cuda.nccl.get_unique_id()
        # CuPy 14+: get_unique_id() returns bytes directly
        uid_bytes = nccl_uid if isinstance(nccl_uid, bytes) else np.array(nccl_uid).tobytes()
        dist_store.set(f'group-{pp_comm_name}-unique-id', uid_bytes)
    torch.distributed.barrier()  # ensure rank 0 has stored the ID
    if pp_rank != 0:
        uid_bytes = dist_store.get(f'group-{pp_comm_name}-unique-id')
    # CuPy 14+: NcclCommunicator expects bytes for commId
    pp_nccl_id = uid_bytes if isinstance(uid_bytes, bytes) else bytes(uid_bytes)
    pp_nccl_comm = cupy.cuda.nccl.NcclCommunicator(cfg.pp_size, pp_nccl_id, pp_rank)
    print(f"[RANK {rank}] PP NCCL communicator '{pp_comm_name}' initialized (pp_rank={pp_rank})", flush=True)

    # DP NCCL communicator: one per data-parallel group
    dp_nccl_comm = None
    if cfg.dp_size > 1:
        dp_group_id = pp_rank  # data_group_0, data_group_1, ...
        dp_comm_name = f"data_group_{dp_group_id}"
        if dp_rank == 0:
            nccl_uid_dp = cupy.cuda.nccl.get_unique_id()
            uid_bytes_dp = nccl_uid_dp if isinstance(nccl_uid_dp, bytes) else np.array(nccl_uid_dp).tobytes()
            dist_store.set(f'group-{dp_comm_name}-unique-id', uid_bytes_dp)
        torch.distributed.barrier()  # ensure dp_rank 0 has stored the ID
        if dp_rank != 0:
            uid_bytes_dp = dist_store.get(f'group-{dp_comm_name}-unique-id')
        dp_nccl_id = uid_bytes_dp if isinstance(uid_bytes_dp, bytes) else bytes(uid_bytes_dp)
        dp_nccl_comm = cupy.cuda.nccl.NcclCommunicator(cfg.dp_size, dp_nccl_id, dp_rank)
        print(f"[RANK {rank}] DP NCCL communicator '{dp_comm_name}' initialized (dp_rank={dp_rank})", flush=True)

    # ── CUDA Streams (matches original GpipeAsync: compute/send/recv) ─────
    torch_comp_stream = torch.cuda.default_stream(device=device)
    torch_recv_stream = torch.cuda.Stream(device=device, priority=-1)
    torch_send_stream = torch.cuda.Stream(device=device, priority=-1)
    # Separate stream for DP AllReduce (matches original AllReduceDP)
    dp_comm_stream = torch.cuda.Stream(device=device, priority=-1) if cfg.dp_size > 1 else None

    # L2: Model (SST-2 Classification)
    if is_single:
        model = SST2StageSingle(cfg, num_classes=2).to(device)
    elif is_first:
        model = SST2StageFirst(cfg, layers_per_stage).to(device)
    elif is_last:
        model = SST2StageLast(cfg, layers_per_stage, num_classes=2).to(device)
    else:
        model = SST2StageMiddle(cfg, layers_per_stage).to(device)

    # Broadcast weights from dp_rank=0 → dp_rank=1 to guarantee exact sync
    for param in model.parameters():
        param_cpu = param.data.cpu()
        torch.distributed.broadcast(param_cpu, src=dp_ranks_in_group[0], group=dp_process_group)
        param.data.copy_(param_cpu.to(device))

    # Re-seed with rank-specific seed so each worker samples DIFFERENT data
    torch.manual_seed(cfg.seed * 31 + rank)
    np.random.seed(cfg.seed * 31 + rank)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[RANK {rank}] Model: {type(model).__name__}, {n_params:,} params", flush=True)

    # Flatten parameters for efficient single-AllReduce (matches original DT-FM AllReduceDP)
    flat_param = None
    if cfg.dp_size > 1:
        flat_param = flatten_params(model.parameters())
        print(f"[RANK {rank}] Flattened params: {flat_param.data.numel():,} elements, "
              f"{flat_param.data.numel() * flat_param.data.element_size() / 1024 / 1024:.1f} MB",
              flush=True)

    # L4: Optimizer
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=cfg.lr, betas=cfg.betas)

    # LR schedule
    def get_lr(it):
        if it < cfg.warmup_iters:
            return cfg.lr * (it + 1) / cfg.warmup_iters
        if it > cfg.max_iters:
            return cfg.min_lr
        decay_ratio = (it - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)

    # Activation buffers
    num_micro = cfg.num_microbatches
    act_shape = (cfg.micro_batch_size, cfg.max_seq_len, cfg.d_model)

    prev_rank = pp_ranks_in_group[pp_rank - 1] if pp_rank > 0 else None
    next_rank = pp_ranks_in_group[pp_rank + 1] if pp_rank < cfg.pp_size - 1 else None

    # Pipeline neighbor ranks within the PP NCCL communicator (0-indexed within group)
    pp_prev = pp_rank - 1 if pp_rank > 0 else None
    pp_next = pp_rank + 1 if pp_rank < cfg.pp_size - 1 else None

    # Pre-allocate GPU activation buffers (matches original DT-FM pattern)
    # For non-first ranks: GPU buffers to receive activations from previous stage
    input_micro_batches = [torch.zeros(act_shape, requires_grad=True, device=device)
                           for _ in range(num_micro)] if not is_first else None
    # For non-last ranks: GPU buffers to receive gradients from next stage  
    output_micro_batches_grad = [torch.zeros(act_shape, requires_grad=False, device=device)
                                 for _ in range(num_micro)] if not is_last else None
    # For non-first ranks: alias for zero_input_grad
    input_bufs = input_micro_batches or []

    micro_batch_float_num = cfg.micro_batch_size * cfg.max_seq_len * cfg.d_model
    print(f"[RANK {rank}] Micro-batch send/recv size: "
          f"{micro_batch_float_num * 4 / 1024 / 1024:.1f} MB (fp32), "
          f"{num_micro} micro-batches", flush=True)

    # Shared RNG so pipeline partners sample IDENTICAL batches.
    # Seeded by dp_rank → different data across DP replicas, same within PP group.
    data_rng = torch.Generator()
    data_rng.manual_seed(cfg.seed + dp_rank)

    # Setup SST-2 data loaders (passed from main process)
    train_embeds, train_labels = train_data
    val_embeds, val_labels = val_data
    
    train_loader = SST2DataLoader(
        train_embeds, train_labels,
        cfg.micro_batch_size, generator=data_rng
    )
    eval_rng_init = torch.Generator()
    eval_rng_init.manual_seed(cfg.seed + dp_rank + 999999)
    val_loader = SST2DataLoader(
        val_embeds, val_labels,
        cfg.micro_batch_size, generator=eval_rng_init
    )
    print(f"[RANK {rank}] SST-2 loaders initialized", flush=True)

    # ── NCCL GPU-direct send/recv helpers (matches original GpipeAsync) ────
    def _nccl_send(tensor, dst_pp_rank, stream):
        """GPU-direct send via CuPy NCCL on the given CUDA stream."""
        cupy_stream = cupy.cuda.ExternalStream(stream.cuda_stream)
        pp_nccl_comm.send(
            tensor.data_ptr(), torch.numel(tensor),
            _type_torch_to_cupy(tensor.dtype), dst_pp_rank, cupy_stream.ptr
        )

    def _nccl_recv(tensor, src_pp_rank, stream):
        """GPU-direct recv via CuPy NCCL on the given CUDA stream."""
        cupy_stream = cupy.cuda.ExternalStream(stream.cuda_stream)
        pp_nccl_comm.recv(
            tensor.data_ptr(), torch.numel(tensor),
            _type_torch_to_cupy(tensor.dtype), src_pp_rank, cupy_stream.ptr
        )

    def _nccl_allreduce(tensor, stream):
        """GPU-direct AllReduce via CuPy NCCL on the given CUDA stream."""
        cupy_stream = cupy.cuda.ExternalStream(stream.cuda_stream)
        dp_nccl_comm.allReduce(
            tensor.data_ptr(), tensor.data_ptr(), torch.numel(tensor),
            _type_torch_to_cupy(tensor.dtype),
            cupy.cuda.nccl.NCCL_SUM, cupy_stream.ptr
        )

    # Helper for dtype mapping (already defined at module level, also used here)
    def _type_torch_to_cupy(torch_type):
        import cupy.cuda.nccl as nccl
        return {torch.float32: nccl.NCCL_FLOAT32, torch.float16: nccl.NCCL_FLOAT16,
                torch.float64: nccl.NCCL_FLOAT64, torch.int32: nccl.NCCL_INT32,
                torch.int: nccl.NCCL_INT, torch.uint8: nccl.NCCL_UINT8,
                torch.float: nccl.NCCL_FLOAT}[torch_type]

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        results = {}
        eval_buf = torch.zeros(act_shape, device=device) if not is_first else None
        for split_name, split_loader in [('train', train_loader), ('val', val_loader)]:
            losses = torch.zeros(cfg.eval_iters)
            for k in range(cfg.eval_iters):
                x, y = split_loader.get_batch()
                x, y = x.to(device), y.to(device)
                if is_single:
                    # Single GPU: compute loss directly, no communication
                    loss = model(x, y)
                    losses[k] = loss.item()
                elif is_first:
                    hidden = model(x)
                    # Synchronous send on default stream for eval
                    _nccl_send(hidden.data, pp_next, torch_comp_stream)
                    torch.cuda.synchronize()
                elif is_last:
                    eval_buf.zero_()
                    _nccl_recv(eval_buf, pp_prev, torch_comp_stream)
                    torch.cuda.synchronize()
                    loss = model(eval_buf, y)
                    losses[k] = loss.item()
                else:
                    # Middle stage: recv → compute → send
                    eval_buf.zero_()
                    _nccl_recv(eval_buf, pp_prev, torch_comp_stream)
                    torch.cuda.synchronize()
                    hidden = model(eval_buf)
                    _nccl_send(hidden.data, pp_next, torch_comp_stream)
                    torch.cuda.synchronize()
            if is_last or is_single:
                results[split_name] = losses.mean().item()
        model.train()
        return results

    # ── Training Loop ─────────────────────────────────────────────────
    # Pre-allocate CUDA timing events (matches DT-FM tidy_profiling)
    # These use enable_timing=True for precise GPU-side measurement via elapsed_time()
    enable_tidy = True  # Set to False to disable per-micro-batch profiling events

    fwd_recv_start_events = [torch.cuda.Event(enable_timing=enable_tidy) for _ in range(num_micro)]
    fwd_recv_ready_events = [torch.cuda.Event(enable_timing=enable_tidy) for _ in range(num_micro)]
    fwd_comp_start_events = [torch.cuda.Event(enable_timing=enable_tidy) for _ in range(num_micro)]
    fwd_comp_ready_events = [torch.cuda.Event(enable_timing=enable_tidy) for _ in range(num_micro)]
    fwd_send_start_events = [torch.cuda.Event(enable_timing=enable_tidy) for _ in range(num_micro)]
    fwd_send_end_events   = [torch.cuda.Event(enable_timing=enable_tidy) for _ in range(num_micro)]

    bwd_recv_start_events = [torch.cuda.Event(enable_timing=enable_tidy) for _ in range(num_micro)]
    bwd_recv_ready_events = [torch.cuda.Event(enable_timing=enable_tidy) for _ in range(num_micro)]
    bwd_comp_start_events = [torch.cuda.Event(enable_timing=enable_tidy) for _ in range(num_micro)]
    bwd_comp_ready_events = [torch.cuda.Event(enable_timing=enable_tidy) for _ in range(num_micro)]
    bwd_send_start_events = [torch.cuda.Event(enable_timing=enable_tidy) for _ in range(num_micro)]
    bwd_send_end_events   = [torch.cuda.Event(enable_timing=enable_tidy) for _ in range(num_micro)]

    opt_allreduce_start_event = torch.cuda.Event(enable_timing=enable_tidy)
    opt_allreduce_end_event   = torch.cuda.Event(enable_timing=enable_tidy)
    opt_step_start_event      = torch.cuda.Event(enable_timing=enable_tidy)
    opt_step_end_event        = torch.cuda.Event(enable_timing=enable_tidy)
    init_event                = torch.cuda.Event(enable_timing=enable_tidy)

    torch.distributed.barrier()
    print(f"[RANK {rank}] Starting training loop ({cfg.max_iters} iters)", flush=True)
    sys.stdout.flush()
    model.train()
    best_val_loss = float('inf')
    EVENT_LOGGER.set_epoch_start(time.time())
    t0 = time.time()

    for iter_num in range(cfg.max_iters):
        # Barrier at start of iteration (matches original sgd_iter)
        torch.distributed.barrier()
        iter_start = time.time()

        # Record init event for tidy profiling timestamp base
        if enable_tidy:
            torch.cuda.synchronize()
            init_time_stamp = time.time() * 1e+6  # microseconds
            init_event.record()

        lr = get_lr(iter_num)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.zero_grad(set_to_none=False)

        # zero pre-allocated input grads (match original zero_input_grad())
        for buf in input_bufs:
            if buf.grad is not None:
                buf.grad.zero_()

        micro_losses = []

        # ── Gradient Accumulation Loop (matches original sgd_iter) ────
        for ga_step in range(cfg.gradient_accumulate_step):

            # Collect micro-batches (from shared loader → PP partners get identical data)
            micro_inputs, micro_targets = [], []
            for _ in range(num_micro):
                x, y = train_loader.get_batch()
                x, y = x.to(device), y.to(device)
                micro_inputs.append(x)
                micro_targets.append(y)

            # ── GPipe FORWARD ──────────────────────────────────────────
            # Async streams with CUDA event sync (NO torch.cuda.synchronize per micro-batch).
            # Matches DT-FM GpipeAsync.forward_stage() exactly.
            cached_outputs = []

            for m in range(num_micro):
                if is_single:
                    # Single GPU: compute forward+loss, no communication
                    with torch.cuda.stream(torch_comp_stream):
                        torch_comp_stream.record_event(fwd_comp_start_events[m])
                        loss = model(micro_inputs[m], micro_targets[m])
                        torch_comp_stream.record_event(fwd_comp_ready_events[m])
                    micro_losses.append(loss.item())
                    cached_outputs.append(loss / (num_micro * cfg.gradient_accumulate_step))

                elif is_first:
                    # First stage: compute on comp stream, async send on send stream
                    with torch.cuda.stream(torch_comp_stream):
                        torch_comp_stream.record_event(fwd_comp_start_events[m])
                        hidden = model(micro_inputs[m])
                        torch_comp_stream.record_event(fwd_comp_ready_events[m])
                    cached_outputs.append(hidden)
                    if pp_next is not None:
                        with torch.cuda.stream(torch_send_stream):
                            torch_send_stream.wait_event(fwd_comp_ready_events[m])
                            torch_send_stream.record_event(fwd_send_start_events[m])
                            _nccl_send(hidden.data, pp_next, torch_send_stream)
                            torch_send_stream.record_event(fwd_send_end_events[m])

                elif is_last:
                    # Last stage: recv on recv stream, compute on comp stream
                    with torch.cuda.stream(torch_recv_stream):
                        torch_recv_stream.record_event(fwd_recv_start_events[m])
                        _nccl_recv(input_micro_batches[m], pp_prev, torch_recv_stream)
                        torch_recv_stream.record_event(fwd_recv_ready_events[m])
                    with torch.cuda.stream(torch_comp_stream):
                        torch_comp_stream.wait_event(fwd_recv_ready_events[m])
                        torch_comp_stream.record_event(fwd_comp_start_events[m])
                        loss = model(input_micro_batches[m], micro_targets[m])
                        torch_comp_stream.record_event(fwd_comp_ready_events[m])
                    micro_losses.append(loss.item())
                    cached_outputs.append(loss / (num_micro * cfg.gradient_accumulate_step))

                else:
                    # Middle stage: recv → compute → send
                    with torch.cuda.stream(torch_recv_stream):
                        torch_recv_stream.record_event(fwd_recv_start_events[m])
                        _nccl_recv(input_micro_batches[m], pp_prev, torch_recv_stream)
                        torch_recv_stream.record_event(fwd_recv_ready_events[m])
                    with torch.cuda.stream(torch_comp_stream):
                        torch_comp_stream.wait_event(fwd_recv_ready_events[m])
                        torch_comp_stream.record_event(fwd_comp_start_events[m])
                        current_output = model(input_micro_batches[m])
                        torch_comp_stream.record_event(fwd_comp_ready_events[m])
                    cached_outputs.append(current_output)
                    if pp_next is not None:
                        with torch.cuda.stream(torch_send_stream):
                            torch_send_stream.wait_event(fwd_comp_ready_events[m])
                            torch_send_stream.record_event(fwd_send_start_events[m])
                            _nccl_send(current_output.data, pp_next, torch_send_stream)
                            torch_send_stream.record_event(fwd_send_end_events[m])

            # Record forward profiling using CUDA event elapsed_time (matches DT-FM)
            if enable_tidy:
                torch.cuda.synchronize()
                for m in range(num_micro):
                    if not is_first and not is_single:
                        recv_us = fwd_recv_start_events[m].elapsed_time(fwd_recv_ready_events[m]) * 1e+3
                        EVENT_LOGGER.record_event(rank, "forward-recv", m, "forward",
                                                  recv_us / 1000.0,
                                                  timestamp=init_time_stamp + init_event.elapsed_time(fwd_recv_start_events[m]) * 1e+3)
                    comp_us = fwd_comp_start_events[m].elapsed_time(fwd_comp_ready_events[m]) * 1e+3
                    EVENT_LOGGER.record_event(rank, "forward-compute", m, "forward",
                                              comp_us / 1000.0,
                                              timestamp=init_time_stamp + init_event.elapsed_time(fwd_comp_start_events[m]) * 1e+3)
                    if not is_last and not is_single:
                        send_us = fwd_send_start_events[m].elapsed_time(fwd_send_end_events[m]) * 1e+3
                        EVENT_LOGGER.record_event(rank, "forward-send", m, "forward",
                                                  send_us / 1000.0,
                                                  timestamp=init_time_stamp + init_event.elapsed_time(fwd_send_start_events[m]) * 1e+3)

            # Barrier between forward and backward (matches original sgd_iter)
            torch.distributed.barrier()

            # ── GPipe BACKWARD ─────────────────────────────────────────
            # Reversed micro-batch order (standard GPipe convention).
            # Async streams with CUDA event sync — NO torch.cuda.synchronize per micro-batch.
            # Matches DT-FM GpipeAsync.backward_stage().

            for m in reversed(range(num_micro)):
                if is_single:
                    # Single GPU: backward loss directly
                    with torch.cuda.stream(torch_comp_stream):
                        torch_comp_stream.record_event(bwd_comp_start_events[m])
                        cached_outputs[m].backward()
                        torch_comp_stream.record_event(bwd_comp_ready_events[m])

                elif is_last:
                    # Last stage: backward loss, async send input grad to prev
                    with torch.cuda.stream(torch_comp_stream):
                        torch_comp_stream.record_event(bwd_comp_start_events[m])
                        cached_outputs[m].backward()
                        torch_comp_stream.record_event(bwd_comp_ready_events[m])
                    if pp_prev is not None:
                        with torch.cuda.stream(torch_send_stream):
                            torch_send_stream.wait_event(bwd_comp_ready_events[m])
                            torch_send_stream.record_event(bwd_send_start_events[m])
                            _nccl_send(input_micro_batches[m].grad, pp_prev, torch_send_stream)
                            torch_send_stream.record_event(bwd_send_end_events[m])

                elif is_first:
                    # First stage: recv grad from next, backward
                    if pp_next is not None:
                        with torch.cuda.stream(torch_recv_stream):
                            torch_recv_stream.record_event(bwd_recv_start_events[m])
                            _nccl_recv(output_micro_batches_grad[m], pp_next, torch_recv_stream)
                            torch_recv_stream.record_event(bwd_recv_ready_events[m])
                        with torch.cuda.stream(torch_comp_stream):
                            torch_comp_stream.wait_event(bwd_recv_ready_events[m])
                            torch_comp_stream.record_event(bwd_comp_start_events[m])
                            cached_outputs[m].backward(gradient=output_micro_batches_grad[m])
                            torch_comp_stream.record_event(bwd_comp_ready_events[m])
                    else:
                        # pp_size=1 fallback (shouldn't reach here if is_single, but safe)
                        with torch.cuda.stream(torch_comp_stream):
                            torch_comp_stream.record_event(bwd_comp_start_events[m])
                            cached_outputs[m].backward()
                            torch_comp_stream.record_event(bwd_comp_ready_events[m])

                else:
                    # Middle stage: recv grad → backward → send grad
                    with torch.cuda.stream(torch_recv_stream):
                        torch_recv_stream.record_event(bwd_recv_start_events[m])
                        _nccl_recv(output_micro_batches_grad[m], pp_next, torch_recv_stream)
                        torch_recv_stream.record_event(bwd_recv_ready_events[m])
                    with torch.cuda.stream(torch_comp_stream):
                        torch_comp_stream.wait_event(bwd_recv_ready_events[m])
                        torch_comp_stream.record_event(bwd_comp_start_events[m])
                        cached_outputs[m].backward(gradient=output_micro_batches_grad[m])
                        torch_comp_stream.record_event(bwd_comp_ready_events[m])
                    if pp_prev is not None:
                        with torch.cuda.stream(torch_send_stream):
                            torch_send_stream.wait_event(bwd_comp_ready_events[m])
                            torch_send_stream.record_event(bwd_send_start_events[m])
                            _nccl_send(input_micro_batches[m].grad, pp_prev, torch_send_stream)
                            torch_send_stream.record_event(bwd_send_end_events[m])

            # Record backward profiling using CUDA event elapsed_time (matches DT-FM)
            if enable_tidy:
                torch.cuda.synchronize()
                for m in range(num_micro):
                    if not is_last and not is_single:
                        recv_us = bwd_recv_start_events[m].elapsed_time(bwd_recv_ready_events[m]) * 1e+3
                        EVENT_LOGGER.record_event(rank, "backward-recv", m, "backward",
                                                  recv_us / 1000.0,
                                                  timestamp=init_time_stamp + init_event.elapsed_time(bwd_recv_start_events[m]) * 1e+3)
                    comp_us = bwd_comp_start_events[m].elapsed_time(bwd_comp_ready_events[m]) * 1e+3
                    EVENT_LOGGER.record_event(rank, "backward-compute", m, "backward",
                                              comp_us / 1000.0,
                                              timestamp=init_time_stamp + init_event.elapsed_time(bwd_comp_start_events[m]) * 1e+3)
                    if not is_first and not is_single:
                        send_us = bwd_send_start_events[m].elapsed_time(bwd_send_end_events[m]) * 1e+3
                        EVENT_LOGGER.record_event(rank, "backward-send", m, "backward",
                                                  send_us / 1000.0,
                                                  timestamp=init_time_stamp + init_event.elapsed_time(bwd_send_start_events[m]) * 1e+3)

        # ── Optimizer step (matches DT-FM GpipeAsync.optimizer_step + AllReduceDP) ──
        # DP AllReduce using flattened params + NCCL on dedicated stream
        if cfg.dp_size > 1 and flat_param is not None and dp_nccl_comm is not None:
            # Signal that backward is done on comp stream
            backward_ready_event = torch.cuda.Event()
            torch_comp_stream.record_event(backward_ready_event)
            with torch.cuda.stream(dp_comm_stream):
                dp_comm_stream.wait_event(backward_ready_event)
                dp_comm_stream.record_event(opt_allreduce_start_event)
                _nccl_allreduce(flat_param.grad.data, dp_comm_stream)
                dp_comm_stream.record_event(opt_allreduce_end_event)
            # Wait for allreduce to finish before optimizer.step()
            with torch.cuda.stream(torch_comp_stream):
                torch_comp_stream.wait_event(opt_allreduce_end_event)
            # Average gradients
            flat_param.grad.data.div_(cfg.dp_size)

        # Grad clip + step
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        with torch.cuda.stream(torch_comp_stream):
            torch_comp_stream.record_event(opt_step_start_event)
            optimizer.step()
            torch_comp_stream.record_event(opt_step_end_event)

        # Single synchronize at end of iteration (matches original sgd_iter)
        torch.cuda.synchronize()
        torch.distributed.barrier()

        # Record optimizer profiling using CUDA events
        if enable_tidy:
            if cfg.dp_size > 1 and flat_param is not None and dp_nccl_comm is not None:
                ar_us = opt_allreduce_start_event.elapsed_time(opt_allreduce_end_event) * 1e+3
                EVENT_LOGGER.record_event(rank, "optimizer-allreduce", iter_num, "optimizer",
                                          ar_us / 1000.0,
                                          timestamp=init_time_stamp + init_event.elapsed_time(opt_allreduce_start_event) * 1e+3,
                                          para="flattened_grad", size=flat_param.grad.numel())
            opt_us = opt_step_start_event.elapsed_time(opt_step_end_event) * 1e+3
            EVENT_LOGGER.record_event(rank, "optimizer-step", iter_num, "optimizer",
                                      opt_us / 1000.0,
                                      timestamp=init_time_stamp + init_event.elapsed_time(opt_step_start_event) * 1e+3)

        # Logging
        t1 = time.time()
        dt = t1 - t0
        iter_time = t1 - iter_start
        t0 = t1
        if (is_last or is_single) and dp_rank == 0 and iter_num % cfg.log_interval == 0:
            avg_loss = sum(micro_losses) / len(micro_losses) if micro_losses else 0
            tokens_per_sec = cfg.batch_size * cfg.max_seq_len / dt if dt > 0 else 0
            print(f"  iter {iter_num:>5d} | loss={avg_loss:.4f} | lr={lr:.2e} | "
                  f"{tokens_per_sec:,.0f} tok/s | dt={dt*1000:.1f}ms", flush=True)

        # Evaluation
        if iter_num > 0 and iter_num % cfg.eval_interval == 0:
            losses = estimate_loss()
            if (is_last or is_single) and dp_rank == 0 and losses:
                val_loss = losses['val']
                print(f"\n  [EVAL] iter {iter_num} | train={losses['train']:.4f} | "
                      f"val={val_loss:.4f} | best={best_val_loss:.4f}", flush=True)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    ckpt_path = os.path.join(cfg.output_dir, "checkpoints", f"best_rank{rank}.pt")
                    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                'iter': iter_num, 'val_loss': val_loss}, ckpt_path)
                    print(f"  * New best! Saved to {ckpt_path}", flush=True)
            # Also save first-stage checkpoint (needed for full-pipeline inference)
            if is_first and not is_single:
                ckpt_path = os.path.join(cfg.output_dir, "checkpoints", f"best_rank{rank}.pt")
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'iter': iter_num, 'val_loss': float('nan')}, ckpt_path)
                print(f"  [RANK {rank}] First-stage checkpoint saved to {ckpt_path}", flush=True)

    # Export profiling trace (matches original DT-FM Chrome trace format)
    trace_path = os.path.join(cfg.output_dir, f"rank{rank}_trace.json")
    EVENT_LOGGER.to_chrome_trace(trace_path)
    print(f"[RANK {rank}] Profiling trace saved to {trace_path}", flush=True)

    # Cleanup
    torch.distributed.barrier()
    print(f"[RANK {rank}] Training complete. Best val loss: {best_val_loss:.4f}", flush=True)
    torch.distributed.destroy_process_group()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    cfg = GPT2DTFMConfig(
        d_model=768, n_heads=12, n_layers=12, d_ff=3072,
        max_seq_len=128, vocab_size=50257,
        batch_size=16, micro_batch_size=4, lr=3e-4,
        max_iters=2000, warmup_iters=100,
        eval_interval=200, log_interval=10, checkpoint_interval=500,
        world_size=4, pp_size=2, dp_size=2, dp_mode="allreduce",
        dist_url="tcp://127.0.0.1:29500",
        output_dir="./dtfm_gpt2_sst2_output", dataset="sst2",
    )

    print(f"\n{'='*60}")
    print(f"  DT-FM GPT-2 SST-2 Sentiment Classification")
    print(f"  world_size={cfg.world_size}  PP={cfg.pp_size}  DP={cfg.dp_size}")
    print(f"  batch_size={cfg.batch_size}  micro_batch={cfg.micro_batch_size}  "
          f"num_micro={cfg.num_microbatches}")
    print(f"  Layers: {cfg.n_layers} total")
    print(f"  Profiling: {'ENABLED' if cfg.enable_profiling else 'DISABLED'}")
    print(f"{'='*60}\n")

    # Default GPU map (fallback — used as initial guess for GCMA).
    # Pipeline partners should be on the same PCIe bus when possible.
    # CUDA_VISIBLE_DEVICES=1,2,3,6 → logical [0,1,2,3]
    default_gpu_map = {
        0: 1,  # RANK 0 (pp=0, dp=0) → logical 1 (physical GPU 2, GEN4)
        1: 2,  # RANK 1 (pp=1, dp=0) → logical 2 (physical GPU 3, GEN4)
        2: 0,  # RANK 2 (pp=0, dp=1) → logical 0 (physical GPU 1, GEN1)
        3: 3,  # RANK 3 (pp=1, dp=1) → logical 3 (physical GPU 6, GEN1)
    }

    # ── L5: Pre-training profiling & GCMA scheduling ──────────────────
    if cfg.layers_per_stage_list is not None:
        # User provided an explicit partition — use it directly
        gpu_map = cfg.gpu_map or default_gpu_map
        layers_per_stage_list = cfg.layers_per_stage_list
        print(f"Using user-specified partition: {layers_per_stage_list}")
        print(f"Using gpu_map: {gpu_map}")
    elif cfg.enable_profiling:
        try:
            gpu_map, layers_per_stage_list = run_profiling(cfg, default_gpu_map)
            print(f"GCMA-optimised gpu_map: {gpu_map}")
            print(f"Profiler-optimised partition: {layers_per_stage_list}")
        except Exception as e:
            print(f"WARNING: Profiling failed ({e}), falling back to defaults")
            import traceback; traceback.print_exc()
            gpu_map = default_gpu_map
            layers_per_stage_list = [cfg.n_layers // cfg.pp_size] * cfg.pp_size
    else:
        gpu_map = default_gpu_map
        layers_per_stage_list = [cfg.n_layers // cfg.pp_size] * cfg.pp_size
        print(f"Even split (profiling disabled): {layers_per_stage_list}")

    # Store on cfg so workers can read it
    cfg.gpu_map = gpu_map
    cfg.layers_per_stage_list = layers_per_stage_list
    print(f"  gpu_map: {cfg.gpu_map}")
    print(f"  Layers per stage: {layers_per_stage_list}")

    # Prepare SST-2 data (on main process, before spawning)
    print("Preparing SST-2 dataset...")
    train_data, val_data = prepare_sst2(cfg)
    train_embeds, train_labels = train_data
    val_embeds, val_labels = val_data
    
    print(f"Train: {train_embeds.shape}, Val: {val_embeds.shape}")

    # Spawn 4 workers (one per GPU) with data as arguments
    mp.set_start_method('fork', force=True)
    mp.spawn(worker, args=(cfg, (train_embeds, train_labels), (val_embeds, val_labels)), 
             nprocs=cfg.world_size, join=True)

    print("\nAll workers finished.")

    # ── Auto-merge pipeline stage checkpoints into a single file ──────────
    # Generic: works with any pp_size and dp_size.
    # For each pipeline stage s (0..pp_size-1), the global ranks holding that
    # stage are: [d * pp_size + s for d in range(dp_size)].
    # We pick the first available DP replica's checkpoint for each stage.
    ckpt_dir = os.path.join(cfg.output_dir, "checkpoints")
    stage_paths = {}  # {pp_stage_index: path}
    for stage_idx in range(cfg.pp_size):
        # Try each DP replica for this stage (prefer dp_rank=0, then 1, ...)
        for dp_idx in range(cfg.dp_size):
            global_rank = dp_idx * cfg.pp_size + stage_idx
            candidate = os.path.join(ckpt_dir, f"best_rank{global_rank}.pt")
            if os.path.exists(candidate):
                stage_paths[stage_idx] = candidate
                break

    missing_stages = [s for s in range(cfg.pp_size) if s not in stage_paths]
    if not missing_stages:
        print(f"\nMerging {cfg.pp_size} pipeline stages into single checkpoint...")
        merged = {}
        last_ckpt_data = None
        for stage_idx in range(cfg.pp_size):
            path = stage_paths[stage_idx]
            ckpt_data = torch.load(path, map_location='cpu', weights_only=False)
            sd = ckpt_data['model']
            print(f"  Stage {stage_idx}: {path} ({len(sd)} tensors)")
            for k, v in sd.items():
                merged[f"stage{stage_idx}.{k}"] = v
            # Keep the last stage's metadata (it has val_loss)
            if stage_idx == cfg.pp_size - 1:
                last_ckpt_data = ckpt_data
        merged_path = os.path.join(ckpt_dir, "merged_full_model.pt")
        torch.save({
            'model': merged,
            'pp_size': cfg.pp_size,
            'iter': last_ckpt_data.get('iter', -1) if last_ckpt_data else -1,
            'val_loss': last_ckpt_data.get('val_loss', float('nan')) if last_ckpt_data else float('nan'),
            'config': {
                'd_model': cfg.d_model, 'n_heads': cfg.n_heads,
                'n_layers': cfg.n_layers, 'd_ff': cfg.d_ff,
                'max_seq_len': cfg.max_seq_len, 'pp_size': cfg.pp_size,
                'layers_per_stage': cfg.layers_per_stage_list or [cfg.n_layers // cfg.pp_size] * cfg.pp_size,
                'gpu_map': cfg.gpu_map,
            },
        }, merged_path)
        n_params = sum(v.numel() for v in merged.values())
        print(f"  Merged model saved to {merged_path}")
        print(f"  Total parameters: {n_params:,}")
        print(f"  File size: {os.path.getsize(merged_path) / 1024 / 1024:.1f} MB")
    else:
        print(f"\n  Warning: Cannot merge — missing stage(s): {missing_stages}")
        print(f"  Found stages: {list(stage_paths.keys())}")
        print(f"  Re-run training to save all {cfg.pp_size} pipeline stage checkpoints.")
