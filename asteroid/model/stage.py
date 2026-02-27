from typing import Dict, Any
import torch
import torch.nn as nn

from ..core.config import AsteroidConfig
from .blocks import GPT2Block, EncoderBlock
from .heads import ClassificationHead, LMHead

# ── Model registry & factory (makes Asteroid LLM-agnostic) ──────────

# Each entry: { "block": BlockClass, "causal": bool }
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gpt2": {"block": GPT2Block, "causal": True},
    "encoder": {"block": EncoderBlock, "causal": False},
}

TASK_REGISTRY: Dict[str, type] = {
    "classification": ClassificationHead,
    "lm": LMHead,
}

def register_model(name: str, block_cls: type, causal: bool = True):
    """Register a custom transformer block for use with Asteroid."""
    MODEL_REGISTRY[name] = {"block": block_cls, "causal": causal}

def register_task(name: str, head_cls: type):
    """Register a custom task head."""
    TASK_REGISTRY[name] = head_cls

def _create_block(cfg: AsteroidConfig) -> nn.Module:
    """Factory: instantiate one transformer block from config."""
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
    """Factory: instantiate a task head from config."""
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
    """A pipeline stage holding a contiguous slice of transformer layers."""
    def __init__(self, cfg: AsteroidConfig, start_layer: int, end_layer: int,
                 is_first: bool = False, is_last: bool = False):
        super().__init__()
        self.is_first = is_first
        self.is_last = is_last
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.task_type = cfg.task_type

        # ── Embedding ──
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
        return n_blocks * batch_size * seq_len * d_model * 4