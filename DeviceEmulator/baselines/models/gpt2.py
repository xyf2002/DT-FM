from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CausalSelfAttention(nn.Module):
    """GPT-2 style multi-head self-attention with causal masking."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        use_flash: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(
                1,
                1,
                max_seq_len,
                max_seq_len,
            ),
        )
        self.use_flash = use_flash and hasattr(
            F,
            "scaled_dot_product_attention",
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, channels = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(channels, dim=2)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.transpose(1, 2)
        if self.use_flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(
                self.bias[:, :, :seq_len, :seq_len] == 0,
                float("-inf"),
            )
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_dropout(self.c_proj(y))


class GPT2MLP(nn.Module):
    """GPT-2 MLP block using c_fc/c_proj parameter names."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_ff)
        self.c_proj = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class GPT2Block(nn.Module):
    """GPT-2 pre-norm transformer block."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
        use_flash: bool = True,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model,
            n_heads,
            max_seq_len,
            dropout,
            use_flash=use_flash,
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = GPT2MLP(d_model, d_ff, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ClassificationHead(nn.Module):
    """Sequence-classification head with configurable pooling."""

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        pool: str = "mean",
    ) -> None:
        super().__init__()
        if pool not in {"mean", "first", "last"}:
            raise ValueError("pool must be one of: mean, first, last")
        self.pool = pool
        self.ln_f = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: Tensor, targets: Tensor | None = None) -> Tensor:
        x = self.ln_f(x)
        if self.pool == "first":
            pooled = x[:, 0]
        elif self.pool == "last":
            pooled = x[:, -1]
        else:
            pooled = x.mean(dim=1)
        logits = self.classifier(pooled)
        if targets is not None:
            return F.cross_entropy(logits, targets, reduction="mean")
        return logits


class LMHead(nn.Module):
    """Language-modeling head with causal next-token loss."""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: Tensor, targets: Tensor | None = None) -> Tensor:
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            return F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                reduction="mean",
            )
        return logits


__all__ = [
    "CausalSelfAttention",
    "GPT2MLP",
    "GPT2Block",
    "ClassificationHead",
    "LMHead",
]
