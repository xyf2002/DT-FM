from __future__ import annotations

import math

import torch.nn.functional as F
from torch import Tensor, nn

from .gpt2 import GPT2MLP


class BidirectionalAttention(nn.Module):
    """BERT-style self-attention without causal masking."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
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
                is_causal=False,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_dropout(self.c_proj(y))


class EncoderBlock(nn.Module):
    """BERT-style pre-norm transformer encoder block."""

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
        del max_seq_len
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = BidirectionalAttention(
            d_model,
            n_heads,
            dropout,
            use_flash=use_flash,
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = GPT2MLP(d_model, d_ff, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


__all__ = [
    "BidirectionalAttention",
    "EncoderBlock",
]
