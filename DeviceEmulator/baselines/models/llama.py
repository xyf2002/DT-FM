from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary position embedding cache and application."""

    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Rotary dim must be even")
        self.dim = dim
        self.max_seq_len_cached = max_seq_len
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int) -> None:
        positions = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        self.max_seq_len_cached = seq_len

    def forward(self, x: Tensor, seq_len: int) -> Tensor:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        cos = self.cos_cached[:, :, :seq_len, :].to(
            device=x.device,
            dtype=x.dtype,
        )
        sin = self.sin_cached[:, :, :seq_len, :].to(
            device=x.device,
            dtype=x.dtype,
        )
        return (x * cos) + (_rotate_half(x) * sin)


class LlamaMLP(nn.Module):
    """LLaMA feed-forward block with SwiGLU gating."""

    def __init__(
        self,
        d_model: int = 4096,
        d_ff: int = 11008,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))


class LlamaAttention(nn.Module):
    """LLaMA causal attention with optional grouped-query attention."""

    def __init__(
        self,
        d_model: int = 4096,
        n_heads: int = 32,
        n_kv_heads: int = 32,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
        use_flash: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            d_model,
            n_kv_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            d_model,
            n_kv_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len)
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
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim,
        )
        q = q.transpose(1, 2)
        k = self.k_proj(x).view(
            batch_size,
            seq_len,
            self.n_kv_heads,
            self.head_dim,
        )
        k = k.transpose(1, 2)
        v = self.v_proj(x).view(
            batch_size,
            seq_len,
            self.n_kv_heads,
            self.head_dim,
        )
        v = v.transpose(1, 2)

        q = self.rotary(q, seq_len)
        k = self.rotary(k, seq_len)

        if self.n_kv_heads < self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

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

        y = y.transpose(1, 2).contiguous()
        y = y.view(batch_size, seq_len, self.n_heads * self.head_dim)
        return self.resid_dropout(self.o_proj(y))


class LlamaBlock(nn.Module):
    """LLaMA pre-norm transformer block using RMSNorm."""

    def __init__(
        self,
        d_model: int = 4096,
        n_heads: int = 32,
        d_ff: int = 11008,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
        use_flash: bool = True,
        n_kv_heads: int | None = None,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.attn_norm = RMSNorm(d_model, eps=rms_norm_eps)
        self.attn = LlamaAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=kv_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            use_flash=use_flash,
        )
        self.ffn_norm = RMSNorm(d_model, eps=rms_norm_eps)
        self.mlp = LlamaMLP(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x


__all__ = [
    "RMSNorm",
    "RotaryEmbedding",
    "LlamaMLP",
    "LlamaAttention",
    "LlamaBlock",
]
