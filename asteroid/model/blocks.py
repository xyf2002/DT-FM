import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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