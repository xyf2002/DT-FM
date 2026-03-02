from __future__ import annotations

import importlib
import inspect

import pytest

from baselines.models.bert import EncoderBlock
from baselines.models.gpt2 import (
    CausalSelfAttention,
    ClassificationHead,
    GPT2Block,
    LMHead,
)
from baselines.models.llama import LlamaBlock, RMSNorm, RotaryEmbedding

torch = importlib.import_module("torch")


B = 2
S = 16
D = 64
N_HEADS = 4
D_FF = 128
MAX_SEQ_LEN = 32
DROPOUT = 0.0
USE_FLASH = False


def _build_module(cls, **overrides):
    signature = inspect.signature(cls)
    defaults = {
        "d_model": D,
        "embedding_dim": D,
        "hidden_size": D,
        "n_embd": D,
        "dim": D,
        "n_heads": N_HEADS,
        "num_heads": N_HEADS,
        "d_ff": D_FF,
        "max_seq_len": MAX_SEQ_LEN,
        "seq_len": MAX_SEQ_LEN,
        "seq_length": MAX_SEQ_LEN,
        "dropout": DROPOUT,
        "use_flash": USE_FLASH,
        "use_flash_attention": USE_FLASH,
        "n_kv_heads": N_HEADS,
    }
    kwargs = {}
    for name in signature.parameters:
        if name == "self":
            continue
        if name in overrides:
            kwargs[name] = overrides[name]
        elif name in defaults:
            kwargs[name] = defaults[name]
    return cls(**kwargs)


def _extract_logits(output):
    if isinstance(output, tuple):
        return output[0]
    if isinstance(output, dict):
        return output["logits"]
    return output


def _extract_loss(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple):
        for value in output:
            if torch.is_tensor(value) and value.ndim == 0:
                return value
    if isinstance(output, dict):
        loss = output.get("loss")
        if torch.is_tensor(loss):
            return loss
    raise AssertionError("Unable to extract scalar loss from model output")


def _apply_rotary(rotary, x, seq_len: int):
    signature = inspect.signature(rotary.forward)
    if "seq_len" in signature.parameters:
        return rotary(x, seq_len=seq_len)
    return rotary(x)


def test_gpt2_block_shape(cpu_device) -> None:
    """GPT2Block forward pass should preserve (B, S, D) tensor shape."""
    block = _build_module(GPT2Block).to(cpu_device)
    x = torch.randn(B, S, D, device=cpu_device)
    y = block(x)
    assert y.shape == (B, S, D)


def test_llama_block_shape(cpu_device) -> None:
    """LlamaBlock forward pass should preserve (B, S, D) tensor shape."""
    block = _build_module(LlamaBlock, n_kv_heads=N_HEADS).to(cpu_device)
    x = torch.randn(B, S, D, device=cpu_device)
    y = block(x)
    assert y.shape == (B, S, D)


def test_encoder_block_shape(cpu_device) -> None:
    """EncoderBlock forward pass should preserve (B, S, D) shape."""
    block = _build_module(EncoderBlock).to(cpu_device)
    x = torch.randn(B, S, D, device=cpu_device)
    y = block(x)
    assert y.shape == (B, S, D)


def test_classification_head_logits(cpu_device) -> None:
    """ClassificationHead should return logits of shape (B, num_classes)."""
    head = ClassificationHead(D, 5).to(cpu_device)
    x = torch.randn(B, S, D, device=cpu_device)
    output = head(x)
    logits = _extract_logits(output)
    assert logits.shape == (B, 5)


def test_classification_head_loss(cpu_device) -> None:
    """ClassificationHead should produce a scalar loss with targets."""
    head = ClassificationHead(D, 5).to(cpu_device)
    x = torch.randn(B, S, D, device=cpu_device)
    targets = torch.zeros(B, dtype=torch.long, device=cpu_device)
    output = head(x, targets=targets)
    loss = _extract_loss(output)
    assert getattr(loss, "ndim", None) == 0


def test_lm_head_logits(cpu_device) -> None:
    """LMHead should return token logits of shape (B, S, vocab_size)."""
    head = LMHead(D, 100).to(cpu_device)
    x = torch.randn(B, S, D, device=cpu_device)
    output = head(x)
    logits = _extract_logits(output)
    assert logits.shape == (B, S, 100)


def test_lm_head_loss(cpu_device) -> None:
    """LMHead should produce a scalar loss when token targets are provided."""
    head = LMHead(D, 100).to(cpu_device)
    x = torch.randn(B, S, D, device=cpu_device)
    targets = torch.zeros(B, S, dtype=torch.long, device=cpu_device)
    output = head(x, targets=targets)
    loss = _extract_loss(output)
    assert getattr(loss, "ndim", None) == 0


def test_causal_attention_invalid_dims() -> None:
    """CausalSelfAttention should reject d_model not divisible by n_heads."""
    with pytest.raises(ValueError):
        _build_module(CausalSelfAttention, d_model=63)


def test_rmsnorm_shape(cpu_device) -> None:
    """RMSNorm forward pass should preserve input tensor shape."""
    norm = RMSNorm(D).to(cpu_device)
    x = torch.randn(B, S, D, device=cpu_device)
    y = norm(x)
    assert y.shape == (B, S, D)


def test_rotary_embedding_shape(cpu_device) -> None:
    """RotaryEmbedding should return the same tensor shape for valid seq_len."""
    rotary = RotaryEmbedding(dim=16, max_seq_len=MAX_SEQ_LEN).to(cpu_device)
    x = torch.randn(B, N_HEADS, S, 16, device=cpu_device)
    y = _apply_rotary(rotary, x, seq_len=S)
    assert y.shape == x.shape


def test_rotary_embedding_extends_cache(cpu_device) -> None:
    """RotaryEmbedding should extend cache when sequence exceeds initial limit."""
    rotary = RotaryEmbedding(dim=16, max_seq_len=8).to(cpu_device)
    x = torch.randn(B, N_HEADS, S, 16, device=cpu_device)
    y = _apply_rotary(rotary, x, seq_len=S)
    assert y.shape == x.shape
