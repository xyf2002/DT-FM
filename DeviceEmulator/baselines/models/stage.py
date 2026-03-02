from __future__ import annotations

import torch
from torch import Tensor, nn

from baselines.core.config import BaseConfig, ModelConfig

from .registry import create_block, create_head


def _coerce_model_config(model_config: ModelConfig | BaseConfig) -> ModelConfig:
    if isinstance(model_config, BaseConfig):
        return model_config.model
    return model_config


class PipelineStage(nn.Module):
    """Pipeline stage over a contiguous transformer-layer range."""

    def __init__(
        self,
        model_config: ModelConfig,
        start_layer: int,
        end_layer: int,
        is_first: bool = False,
        is_last: bool = False,
    ) -> None:
        super().__init__()
        cfg = _coerce_model_config(model_config)
        self.model_config = cfg
        self.is_first = is_first
        self.is_last = is_last
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.task_type = cfg.task_type

        if is_first:
            self.embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
            self.pos_embedding = nn.Embedding(
                cfg.max_seq_len,
                cfg.embedding_dim,
            )
            self.drop = nn.Dropout(cfg.dropout)

        blocks = []
        for _ in range(start_layer, end_layer):
            blocks.append(create_block(cfg))
        self.blocks = nn.ModuleList(blocks)

        if is_last:
            self.head = create_head(cfg)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: Tensor,
        targets: Tensor | None = None,
    ) -> Tensor:
        if self.is_first:
            if x.dtype == torch.long:
                _, seq_len = x.shape[:2]
                pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
                x = self.embedding(x) + self.pos_embedding(pos)
                x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        if self.is_last:
            return self.head(x, targets)
        return x

    def num_params(self) -> int:
        return sum(param.numel() for param in self.parameters())

    def weight_size_bytes(self) -> int:
        return sum(
            param.numel() * param.element_size()
            for param in self.parameters()
        )

    def activation_size_bytes(
        self,
        batch_size: int,
        seq_len: int,
        d_model: int,
    ) -> int:
        n_blocks = len(self.blocks)
        return n_blocks * batch_size * seq_len * d_model * 4


__all__ = ["PipelineStage"]
