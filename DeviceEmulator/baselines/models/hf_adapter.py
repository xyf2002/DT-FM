from __future__ import annotations

from dataclasses import replace
from typing import cast

import torch
from torch import Tensor, nn

from baselines.core.config import BaseConfig, ModelConfig

from .registry import create_head
from .stage import PipelineStage


def _coerce_model_config(model_config: ModelConfig | BaseConfig) -> ModelConfig:
    if isinstance(model_config, BaseConfig):
        return model_config.model
    return model_config


class _HFEmbeddingAdapter(nn.Module):
    """Adapter that converts token ids to hidden states."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.source = source

    def forward(self, x: Tensor) -> Tensor:
        if x.dtype != torch.long:
            return x
        embeddings = getattr(self.source, "embeddings", None)
        if embeddings is not None and callable(embeddings):
            return embeddings(input_ids=x)

        wte = getattr(self.source, "wte", None)
        if wte is not None and callable(wte):
            hidden = cast(Tensor, wte(x))
            wpe = getattr(self.source, "wpe", None)
            if wpe is not None and callable(wpe):
                seq_len = x.shape[1]
                pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
                hidden = cast(Tensor, hidden + cast(Tensor, wpe(pos)))
            drop = getattr(self.source, "drop", None)
            if drop is not None and callable(drop):
                hidden = cast(Tensor, drop(hidden))
            return hidden

        embed_tokens = getattr(self.source, "embed_tokens", None)
        if embed_tokens is not None and callable(embed_tokens):
            return embed_tokens(x)

        if hasattr(self.source, "get_input_embeddings"):
            layer = self.source.get_input_embeddings()
            if isinstance(layer, nn.Module):
                return layer(x)
        raise ValueError("Unable to find HuggingFace embedding module")


class _HFBlockAdapter(nn.Module):
    """Adapter that normalizes HuggingFace layer outputs to Tensor."""

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        out = self.block(x)
        if isinstance(out, tuple):
            return out[0]
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        return out


class HFModelAdapter:
    """Adapter for converting HuggingFace models into PipelineStage blocks."""

    def __init__(
        self,
        model_name_or_path: str,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.trust_remote_code = trust_remote_code
        self._model: nn.Module | None = None

    def _load_model(self) -> nn.Module:
        if self._model is None:
            from transformers import AutoModel

            self._model = AutoModel.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=self.trust_remote_code,
            )
        return self._model

    def _find_embedding_source(self, model: nn.Module) -> nn.Module:
        candidates = [
            model,
            getattr(model, "model", None),
            getattr(model, "transformer", None),
        ]
        for candidate in candidates:
            if not isinstance(candidate, nn.Module):
                continue
            if any(
                hasattr(candidate, attr)
                for attr in (
                    "embeddings",
                    "wte",
                    "embed_tokens",
                    "get_input_embeddings",
                )
            ):
                return candidate
        return model

    @staticmethod
    def extract_layers(model: nn.Module) -> nn.ModuleList:
        """Extract transformer layers across common HuggingFace layouts."""
        candidates = [
            model,
            getattr(model, "model", None),
            getattr(model, "transformer", None),
            getattr(model, "encoder", None),
        ]
        for container in candidates:
            if not isinstance(container, nn.Module):
                continue
            for attr in ("layers", "h", "layer"):
                layers = getattr(container, attr, None)
                if isinstance(layers, nn.ModuleList):
                    return layers
        encoder = getattr(model, "encoder", None)
        if isinstance(encoder, nn.Module):
            layers = getattr(encoder, "layer", None)
            if isinstance(layers, nn.ModuleList):
                return layers
        raise ValueError("Unable to locate transformer layers in HF model")

    def get_model_config(self) -> ModelConfig:
        model = self._load_model()
        hf_cfg = model.config

        model_type = getattr(hf_cfg, "model_type", "gpt2")
        embedding_dim = getattr(
            hf_cfg,
            "hidden_size",
            getattr(hf_cfg, "n_embd", 768),
        )
        num_layers = getattr(
            hf_cfg,
            "num_hidden_layers",
            getattr(hf_cfg, "n_layer", 12),
        )
        num_heads = getattr(
            hf_cfg,
            "num_attention_heads",
            getattr(hf_cfg, "n_head", 12),
        )
        d_ff = getattr(
            hf_cfg,
            "intermediate_size",
            getattr(hf_cfg, "n_inner", embedding_dim * 4),
        )
        max_seq_len = getattr(
            hf_cfg,
            "max_position_embeddings",
            getattr(hf_cfg, "n_positions", 2048),
        )
        vocab_size = getattr(hf_cfg, "vocab_size", 50257)
        dropout = getattr(
            hf_cfg,
            "hidden_dropout_prob",
            getattr(hf_cfg, "resid_pdrop", 0.1),
        )
        use_flash = bool(getattr(hf_cfg, "_attn_implementation", "") == "flash")

        lm_types = {
            "gpt2",
            "gptj",
            "gpt_neox",
            "llama",
            "mistral",
            "qwen2",
        }
        task_type = "lm" if model_type in lm_types else "classification"
        task = (
            "Seq2SeqClassification"
            if task_type == "lm"
            else "SeqClassification"
        )

        return ModelConfig(
            model_name=self.model_name_or_path,
            model_type=model_type,
            task_type=task_type,
            task=task,
            seq_length=max_seq_len,
            max_seq_len=max_seq_len,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            vocab_size=vocab_size,
            dropout=dropout,
            use_flash_attention=use_flash,
        )

    def to_pipeline_stages(
        self,
        num_stages: int,
        model_config: ModelConfig | None = None,
    ) -> list[PipelineStage]:
        if num_stages <= 0:
            raise ValueError("num_stages must be positive")

        model = self._load_model()
        base_cfg = (
            self.get_model_config()
            if model_config is None
            else model_config
        )
        cfg = _coerce_model_config(base_cfg)

        layers = self.extract_layers(model)
        total_layers = len(layers)
        base, extra = divmod(total_layers, num_stages)

        stages: list[PipelineStage] = []
        start = 0
        for idx in range(num_stages):
            span = base + (1 if idx < extra else 0)
            end = start + span

            stub_cfg = replace(cfg, model_type="gpt2")
            stage = PipelineStage(
                model_config=stub_cfg,
                start_layer=0,
                end_layer=0,
                is_first=False,
                is_last=False,
            )
            stage.start_layer = start
            stage.end_layer = end
            stage.blocks = nn.ModuleList(
                [_HFBlockAdapter(block) for block in layers[start:end]]
            )

            if idx == 0:
                source = self._find_embedding_source(model)
                embedding = _HFEmbeddingAdapter(source)
                stage.blocks = nn.ModuleList([embedding, *stage.blocks])

            if idx == num_stages - 1:
                stage.is_last = True
                stage.head = create_head(cfg)

            stages.append(stage)
            start = end

        return stages


__all__ = ["HFModelAdapter"]
