"""HuggingFace model adapter for pipeline-parallel Asteroid stages.

Loads any HuggingFace ``AutoModel``, splits it into pipeline stages,
and wraps layers with adapters so the forward signature matches
``AsteroidStage.forward()``.

Usage::

    from asteroid.model.hf_adapter import HFModelAdapter
    adapter = HFModelAdapter("gpt2")
    stages = adapter.to_pipeline_stages(num_stages=3, cfg=cfg)
"""
from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from ..core.config import AsteroidConfig
from .heads import ClassificationHead, LMHead
from .stage import AsteroidStage, TASK_REGISTRY, _create_head


# ---------------------------------------------------------------------------
# Internal wrapper modules
# ---------------------------------------------------------------------------

class _HFEmbeddingAdapter(nn.Module):
    """Convert token ids to hidden states using an HF model's embeddings."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.source = source

    def forward(self, x: Tensor) -> Tensor:
        if x.dtype != torch.long:
            return x

        # Try common HF embedding patterns
        embeddings = getattr(self.source, "embeddings", None)
        if embeddings is not None and callable(embeddings):
            return embeddings(input_ids=x)

        # GPT-2 style: wte + wpe + drop
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

        # LLaMA / Mistral style
        embed_tokens = getattr(self.source, "embed_tokens", None)
        if embed_tokens is not None and callable(embed_tokens):
            return embed_tokens(x)

        # Generic fallback
        if hasattr(self.source, "get_input_embeddings"):
            layer = self.source.get_input_embeddings()
            if isinstance(layer, nn.Module):
                return layer(x)

        raise ValueError("Unable to find HuggingFace embedding module")


class _HFBlockAdapter(nn.Module):
    """Normalize HuggingFace layer outputs to a plain Tensor."""

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


# ---------------------------------------------------------------------------
# Public adapter
# ---------------------------------------------------------------------------

class HFModelAdapter:
    """Convert a HuggingFace model into a list of ``AsteroidStage`` objects."""

    def __init__(
        self,
        model_name_or_path: str,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.trust_remote_code = trust_remote_code
        self._model: nn.Module | None = None

    # ---- internal helpers ------------------------------------------------

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
        for c in candidates:
            if not isinstance(c, nn.Module):
                continue
            if any(
                hasattr(c, attr)
                for attr in ("embeddings", "wte", "embed_tokens",
                             "get_input_embeddings")
            ):
                return c
        return model

    @staticmethod
    def extract_layers(model: nn.Module) -> nn.ModuleList:
        """Extract transformer layer list across common HF layouts."""
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
        raise ValueError("Unable to locate transformer layers in HF model")

    # ---- config extraction -----------------------------------------------

    def get_model_config_overrides(self) -> dict:
        """Return config fields inferred from the HF model."""
        model = self._load_model()
        hf_cfg = model.config

        model_type = getattr(hf_cfg, "model_type", "gpt2")
        embedding_dim = getattr(hf_cfg, "hidden_size",
                                getattr(hf_cfg, "n_embd", 768))
        num_layers = getattr(hf_cfg, "num_hidden_layers",
                             getattr(hf_cfg, "n_layer", 12))
        num_heads = getattr(hf_cfg, "num_attention_heads",
                            getattr(hf_cfg, "n_head", 12))
        d_ff = getattr(hf_cfg, "intermediate_size",
                       getattr(hf_cfg, "n_inner", embedding_dim * 4))
        max_seq_len = getattr(hf_cfg, "max_position_embeddings",
                              getattr(hf_cfg, "n_positions", 2048))
        vocab_size = getattr(hf_cfg, "vocab_size", 50257)
        dropout = getattr(hf_cfg, "hidden_dropout_prob",
                          getattr(hf_cfg, "resid_pdrop", 0.1))

        lm_types = {"gpt2", "gptj", "gpt_neox", "llama", "mistral", "qwen2"}
        task_type = "lm" if model_type in lm_types else "classification"

        return {
            "model_type": model_type,
            "embedding_dim": embedding_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "max_seq_len": max_seq_len,
            "vocab_size": vocab_size,
            "dropout": dropout,
            "task_type": task_type,
        }

    # ---- main entry point ------------------------------------------------

    def to_pipeline_stages(
        self,
        num_stages: int,
        cfg: AsteroidConfig,
    ) -> list[AsteroidStage]:
        """Split the HF model into ``num_stages`` AsteroidStage objects.

        The stages reuse the HF model's pretrained weights via adapters
        instead of re-initializing from scratch.
        """
        if num_stages <= 0:
            raise ValueError("num_stages must be positive")

        model = self._load_model()
        layers = self.extract_layers(model)
        total_layers = len(layers)
        base, extra = divmod(total_layers, num_stages)

        stages: list[AsteroidStage] = []
        start = 0
        for idx in range(num_stages):
            span = base + (1 if idx < extra else 0)
            end = start + span
            is_first = idx == 0
            is_last = idx == num_stages - 1

            # Build a lightweight AsteroidStage shell (skip __init__ block
            # creation since we supply our own wrapped HF blocks).
            stage = AsteroidStage.__new__(AsteroidStage)
            nn.Module.__init__(stage)
            stage.is_first = is_first
            stage.is_last = is_last
            stage.start_layer = start
            stage.end_layer = end
            stage.task_type = cfg.task_type

            blocks: list[nn.Module] = []
            if is_first:
                source = self._find_embedding_source(model)
                blocks.append(_HFEmbeddingAdapter(source))

            blocks.extend(_HFBlockAdapter(layers[i]) for i in range(start, end))
            stage.blocks = nn.ModuleList(blocks)

            if is_last:
                stage.head = _create_head(cfg)

            stages.append(stage)
            start = end

        return stages


__all__ = ["HFModelAdapter"]
