"""YAML config loader for the unified ``asteroid.yaml``.

Provides ``load_config()`` to parse a YAML file into an
``AsteroidConfig`` dataclass, and ``save_config()`` to write
a config back to disk.  Also exposes ``load_cluster_nodes()``
for deploy scripts that only need the cluster section.

Usage::

    from asteroid.utils.config_loader import load_config
    cfg = load_config("asteroid.yaml")
    print(cfg.model_type, cfg.strategy, cfg.schedule_type)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, cast

logger = logging.getLogger(__name__)


def _section(raw: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Return a sub-dict from *raw*, defaulting to ``{}`` if missing."""
    val = raw.get(key)
    if val is None:
        return {}
    if not isinstance(val, dict):
        logger.warning("Expected dict for section %r, got %s", key, type(val).__name__)
        return {}
    return val


def load_config(path: str | Path) -> "AsteroidConfig":  # noqa: F821
    """Load ``asteroid.yaml`` and return a populated ``AsteroidConfig``."""
    import yaml
    from asteroid.core.config import AsteroidConfig

    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = cast(dict, yaml.safe_load(f)) or {}

    model = _section(raw, "model")
    training = _section(raw, "training")
    parallelism = _section(raw, "parallelism")
    ft = _section(raw, "fault_tolerance")
    mps_sec = _section(raw, "mps")
    deploy = _section(raw, "deploy")

    cfg = AsteroidConfig(
        # Model
        model_name=model.get("model_name", "gpt2"),
        model_type=model.get("model_type", "gpt2"),
        task_type=model.get("task_type", "classification"),
        num_layers=model.get("num_layers", 12),
        embedding_dim=model.get("embedding_dim", 768),
        num_heads=model.get("num_heads", 12),
        n_kv_heads=model.get("n_kv_heads", 0),
        d_ff=model.get("d_ff", 3072),
        max_seq_len=model.get("max_seq_len", 128),
        vocab_size=model.get("vocab_size", 50257),
        num_classes=model.get("num_classes", 2),
        dropout=model.get("dropout", 0.1),
        use_flash_attention=model.get("use_flash_attention", True),
        hf_model_name=model.get("hf_model_name", ""),
        # Training
        global_batch_size=training.get("global_batch_size", 256),
        micro_batch_size=training.get("micro_batch_size", 4),
        lr=training.get("lr", 3e-4),
        min_lr=training.get("min_lr", 1e-5),
        weight_decay=training.get("weight_decay", 0.01),
        max_iters=training.get("max_iters", 500),
        warmup_iters=training.get("warmup_iters", 50),
        grad_clip=training.get("grad_clip", 1.0),
        eval_interval=training.get("eval_interval", 100),
        log_interval=training.get("log_interval", 10),
        seed=training.get("seed", 42),
        dataset=training.get("dataset", "sst2"),
        # Parallelism
        strategy=parallelism.get("strategy", "asteroid"),
        schedule_type=parallelism.get("schedule_type", ""),
        world_size=parallelism.get("world_size", 3),
        num_stages=parallelism.get("num_stages", 2),
        comm_backend=parallelism.get("comm_backend", "torch_dist"),
        dist_url=parallelism.get("dist_url", "tcp://127.0.0.1:29600"),
        d2d_bandwidth_mbps=parallelism.get("d2d_bandwidth_mbps", 100.0),
        # Fault tolerance (unified)
        heartbeat_interval_s=ft.get("heartbeat_interval_s", 2.0),
        heartbeat_timeout_s=ft.get("heartbeat_timeout_s", 8.0),
        backward_timeout_ms=ft.get("backward_timeout_ms", 30000.0),
        replication_mode=ft.get("replication_mode", "all"),
        replication_interval=ft.get("replication_interval", 25),
        ft_check_interval=ft.get("ft_check_interval", 5),
        checkpoint_strategy=ft.get("checkpoint_strategy", "async"),
        checkpoint_dir=ft.get("checkpoint_dir", "./checkpoints"),
        checkpoint_interval=ft.get("checkpoint_interval", 100),
        # MPS
        mps_enabled=mps_sec.get("enabled", False),
        mps_active_thread_percentage=mps_sec.get("active_thread_percentage", 100),
        # Deploy / I/O
        output_dir=deploy.get("output_dir", "./asteroid_output"),
    )
    return cfg


def load_cluster_nodes(path: str | Path) -> List[Dict[str, Any]]:
    """Return the ``cluster.nodes`` list from ``asteroid.yaml``.

    Each entry is a dict with keys: ip, hostname, nic, gpu_id,
    memory_mb, rank, and optionally role.
    """
    import yaml

    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = cast(dict, yaml.safe_load(f)) or {}
    return raw.get("cluster", {}).get("nodes", [])


def save_config(cfg: "AsteroidConfig", path: str | Path) -> None:  # noqa: F821
    """Serialize an ``AsteroidConfig`` back to YAML."""
    import dataclasses
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    raw = {
        "model": {
            "model_name": cfg.model_name,
            "model_type": cfg.model_type,
            "task_type": cfg.task_type,
            "num_layers": cfg.num_layers,
            "embedding_dim": cfg.embedding_dim,
            "num_heads": cfg.num_heads,
            "n_kv_heads": cfg.n_kv_heads,
            "d_ff": cfg.d_ff,
            "max_seq_len": cfg.max_seq_len,
            "vocab_size": cfg.vocab_size,
            "num_classes": cfg.num_classes,
            "dropout": cfg.dropout,
            "use_flash_attention": cfg.use_flash_attention,
            "hf_model_name": cfg.hf_model_name,
        },
        "training": {
            "global_batch_size": cfg.global_batch_size,
            "micro_batch_size": cfg.micro_batch_size,
            "lr": cfg.lr,
            "min_lr": cfg.min_lr,
            "weight_decay": cfg.weight_decay,
            "max_iters": cfg.max_iters,
            "warmup_iters": cfg.warmup_iters,
            "grad_clip": cfg.grad_clip,
            "eval_interval": cfg.eval_interval,
            "log_interval": cfg.log_interval,
            "seed": cfg.seed,
            "dataset": cfg.dataset,
        },
        "parallelism": {
            "strategy": cfg.strategy,
            "schedule_type": cfg.schedule_type,
            "num_stages": cfg.num_stages,
            "world_size": cfg.world_size,
            "comm_backend": cfg.comm_backend,
            "dist_url": cfg.dist_url,
            "d2d_bandwidth_mbps": cfg.d2d_bandwidth_mbps,
        },
        "fault_tolerance": {
            "heartbeat_interval_s": cfg.heartbeat_interval_s,
            "heartbeat_timeout_s": cfg.heartbeat_timeout_s,
            "backward_timeout_ms": cfg.backward_timeout_ms,
            "replication_mode": cfg.replication_mode,
            "replication_interval": cfg.replication_interval,
            "ft_check_interval": cfg.ft_check_interval,
            "checkpoint_strategy": cfg.checkpoint_strategy,
            "checkpoint_dir": cfg.checkpoint_dir,
            "checkpoint_interval": cfg.checkpoint_interval,
        },
        "mps": {
            "enabled": cfg.mps_enabled,
            "active_thread_percentage": cfg.mps_active_thread_percentage,
        },
        "deploy": {
            "output_dir": cfg.output_dir,
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, default_flow_style=False, sort_keys=False)
    logger.info("Config saved to %s", path)


__all__ = ["load_config", "load_cluster_nodes", "save_config"]
