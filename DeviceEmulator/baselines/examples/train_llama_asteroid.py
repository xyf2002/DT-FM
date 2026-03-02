"""Example: Llama2 planning with Asteroid strategy.

Usage:
    python -m baselines.examples.train_llama_asteroid \
        --config baselines/configs/asteroid_default.yaml \
        --model meta-llama/Llama-2-7b-hf --dry-run
"""
from __future__ import annotations

import argparse
import logging

from baselines.core.config import DeviceConfig, DeviceTopology
from baselines.fault_tolerance import AsyncCheckpoint, HeartbeatDetector
from baselines.models import HFModelAdapter
from baselines.strategies import AsteroidStrategy
from baselines.utils.config_loader import load_config
from baselines.utils.seed import seed_everything as set_deterministic_seed

logger = logging.getLogger(__name__)


class _Args(argparse.Namespace):
    config: str = ""
    model: str = ""
    dry_run: bool = False


def _build_demo_topology(world_size: int) -> DeviceTopology:
    return DeviceTopology(
        device_specs=[
            DeviceConfig(
                device_id=i,
                compute_capacity=1.0,
                memory_budget_mb=32768.0,
            )
            for i in range(world_size)
        ],
        bandwidths={
            (i, j): 80.0
            for i in range(world_size)
            for j in range(world_size)
            if i != j
        },
        latencies={
            (i, j): 0.2
            for i in range(world_size)
            for j in range(world_size)
            if i != j
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Llama2 Asteroid planning")
    _ = parser.add_argument(
        "--config",
        type=str,
        default="baselines/configs/asteroid_default.yaml",
    )
    _ = parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace model name",
    )
    _ = parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan only, no training",
    )
    args = parser.parse_args(namespace=_Args())

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = load_config(args.config)
    set_deterministic_seed(cfg.training.seed)
    logger.info("Loaded config from %s", args.config)

    adapter = HFModelAdapter(model_name_or_path=args.model)
    hf_model_cfg = adapter.get_model_config()
    cfg.model = hf_model_cfg
    logger.info(
        "Resolved HF config: model=%s layers=%d dim=%d",
        cfg.model.model_name,
        cfg.model.num_layers,
        cfg.model.embedding_dim,
    )

    topology = _build_demo_topology(cfg.distributed.world_size)
    strategy = AsteroidStrategy(
        num_stages=cfg.parallel.pp_size,
        micro_batch_size=cfg.training.micro_batch_size,
        num_microbatches=cfg.training.num_microbatches,
    )
    plan = strategy.create_plan(cfg.model, topology)
    logger.info(
        "Plan: partition=%s schedule=%s latency=%.2fms",
        plan.partition_points,
        plan.schedule_type,
        plan.estimated_latency_ms,
    )

    _ = AsyncCheckpoint(
        checkpoint_dir=cfg.fault_tolerance.checkpoint_dir,
        interval=cfg.fault_tolerance.checkpoint_interval,
    )
    heartbeat = HeartbeatDetector(
        device_id=0,
        interval_s=cfg.fault_tolerance.heartbeat_interval_s,
        timeout_s=cfg.fault_tolerance.heartbeat_timeout_s,
    )
    heartbeat.start()
    alive = heartbeat.check_alive(0)
    heartbeat.stop()
    logger.info("Heartbeat self-check: alive=%s", alive)

    if args.dry_run:
        logger.info("Dry run complete.")
        return

    stages = adapter.to_pipeline_stages(
        num_stages=max(1, len(plan.partition_points) + 1),
        model_config=cfg.model,
    )
    logger.info("Setup complete. %d HF stages ready for training.", len(stages))


if __name__ == "__main__":
    main()
