"""Example: GPT-2 training with DT-FM strategy.

Usage:
    python -m baselines.examples.train_gpt2_dtfm \
        --config baselines/configs/dtfm_default.yaml
"""
from __future__ import annotations

import argparse
import logging

from baselines.core.config import DeviceConfig, DeviceTopology
from baselines.fault_tolerance import BasicCheckpoint
from baselines.models import PipelineStage
from baselines.strategies import DTFMStrategy
from baselines.utils.config_loader import load_config
from baselines.utils.seed import seed_everything as set_deterministic_seed

logger = logging.getLogger(__name__)


class _Args(argparse.Namespace):
    config: str = ""
    dry_run: bool = False


def main() -> None:
    parser = argparse.ArgumentParser(description="GPT-2 DT-FM training")
    _ = parser.add_argument(
        "--config",
        type=str,
        default="baselines/configs/dtfm_default.yaml",
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

    topology = DeviceTopology(
        device_specs=[
            DeviceConfig(
                device_id=i,
                compute_capacity=1.0,
                memory_budget_mb=8192.0,
            )
            for i in range(cfg.distributed.world_size)
        ],
        bandwidths={
            (i, j): 100.0
            for i in range(cfg.distributed.world_size)
            for j in range(cfg.distributed.world_size)
            if i != j
        },
        latencies={
            (i, j): 0.1
            for i in range(cfg.distributed.world_size)
            for j in range(cfg.distributed.world_size)
            if i != j
        },
    )

    strategy = DTFMStrategy(
        pp_size=cfg.parallel.pp_size,
        dp_size=cfg.parallel.dp_size,
    )
    plan = strategy.create_plan(cfg.model, topology)
    logger.info(
        "Plan: partition=%s schedule=%s latency=%.2fms",
        plan.partition_points,
        plan.schedule_type,
        plan.estimated_latency_ms,
    )

    if args.dry_run:
        logger.info("Dry run complete.")
        return

    num_stages = len(plan.partition_points) + 1
    boundaries = [0, *plan.partition_points, cfg.model.num_layers]
    stages: list[PipelineStage] = []
    for i in range(num_stages):
        stage = PipelineStage(
            model_config=cfg.model,
            start_layer=boundaries[i],
            end_layer=boundaries[i + 1],
            is_first=(i == 0),
            is_last=(i == num_stages - 1),
        )
        stages.append(stage)
        logger.info(
            "Stage %d: layers [%d, %d) params=%d",
            i,
            boundaries[i],
            boundaries[i + 1],
            stage.num_params(),
        )

    _ = BasicCheckpoint(
        checkpoint_dir=cfg.fault_tolerance.checkpoint_dir,
        interval=cfg.fault_tolerance.checkpoint_interval,
    )
    logger.info("Setup complete. %d stages ready for training.", len(stages))


if __name__ == "__main__":
    main()
