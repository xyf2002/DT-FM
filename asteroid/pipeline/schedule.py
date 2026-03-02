"""Pipeline schedule implementations: GPipe and 1F1B.

Provides ``build_schedule()`` factory to generate a micro-batch
execution timeline for a pipeline-parallel training step.

Each schedule is returned as a list-of-lists:
``schedule[stage_id]`` is a list of ``(action, micro_batch_id)``
tuples where *action* is ``"F"`` (forward) or ``"B"`` (backward).

Usage::

    from asteroid.pipeline.schedule import build_schedule
    timeline = build_schedule("1f1b", num_stages=3, num_microbatches=6)
    for stage, ops in enumerate(timeline):
        print(f"Stage {stage}: {ops}")
"""
from __future__ import annotations

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

Action = str  # "F" or "B"
Op = Tuple[Action, int]  # (action, micro_batch_id)
StageTimeline = List[Op]
Schedule = List[StageTimeline]


def gpipe_schedule(num_stages: int, num_microbatches: int) -> Schedule:
    """GPipe (all-forward then all-backward).

    Low complexity, but high pipeline bubble.
    """
    schedule: Schedule = [[] for _ in range(num_stages)]
    for stage in range(num_stages):
        # All forwards first
        for mb in range(num_microbatches):
            schedule[stage].append(("F", mb))
        # Then all backwards (in reverse micro-batch order)
        for mb in reversed(range(num_microbatches)):
            schedule[stage].append(("B", mb))
    return schedule


def one_f_one_b_schedule(num_stages: int, num_microbatches: int) -> Schedule:
    """1F1B (one forward one backward) interleaved schedule.

    The steady state alternates F and B to keep the pipeline full
    and minimize the bubble.  Structure per stage *s*:

    1. **Warmup** — *s* forward passes (filling the pipeline)
    2. **Steady state** — alternating (F, B) pairs
    3. **Cooldown** — remaining backward passes
    """
    if num_microbatches < num_stages:
        logger.warning(
            "1F1B: num_microbatches (%d) < num_stages (%d); "
            "falling back to GPipe",
            num_microbatches, num_stages,
        )
        return gpipe_schedule(num_stages, num_microbatches)

    schedule: Schedule = [[] for _ in range(num_stages)]

    for stage in range(num_stages):
        warmup = num_stages - stage - 1  # forwards before first backward
        fwd_idx = 0
        bwd_idx = 0

        # Phase 1: warmup forwards
        for _ in range(warmup):
            if fwd_idx < num_microbatches:
                schedule[stage].append(("F", fwd_idx))
                fwd_idx += 1

        # Phase 2: steady state (1F1B pairs)
        steady = num_microbatches - warmup
        for _ in range(steady):
            if fwd_idx < num_microbatches:
                schedule[stage].append(("F", fwd_idx))
                fwd_idx += 1
            schedule[stage].append(("B", bwd_idx))
            bwd_idx += 1

        # Phase 3: cooldown backwards
        while bwd_idx < num_microbatches:
            schedule[stage].append(("B", bwd_idx))
            bwd_idx += 1

    return schedule


def build_schedule(
    schedule_type: str,
    num_stages: int,
    num_microbatches: int,
) -> Schedule:
    """Factory: build a pipeline schedule by name.

    Args:
        schedule_type: ``"gpipe"`` or ``"1f1b"``.
        num_stages: number of pipeline stages.
        num_microbatches: number of micro-batches per global step.

    Returns:
        A list indexed by stage, each containing ``(action, mb_id)`` tuples.
    """
    t = schedule_type.lower().replace("-", "").replace("_", "")
    if t in ("gpipe", ""):
        return gpipe_schedule(num_stages, num_microbatches)
    if t in ("1f1b", "onef1b", "oneforwardoneback"):
        return one_f_one_b_schedule(num_stages, num_microbatches)
    raise ValueError(
        f"Unknown schedule_type '{schedule_type}'. Use 'gpipe' or '1f1b'."
    )


__all__ = [
    "gpipe_schedule",
    "one_f_one_b_schedule",
    "build_schedule",
]
