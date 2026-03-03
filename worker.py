#!/usr/bin/env python3
"""
Asteroid Worker - Multi-node distributed training worker.

This script is designed to be launched by Kubernetes with environment variables:
  - RANK: Global rank of this worker
  - WORLD_SIZE: Total number of workers
  - MASTER_ADDR: Hostname/IP of rank 0 (headless service DNS)
  - MASTER_PORT: Port for distributed communication
  - NCCL_SOCKET_IFNAME: Network interface for NCCL
  - CUDA_VISIBLE_DEVICES: GPU to use
  - HPP_PLAN_PATH: Path to hpp_plan.json

For local testing with mp.spawn, run with --local flag:
  python worker.py --local --world-size 3

Usage (K8s mode - env vars injected by K8s):
  python worker.py

Usage (local testing):
  python worker.py --local --world-size 3 --num-stages 2
"""

import os
import sys
import json
import argparse

# =============================================================================
# Environment Setup - Must happen BEFORE importing torch
# =============================================================================

def _setup_multinode_env():
    """
    Setup environment for multi-node execution.
    Reads from K8s-injected environment variables.
    """
    # Validate required environment variables
    required = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
    missing = [var for var in required if var not in os.environ]
    
    if missing:
        print(f"[ENV] Missing required vars: {missing}", file=sys.stderr)
        print("[ENV] Running in local mode with defaults...", file=sys.stderr)
        return False
    
    # Set NCCL configuration from env (injected by K8s template)
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_SHM_DISABLE", "0")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    
    return True


def _setup_local_env(world_size: int, port: int = None):
    """
    Setup environment for local mp.spawn testing.
    """
    import random
    
    if port is None:
        port = random.randint(20000, 30000)
    
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(world_size)))
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_SHM_DISABLE", "0")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    
    return port


# Determine execution mode before importing torch
_MULTINODE_MODE = _setup_multinode_env()

# Now safe to import torch
import time
import random
from datetime import timedelta
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.multiprocessing as mp

# Modular imports
from asteroid.core.config import AsteroidConfig, HPPPlanConfig, DeviceSpec
from asteroid.core.state import AsteroidStateManager
from asteroid.utils.logger import EVENT_LOGGER, logger
from asteroid.model.stage import AsteroidStage
from asteroid.optim.optim_utils import flatten_params, get_lr
from asteroid.ft.fault_tolerance import AsteroidFaultTolerance
from asteroid.utils.data_utils import prepare_data, prepare_sst2
from asteroid.planner.profiler import AsteroidProfiler
from asteroid.planner.dp_planner import AsteroidPlanner
from asteroid.pipeline.schedule import build_schedule

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

def worker(rank: int, cfg: AsteroidConfig,
           train_data: Tuple[torch.Tensor, torch.Tensor],
           val_data: Tuple[torch.Tensor, torch.Tensor],
           plan: Optional[HPPPlanConfig] = None,
           use_env_init: bool = False):
    """
    Main worker function for distributed training.
    
    Args:
        rank: Global rank of this worker
        cfg: Asteroid configuration
        train_data: (embeddings, labels) training data
        val_data: (embeddings, labels) validation data
        plan: HPP execution plan (optional)
        use_env_init: If True, use env:// init method (for K8s)
    """
    # Device selection - use CUDA_VISIBLE_DEVICES if set, otherwise map by rank
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        # K8s sets CUDA_VISIBLE_DEVICES to single GPU, so use cuda:0
        visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if len(visible_devices) == 1:
            cuda_id = 0
        else:
            cuda_id = rank % len(visible_devices)
    else:
        cuda_id = rank % torch.cuda.device_count()
    
    device = torch.device(f'cuda:{cuda_id}')
    torch.cuda.set_device(device)
    torch.manual_seed(cfg.seed + rank)
    
    print(f"[RANK {rank}] Device: cuda:{cuda_id} ({torch.cuda.get_device_name(device)})", flush=True)

    if plan is not None and plan.device_groups:
        _rank_to_stage, _rank_to_dp_pos, _stage_sizes = {}, {}, {}
        for _s, _devs in plan.device_groups.items():
            _stage_sizes[_s] = len(_devs)
            for _dp, _dev in enumerate(_devs):
                _rank_to_stage[_dev] = _s
                _rank_to_dp_pos[_dev] = _dp

        pp_size = plan.num_stages
        pp_rank = _rank_to_stage[rank]
        dp_rank = _rank_to_dp_pos[rank]
        dp_size = _stage_sizes[pp_rank]
        print(f"[RANK {rank}] Using planner device_groups: stage_sizes={dict(_stage_sizes)}", flush=True)
    else:
        pp_size = cfg.num_stages
        dp_size = cfg.world_size // pp_size
        pp_rank = rank % pp_size
        dp_rank = rank // pp_size

    # Stage Booleans (Strict isolation for 1-stage vs Multi-stage)
    is_single = (pp_size == 1)
    is_first = (pp_rank == 0) and not is_single
    is_last = (pp_rank == pp_size - 1) and not is_single

    state = AsteroidStateManager()
    state.global_rank = rank
    state.stage_idx = pp_rank
    state.device = device
    state._dp_rank = dp_rank
    state._dp_size = dp_size
    state.plan = plan

    if not torch.distributed.is_initialized():
        # Use split backends: gloo for CPU ops (barrier), NCCL for GPU ops (allreduce, send/recv)
        dist_backend = 'cpu:gloo,cuda:nccl'
        if use_env_init:
            # Multi-node mode: use env:// which reads MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE
            print(f"[RANK {rank}] Initializing distributed (env://, backend={dist_backend})...", flush=True)
            torch.distributed.init_process_group(
                backend=dist_backend,
                init_method='env://',
                timeout=timedelta(seconds=300))
        else:
            # Local mode: use explicit URL
            print(f"[RANK {rank}] Initializing distributed ({cfg.dist_url}, backend={dist_backend})...", flush=True)
            torch.distributed.init_process_group(
                backend=dist_backend,
                init_method=cfg.dist_url,
                world_size=cfg.world_size, 
                rank=rank,
                timeout=timedelta(seconds=120))
        print(f"[RANK {rank}] Distributed initialized successfully", flush=True)

    pp_process_group, pp_ranks_in_group = None, None
    dp_process_group, dp_ranks_in_group = None, None

    if plan is not None and plan.device_groups:
        max_dp = max(_stage_sizes.values())
        for d in range(max_dp):
            pp_ranks = []
            for s in range(pp_size):
                devs = plan.device_groups[s]
                if d < len(devs):
                    pp_ranks.append(devs[d])
            if len(pp_ranks) >= 1:
                grp = torch.distributed.new_group(ranks=pp_ranks)
                if rank in pp_ranks:
                    pp_process_group = grp
                    pp_ranks_in_group = pp_ranks

        for s in range(pp_size):
            dp_ranks = list(plan.device_groups[s])
            grp = torch.distributed.new_group(ranks=dp_ranks)
            if rank in dp_ranks:
                dp_process_group = grp
                dp_ranks_in_group = dp_ranks
    else:
        for d in range(dp_size):
            pp_ranks = [d * pp_size + s for s in range(pp_size)]
            grp = torch.distributed.new_group(ranks=pp_ranks)
            if rank in pp_ranks:
                pp_process_group = grp
                pp_ranks_in_group = pp_ranks

        for s in range(pp_size):
            dp_ranks = [d * pp_size + s for d in range(dp_size)]
            grp = torch.distributed.new_group(ranks=dp_ranks)
            if rank in dp_ranks:
                dp_process_group = grp
                dp_ranks_in_group = dp_ranks

    dist_store = torch.distributed.distributed_c10d._get_default_store()
    # Skip NCCL communicators - use torch.distributed instead for pipeline comm
    pp_nccl, dp_nccl = None, None

    # CUDA Streams for Asynchronous Execution (Matched to DT-FM)
    comp_stream = torch.cuda.default_stream(device=device)
    recv_stream = torch.cuda.Stream(device=device, priority=-1)
    send_stream = torch.cuda.Stream(device=device, priority=-1)
    dp_stream = torch.cuda.Stream(device=device, priority=-1) if dp_size > 1 else None

    if plan is not None and plan.partition_points:
        boundaries = [0] + list(plan.partition_points) + [cfg.num_layers]
        start_layer, end_layer = boundaries[pp_rank], boundaries[pp_rank + 1]
    else:
        layers_per_stage = cfg.num_layers // pp_size
        start_layer, end_layer = pp_rank * layers_per_stage, (pp_rank + 1) * layers_per_stage

    # AsteroidStage inherently acts as First+Last if pp_rank==0 and pp_rank==pp_size-1
    model = AsteroidStage(cfg, start_layer, end_layer, 
                          is_first=(pp_rank == 0), is_last=(pp_rank == pp_size - 1)).to(device)

    if dp_size > 1:
        for param in model.parameters():
            p_cpu = param.data.cpu()
            torch.distributed.broadcast(p_cpu, src=dp_ranks_in_group[0], group=dp_process_group)
            param.data.copy_(p_cpu.to(device))

    flat_param = flatten_params(model.parameters()) if dp_size > 1 else None

    decay_p = [p for p in model.parameters() if p.dim() >= 2]
    nodecay_p = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_p, "weight_decay": cfg.weight_decay},
        {"params": nodecay_p, "weight_decay": 0.0},
    ], lr=cfg.lr, betas=(0.9, 0.95))

    ft = AsteroidFaultTolerance(state, str(Path(cfg.output_dir) / "checkpoints"))
    ft.start_heartbeat(rank, cfg.heartbeat_interval_s)

    train_embeds, train_labels = train_data
    val_embeds, val_labels = val_data

    is_token_input = (train_embeds.dtype == torch.long)  # LM uses token IDs

    def sample_batch_for(embeds, labels, bs, iter_num, micro_idx):
        seed = cfg.seed * 1000003 + dp_rank * 100003 + iter_num * 997 + micro_idx
        g = torch.Generator()
        g.manual_seed(seed)
        ix = torch.randint(len(embeds), (bs,), generator=g)
        if is_token_input:
            return embeds[ix].long().to(device), labels[ix].long().to(device)
        return embeds[ix].float().to(device), labels[ix].long().to(device)

    num_micro = cfg.num_microbatches
    act_shape = (cfg.micro_batch_size, cfg.max_seq_len, cfg.embedding_dim)

    pp_prev = pp_rank - 1 if pp_rank > 0 else None
    pp_next = pp_rank + 1 if pp_rank < pp_size - 1 else None

    # Map pp_rank to global rank for send/recv
    # pp_ranks_in_group[pp_rank] gives the global rank
    def get_global_rank(pp_r):
        if pp_ranks_in_group is not None:
            return pp_ranks_in_group[pp_r]
        return pp_r

    pp_prev_global = get_global_rank(pp_prev) if pp_prev is not None else None
    pp_next_global = get_global_rank(pp_next) if pp_next is not None else None

    # Exact buffer setup from DT-FM
    # Note: input_bufs need requires_grad=True but will be recreated each microbatch
    input_bufs = [torch.zeros(act_shape, device=device) for _ in range(num_micro)] if not is_first and not is_single else None
    grad_bufs = [torch.zeros(act_shape, device=device) for _ in range(num_micro)] if not is_last and not is_single else None

    # Events for stream synchronization
    fwd_recv_ready_events = [torch.cuda.Event() for _ in range(num_micro)]
    fwd_comp_ready_events = [torch.cuda.Event() for _ in range(num_micro)]
    bwd_recv_ready_events = [torch.cuda.Event() for _ in range(num_micro)]
    bwd_comp_ready_events = [torch.cuda.Event() for _ in range(num_micro)]

    EVENT_LOGGER.set_epoch_start(time.time())
    best_val_loss = float('inf')
    t0 = time.time()

    # TensorBoard writer — every rank logs per-node metrics
    tb_writer = None
    if HAS_TENSORBOARD:
        tb_base_dir = os.environ.get("TENSORBOARD_LOG_DIR", "/tmp/asteroid_tb_logs")
        tb_log_dir = os.path.join(tb_base_dir, f"rank-{rank}")
        os.makedirs(tb_log_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tb_log_dir, flush_secs=30)
        print(f"[RANK {rank}] TensorBoard logging to {tb_log_dir}", flush=True)

    # ── Resolve pipeline schedule ──────────────────────────────────────
    schedule_type = cfg.schedule_type or "gpipe"
    use_1f1b = schedule_type.lower().replace("-", "").replace("_", "") == "1f1b"

    if not is_single:
        schedule_timeline = build_schedule(schedule_type, pp_size, num_micro)
        stage_ops = schedule_timeline[pp_rank]
        print(f"[RANK {rank}] Schedule: {schedule_type} | "
              f"{len(stage_ops)} ops for stage {pp_rank}", flush=True)
    else:
        stage_ops = None

    # ── Per-microbatch pipeline operations (used by GPipe and 1F1B) ──────
    def do_forward_micro(m):
        """Execute forward pass for microbatch m on this stage."""
        if is_first:
            x, _ = sample_batch_for(train_embeds, train_labels,
                                    cfg.micro_batch_size, iter_num, m)
            out = model(x)
            cached_outputs[m] = out
            if pp_next_global is not None:
                torch.distributed.send(out.data.contiguous(),
                                       dst=pp_next_global)
        elif is_last:
            torch.distributed.recv(input_bufs[m], src=pp_prev_global)
            input_bufs[m].requires_grad_(True)
            _, y = sample_batch_for(train_embeds, train_labels,
                                    cfg.micro_batch_size, iter_num, m)
            loss = model(input_bufs[m], y) * loss_scale
            micro_losses.append(loss)
            cached_outputs[m] = loss
        else:  # Middle stage
            torch.distributed.recv(input_bufs[m], src=pp_prev_global)
            input_bufs[m].requires_grad_(True)
            out = model(input_bufs[m])
            cached_outputs[m] = out
            if pp_next_global is not None:
                torch.distributed.send(out.data.contiguous(),
                                       dst=pp_next_global)

    def do_backward_micro(m):
        """Execute backward pass for microbatch m on this stage."""
        if is_last:
            out = cached_outputs[m]
            if out is not None:
                if out.dim() == 0:
                    out.backward()
                else:
                    out.backward(torch.ones_like(out))
            if pp_prev_global is not None:
                torch.distributed.send(input_bufs[m].grad.contiguous(),
                                       dst=pp_prev_global)
        elif is_first:
            if pp_next_global is not None:
                torch.distributed.recv(grad_bufs[m], src=pp_next_global)
                cached_outputs[m].backward(gradient=grad_bufs[m])
        else:  # Middle stage
            torch.distributed.recv(grad_bufs[m], src=pp_next_global)
            cached_outputs[m].backward(gradient=grad_bufs[m])
            if pp_prev_global is not None:
                torch.distributed.send(input_bufs[m].grad.contiguous(),
                                       dst=pp_prev_global)

    sched_label = "1F1B" if use_1f1b else "GPipe"
    print(f"[RANK {rank}] Ready. Starting {sched_label} training loop...",
          flush=True)

    for iter_num in range(cfg.max_iters):
        model.train()
        optimizer.zero_grad()

        lr = get_lr(iter_num, cfg)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        if input_bufs:
            for buf in input_bufs:
                if buf.grad is not None: buf.grad.zero_()

        micro_losses = []
        cached_outputs = [None] * num_micro
        loss_scale = 1.0 / num_micro

        # =====================================================================
        # Pipeline Execution (schedule-driven: GPipe or 1F1B)
        # =====================================================================
        fwd_start = time.time()

        if is_single:
            # ── Single stage: no pipeline communication ────────────
            for m in range(num_micro):
                x, y = sample_batch_for(train_embeds, train_labels,
                                        cfg.micro_batch_size, iter_num, m)
                loss = model(x, y) * loss_scale
                micro_losses.append(loss)
                cached_outputs[m] = loss
            fwd_end = time.time()
            barrier_start = barrier_fwd_end = time.time()
            bwd_start = time.time()
            for m in reversed(range(num_micro)):
                cached_outputs[m].backward()

        elif use_1f1b:
            # ── 1F1B Interleaved Schedule ──────────────────────────
            # Warmup forwards → steady-state (F,B) pairs → cooldown
            # No mid-step barrier — send/recv provides sync.
            for action, m in stage_ops:
                if action == "F":
                    do_forward_micro(m)
                else:
                    do_backward_micro(m)
            fwd_end = bwd_start = time.time()
            barrier_start = barrier_fwd_end = time.time()

        else:
            # ── GPipe Schedule ─────────────────────────────────────
            # ALL Forwards → Barrier → ALL Backwards
            for m in range(num_micro):
                do_forward_micro(m)
            fwd_end = time.time()
            barrier_start = time.time()
            torch.distributed.barrier()
            barrier_fwd_end = time.time()
            bwd_start = time.time()
            for m in reversed(range(num_micro)):
                do_backward_micro(m)

        # Synchronize GPU at end of step before optimizer
        torch.cuda.synchronize()
        bwd_end = time.time()

        # Late loss scaling to avoid async stream blockage
        if is_last or is_single:
            avg_loss = sum([l.item() / loss_scale for l in micro_losses]) / len(micro_losses) if micro_losses else 0

        # DP gradient sync using torch.distributed.all_reduce
        if dp_size > 1 and flat_param is not None and dp_process_group is not None:
            torch.distributed.all_reduce(flat_param.grad.data, group=dp_process_group)
            flat_param.grad.data.div_(dp_size)

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        
        optimizer.step()
        barrier_step_start = time.time()
        torch.distributed.barrier()
        barrier_step_end = time.time()

        if iter_num % cfg.ft_check_interval == 0 and iter_num > 0:
            state.record_backward(iter_num)
            if iter_num > cfg.ft_check_interval:
                prev_iter = iter_num - cfg.ft_check_interval
                if ft.detect_failure(prev_iter, cfg.backward_timeout_ms):
                    logger.warning(f"[RANK {rank}] FT: Passive timeout at iter {prev_iter}")
                    ft.handle_passive_timeout(prev_iter)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        # Per-rank TensorBoard logging (all ranks)
        if tb_writer is not None and iter_num % cfg.log_interval == 0:
            fwd_ms = (fwd_end - fwd_start) * 1000
            bwd_ms = (bwd_end - bwd_start) * 1000
            barrier_fwd_ms = (barrier_fwd_end - barrier_start) * 1000
            barrier_step_ms = (barrier_step_end - barrier_step_start) * 1000
            gpu_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            gpu_mem_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 * 1024)

            tb_writer.add_scalar("timing/forward_ms", fwd_ms, iter_num)
            tb_writer.add_scalar("timing/backward_ms", bwd_ms, iter_num)
            tb_writer.add_scalar("timing/barrier_after_fwd_ms", barrier_fwd_ms, iter_num)
            tb_writer.add_scalar("timing/barrier_after_step_ms", barrier_step_ms, iter_num)
            tb_writer.add_scalar("timing/total_step_ms", dt * 1000, iter_num)
            tb_writer.add_scalar("gpu/peak_memory_allocated_MB", gpu_mem_mb, iter_num)
            tb_writer.add_scalar("gpu/peak_memory_reserved_MB", gpu_mem_reserved_mb, iter_num)
            tb_writer.add_scalar("gpu/memory_utilization_pct",
                                 torch.cuda.memory_allocated(device) / torch.cuda.max_memory_reserved(device) * 100
                                 if torch.cuda.max_memory_reserved(device) > 0 else 0, iter_num)

            if iter_num % (cfg.log_interval * 5) == 0:
                tb_writer.flush()

        if (is_last or is_single) and iter_num % cfg.log_interval == 0:
            tps = cfg.global_batch_size * cfg.max_seq_len / dt if dt > 0 else 0
            print(f"  iter {iter_num:>5d} | loss={avg_loss:.4f} | lr={lr:.2e} | {tps:,.0f} tok/s | dt={dt*1000:.1f}ms", flush=True)

            # Log training metrics to TensorBoard (last stage only)
            if tb_writer is not None:
                tb_writer.add_scalar("train/loss", avg_loss, iter_num)
                tb_writer.add_scalar("train/learning_rate", lr, iter_num)
                tb_writer.add_scalar("train/throughput_tok_s", tps, iter_num)

        # ── Periodic Checkpoint ────────────────────────────────────
        if cfg.checkpoint_interval > 0 and (iter_num + 1) % cfg.checkpoint_interval == 0 and iter_num > 0:
            ckpt_path = ft.save_checkpoint(
                epoch=0, iter_id=iter_num + 1, model=model, optimizer=optimizer,
                start_layer=start_layer, end_layer=end_layer,
                is_first=(pp_rank == 0), is_last=(pp_rank == pp_size - 1),
                config=cfg.to_dict() if hasattr(cfg, 'to_dict') else None)
            print(f"[RANK {rank}] Checkpoint saved: {ckpt_path}", flush=True)

    # ── Final Model Save ───────────────────────────────────────────────────
    final_ckpt_path = ft.save_checkpoint(
        epoch=0, iter_id=cfg.max_iters, model=model, optimizer=optimizer,
        start_layer=start_layer, end_layer=end_layer,
        is_first=(pp_rank == 0), is_last=(pp_rank == pp_size - 1),
        config=cfg.to_dict() if hasattr(cfg, 'to_dict') else None)
    print(f"[RANK {rank}] Final checkpoint saved: {final_ckpt_path}", flush=True)

    ft.stop_heartbeat()
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()
        print(f"[RANK {rank}] TensorBoard logs finalized.", flush=True)
    torch.distributed.barrier()
    print(f"[RANK {rank}] Training complete.", flush=True)
    torch.distributed.destroy_process_group()


def _run_isolated_hardware_profiling(cfg, devices, queue):
    import torch
    import gc
    from asteroid.planner.profiler import AsteroidProfiler
    from asteroid.model.stage import _create_block
    
    profiler = AsteroidProfiler(cfg, devices)
    
    # Profile on first device only (assume homogeneous GPUs) - memory efficient
    d = devices[0]
    device = torch.device(f"cuda:{d.device_id}")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    
    # Profile one layer at a time to avoid OOM
    batch_sizes = [4]
    seq_len, d_model = cfg.max_seq_len, cfg.embedding_dim
    num_iters = 10
    
    import numpy as np
    for layer_idx in range(cfg.num_layers):
        # Create single layer
        layer = _create_block(cfg).to(device)
        
        for bs in batch_sizes:
            x = torch.randn(bs, seq_len, d_model, device=device, requires_grad=True)
            
            # Warmup
            for _ in range(2):
                with torch.no_grad():
                    _ = layer(x)
            torch.cuda.synchronize()
            
            # Forward timing
            fwd_times = []
            for _ in range(num_iters):
                torch.cuda.synchronize()
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                with torch.no_grad():
                    _ = layer(x)
                e.record()
                torch.cuda.synchronize()
                fwd_times.append(s.elapsed_time(e))
            
            # Backward timing
            bwd_times = []
            for _ in range(num_iters):
                torch.cuda.synchronize()
                layer.zero_grad()
                if x.grad is not None:
                    x.grad.zero_()
                out = layer(x)
                grad_out = torch.ones_like(out)
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                out.backward(gradient=grad_out)
                e.record()
                torch.cuda.synchronize()
                bwd_times.append(s.elapsed_time(e))
            
            fwd_ms = float(np.median(fwd_times[2:]))
            bwd_ms = float(np.median(bwd_times[2:]))
            profiler.exec_times[d.device_id][layer_idx][bs] = (fwd_ms, bwd_ms)
        
        # Store sizes for this layer
        if layer_idx == 0:
            profiler.activation_sizes = [seq_len * d_model * 4] * cfg.num_layers
            weight_size = sum(p.numel() * p.element_size() for p in layer.parameters())
            profiler.weight_sizes = [weight_size] * cfg.num_layers
        
        # Cleanup immediately
        del layer, x
        torch.cuda.empty_cache()
        gc.collect()
    
    # Copy profiled times to other devices (assume homogeneous)
    for other_d in devices[1:]:
        profiler.exec_times[other_d.device_id] = dict(profiler.exec_times[d.device_id])
        
    def sanitize_dict(d):
        if isinstance(d, dict):
            return {k: sanitize_dict(v) for k, v in d.items()}
        return d
        
    queue.put({
        'exec_times': sanitize_dict(profiler.exec_times),
        'weight_sizes': profiler.weight_sizes,
        'activation_sizes': profiler.activation_sizes
    })


def load_hpp_plan(path: str) -> HPPPlanConfig:
    """Load HPP plan from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return HPPPlanConfig.from_json(data)


def _worker_wrapper(rank: int, cfg: AsteroidConfig, train_data, val_data, plan, use_env_init: bool):
    """Wrapper for mp.spawn that passes all arguments."""
    worker(rank, cfg, train_data, val_data, plan, use_env_init)


def main_multinode():
    """
    Entry point for multi-node execution (K8s mode).
    Reads configuration from environment variables.
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    print(f"[RANK {rank}] Starting Asteroid worker (multi-node mode)")
    print(f"[RANK {rank}] MASTER_ADDR={os.environ['MASTER_ADDR']}")
    print(f"[RANK {rank}] MASTER_PORT={os.environ['MASTER_PORT']}")
    print(f"[RANK {rank}] WORLD_SIZE={world_size}")
    print(f"[RANK {rank}] NCCL_SOCKET_IFNAME={os.environ.get('NCCL_SOCKET_IFNAME', 'not set')}")
    
    # Load HPP plan
    plan_path = os.environ.get("HPP_PLAN_PATH", "./hpp_plan.json")
    plan = None
    
    if os.path.exists(plan_path):
        print(f"[RANK {rank}] Loading HPP plan from {plan_path}")
        plan = load_hpp_plan(plan_path)
        print(f"[RANK {rank}] Plan: {plan.num_stages} stages, partition={plan.partition_points}")
    else:
        print(f"[RANK {rank}] No HPP plan found at {plan_path}, using default configuration")
    
    # Build config
    cfg = AsteroidConfig(
        world_size=world_size,
        num_stages=plan.num_stages if plan else 2,
    )
    
    # Load data
    print(f"[RANK {rank}] Loading dataset...")
    train_data, val_data = prepare_data(cfg)
    print(f"[RANK {rank}] Data loaded: train={train_data[0].shape}")
    
    # Run worker with env:// init
    worker(rank, cfg, train_data, val_data, plan, use_env_init=True)
    
    print(f"[RANK {rank}] Worker finished")


def main_local(args):
    """
    Entry point for local testing with mp.spawn.
    """
    from asteroid.model.stage import _create_block
    
    port = _setup_local_env(args.world_size, args.port)
    
    cfg = AsteroidConfig(
        world_size=args.world_size, 
        num_stages=args.num_stages,
        dist_url=f"tcp://127.0.0.1:{port}"
    )
    
    devices = [DeviceSpec(device_id=i, memory_budget_mb=8192.0) for i in range(cfg.world_size)]
    plan = None
    
    if args.plan:
        print(f"Loading HPP plan from {args.plan}...")
        plan = load_hpp_plan(args.plan)
    elif not args.skip_profiling:
        print(f"Phase 1: Running Hardware Profiling...")
        
        ctx = mp.get_context('spawn')
        q = ctx.Queue()
        p = ctx.Process(target=_run_isolated_hardware_profiling, args=(cfg, devices, q))
        p.start()
        profiler_data = q.get()
        p.join()

        profiler = AsteroidProfiler(cfg, devices)
        profiler.exec_times = profiler_data['exec_times']
        profiler.weight_sizes = profiler_data['weight_sizes']
        profiler.activation_sizes = profiler_data['activation_sizes']

        print("Phase 2: Planning Stage...")
        plan = AsteroidPlanner(profiler, cfg, devices).plan()
    
    print("Phase 3: Dataset Preparation...")
    train_data, val_data = prepare_data(cfg)

    print(f"\nPhase 4: Spawning {cfg.world_size} workers on port {port}...")
    
    mp.spawn(
        _worker_wrapper, 
        args=(cfg, train_data, val_data, plan, False),
        nprocs=cfg.world_size, 
        join=True
    )


def main():
    """Main entry point - dispatches to local or multi-node mode."""
    parser = argparse.ArgumentParser(
        description="Asteroid Distributed Training Worker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--local", action="store_true",
        help="Run in local mode with mp.spawn (for testing)"
    )
    parser.add_argument(
        "--world-size", type=int, default=3,
        help="Number of workers (local mode only, default: 3)"
    )
    parser.add_argument(
        "--num-stages", type=int, default=2,
        help="Number of pipeline stages (local mode only, default: 2)"
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Master port (local mode only, default: random)"
    )
    parser.add_argument(
        "--plan", type=str, default=None,
        help="Path to hpp_plan.json (optional)"
    )
    parser.add_argument(
        "--skip-profiling", action="store_true",
        help="Skip hardware profiling (local mode only)"
    )
    
    args = parser.parse_args()
    
    if args.local or not _MULTINODE_MODE:
        # Local testing mode with mp.spawn
        main_local(args)
    else:
        # K8s multi-node mode
        main_multinode()


if __name__ == "__main__":
    main()