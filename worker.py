import os

# =============================================================================
# 1. HARDWARE ISOLATION (MUST BE AT THE VERY TOP)
# Hides all GPUs except 5, 6, and 7. PyTorch will see them as cuda:0, cuda:1, cuda:2.
# =============================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"

# IMPORTANT: Do NOT set CUDA_LAUNCH_BLOCKING=1 with NCCL - it causes deadlocks
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["NCCL_SOCKET_IFNAME"] = "lo"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "0"
os.environ["NCCL_DEBUG"] = "WARN"
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")

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
# Note: Using torch.distributed for pipeline communication instead of raw NCCL
from asteroid.ft.fault_tolerance import AsteroidFaultTolerance
from asteroid.utils.data_utils import prepare_sst2
from asteroid.planner.profiler import AsteroidProfiler
from asteroid.planner.dp_planner import AsteroidPlanner

def worker(rank: int, cfg: AsteroidConfig,
           train_data: Tuple[torch.Tensor, torch.Tensor],
           val_data: Tuple[torch.Tensor, torch.Tensor],
           plan: Optional[HPPPlanConfig] = None):
    
    cuda_id = rank % torch.cuda.device_count()
    device = torch.device(f'cuda:{cuda_id}')
    torch.cuda.set_device(device)
    torch.manual_seed(cfg.seed + rank)

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
        torch.distributed.init_process_group(
            backend='nccl', init_method=cfg.dist_url,
            world_size=cfg.world_size, rank=rank,
            timeout=timedelta(seconds=120))

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

    def sample_batch_for(embeds, labels, bs, iter_num, micro_idx):
        seed = cfg.seed * 1000003 + dp_rank * 100003 + iter_num * 997 + micro_idx
        g = torch.Generator()
        g.manual_seed(seed)
        ix = torch.randint(len(embeds), (bs,), generator=g)
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

    print(f"[RANK {rank}] Ready. Starting GPIPE Loop...", flush=True)

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
        # EXACT GPIPE LOOP (Adapted strictly from dtfm_gpt2_train copy.py)
        # ALL Forwards -> Barrier -> ALL Backwards -> Sync
        # =====================================================================

        # ── GPipe FORWARD ──────────────────────────────────────────
        # Use torch.distributed.send/recv for reliable pipeline communication
        for m in range(num_micro):
            if is_single:
                x, y = sample_batch_for(train_embeds, train_labels, cfg.micro_batch_size, iter_num, m)
                loss = model(x, y) * loss_scale
                micro_losses.append(loss)
                cached_outputs[m] = loss

            elif is_first:
                x, y = sample_batch_for(train_embeds, train_labels, cfg.micro_batch_size, iter_num, m)
                out = model(x)
                cached_outputs[m] = out
                if pp_next_global is not None:
                    torch.distributed.send(out.data.contiguous(), dst=pp_next_global)

            elif is_last:
                torch.distributed.recv(input_bufs[m], src=pp_prev_global)
                input_bufs[m].requires_grad_(True)  # Enable grad tracking after recv
                _, y = sample_batch_for(train_embeds, train_labels, cfg.micro_batch_size, iter_num, m)
                loss = model(input_bufs[m], y) * loss_scale
                micro_losses.append(loss)
                cached_outputs[m] = loss

            else:  # Middle stage
                torch.distributed.recv(input_bufs[m], src=pp_prev_global)
                input_bufs[m].requires_grad_(True)  # Enable grad tracking after recv
                out = model(input_bufs[m])
                cached_outputs[m] = out
                if pp_next_global is not None:
                    torch.distributed.send(out.data.contiguous(), dst=pp_next_global)

        # Barrier ensures all forwards are queued safely
        torch.distributed.barrier()

        # ── GPipe BACKWARD ─────────────────────────────────────────
        # Use torch.distributed.send/recv for reliable pipeline communication
        for m in reversed(range(num_micro)):
            if is_single:
                cached_outputs[m].backward()

            elif is_last:
                out = cached_outputs[m]
                if out is not None:
                    if out.dim() == 0:
                        out.backward()
                    else:
                        out.backward(torch.ones_like(out))
                if pp_prev_global is not None:
                    torch.distributed.send(input_bufs[m].grad.contiguous(), dst=pp_prev_global)

            elif is_first:
                if pp_next_global is not None:
                    torch.distributed.recv(grad_bufs[m], src=pp_next_global)
                    cached_outputs[m].backward(gradient=grad_bufs[m])

            else:  # Middle stage
                torch.distributed.recv(grad_bufs[m], src=pp_next_global)
                cached_outputs[m].backward(gradient=grad_bufs[m])
                if pp_prev_global is not None:
                    torch.distributed.send(input_bufs[m].grad.contiguous(), dst=pp_prev_global)

        # Synchronize GPU ONCE at the end of the iteration before optimizer step
        torch.cuda.synchronize()

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
        torch.distributed.barrier()

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
        if (is_last or is_single) and iter_num % cfg.log_interval == 0:
            tps = cfg.global_batch_size * cfg.max_seq_len / dt if dt > 0 else 0
            print(f"  iter {iter_num:>5d} | loss={avg_loss:.4f} | lr={lr:.2e} | {tps:,.0f} tok/s | dt={dt*1000:.1f}ms", flush=True)

    ft.stop_heartbeat()
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

if __name__ == "__main__":
    from asteroid.model.stage import _create_block
    import random
    
    port = random.randint(20000, 30000)
    cfg = AsteroidConfig(
        world_size=3, 
        num_stages=3,
        dist_url=f"tcp://127.0.0.1:{port}"
    )

    devices = [DeviceSpec(device_id=i, memory_budget_mb=8192.0) for i in range(cfg.world_size)]

    print(f"Phase 1: Running Actual Hardware Profiling (Mapping to Physical 4,5,6)...")
    
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
    train_data, val_data = prepare_sst2(cfg)

    print(f"\nPhase 4: Spawning {cfg.world_size} workers on port {port}...")
    
    mp.spawn(worker, args=(cfg, train_data, val_data, plan), nprocs=cfg.world_size, join=True)