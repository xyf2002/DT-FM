import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.multiprocessing as mp

# Import from our new modular structure
from asteroid.core.config import AsteroidConfig, HPPPlanConfig, DeviceSpec
from asteroid.core.state import AsteroidStateManager
from asteroid.utils.logger import EVENT_LOGGER, logger
from asteroid.model.stage import AsteroidStage
from asteroid.optim.optim_utils import flatten_params, get_lr
from asteroid.comm.nccl_utils import setup_nccl_communicators, _nccl_send, _nccl_recv, _nccl_allreduce
from asteroid.ft.fault_tolerance import AsteroidFaultTolerance
from asteroid.utils.data_utils import prepare_sst2
from asteroid.planner.profiler import AsteroidProfiler
from asteroid.planner.dp_planner import AsteroidPlanner

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ["NCCL_SOCKET_IFNAME"] = "lo"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "0"
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")

def worker(rank: int, cfg: AsteroidConfig,
           train_data: Tuple[torch.Tensor, torch.Tensor],
           val_data: Tuple[torch.Tensor, torch.Tensor],
           plan: Optional[HPPPlanConfig] = None):
    """Baseline Asteroid Worker — modularized from astroid.py"""
    
    cuda_id = rank % torch.cuda.device_count()
    device = torch.device(f'cuda:{cuda_id}')
    torch.cuda.set_device(device)
    torch.manual_seed(cfg.seed + rank)

    # 1. Parse Topology from the HPP Plan
    if plan is not None and plan.device_groups:
        _rank_to_stage = {}
        _rank_to_dp_pos = {}
        _stage_sizes = {}
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
        _stage_sizes = {s: dp_size for s in range(pp_size)}

    is_first = (pp_rank == 0)
    is_last = (pp_rank == pp_size - 1)

    state = AsteroidStateManager()
    state.global_rank = rank
    state.stage_idx = pp_rank
    state.device = device
    state._dp_rank = dp_rank
    state._dp_size = dp_size
    state.plan = plan

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend='gloo', init_method=cfg.dist_url,
            world_size=cfg.world_size, rank=rank,
            timeout=timedelta(seconds=120))

    # 2. Setup Process Groups & Communicators
    pp_process_group, pp_ranks_in_group = None, None
    dp_process_group, dp_ranks_in_group = None, None

    if plan is not None and plan.device_groups:
        # Form PP columns (assumes symmetrical DP degrees for baseline functionality)
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

        # Form DP rows
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
    pp_nccl, dp_nccl = setup_nccl_communicators(
        rank, cfg, pp_rank, dp_rank, cuda_id, dist_store,
        pp_size_override=pp_size, dp_size_override=dp_size)

    comp_stream = torch.cuda.default_stream(device=device)
    recv_stream = torch.cuda.Stream(device=device, priority=-1)
    send_stream = torch.cuda.Stream(device=device, priority=-1)
    dp_stream = torch.cuda.Stream(device=device, priority=-1) if dp_size > 1 else None

    # 3. Model Initialization based on Planner Partitions
    if plan is not None and plan.partition_points:
        boundaries = [0] + list(plan.partition_points) + [cfg.num_layers]
        start_layer = boundaries[pp_rank]
        end_layer = boundaries[pp_rank + 1]
    else:
        layers_per_stage = cfg.num_layers // pp_size
        start_layer = pp_rank * layers_per_stage
        end_layer = start_layer + layers_per_stage if pp_rank < pp_size - 1 else cfg.num_layers

    model = AsteroidStage(cfg, start_layer, end_layer, is_first=is_first, is_last=is_last).to(device)

    # Broadcast initial weights across DP group
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

    # 4. Standard Homogenous Micro-Batching Setup
    def sample_batch_for(embeds, labels, bs, iter_num, micro_idx):
        seed = cfg.seed * 1000003 + dp_rank * 100003 + iter_num * 997 + micro_idx
        g = torch.Generator()
        g.manual_seed(seed)
        ix = torch.randint(len(embeds), (bs,), generator=g)
        return embeds[ix].float().to(device), labels[ix].long().to(device)

    num_micro = cfg.num_microbatches
    act_shape = (cfg.micro_batch_size, cfg.max_seq_len, cfg.embedding_dim)

    # Establish Point-to-Point links strictly along the PP column
    if pp_ranks_in_group:
        my_idx = pp_ranks_in_group.index(rank)
        pp_prev = pp_ranks_in_group[my_idx - 1] if my_idx > 0 else None
        pp_next = pp_ranks_in_group[my_idx + 1] if my_idx < len(pp_ranks_in_group) - 1 else None
    else:
        pp_prev = pp_rank - 1 if pp_rank > 0 else None
        pp_next = pp_rank + 1 if pp_rank < pp_size - 1 else None

    input_bufs = [torch.zeros(act_shape, requires_grad=True, device=device) for _ in range(num_micro)] if not is_first else None
    grad_bufs = [torch.zeros(act_shape, device=device) for _ in range(num_micro)] if not is_last else None

    fwd_recv_ready = [torch.cuda.Event() for _ in range(num_micro)]
    fwd_comp_ready = [torch.cuda.Event() for _ in range(num_micro)]
    bwd_recv_ready = [torch.cuda.Event() for _ in range(num_micro)]
    bwd_comp_ready = [torch.cuda.Event() for _ in range(num_micro)]

    P = pp_size
    warmup_microbatches = min(num_micro, P - pp_rank - 1) if pp_rank < P - 1 else 0
    steady_microbatches = num_micro - warmup_microbatches

    EVENT_LOGGER.set_epoch_start(time.time())
    best_val_loss = float('inf')
    t0 = time.time()

    # 5. The 1F1B Loop
    for iter_num in range(cfg.max_iters):
        model.train()
        optimizer.zero_grad()

        lr = get_lr(iter_num, cfg)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        if input_bufs:
            for buf in input_bufs:
                if buf.grad is not None:
                    buf.grad.zero_()

        micro_losses = []
        cached_outputs = [None] * num_micro
        loss_scale = 1.0 / num_micro

        # Phase 1: Warmup FWD
        for m in range(warmup_microbatches):
            if is_first:
                x, y = sample_batch_for(train_embeds, train_labels, cfg.micro_batch_size, iter_num, m)
                with torch.cuda.stream(comp_stream):
                    out = model(x)
                    cached_outputs[m] = out
                    comp_stream.record_event(fwd_comp_ready[m])
                torch.cuda.synchronize()
                if pp_next is not None:
                    with torch.cuda.stream(send_stream):
                        send_stream.wait_event(fwd_comp_ready[m])
                        _nccl_send(out.detach().contiguous(), pp_next, pp_nccl, send_stream)
                    torch.cuda.synchronize()
            else:
                with torch.cuda.stream(recv_stream):
                    _nccl_recv(input_bufs[m], pp_prev, pp_nccl, recv_stream)
                    recv_stream.record_event(fwd_recv_ready[m])
                torch.cuda.synchronize()
                with torch.cuda.stream(comp_stream):
                    comp_stream.wait_event(fwd_recv_ready[m])
                    inp = input_bufs[m]
                    if not inp.requires_grad:
                        inp = inp.requires_grad_(True)
                        input_bufs[m] = inp
                    if is_last:
                        _, y = sample_batch_for(train_embeds, train_labels, cfg.micro_batch_size, iter_num, m)
                        loss = model(inp, y) * loss_scale
                        micro_losses.append(loss.item() / loss_scale)
                        cached_outputs[m] = loss
                    else:
                        out = model(inp)
                        cached_outputs[m] = out
                    comp_stream.record_event(fwd_comp_ready[m])
                torch.cuda.synchronize()
                if pp_next is not None:
                    with torch.cuda.stream(send_stream):
                        send_stream.wait_event(fwd_comp_ready[m])
                        _nccl_send(cached_outputs[m].detach().contiguous(), pp_next, pp_nccl, send_stream)
                    torch.cuda.synchronize()

        # Phase 2: Steady-state 1F1B
        for m_idx in range(steady_microbatches):
            fwd_m = warmup_microbatches + m_idx
            bwd_m = m_idx

            if is_first:
                x, y = sample_batch_for(train_embeds, train_labels, cfg.micro_batch_size, iter_num, fwd_m)
                with torch.cuda.stream(comp_stream):
                    out = model(x)
                    cached_outputs[fwd_m] = out
                    comp_stream.record_event(fwd_comp_ready[fwd_m])
                torch.cuda.synchronize()
                if pp_next is not None:
                    with torch.cuda.stream(send_stream):
                        send_stream.wait_event(fwd_comp_ready[fwd_m])
                        _nccl_send(out.detach().contiguous(), pp_next, pp_nccl, send_stream)
                    torch.cuda.synchronize()
            else:
                with torch.cuda.stream(recv_stream):
                    _nccl_recv(input_bufs[fwd_m], pp_prev, pp_nccl, recv_stream)
                    recv_stream.record_event(fwd_recv_ready[fwd_m])
                torch.cuda.synchronize()
                with torch.cuda.stream(comp_stream):
                    comp_stream.wait_event(fwd_recv_ready[fwd_m])
                    inp = input_bufs[fwd_m]
                    if not inp.requires_grad:
                        inp = inp.requires_grad_(True)
                        input_bufs[fwd_m] = inp
                    if is_last:
                        _, y = sample_batch_for(train_embeds, train_labels, cfg.micro_batch_size, iter_num, fwd_m)
                        loss = model(inp, y) * loss_scale
                        micro_losses.append(loss.item() / loss_scale)
                        cached_outputs[fwd_m] = loss
                    else:
                        out = model(inp)
                        cached_outputs[fwd_m] = out
                    comp_stream.record_event(fwd_comp_ready[fwd_m])
                torch.cuda.synchronize()
                if pp_next is not None:
                    with torch.cuda.stream(send_stream):
                        send_stream.wait_event(fwd_comp_ready[fwd_m])
                        _nccl_send(cached_outputs[fwd_m].detach().contiguous(), pp_next, pp_nccl, send_stream)
                    torch.cuda.synchronize()

            if is_last:
                with torch.cuda.stream(comp_stream):
                    out = cached_outputs[bwd_m]
                    if out is not None:
                        if out.dim() == 0:
                            out.backward()
                        else:
                            out.backward(torch.ones_like(out))
                        comp_stream.record_event(bwd_comp_ready[bwd_m])
                torch.cuda.synchronize()
                if pp_prev is not None and input_bufs and input_bufs[bwd_m].grad is not None:
                    with torch.cuda.stream(send_stream):
                        send_stream.wait_event(bwd_comp_ready[bwd_m])
                        _nccl_send(input_bufs[bwd_m].grad.contiguous(), pp_prev, pp_nccl, send_stream)
                    torch.cuda.synchronize()
            else:
                with torch.cuda.stream(recv_stream):
                    _nccl_recv(grad_bufs[bwd_m], pp_next, pp_nccl, recv_stream)
                    recv_stream.record_event(bwd_recv_ready[bwd_m])
                torch.cuda.synchronize()
                with torch.cuda.stream(comp_stream):
                    comp_stream.wait_event(bwd_recv_ready[bwd_m])
                    out = cached_outputs[bwd_m]
                    if out is not None:
                        out.backward(gradient=grad_bufs[bwd_m])
                    comp_stream.record_event(bwd_comp_ready[bwd_m])
                torch.cuda.synchronize()
                if pp_prev is not None and input_bufs and input_bufs[bwd_m].grad is not None:
                    with torch.cuda.stream(send_stream):
                        send_stream.wait_event(bwd_comp_ready[bwd_m])
                        _nccl_send(input_bufs[bwd_m].grad.contiguous(), pp_prev, pp_nccl, send_stream)
                    torch.cuda.synchronize()
            cached_outputs[bwd_m] = None

        # Phase 3: Cooldown BWD
        for m in range(steady_microbatches, num_micro):
            if is_last:
                with torch.cuda.stream(comp_stream):
                    out = cached_outputs[m]
                    if out is not None:
                        if out.dim() == 0:
                            out.backward()
                        else:
                            out.backward(torch.ones_like(out))
                        comp_stream.record_event(bwd_comp_ready[m])
                torch.cuda.synchronize()
                if pp_prev is not None and input_bufs and input_bufs[m].grad is not None:
                    with torch.cuda.stream(send_stream):
                        send_stream.wait_event(bwd_comp_ready[m])
                        _nccl_send(input_bufs[m].grad.contiguous(), pp_prev, pp_nccl, send_stream)
                    torch.cuda.synchronize()
            else:
                with torch.cuda.stream(recv_stream):
                    _nccl_recv(grad_bufs[m], pp_next, pp_nccl, recv_stream)
                    recv_stream.record_event(bwd_recv_ready[m])
                torch.cuda.synchronize()
                with torch.cuda.stream(comp_stream):
                    comp_stream.wait_event(bwd_recv_ready[m])
                    out = cached_outputs[m]
                    if out is not None:
                        out.backward(gradient=grad_bufs[m])
                    comp_stream.record_event(bwd_comp_ready[m])
                torch.cuda.synchronize()
                if pp_prev is not None and input_bufs and input_bufs[m].grad is not None:
                    with torch.cuda.stream(send_stream):
                        send_stream.wait_event(bwd_comp_ready[m])
                        _nccl_send(input_bufs[m].grad.contiguous(), pp_prev, pp_nccl, send_stream)
                    torch.cuda.synchronize()
            cached_outputs[m] = None

        # 6. DP Sync & Optimizer Step
        if dp_size > 1 and flat_param is not None and dp_nccl is not None:
            bwd_ready = torch.cuda.Event()
            comp_stream.record_event(bwd_ready)
            with torch.cuda.stream(dp_stream):
                dp_stream.wait_event(bwd_ready)
                _nccl_allreduce(flat_param.grad.data, dp_nccl, dp_stream)
            torch.cuda.synchronize()
            flat_param.grad.data.div_(dp_size)

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        torch.cuda.synchronize()
        torch.distributed.barrier()

        # Fault Tolerance Checks
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
        if is_last and iter_num % cfg.log_interval == 0:
            avg_loss = sum(micro_losses) / len(micro_losses) if micro_losses else 0
            tps = cfg.global_batch_size * cfg.max_seq_len / dt if dt > 0 else 0
            print(f"  iter {iter_num:>5d} | loss={avg_loss:.4f} | lr={lr:.2e} | {tps:,.0f} tok/s | dt={dt*1000:.1f}ms", flush=True)

    ft.stop_heartbeat()
    torch.distributed.barrier()
    print(f"[RANK {rank}] Training complete.", flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    cfg = AsteroidConfig(
        embedding_dim=768, num_heads=12, num_layers=12, d_ff=3072,
        max_seq_len=128, vocab_size=50257, num_classes=2,
        global_batch_size=16, micro_batch_size=4, num_microbatches=4,
        lr=3e-4, max_iters=50, warmup_iters=5, log_interval=5,
        world_size=4, num_stages=2,
        dist_url="tcp://127.0.0.1:29600"
    )

    devices = [DeviceSpec(device_id=i, compute_capacity=1.0 + 0.5 * (i % 2)) for i in range(cfg.world_size)]
    profiler = AsteroidProfiler(cfg, devices)
    profiler.set_synthetic_profiles(cfg.num_layers, cfg.world_size, batch_sizes=[1, 2, 4, 8, 16])
    
    plan = AsteroidPlanner(profiler, cfg, devices).plan()
    train_data, val_data = prepare_sst2(cfg)

    print("\nSpawning workers...")
    mp.set_start_method('fork', force=True)
    mp.spawn(worker, args=(cfg, train_data, val_data, plan), nprocs=cfg.world_size, join=True)
    print("\nAll workers finished.")