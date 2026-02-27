import numpy as np
import torch

try:
    import cupy
    import cupy.cuda.nccl
    CUPY_NCCL_AVAILABLE = True
except (ImportError, AttributeError):
    cupy = None
    CUPY_NCCL_AVAILABLE = False

from ..core.config import AsteroidConfig

def _type_torch_to_nccl(torch_dtype):
    """Map torch dtype to CuPy NCCL dtype — from DT-FM."""
    import cupy.cuda.nccl as nccl
    return {
        torch.float32: nccl.NCCL_FLOAT32,
        torch.float:   nccl.NCCL_FLOAT32,
        torch.float16: nccl.NCCL_FLOAT16,
        torch.float64: nccl.NCCL_FLOAT64,
        torch.int32:   nccl.NCCL_INT32,
        torch.int:     nccl.NCCL_INT,
        torch.uint8:   nccl.NCCL_UINT8,
    }[torch_dtype]


def _nccl_send(tensor: torch.Tensor, dst_rank: int,
               nccl_comm, stream: torch.cuda.Stream):
    """Point-to-point send using CuPy NCCL — from DT-FM."""
    assert tensor.is_contiguous() and tensor.is_cuda
    cupy_stream = cupy.cuda.ExternalStream(stream.cuda_stream)
    nccl_comm.send(tensor.data_ptr(), tensor.numel(),
                   _type_torch_to_nccl(tensor.dtype), dst_rank,
                   cupy_stream.ptr)


def _nccl_recv(tensor: torch.Tensor, src_rank: int,
               nccl_comm, stream: torch.cuda.Stream):
    """Point-to-point recv using CuPy NCCL — from DT-FM."""
    assert tensor.is_contiguous() and tensor.is_cuda
    cupy_stream = cupy.cuda.ExternalStream(stream.cuda_stream)
    nccl_comm.recv(tensor.data_ptr(), tensor.numel(),
                   _type_torch_to_nccl(tensor.dtype), src_rank,
                   cupy_stream.ptr)


def _nccl_allreduce(tensor: torch.Tensor, nccl_comm,
                    stream: torch.cuda.Stream):
    """In-place AllReduce sum using CuPy NCCL — from DT-FM."""
    assert tensor.is_contiguous() and tensor.is_cuda
    cupy_stream = cupy.cuda.ExternalStream(stream.cuda_stream)
    nccl_comm.allReduce(tensor.data_ptr(), tensor.data_ptr(),
                        tensor.numel(), _type_torch_to_nccl(tensor.dtype),
                        cupy.cuda.nccl.NCCL_SUM,
                        cupy_stream.ptr)


def setup_nccl_communicators(rank: int, cfg: AsteroidConfig, pp_rank: int, dp_rank: int, 
                             cuda_id: int, dist_store,
                             pp_size_override=None, dp_size_override=None):
    """Create PP and DP NCCL communicators.

    Args:
        pp_size_override: If provided, use this as the PP communicator size
                          instead of cfg.num_stages.  Needed when the planner
                          produces unequal device groups.
        dp_size_override:  If provided, use this as the DP communicator size
                          instead of world_size // num_stages.  Allows per-stage
                          DP group sizing from the planner.
    """
    cupy.cuda.Device(cuda_id).use()

    # PP communicator
    pp_size = pp_size_override if pp_size_override is not None else cfg.num_stages
    pp_group_id = dp_rank
    pp_comm_name = f"asteroid_pp_{pp_group_id}"
    if pp_rank == 0:
        uid = cupy.cuda.nccl.get_unique_id()
        uid_bytes = uid if isinstance(uid, bytes) else np.array(uid).tobytes()
        dist_store.set(f'group-{pp_comm_name}-uid', uid_bytes)
    torch.distributed.barrier()
    if pp_rank != 0:
        uid_bytes = dist_store.get(f'group-{pp_comm_name}-uid')
    pp_nccl_id = uid_bytes if isinstance(uid_bytes, bytes) else bytes(uid_bytes)

    pp_nccl = cupy.cuda.nccl.NcclCommunicator(pp_size, pp_nccl_id, pp_rank)

    # DP communicator (within a stage's device group)
    dp_nccl = None
    dp_size = dp_size_override if dp_size_override is not None \
        else cfg.world_size // cfg.num_stages
    if dp_size > 1:
        dp_comm_name = f"asteroid_dp_{pp_rank}"
        if dp_rank == 0:
            uid_dp = cupy.cuda.nccl.get_unique_id()
            uid_bytes_dp = uid_dp if isinstance(uid_dp, bytes) \
                else np.array(uid_dp).tobytes()
            dist_store.set(f'group-{dp_comm_name}-uid', uid_bytes_dp)
        torch.distributed.barrier()
        if dp_rank != 0:
            uid_bytes_dp = dist_store.get(f'group-{dp_comm_name}-uid')
        dp_nccl_id = uid_bytes_dp if isinstance(uid_bytes_dp, bytes) \
            else bytes(uid_bytes_dp)
        dp_nccl = cupy.cuda.nccl.NcclCommunicator(dp_size, dp_nccl_id, dp_rank)

    return pp_nccl, dp_nccl