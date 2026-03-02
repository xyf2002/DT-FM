from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DeviceConfig:
    """Per-device runtime and hardware capability settings."""

    use_cuda: bool = True
    cuda_id: int = 0
    cuda_num: int = 1
    debug_mem: bool = True
    device_id: int = 0
    device_type: str = "gpu"
    memory_budget_mb: float = 4096.0
    compute_capacity: float = 1.0
    mps_enabled: bool = False
    mps_active_thread_percentage: int = 100
    mps_pipe_directory: str = ""
    mps_log_directory: str = ""
    mps_pinned_device_mem_limit: str = ""

    def validate(self) -> None:
        """Validate MPS configuration fields."""
        if not (1 <= self.mps_active_thread_percentage <= 100):
            raise ValueError(
                "mps_active_thread_percentage must be 1-100,"
                f" got {self.mps_active_thread_percentage}"
            )


@dataclass
class DistributedConfig:
    """Distributed-process and communication-group configuration."""

    dist_backend: str = "nccl"
    dist_url: str = "tcp://127.0.0.1:29500"
    world_size: int = 1
    pipeline_group_size: int = 1
    data_group_size: int = 1
    rank: int = 0
    timeout_s: float = 120.0


@dataclass
class ModelConfig:
    """Model architecture and task-head configuration."""

    model_name: str = "gpt2"
    model_type: str = "gpt2"
    task_type: str = "classification"
    task: str = "SeqClassification"
    seq_length: int = 2048
    max_seq_len: int = 2048
    embedding_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int = 3072
    vocab_size: int = 50257
    num_classes: int = 2
    dropout: float = 0.1
    use_flash_attention: bool = True


@dataclass
class TrainingConfig:
    """Optimization, batch sizing, and training-loop hyperparameters."""

    batch_size: int = 16
    global_batch_size: int = 16
    micro_batch_size: int = 4
    num_microbatches: int = 4
    lr: float = 3e-4
    min_lr: float = 1e-5
    warmup_iters: int = 50
    max_iters: int = 500
    num_iters: int = 500
    num_epochs: int = 1
    gradient_accumulate_step: int = 1
    seed: int = 42
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_interval: int = 100
    log_interval: int = 10
    betas: tuple[float, float] = (0.9, 0.95)

    def __post_init__(self) -> None:
        if self.global_batch_size < self.micro_batch_size:
            self.global_batch_size = self.micro_batch_size
        if self.batch_size <= 0:
            self.batch_size = self.global_batch_size
        if self.micro_batch_size > 0:
            self.num_microbatches = max(
                1,
                self.global_batch_size // self.micro_batch_size,
            )


@dataclass
class ParallelConfig:
    """Parallelism and schedule configuration for PP/DP execution."""

    pp_mode: str = "gpipe"
    dp_mode: str = "allreduce"
    gradient_accumulate_step: int = 1
    world_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    num_stages: int = 1
    schedule_type: str = "gpipe"

    def __post_init__(self) -> None:
        if self.num_stages <= 0:
            self.num_stages = max(self.pp_size, 1)


@dataclass
class FaultToleranceConfig:
    """Checkpoint, heartbeat, and recovery-policy configuration."""

    checkpoint_dir: str = "./checkpoints"
    checkpoint_interval: int = 100
    heartbeat_interval_s: float = 5.0
    heartbeat_timeout_s: float = 15.0
    backward_timeout_ms: float = 30000.0
    replication_mode: str = "none"
    replication_interval: int = 50
    ft_check_interval: int = 10


@dataclass
class ParallelismPlan:
    """Planner output describing partitioning and micro-batch allocation."""

    partition_points: list[int] = field(default_factory=list)
    device_groups: dict[int, list[int]] = field(default_factory=dict)
    micro_batch_alloc: dict[int, dict[int, int]] = field(default_factory=dict)
    schedule_type: str = "gpipe"
    estimated_latency_ms: float = float("inf")


@dataclass
class DeviceTopology:
    """Device-level topology and communication characteristics."""

    device_specs: list[DeviceConfig] = field(default_factory=list)
    bandwidths: dict[tuple[int, int], float] = field(default_factory=dict)
    latencies: dict[tuple[int, int], float] = field(default_factory=dict)


@dataclass
class BaseConfig:
    """Unified baseline config combining model, training, and runtime settings."""

    device: DeviceConfig = field(default_factory=DeviceConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    fault_tolerance: FaultToleranceConfig = field(
        default_factory=FaultToleranceConfig,
    )
    topology: DeviceTopology | None = None
    plan: ParallelismPlan | None = None
    metadata: dict[str, object] = field(default_factory=dict)


__all__ = [
    "BaseConfig",
    "DeviceConfig",
    "DistributedConfig",
    "ModelConfig",
    "TrainingConfig",
    "ParallelConfig",
    "FaultToleranceConfig",
    "ParallelismPlan",
    "DeviceTopology",
]
