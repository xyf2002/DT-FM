import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class DeviceSpec:
    """Hardware specification for a single edge device."""
    device_id: int = 0
    device_type: str = "jetson_nano"  # nano, tx2, nx, gpu
    memory_budget_mb: float = 4096.0
    cuda_id: int = 0
    compute_capacity: float = 1.0     # relative throughput factor


@dataclass
class NodeInfo:
    """Physical node information for Kubernetes deployment."""
    hostname: str = "localhost"
    ip: str = "127.0.0.1"
    nic: str = "eth0"
    gpu_id: int = 0
    memory_mb: int = 4096
    architecture: str = "x86_64"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hostname": self.hostname,
            "ip": self.ip,
            "nic": self.nic,
            "gpu_id": self.gpu_id,
            "memory_mb": self.memory_mb,
            "architecture": self.architecture,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeInfo":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

@dataclass
class AsteroidConfig:
    """Unified configuration for the Asteroid system."""
    # Model
    model_name: str = "gpt2"
    model_type: str = "gpt2"          # key into MODEL_REGISTRY
    task_type: str = "classification"  # classification | lm
    num_layers: int = 12
    embedding_dim: int = 768
    num_heads: int = 12
    d_ff: int = 3072
    max_seq_len: int = 128
    vocab_size: int = 50257
    num_classes: int = 2
    dropout: float = 0.1
    use_flash_attention: bool = True
    # Training
    global_batch_size: int = 256
    micro_batch_size: int = 4
    num_microbatches: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_iters: int = 500
    warmup_iters: int = 50
    min_lr: float = 1e-5
    grad_clip: float = 1.0
    eval_interval: int = 100
    log_interval: int = 10
    seed: int = 42
    # Parallelism (HPP)
    world_size: int = 3
    num_stages: int = 2       # P in the paper
    # Communication
    dist_url: str = "tcp://127.0.0.1:29600"
    d2d_bandwidth_mbps: float = 100.0  # default edge bandwidth
    # Fault Tolerance
    heartbeat_interval_s: float = 5.0
    heartbeat_timeout_s: float = 15.0
    backward_timeout_ms: float = 30000.0  # passive FT backward timeout (ms)
    replication_mode: str = "topology"  # topology | local | global | none
    replication_interval: int = 50       # replicate weights every N iters
    ft_check_interval: int = 10          # check for failures every N iters
    # I/O
    output_dir: str = "./asteroid_output"
    dataset: str = "sst2"
    gpu_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    def __post_init__(self):
        self.num_microbatches = self.global_batch_size // self.micro_batch_size
        os.makedirs(self.output_dir, exist_ok=True)

@dataclass
class HPPPlanConfig:
    """Output of the Asteroid Planner — the HPP execution plan."""
    num_stages: int = 2
    # partition_points[i] = first layer index of stage i+1 (stage 0 starts at 0)
    partition_points: List[int] = field(default_factory=list)
    # device_groups[stage_idx] = list of device_ids assigned to that stage
    device_groups: Dict[int, List[int]] = field(default_factory=dict)
    # micro_batch_alloc[stage_idx][device_id] = num samples for that device
    micro_batch_alloc: Dict[int, Dict[int, int]] = field(default_factory=dict)
    # dominant_step index
    dominant_step: int = 0
    # estimated HPP-Round latency (ms)
    estimated_latency_ms: float = float('inf')
    # node_mapping: device_id (rank) -> physical node information for K8s deployment
    node_mapping: Dict[int, NodeInfo] = field(default_factory=dict)
    
    def to_json(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict for hpp_plan.json."""
        return {
            "num_stages": self.num_stages,
            "partition_points": self.partition_points,
            "device_groups": {str(k): v for k, v in self.device_groups.items()},
            "micro_batch_alloc": {
                str(s): {str(d): samples for d, samples in alloc.items()}
                for s, alloc in self.micro_batch_alloc.items()
            },
            "dominant_step": self.dominant_step,
            "estimated_latency_ms": self.estimated_latency_ms,
            "node_mapping": {
                str(k): v.to_dict() for k, v in self.node_mapping.items()
            },
            "world_size": sum(len(devs) for devs in self.device_groups.values()),
        }
    
    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "HPPPlanConfig":
        """Deserialize from JSON dict (e.g., loaded hpp_plan.json)."""
        node_mapping = {}
        for k, v in data.get("node_mapping", {}).items():
            node_mapping[int(k)] = NodeInfo.from_dict(v)
        
        return cls(
            num_stages=data.get("num_stages", 2),
            partition_points=data.get("partition_points", []),
            device_groups={int(k): v for k, v in data.get("device_groups", {}).items()},
            micro_batch_alloc={
                int(s): {int(d): samples for d, samples in alloc.items()}
                for s, alloc in data.get("micro_batch_alloc", {}).items()
            },
            dominant_step=data.get("dominant_step", 0),
            estimated_latency_ms=data.get("estimated_latency_ms", float('inf')),
            node_mapping=node_mapping,
        )