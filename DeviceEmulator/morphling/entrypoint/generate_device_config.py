import json
from dataclasses import dataclass, field

import numpy as np
from transformers import HfArgumentParser, set_seed

from morphling.common import DeviceConfig, EnhancedJSONEncoder
from morphling.common.config import SYMBOLS, bytes2human, human2bytes


@dataclass
class ModelConfigArguments:
    model_name: str = field(
        default="facebook/opt-125m", metadata={"help": "Huggingface model name"}
    )
    batch_size: int = field(default=128, metadata={"help": "Batch size"})
    seq_length: int = field(default=1024, metadata={"help": "Sequence length"})
    backend: str = field(
        default="rabbitmq",
        metadata={"help": "The backend to use for the device"},
    )
    block_size: int = field(default=128, metadata={"help": "Block size"})
    cfg: str = field(
        default="",
        metadata={"help": "Proxy config file path"},
    )
    # redis_host: str = field(
    #     default="127.0.0.1:6379",
    #     metadata={"help": "Redis server host:port for proxy backend"},
    # )
    proxy_host: str = field(
        default="",
        metadata={
            "help": "Proxy server host:port (e.g., 155.98.37.203:39000), overrides config file"
        },
    )
    log_level: str = field(default="info", metadata={"help": "Log level"})


@dataclass
class DeviceConfigArguments:
    num_devices: int = field(
        default=256, metadata={"help": "Number of devices"}
    )
    flops_lb: str = field(
        default="5T", metadata={"help": "Lower bound of device FLOPS"}
    )
    flops_ub: str = field(
        default="7T", metadata={"help": "Upper bound of device FLOPS"}
    )
    ul_bw_lb: str = field(
        default="5M",
        metadata={"help": "Lower bound of device uplink bandwidth (B/s)"},
    )
    ul_bw_ub: str = field(
        default="10M",
        metadata={"help": "Upper bound of device uplink bandwidth (B/s)"},
    )
    dl_bw_lb: str = field(
        default="10M",
        metadata={"help": "Lower bound of device downlink bandwidth (B/s)"},
    )
    dl_bw_ub: str = field(
        default="100M",
        metadata={"help": "Upper bound of device downlink bandwidth (B/s)"},
    )
    ul_latency_lb: float = field(
        default=0, metadata={"help": "Lower bound of device uplink latency"}
    )
    ul_latency_ub: float = field(
        default=0.5, metadata={"help": "Upper bound of device uplink latency"}
    )
    dl_latency_lb: float = field(
        default=0, metadata={"help": "Lower bound of device downlink latency"}
    )
    dl_latency_ub: float = field(
        default=0.5, metadata={"help": "Upper bound of device downlink latency"}
    )

    mem_lb: str = field(
        default="1G", metadata={"help": "Lower bound of device memory"}
    )
    mem_ub: str = field(
        default="3G", metadata={"help": "Upper bound of device memory"}
    )

    seed: int = field(default=42, metadata={"help": "Random seed"})

    straggler_ratio: float = field(
        default=-1, metadata={"help": "Straggler ratio"}
    )
    straggler_num: int = field(
        default=-1, metadata={"help": "Number of stragglers"}
    )
    straggler_ul_scale: float = field(
        default=0.1, metadata={"help": "Straggler uplink scale"}
    )
    straggler_dl_scale: float = field(
        default=0.1, metadata={"help": "Straggler downlink scale"}
    )
    straggler_flops_scale: float = field(
        default=0.1, metadata={"help": "Strafficking FLOPS scale"}
    )

    output: str = field(
        default="device_config.json", metadata={"help": "Output file path"}
    )

    def __post_init__(self):
        # convert the human-readable units to bytes
        self.flops_lb = human2bytes(self.flops_lb)
        self.flops_ub = human2bytes(self.flops_ub)
        self.ul_bw_lb = human2bytes(self.ul_bw_lb)
        self.ul_bw_ub = human2bytes(self.ul_bw_ub)
        self.dl_bw_lb = human2bytes(self.dl_bw_lb)
        self.dl_bw_ub = human2bytes(self.dl_bw_ub)
        self.mem_lb = human2bytes(self.mem_lb)
        self.mem_ub = human2bytes(self.mem_ub)

        # all upper bounds must be greater than lower bounds
        assert self.flops_ub > self.flops_lb, (
            "Upper bound of device FLOPS must be greater than lower bound"
        )
        assert self.ul_bw_ub > self.ul_bw_lb, (
            "Upper bound of device uplink bandwidth must be greater than lower bound"
        )
        assert self.dl_bw_ub > self.dl_bw_lb, (
            "Upper bound of device downlink bandwidth must be greater than lower bound"
        )
        assert self.ul_latency_ub > self.ul_latency_lb, (
            "Upper bound of device uplink latency must be greater than lower bound"
        )
        assert self.dl_latency_ub > self.dl_latency_lb, (
            "Upper bound of device downlink latency must be greater than lower bound"
        )

        set_seed(42)
        self.device_flops = np.random.randint(
            self.flops_lb, self.flops_ub, self.num_devices
        )
        self.ul_bw = np.random.randint(
            self.ul_bw_lb, self.ul_bw_ub, self.num_devices
        )
        self.dl_bw = np.random.randint(
            self.dl_bw_lb, self.dl_bw_ub, self.num_devices
        )
        self.device_mem = np.random.randint(
            self.mem_lb, self.mem_ub, self.num_devices
        )
        self.ul_lat = np.zeros(self.num_devices)
        self.dl_lat = np.zeros(self.num_devices)

        # self.byte_size = 2
        # self.num_tokens = self.batch_size * self.seq_length

        # self.device_flops = self.device_flops / 1000  # TFLOPS / ms
        # self.ul_bw = self.ul_bw / 1000  # MB / ms
        # self.dl_bw = self.dl_bw / 1000  # MB / ms
        # self.ul_lat = self.ul_lat * 1000  # ms
        # self.dl_lat = self.dl_lat * 1000  # ms

        # straggler_ratio and straggler_num can only be one of them positive
        # assert self.straggler_ratio >= 0 or self.straggler_num >= 0, "Only one of straggler_ratio and straggler_num can be positive"
        self.num_stragglers = 0
        if self.straggler_ratio > 0:
            self.num_stragglers = int(
                np.ceil(self.num_devices * self.straggler_ratio)
            )
        if self.straggler_num > 0:
            self.num_stragglers = self.straggler_num
        print(
            f"Number of stragglers: {self.num_stragglers}, out of {self.num_devices} devices"
        )
        if self.num_stragglers > 0:
            self.straggler_idx = np.random.choice(
                self.num_devices, self.num_stragglers, replace=False
            )
            self.ul_bw[self.straggler_idx] = (
                self.ul_bw[self.straggler_idx] * self.straggler_ul_scale
            )
            self.dl_bw[self.straggler_idx] = (
                self.dl_bw[self.straggler_idx] * self.straggler_dl_scale
            )
            self.device_flops[self.straggler_idx] = (
                self.device_flops[self.straggler_idx]
                * self.straggler_flops_scale
            )


if __name__ == "__main__":
    parser = HfArgumentParser(DeviceConfigArguments)
    args = parser.parse_args_into_dataclasses()[0]
    print(args)

    # save the device config
    meta_list = []
    for id in range(args.num_devices):
        meta = DeviceConfig(
            rank=id,
            flops=int(args.device_flops[id]),
            memory=int(args.device_mem[id]),
            ul_bw=float(args.ul_bw[id]),
            dl_bw=float(args.dl_bw[id]),
            ul_lat=float(args.ul_lat[id]),
            dl_lat=float(args.dl_lat[id]),
        )
        meta_list.append(meta)

    with open(args.output, "w") as f:
        json.dump(meta_list, f, indent=4, cls=EnhancedJSONEncoder)
