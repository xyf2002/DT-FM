import dataclasses
import json
import logging
from dataclasses import dataclass, field
from typing import Union

import psutil
import torch
from transformers import HfArgumentParser

from morphling.common.types_and_defs import *

KB = 1024
MB = 1024 * KB
GB = 1024 * MB

SYMBOLS = {
    "customary": ("B", "K", "M", "G", "T", "P", "E", "Z", "Y"),
    "customary_ext": (
        "byte",
        "kilo",
        "mega",
        "giga",
        "tera",
        "peta",
        "exa",
        "zetta",
        "iotta",
    ),
    "iec": ("Bi", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi", "Yi"),
    "iec_ext": (
        "byte",
        "kibi",
        "mebi",
        "gibi",
        "tebi",
        "pebi",
        "exbi",
        "zebi",
        "yobi",
    ),
}


def bytes2human(n, format="%(value).1f %(symbol)s", symbols="customary"):
    """
    Convert n bytes into a human readable string based on format.
    symbols can be either "customary", "customary_ext", "iec" or "iec_ext",
    see: http://goo.gl/kTQMs

      >>> bytes2human(0)
      '0.0 B'
      >>> bytes2human(0.9)
      '0.0 B'
      >>> bytes2human(1)
      '1.0 B'
      >>> bytes2human(1.9)
      '1.0 B'
      >>> bytes2human(1024)
      '1.0 K'
      >>> bytes2human(1048576)
      '1.0 M'
      >>> bytes2human(1099511627776127398123789121)
      '909.5 Y'

      >>> bytes2human(9856, symbols="customary")
      '9.6 K'
      >>> bytes2human(9856, symbols="customary_ext")
      '9.6 kilo'
      >>> bytes2human(9856, symbols="iec")
      '9.6 Ki'
      >>> bytes2human(9856, symbols="iec_ext")
      '9.6 kibi'

      >>> bytes2human(10000, "%(value).1f %(symbol)s/sec")
      '9.8 K/sec'

      >>> # precision can be adjusted by playing with %f operator
      >>> bytes2human(10000, format="%(value).5f %(symbol)s")
      '9.76562 K'
    """
    n = int(n)
    if n < 0:
        raise ValueError("n < 0")
    symbols = SYMBOLS[symbols]
    prefix = {}
    for i, s in enumerate(symbols[1:]):
        prefix[s] = 1 << (i + 1) * 10
    for symbol in reversed(symbols[1:]):
        if n >= prefix[symbol]:
            value = float(n) / prefix[symbol]
            return format % locals()
    return format % dict(symbol=symbols[0], value=n)


def human2bytes(s):
    """
    Attempts to guess the string format based on default symbols
    set and return the corresponding bytes as an integer.
    When unable to recognize the format ValueError is raised.

      >>> human2bytes('0 B')
      0
      >>> human2bytes('1 K')
      1024
      >>> human2bytes('1 M')
      1048576
      >>> human2bytes('1 Gi')
      1073741824
      >>> human2bytes('1 tera')
      1099511627776

      >>> human2bytes('0.5kilo')
      512
      >>> human2bytes('0.1  byte')
      0
      >>> human2bytes('1 k')  # k is an alias for K
      1024
      >>> human2bytes('12 foo')
      Traceback (most recent call last):
          ...
      ValueError: can't interpret '12 foo'
    """

    # if string contains all numbers, return as int
    if s.isdigit():
        return int(s)

    # if string contains all numbers and a dot, return as float
    if s.replace(".", "", 1).isdigit():
        return float(s)

    init = s
    num = ""
    while s and s[0:1].isdigit() or s[0:1] == ".":
        num += s[0]
        s = s[1:]
    num = float(num)
    letter = s.strip()
    for name, sset in SYMBOLS.items():
        if letter in sset:
            break
    else:
        if letter == "k":
            # treat 'k' as an alias for 'K' as per: http://goo.gl/kTQMs
            sset = SYMBOLS["customary"]
            letter = letter.upper()
        else:
            raise ValueError("can't interpret %r" % init)
    prefix = {sset[0]: 1}
    for i, s in enumerate(sset[1:]):
        prefix[s] = 1 << (i + 1) * 10
    return int(num * prefix[letter])


@dataclass
class TrainigConfig:
    batch_size: int = field(default=128, metadata={"help": "Batch size"})
    seq_length: int = field(default=1024, metadata={"help": "Sequence length"})
    model: str = field(default="gpt2", metadata={"help": "Model name"})


@dataclass
class DeviceConfig:
    rank: int
    # ip: str
    # port: int
    flops: int = field(
        default="1e12",
        metadata={
            "help": "The number of FLOPs of the device, support customary units"
        },
    )
    memory: int = field(
        default="1e12",
        metadata={
            "help": "The memory size in bytes of the device, support customary units"
        },
    )
    ul_bw: float = field(
        default="1e12",
        metadata={
            "help": "The uplink bandwidth in bytes of the device, support customary units"
        },
    )
    dl_bw: float = field(
        default="1e12",
        metadata={
            "help": "The downlink bandwidth in bytes of the device, support customary units"
        },
    )

    ul_lat: float = field(
        default=0,
        metadata={"help": "The uplink latency of the device"},
    )

    dl_lat: float = field(
        default=0,
        metadata={"help": "The downlink latency of the device"},
    )

    # def __post_init__(self):
    #     self.flops = human2bytes(self.flops)
    #     self.memory = human2bytes(self.memory)
    #     self.ul_bw = human2bytes(self.ul_bw)
    #     self.dl_bw = human2bytes(self.dl_bw)


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


@dataclass
class EmulatorConfig:
    gpu_memory: int = field(
        default=10,
        metadata={"help": "an integer in GB"},
    )
    cpu_memory: int = field(
        default=20,
        metadata={"help": "an integer in GB"},
    )
    ckpt_path: str = field(
        default="checkpoints",
        metadata={"help": "The path to save the emulator model checkpoints"},
    )
    listen_port: int = field(
        default=50051,
        metadata={"help": "The port to listen to the incoming requests"},
    )
    listen_ip: str = field(
        default="localhost",
        metadata={"help": "The address to listen to the incoming requests"},
    )
    debug: bool = field(
        default=False,
        metadata={"help": "Enable debug mode"},
    )

    def __post_init__(self):
        # total_gpu_memory = int(
        #     torch.cuda.get_device_properties(0).total_memory / 1024**3
        # )
        # total_cpu_memory = int(psutil.virtual_memory().total / 1024**3)

        # if isinstance(self.gpu_memory, float):
        #     self.gpu_memory = int(total_gpu_memory * self.gpu_memory)

        # if isinstance(self.cpu_memory, float):
        #     self.cpu_memory = int(total_cpu_memory * self.cpu_memory)

        # if self.gpu_memory > total_gpu_memory:
        #     raise ValueError(f"gpu_memory should not exceed {total_gpu_memory} GB")
        # if self.cpu_memory > total_cpu_memory:
        #     raise ValueError(f"cpu_memory should not exceed {total_cpu_memory} GB")

        self.gpu_memory *= GB
        self.cpu_memory *= GB

        os.environ["MORPHLING_CKPT_PATH"] = self.ckpt_path
        os.environ["MORPHLING_GPU_SIZE"] = str(self.gpu_memory)

        param_meta_map_file = os.path.join(
            self.ckpt_path, "param_meta_map.json"
        )

        with open(param_meta_map_file, "r") as f:
            param_meta_map = json.load(f)

        shm_mem_size, shm_mem_offsets = compute_shm_offsets(param_meta_map)
        pin_mem_size, pin_mem_offsets = compute_pin_offsets(param_meta_map)

        os.environ["MORPHLING_PIN_SIZE"] = str(pin_mem_size)
        os.environ["MORPHLING_SHM_SIZE"] = str(self.cpu_memory - pin_mem_size)

        assert self.cpu_memory > pin_mem_size, (
            "CPU memory should be greater than model size"
        )

    @classmethod
    def load_from_file(self, config_path):
        parser = HfArgumentParser(self)
        self = parser.parse_json_file(json_file=config_path)[0]
        return self

    @classmethod
    def load_from_json(self, config_json):
        parser = HfArgumentParser(self)
        self = parser.parse_dict(config_json)[0]
        return self


# a universal logger for all modules
# Usage:
# from morphling.common.logger import logger
# logger.info("hello world")
import os

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_LEVEL = getattr(logging, LOG_LEVEL.upper())

from logging import Formatter, StreamHandler, getLogger


def get_logger():
    logger = getLogger("morphling")
    logger.setLevel(LOG_LEVEL)
    handler = StreamHandler()
    handler.setLevel(LOG_LEVEL)
    formatter = Formatter(
        "%(asctime)s - %(className)s.%(funcName)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
