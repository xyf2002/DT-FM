from collections import Counter
from typing import Dict, Tuple

import numpy as np
import torch


def find_tensor_same_size(
    param_meta_map: Dict[str, dict], size: int
) -> np.ndarray:
    names_of_size = [
        name for name, param in param_meta_map.items() if param["size"] == size
    ]
    ids_of_size = [param_meta_map[name]["id"] for name in names_of_size]
    ids_of_size = np.array(ids_of_size, dtype=np.uint32)
    return ids_of_size


def compute_shm_offsets(
    param_meta_map: Dict[str, dict],
) -> Tuple[int, Dict[str, int]]:
    unique_sizes_counter = Counter(
        [param["size"] for param in param_meta_map.values()]
    )
    shm_mem_size = sum(
        [size + 4 * count for size, count in unique_sizes_counter.items()]
    )

    shm_mem_size_cum = np.cumsum(
        [size + 4 * count for size, count in unique_sizes_counter.items()]
    )
    shm_mem_size_cum = shm_mem_size_cum - shm_mem_size_cum[0]
    shm_mem_offsets = dict(zip(unique_sizes_counter.keys(), shm_mem_size_cum))
    shm_mem_offsets = {k: int(v) for k, v in shm_mem_offsets.items()}

    return shm_mem_size, shm_mem_offsets


def compute_pin_offsets(
    param_meta_map: Dict[str, dict],
) -> Tuple[int, Dict[str, int]]:
    pin_mem_size = sum([meta["size"] for _, meta in param_meta_map.items()])
    offset = 0
    pin_mem_offsets = {}
    for name, meta in param_meta_map.items():
        pin_mem_offsets[name] = offset
        offset += meta["size"]
    return pin_mem_size, pin_mem_offsets


def update_shm_offsets(
    param_meta_map: Dict[str, dict],
) -> Tuple[int, Dict[str, int]]:
    _, shm_mem_offsets = compute_shm_offsets(param_meta_map)
    unique_sizes_counter = Counter(
        [param["size"] for param in param_meta_map.values()]
    )
    for size, count in unique_sizes_counter.items():
        # find all tensor name and id with the same size
        names_of_size = [
            name
            for name, param in param_meta_map.items()
            if param["size"] == size
        ]

        for i, name in enumerate(names_of_size):
            param_meta_map[name]["shm_offset"] = shm_mem_offsets[size] + i * 4

    return param_meta_map
