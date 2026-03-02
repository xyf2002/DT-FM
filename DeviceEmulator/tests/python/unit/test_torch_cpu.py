import time

import torch
from tqdm import trange

latency = []
for _ in trange(100):
    a = torch.randn(1024, 1024)
    b = torch.randn(1024, 1)
    start_time = time.perf_counter()
    c = a.clone()
    end_time = time.perf_counter()
    latency.append(end_time - start_time)

# remove first 10 and last 10 latencies
latency = latency[10:-10]
print(f"Average latency: {sum(latency) / len(latency)} seconds")
print(f"Max latency: {max(latency)} seconds")
print(f"Min latency: {min(latency)} seconds")
