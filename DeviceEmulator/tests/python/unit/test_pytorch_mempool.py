import multiprocessing as mp
import os

import psutil
import torch


def get_memory_usage(size):
    a = torch.zeros(size)
    print(a.numel() * a.element_size() / 1024**2)
    print(psutil.Process(os.getpid()).memory_info().rss / 1024**2)


for size in [
    10**0,
    10**1,
    10**2,
    10**3,
    10**4,
    10**5,
    10**6,
    10**7,
    10**8,
    10**9,
]:
    p1 = mp.Process(target=get_memory_usage, args=(size,))
    p2 = mp.Process(target=get_memory_usage, args=(size,))
    p2.start()
    p1.start()
    p1.join()
    p2.join()
