import argparse
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import torch

import morphling
from morphling.backend import AutoBackend, AutoWorker

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True)
    parser.add_argument("--num_devices", type=int, default=4)
    parser.add_argument("--device_id", type=int, default=-1)

    args = parser.parse_args()
    print(args)

    os.environ["NUM_DEVICES"] = str(args.num_devices)

    tensor_a = torch.randn(2, 128, 4096).contiguous()
    tensor_b = torch.randn(2, 128, 4096).contiguous()

    if args.type == "server":
        backend = AutoBackend.from_name("mqtt")
        backend.start()
        for i in range(args.num_devices * 1):
            # backend.publish(f"/morphling/{i % args.num_devices}", tensor_a)
            start_time = time.time()
            output = backend.sync_dispatch_matmul(tensor_a, tensor_b)
            end_time = time.time()
            print(
                f"Time taken to publish messages: {end_time - start_time}",
                flush=True,
            )
        time.sleep(5)
    else:
        backend = AutoWorker.from_name(
            "mqtt", f"/morphling/req/{args.device_id}"
        )
        backend.start()

        # put to sleep without pulling cpu
        while True:
            time.sleep(1)
