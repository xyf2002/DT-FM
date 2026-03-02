#!/usr/bin/env python3
"""
Run server with fake devices, then wait for real devices to connect manually.

This script starts the server and fake devices. Real devices should be
connected manually using morphling_device command.

Example:
  # Start server with 2 fake devices, wait for 1 real device to connect manually
  python3 scripts/run_mixed_devices.py --num_devices 3 --num_fake_devices 2 --backend proxy

Usage:
  python3 scripts/run_mixed_devices.py --num_devices 3 --num_fake_devices 2 --backend proxy \
    --fake-device-host 127.0.0.1 --fake-device-port 50051

  # Then in another terminal, manually connect the real device:
  # morphling_device --id 2 --flops 1e12 --memory 8G --ul_bw 100M --dl_bw 100M --ul_lat 10 --dl_lat 10 --backend proxy --cfg config/proxy/cli.ini
"""

import argparse
import asyncio
import os
import socket
import struct
import subprocess
import sys
import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import morphling
from morphling.backend import AutoBackend
from morphling.hooks import apply_hooks

torch.autograd.set_detect_anomaly(True)  # type: ignore[attr-defined]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run mixed real and fake devices"
    )

    # Device configuration
    parser.add_argument(
        "--num_devices",
        type=int,
        default=3,
        help="Total number of devices to run",
    )
    parser.add_argument(
        "--num_fake_devices",
        type=int,
        default=2,
        help="Number of fake devices (rest will be real)",
    )
    parser.add_argument(
        "--fake-device-host",
        type=str,
        default="127.0.0.1",
        help="Host for fake devices to connect to",
    )
    parser.add_argument(
        "--fake-device-port",
        type=int,
        default=50051,
        help="Port for fake devices to connect to",
    )

    # Backend configuration
    parser.add_argument(
        "--backend",
        type=str,
        default="proxy",
        help="Backend to use: proxy, mqtt, rabbitmq, amqp",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=1024,
        help="Block size used by some backends",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="Config file path for backend (e.g., proxy config)",
    )

    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/opt-125m",
        help="Model name to load",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for inference"
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=128,
        help="Sequence length for inference",
    )

    # Device specifications (will be replicated for all devices)
    parser.add_argument(
        "--device_flops", type=float, default=1e12, help="FLOPS per device"
    )
    parser.add_argument(
        "--device_mem", type=int, default=8, help="Memory per device (GB)"
    )
    parser.add_argument(
        "--ul_bw", type=float, default=100, help="Upload bandwidth (Mbps)"
    )
    parser.add_argument(
        "--dl_bw", type=float, default=100, help="Download bandwidth (Mbps)"
    )
    parser.add_argument(
        "--ul_lat", type=float, default=10, help="Upload latency (ms)"
    )
    parser.add_argument(
        "--dl_lat", type=float, default=10, help="Download latency (ms)"
    )

    return parser.parse_args()


class FakeDevice:
    """Lightweight fake device that connects but doesn't do real computation"""

    def __init__(
        self, device_id: int, host: str = "127.0.0.1", port: int = 50051
    ):
        self.device_id = device_id
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        self.thread = None

    def connect(self):
        """Connect to the server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            print(
                f"[Fake Device {self.device_id}] Connected to {self.host}:{self.port}"
            )
            return True
        except Exception as e:
            print(f"[Fake Device {self.device_id}] Connection failed: {e}")
            return False

    def handle_task(self, task_data):
        """Handle incoming task - fake devices just return random results"""
        try:
            # Parse task (simplified - adjust based on your protocol)
            # For proxy backend, tasks typically include matrix dimensions
            import random

            import torch

            # Generate random result of appropriate size
            # You may need to adjust this based on actual protocol
            result = torch.randn(512, 512)  # Placeholder result

            print(f"[Fake Device {self.device_id}] Generated fake result")
            return result

        except Exception as e:
            print(f"[Fake Device {self.device_id}] Error handling task: {e}")
            return None

    def run(self):
        """Main loop for the fake device"""
        self.running = True

        while self.running:
            try:
                # Receive task size
                if self.socket is None:
                    break
                size_data = self.socket.recv(8)
                if not size_data:
                    break

                task_size = struct.unpack("Q", size_data)[0]

                # Receive task data
                task_data = b""
                while len(task_data) < task_size:
                    chunk = self.socket.recv(
                        min(task_size - len(task_data), 4096)
                    )
                    if not chunk:
                        break
                    task_data += chunk

                # Process task (fake)
                result = self.handle_task(task_data)

                # Send back result (simplified)
                if result is not None:
                    result_bytes = result.numpy().tobytes()
                    self.socket.sendall(struct.pack("Q", len(result_bytes)))
                    self.socket.sendall(result_bytes)

            except Exception as e:
                print(f"[Fake Device {self.device_id}] Error in main loop: {e}")
                break

        print(f"[Fake Device {self.device_id}] Stopped")

    def start(self):
        """Start the fake device in a separate thread"""
        if self.connect():
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            return True
        return False

    def stop(self):
        """Stop the fake device"""
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        if self.thread:
            self.thread.join(timeout=5)


def print_real_device_command(device_id: int, args):
    """Print the command to manually start a real device"""

    # Convert to proper units for morphling_device
    flops_str = f"{int(args.device_flops)}"
    mem_str = f"{args.device_mem}G"
    ul_bw_str = f"{int(args.ul_bw)}M"
    dl_bw_str = f"{int(args.dl_bw)}M"

    cfg_path = args.cfg or os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config", "proxy", "cli.ini"
    )

    command = [
        "morphling_device",
        "--id",
        str(device_id),
        "--flops",
        flops_str,
        "--memory",
        mem_str,
        "--ul_bw",
        ul_bw_str,
        "--dl_bw",
        dl_bw_str,
        "--ul_lat",
        str(args.ul_lat),
        "--dl_lat",
        str(args.dl_lat),
        "--backend",
        args.backend,
        "--cfg",
        cfg_path,
    ]

    return " ".join(command)


def main():
    args = parse_args()

    # Validate arguments
    if args.num_fake_devices > args.num_devices:
        print(
            f"Error: num_fake_devices ({args.num_fake_devices}) cannot be greater than num_devices ({args.num_devices})"
        )
        sys.exit(1)

    num_real_devices = args.num_devices - args.num_fake_devices

    print(f"=== Mixed Device Configuration ===")
    print(f"Total devices: {args.num_devices}")
    print(f"Fake devices: {args.num_fake_devices}")
    print(f"Real devices: {num_real_devices}")
    print(f"Backend: {args.backend}")
    print(
        f"Fake device endpoint: {args.fake_device_host}:{args.fake_device_port}"
    )
    print(f"Model: {args.model_name}")
    print("=" * 40)

    # Set environment variables
    os.environ["NUM_DEVICES"] = str(args.num_devices)

    # Initialize environment
    this_file_path = os.path.dirname(os.path.realpath(__file__))
    output = subprocess.run(
        ["bash", f"{this_file_path}/env_init.sh"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print("Environment initialized:", output.stdout.decode())

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print("Model loaded")

    # Initialize backend
    print("Initializing backend...")
    if args.backend == "rabbitmq":
        loop = asyncio.get_event_loop()
        backend = AutoBackend.from_name(
            args.backend, loop, block_size=args.block_size
        )
        loop.run_until_complete(backend.connect())
    elif args.backend == "amqp":
        backend = AutoBackend.from_name(
            args.backend, "localhost", args.block_size
        )
    elif args.backend == "mqtt":
        backend = AutoBackend.from_name(args.backend, args.block_size)
        backend.start()  # type: ignore[attr-defined]
    elif args.backend == "proxy":
        backend = AutoBackend.from_name(args.backend)
        cfg_path = args.cfg or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config",
            "proxy",
            "svr.ini",
        )
        backend.initialize(cfg_path)  # type: ignore[attr-defined]
        backend.start()  # type: ignore[attr-defined]
    else:
        print(f"Unknown backend: {args.backend}")
        sys.exit(1)

    morphling.hooks.autograd._backend = backend
    print("Backend initialized")

    # Give backend time to start
    time.sleep(3)

    # Start fake devices
    fake_devices = []
    if args.num_fake_devices > 0:
        print(f"\nStarting {args.num_fake_devices} fake device(s)...")
        for i in range(args.num_fake_devices):
            fake_dev = FakeDevice(
                device_id=i,
                host=args.fake_device_host,
                port=args.fake_device_port,
            )
            if fake_dev.start():
                fake_devices.append(fake_dev)
                print(f"✓ Fake device {i} started")
                time.sleep(0.5)
            else:
                print(f"✗ Failed to start fake device {i}")

    # Print commands for manually connecting real devices
    if num_real_devices > 0:
        print(f"\n{'=' * 60}")
        print(f"Please manually connect {num_real_devices} real device(s)")
        print(f"{'=' * 60}")
        for i in range(args.num_fake_devices, args.num_devices):
            cmd = print_real_device_command(i, args)
            print(f"\n[Real Device {i}] Run this command in a new terminal:")
            print(f"  {cmd}")
        print(f"\n{'=' * 60}\n")

    # Wait for all devices to connect (including manually connected real devices)
    if args.backend == "proxy":
        print(
            f"\nWaiting for all {args.num_devices} device(s) to connect to proxy server..."
        )
        print(f"  - {args.num_fake_devices} fake device(s) already started")
        print(
            f"  - Waiting for {num_real_devices} real device(s) to connect manually\n"
        )

        timeout = 600  # 10 minutes to give time for manual connection
        start_time = time.time()
        last_count = 0

        while time.time() - start_time < timeout:
            try:
                connection_count = backend.get_connection_count()  # type: ignore[attr-defined]

                # Only print if count changed
                if connection_count != last_count:
                    fake_connected = min(
                        connection_count, args.num_fake_devices
                    )
                    real_connected = max(
                        0, connection_count - args.num_fake_devices
                    )
                    print(
                        f"Connected: {connection_count}/{args.num_devices} "
                        f"(fake: {fake_connected}/{args.num_fake_devices}, "
                        f"real: {real_connected}/{num_real_devices})"
                    )
                    last_count = connection_count

                if connection_count >= args.num_devices:
                    print("✓ All devices connected!")
                    break

                time.sleep(3)
            except Exception as e:
                print(f"Error checking connection count: {e}")
                time.sleep(3)
        else:
            connected = (
                backend.get_connection_count()  # type: ignore[attr-defined]
                if hasattr(backend, "get_connection_count")
                else 0
            )
            print(f"\n⚠ Timeout waiting for devices.")
            print(f"Connected: {connected}/{args.num_devices}")
            user_input = input("Continue anyway? (y/n): ")
            if user_input.lower() != "y":
                print("Exiting...")
                for fake_dev in fake_devices:
                    fake_dev.stop()
                sys.exit(0)

    time.sleep(3)

    # Prepare input
    print("\n=== Running Inference ===")
    input_text = [
        "Hello, my dog is cute. He is a good " * 128
    ] * args.batch_size
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.seq_length,
    )

    print(f"Input shape: {inputs['input_ids'].shape}")

    # Apply hooks for distributed execution
    apply_hooks("linear")

    model = model.to("cpu")
    inputs = inputs.to("cpu")

    # Forward pass
    print("Running forward pass...")
    start = time.time()
    outputs = model(
        **inputs,
        return_dict=True,
        output_hidden_states=True,
        output_attentions=True,
    )
    end = time.time()
    print(f"✓ Forward pass completed in {end - start:.2f}s")

    out_logits = outputs.logits
    print(f"Output logits shape: {out_logits.shape}")

    # Backward pass
    labels = inputs["input_ids"]
    loss = torch.nn.functional.cross_entropy(
        out_logits.view(-1, out_logits.size(-1)), labels.view(-1)
    )
    print(f"Loss: {loss.item():.4f}")

    print("Running backward pass...")
    start = time.time()
    loss.backward()
    end = time.time()
    print(f"✓ Backward pass completed in {end - start:.2f}s")

    print("\n=== Test Completed Successfully ===")

    # Cleanup
    print("\nCleaning up...")
    for fake_dev in fake_devices:
        fake_dev.stop()

    print("Done!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
