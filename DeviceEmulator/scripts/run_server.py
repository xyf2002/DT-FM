#!/usr/bin/env python3
"""
Run server-only for the DeviceEmulator project.

This script initializes the selected backend (mqtt/rabbitmq/amqp/proxy)
and, if requested, loads the model. It does NOT spawn any device client
processes — it only starts the server and waits for connections.

Example:
  python3 scripts/run_server.py --backend proxy --model_name facebook/opt-125m
  python3 scripts/run_server.py --backend proxy --model_name facebook/opt-125m --cfg ./config/proxy/svr.ini
"""

import argparse
import asyncio
import os
import signal
import sys
import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import morphling
from morphling.backend import AutoBackend
from morphling.entrypoint import DeviceConfigArguments, ModelConfigArguments
from morphling.hooks import apply_hooks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run server-only for DeviceEmulator"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="proxy",
        help="Backend to use: proxy, mqtt, rabbitmq, amqp",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="Config file path for proxy backend (default: config/proxy/svr.ini)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="(Optional) model name to load for the server",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=1024,
        help="Block size used by some backends",
    )
    parser.add_argument(
        "--no-model",
        dest="load_model",
        action="store_false",
        help="Do not load the model; only start backend",
    )
    parser.add_argument(
        "--no-wait",
        dest="no_wait",
        action="store_true",
        help="Start immediately without waiting for any devices",
    )
    parser.add_argument(
        "--enable-cache",
        dest="enable_cache",
        action="store_true",
        help="Enable client-side caching (for proxy backend)",
    )
    parser.add_argument(
        "--enable-hooks",
        dest="enable_hooks",
        action="store_true",
        help="Enable hooks for distributed computation (apply_hooks)",
    )
    parser.add_argument(
        "--enable-verification",
        dest="enable_verification",
        action="store_true",
        help="Enable output verification in hooks (for debugging)",
    )
    parser.set_defaults(
        load_model=True, enable_cache=False, enable_hooks=False, no_wait=False
    )
    return parser.parse_args()


async def start_backend_async(backend_name: str, block_size: int):
    loop = asyncio.get_event_loop()
    backend = AutoBackend.from_name(backend_name, loop, block_size=block_size)
    await backend.connect()
    return backend


def start_backend_sync(
    backend_name: str,
    block_size: int,
    enable_cache: bool = False,
    cfg_path: Optional[str] = None,
):
    # Some backends (like mqtt/proxy) may use synchronous initialization
    backend = None
    if backend_name in ("mqtt", "proxy"):
        backend = AutoBackend.from_name(backend_name)
        # proxy/mqtt backends expose start() instead of async connect
        if hasattr(backend, "initialize"):
            if backend_name == "proxy":
                # ProxySvr.initialize() expects a config file path and optional cache flag
                import os

                default_config_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "config",
                    "proxy",
                    "svr.ini",
                )
                config_path = cfg_path or default_config_path

                # Check if initialize accepts cache parameter
                if enable_cache and hasattr(backend, "set_cache_enabled"):
                    backend.set_cache_enabled(True)  # type: ignore[attr-defined]
                    print(f"✓ Client-side caching ENABLED")
                else:
                    print(f"Client-side caching disabled")

                backend.initialize(config_path)  # type: ignore[attr-defined]
            else:
                try:
                    backend.initialize()  # type: ignore[attr-defined]
                except TypeError:
                    # initialize may accept args in some versions; ignore
                    backend.initialize(None)  # type: ignore[attr-defined]
        if hasattr(backend, "start"):
            backend.start()  # type: ignore[attr-defined]
    else:
        # Use asyncio for rabbitmq/amqp style backends
        loop = asyncio.get_event_loop()
        backend = loop.run_until_complete(
            start_backend_async(backend_name, block_size)
        )
    return backend


def get_connection_count_safe(backend) -> int:
    """Safely get connection count if backend supports it."""
    if hasattr(backend, "get_connection_count"):
        return backend.get_connection_count()  # type: ignore[attr-defined]
    return 0


def wait_for_devices(
    backend, min_devices: int, timeout: int = 120, no_wait: bool = False
):
    """Wait for minimum number of devices to connect

    Args:
        backend: The backend instance
        min_devices: Minimum devices required before starting
        timeout: Maximum time to wait in seconds
        no_wait: If True, return immediately without waiting
    """
    if no_wait:
        print("No-wait mode: Starting immediately without waiting for devices")
        return get_connection_count_safe(backend)

    print(
        f"Waiting for at least {min_devices} device(s) to connect (timeout: {timeout}s)..."
    )
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            if hasattr(backend, "get_connection_count"):
                connection_count = get_connection_count_safe(backend)
                elapsed = int(time.time() - start_time)
                print(
                    f"[{elapsed}s] Connected devices: {connection_count}/{min_devices}"
                )

                if connection_count >= min_devices:
                    print(f"✓ {connection_count} device(s) connected!")
                    return connection_count
            else:
                print("Backend does not support connection counting")
                return 1

            time.sleep(2)
        except Exception as e:
            print(f"Error checking connection count: {e}")
            time.sleep(2)

    final_count = get_connection_count_safe(backend)
    print(
        f"Timeout after {timeout}s. Connected: {final_count}/{min_devices} devices"
    )
    return final_count


def main():
    args = parse_args()
    print(
        f"Starting server-only with backend={args.backend}, load_model={args.load_model}"
    )

    # Optionally load model (on CPU by default)
    model = None
    tokenizer = None
    if args.load_model and args.model_name:
        print("Loading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=None
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        print("Model loaded")

    # Initialize backend
    print("Initializing backend...")
    backend = start_backend_sync(
        args.backend, args.block_size, args.enable_cache, args.cfg
    )
    print("Backend started. Server is now listening for device connections.")

    # Set backend for morphling hooks
    morphling.hooks.autograd._backend = backend
    morphling.hooks.autograd._enable_verification = args.enable_verification

    if args.enable_verification:
        print("✓ Output verification in hooks ENABLED")

    # Wait for minimum devices to connect
    min_devices = 1
    connected = wait_for_devices(backend, min_devices, no_wait=args.no_wait)

    if connected < min_devices and not args.no_wait:
        print(
            f"Warning: Only {connected} device(s) connected, but {min_devices} required."
        )
        print("Continuing anyway (new devices can join dynamically)...")
    elif connected == 0 and not args.no_wait:
        print("⚠ No devices connected, but starting in dynamic mode")
        print("Inference will proceed when devices connect")

    if connected > 0 and model is not None and tokenizer is not None:
        print("\n=== Running Text Generation Inference ===")
        # Print current device status
        current_devices = (
            get_connection_count_safe(backend)
            if hasattr(backend, "get_connection_count")
            else connected
        )
        print(f"📊 Current connected devices: {current_devices}")
        print(
            f"💡 Note: Backend (C++ RephrasePartitions) will dynamically allocate tasks based on actual device count"
        )
        print()

        input_text = ["Hello, my dog is cute. He is a good " * 128]
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        print("inputs:", inputs)

        # Apply hooks for distributed computation (similar to run_devices.py line 236)
        if args.enable_hooks:
            print(
                "✓ Distributed computation mode: apply_hooks('linear') ENABLED"
            )
            print(
                "  → Linear layer computations will be offloaded to remote devices"
            )
            apply_hooks("linear")
        else:
            print("✗ Local computation mode: apply_hooks('linear') DISABLED")
            print("  → All computations will run locally using PyTorch")

        inputs = inputs.to("cpu")
        model = model.to("cpu")
        model = model.to(torch.float32)

        # Debug: Print input info
        print(f"Input shape: {inputs['input_ids'].shape}")
        print(f"First 10 token IDs: {inputs['input_ids'][0, :10]}")

        start = time.time()
        outputs = model(**inputs, return_dict=True)
        end = time.time()
        print(f"Inference finished in {end - start:.2f}s")

        # Print device info during inference
        final_devices = (
            get_connection_count_safe(backend)
            if hasattr(backend, "get_connection_count")
            else current_devices
        )
        if final_devices > current_devices:
            print(
                f"✓ Device(s) joined during inference: {current_devices} → {final_devices}"
            )
        else:
            print(
                f"  Device count: {final_devices} (consistent throughout inference)"
            )

        if hasattr(outputs, "logits"):
            print("logits shape:", outputs.logits.shape)

            # Save logits to pt file
            os.makedirs("logits_comparison", exist_ok=True)
            suffix = "with_hooks" if args.enable_hooks else "without_hooks"
            logits_path = os.path.join(
                "logits_comparison", f"logits_{suffix}.pt"
            )
            torch.save(outputs.logits.cpu().detach(), logits_path)
            print(f"✓ Saved logits to {logits_path}")
        print("=== Inference Done ===\n")

    # Graceful shutdown handling
    stop = False

    def _signal_handler(sig, frame):
        nonlocal stop
        print(f"\nReceived signal {sig}. Shutting down...")
        stop = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    print("\n=== Server is running ===")
    print("Press Ctrl+C to stop")

    try:
        while not stop:
            time.sleep(1)
    finally:
        print("Stopping backend...")
        try:
            if hasattr(backend, "stop"):
                backend.stop()  # type: ignore[attr-defined]
            if hasattr(backend, "disconnect"):
                # async disconnect if available
                loop = asyncio.get_event_loop()
                loop.run_until_complete(backend.disconnect())  # type: ignore[attr-defined]
        except Exception as e:
            print("Error while stopping backend:", e)
        print("Server shutdown complete.")
        import sys

        sys.exit(0)


if __name__ == "__main__":
    # import torch lazily to avoid heavy import if not loading model
    try:
        import torch  # noqa: F401
    except Exception:
        pass
    main()
