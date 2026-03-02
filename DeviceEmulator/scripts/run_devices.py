"""
Run server and device emulator processes with a configured backend.

Usage:
  python3 scripts/run_devices.py --num_devices 4 --model_name facebook/opt-125m \
    --backend proxy --seq_length 128 --batch_size 1 --cfg config/proxy/svr.ini
  python3 scripts/run_devices.py --num_devices 2 --model_name facebook/opt-125m \
    --backend proxy --enable-hooks
"""

import asyncio
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

import morphling

# from morphling import set_backend
from morphling.backend import AutoBackend
from morphling.entrypoint import DeviceConfigArguments, ModelConfigArguments
from morphling.hooks import apply_hooks

torch.autograd.set_detect_anomaly(True)  # type: ignore[attr-defined]

# # if SIGINT is received, kill all the devices
# def signal_handler(sig, frame):
#     for p in device_processes:
#         p.kill()
#     exit(0)


# import signal

# signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    # Detect enable-hooks flag before using HfArgumentParser because some
    # versions of HfArgumentParser can raise if unknown args remain.
    import sys

    _enable_hooks_flag_names = ("--enable-hooks", "--enable_hooks", "--hooks")
    local_enable_hooks = False
    # Build argv without our local flags so HfArgumentParser won't complain
    orig_argv = sys.argv
    filtered_argv = [orig_argv[0]]
    for a in orig_argv[1:]:
        if a in _enable_hooks_flag_names:
            local_enable_hooks = True
        else:
            filtered_argv.append(a)

    # Temporarily replace sys.argv for HfArgumentParser
    sys.argv = filtered_argv

    parser = HfArgumentParser((DeviceConfigArguments, ModelConfigArguments))
    device_args, model_args = parser.parse_args_into_dataclasses()

    # Restore original argv
    sys.argv = orig_argv
    print(device_args, model_args, flush=True)

    os.environ["NUM_DEVICES"] = str(device_args.num_devices)
    num_gpus = torch.cuda.device_count()

    # read the output of the bash script
    this_file_path = os.path.dirname(os.path.realpath(__file__))
    output = subprocess.run(
        ["bash", f"{this_file_path}/env_init.sh"], stdout=subprocess.PIPE
    )
    print("env_init", output.stdout)

    # time.sleep(15)
    # start model from here
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name, torch_dtype=torch.float32
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)

    print("Model loaded", model)

    # # get output of "docker exec mosquitto mosquitto_sub -t '$SYS/broker/subscriptions/count'  -C 1"
    # while True:
    #     command = [
    #             "docker",
    #             "exec",
    #             "mosquitto",
    #             "mosquitto_sub",
    #             "-t",
    #             "$SYS/broker/subscriptions/count",
    #             "-C",
    #             "1",
    #         ]
    #     # command = [
    #     #         "docker",
    #     #         "exec",
    #     #         "emqx",
    #     #         "emqx_ctl",
    #     #         "broker",
    #     #         "stats",
    #     #     ]
    #     output = subprocess.run(
    #         command,
    #         stdout=subprocess.PIPE,
    #     )
    #     print("Command", " ".join(command))
    #     print("Subscriptions count", output.stdout)
    #     # print("Error", output.stderr)

    #     # # find line of "connections.count" from output.stdout
    #     # for line in output.stdout.decode("utf-8").split("\n"):
    #     #     if "connections.count" in line:
    #     #         print("Connections count", line)
    #     #         break

    #     # if int(line.split(" ")[-1]) >= device_args.num_devices:
    #     #     break

    #     if int(output.stdout) >= device_args.num_devices:
    #         break

    #     time.sleep(1)
    #     print("Waiting for devices to connect")

    if model_args.backend == "rabbitmq":
        loop = asyncio.get_event_loop()
        backend = AutoBackend.from_name(
            model_args.backend, loop, block_size=model_args.block_size
        )
        loop.run_until_complete(backend.connect())

    elif model_args.backend == "amqp":
        backend = AutoBackend.from_name(
            model_args.backend, "localhost", model_args.block_size
        )

    elif model_args.backend == "mqtt":
        backend = AutoBackend.from_name(
            model_args.backend, model_args.block_size
        )
        backend.start()

    elif model_args.backend == "proxy":
        backend = AutoBackend.from_name(model_args.backend)
        backend.initialize(model_args.cfg)
        backend.start()

    # backend = AutoBackend.from_name("amqp", "localhost", model_args.block_size)
    morphling.hooks.autograd._backend = backend

    time.sleep(5)

    device_processes = []
    print("Running devices", device_args.num_devices)
    for i in range(device_args.num_devices):
        # command = [
        #     "morphling_device",
        #     "--id",
        #     str(i),
        #     "--flops",
        #     str(device_args.device_flops[i]),
        #     "--memory",
        #     str(device_args.device_mem[i]),
        #     "--ul_bw",
        #     str(device_args.ul_bw[i]),
        #     "--dl_bw",
        #     str(device_args.dl_bw[i]),
        #     "--ul_lat",
        #     str(device_args.ul_lat[i]),
        #     "--dl_lat",
        #     str(device_args.dl_lat[i]),
        #     "--emulation",
        #     "--backend",
        #     model_args.backend,
        #     "&",
        # ]

        # env = os.environ.copy()
        # env["MORPHLING_PIN_SIZE"] = str(device_args.device_mem[i])
        # env["SPDLOG_LEVEL"] = os.environ.get("SPDLOG_LEVEL", "info")
        # env["TORCH_SHOW_CPP_STACKTRACES"] = "1"

        command = [
            "CUDA_VISIBLE_DEVICES=" + str(i % num_gpus),
            "bash",
            f"{this_file_path}/run_device.sh",
            str(i),
            str(device_args.device_flops[i]),
            str(device_args.device_mem[i]),
            str(device_args.ul_bw[i]),
            str(device_args.dl_bw[i]),
            str(device_args.ul_lat[i]),
            str(device_args.dl_lat[i]),
            model_args.backend,
            getattr(
                model_args, "proxy_host", ""
            ),  # Pass proxy_host as the 10th parameter (optional)
        ]
        # print("Running device", command)
        os.system(" ".join(command))

        # print("Running device", command)

        # subprocess.Popen(command, env=env)

        # # create new process rather than subprocess
        # os.system(" ".join(command))
        # # device_processes.append(p)

    #     / # mosquitto_sub -t '$SYS/broker/subscriptions/count' -v
    # $SYS/broker/subscriptions/count 40
    # $SYS/broker/subscriptions/count 41

    # Wait for devices to connect for proxy backend
    if model_args.backend == "proxy":
        print("Waiting for devices to connect to proxy server...")
        timeout = 120  # 2 minutes timeout
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                connection_count = backend.get_connection_count()
                print(
                    f"Connected devices: {connection_count}/{device_args.num_devices}"
                )

                if connection_count >= device_args.num_devices:
                    print("All devices connected!")
                    break

                time.sleep(2)
            except Exception as e:
                print(f"Error checking connection count: {e}")
                time.sleep(2)
        else:
            print(
                f"Timeout waiting for devices to connect. Connected: {backend.get_connection_count()}/{device_args.num_devices}"
            )
            # Continue anyway

    time.sleep(5)

    # random text for seqlen > 128
    input_text = [
        "".join("Hello, my dog is cute. He is a good ") * 128
    ] * model_args.batch_size
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=model_args.seq_length,
    )

    print("inputs", inputs, flush=True)

    # ref_model = model.to("cuda:0")
    # ref_input_ids = input_ids.to("cuda:0")
    # ref_outputs = ref_model(
    #     **input_ids,
    #     return_dict=True,
    #     output_hidden_states=True,
    #     output_attentions=True,
    # )
    # ref_logits = ref_outputs.logits
    # ref_hidden_states = ref_outputs.hidden_states
    # ref_attentions = ref_outputs.attentions
    # # ref_outputs = [out.cpu() for out in ref_outputs if isinstance(out, torch.Tensor)]

    if local_enable_hooks:
        print("✓ Distributed computation mode: apply_hooks('linear') ENABLED")
        apply_hooks("linear")
    else:
        print("✗ Local computation mode: apply_hooks('linear') DISABLED")

    model = model.to("cpu")
    inputs = inputs.to("cpu")
    start = time.time()
    outputs = model(
        **inputs,
        return_dict=True,
        output_hidden_states=True,
        output_attentions=True,
    )
    end = time.time()
    print("Forward time", end - start)
    out_logits = outputs.logits
    out_hidden_states = outputs.hidden_states
    out_attentions = outputs.attentions
    print("out_logits", out_logits)

    # Save logits to pt file
    os.makedirs("logits_comparison", exist_ok=True)
    suffix = "with_hooks" if local_enable_hooks else "without_hooks"
    logits_path = os.path.join("logits_comparison", f"logits_{suffix}.pt")
    torch.save(out_logits.cpu().detach(), logits_path)
    print(f"✓ Saved logits to {logits_path}")

    labels = inputs["input_ids"]
    loss = torch.nn.functional.cross_entropy(
        out_logits.view(-1, out_logits.size(-1)), labels.view(-1)
    )
    print("Loss", loss)
    start = time.time()
    loss.backward()
    end = time.time()
    print("Backward time", end - start)

    # print("ref_hidden_states", ref_hidden_states)

    # for i, (ref, out) in enumerate(zip(ref_hidden_states, out_hidden_states)):
    #     print("ref", ref)
    #     print("out", out)
    #     assert torch.allclose(
    #         ref.to("cpu"), out.to("cpu"), atol=1e-6
    #     ), f"Attention are not close!, max diff: {torch.max(torch.abs(ref.to('cpu') - out.to('cpu')))}"
    #     print(f"Attention {i} is close!")

    # for i, (ref, out) in enumerate(zip(ref_hidden_states, out_hidden_states)):
    #     assert torch.allclose(
    #         ref.to("cpu"), out.to("cpu"), atol=1e-6
    #     ), f"hidden_states are not close!, max diff: {torch.max(torch.abs(ref.to("cpu") - out.to("cpu")))}"
    #     print(f"Hidden state {i} is close!")

    # assert torch.allclose(
    #     ref_logits.to("cpu"), out_logits.to("cpu"), atol=1e-6
    # ), f"Logits are not close!, max diff: {torch.max(torch.abs(ref_logits.to("cpu") - out_logits.to("cpu")))}"

    # # all elements needs to be close
    # for ref, out in zip(ref_outputs, outputs):
    #     print("ref", ref)
    #     print("out", out)
    #     assert torch.allclose(
    #         ref, out, atol=1e-6
    #     ), f"Outputs are not close!, max diff: {torch.max(torch.abs(ref - out))}"

    # print("All outputs are close!")


# morphling_device_config --num_devices 4 --output build/device_config.json

# morphling_device
