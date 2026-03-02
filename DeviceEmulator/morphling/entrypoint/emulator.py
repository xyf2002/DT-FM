import os
import subprocess
import sys

from transformers import HfArgumentParser

import morphling
from morphling.common import EmulatorConfig

KB = 1024
MB = 1024 * KB
GB = 1024 * MB


def main():
    parser = HfArgumentParser((EmulatorConfig,))
    args = parser.parse_args()

    print(args)

    print(morphling.__path__[0])

    server_executable = os.path.join(morphling.__path__[0], "morphling_server")
    checkpoint_path = args.ckpt_path
    listen_address = f"{args.listen_ip}:{args.listen_port}"

    env = os.environ.copy()
    env["MORPHLING_SERVER_ADDRESS"] = listen_address
    env["MORPHLING_GPU_SIZE"] = str(args.gpu_memory * GB)
    env["MORPHLING_PIN_SIZE"] = str(args.cpu_memory * GB)

    if args.debug:
        env["SPDLOG_LEVEL"] = "DEBUG"

    sys.exit(
        subprocess.call(
            [
                server_executable,
                "--listen",
                listen_address,
                "--path",
                checkpoint_path,
            ],
            env=env,
        )
    )

    print("Emulator exited")

    # param_meta_map_file = os.path.join(args.ckpt_path, "param_meta_map.json")

    # with open(param_meta_map_file, "r") as f:
    #     param_meta_map = json.load(f)

    # shm_mem_size, shm_mem_offsets = compute_shm_offsets(param_meta_map)

    # unique_sizes_counter = Counter(
    #     [param["size"] for param in param_meta_map.values()]
    # )

    # shm_param_buffer = shared_memory.SharedMemory(create=True, size=shm_mem_size)
    # buffer = shm_param_buffer.buf
    # buffer[:] = bytearray(shm_mem_size)

    # for size, count in unique_sizes_counter.items():
    #     # find all tensor name and id with the same size
    #     ids_of_size = find_tensor_same_size(param_meta_map, size)

    #     # print(f"Size: {size}, Count: {count}, Names: {names_of_size}, IDs: {ids_of_size}")

    #     assert (
    #         len(ids_of_size) == count
    #     ), f"Size mismatch: {len(ids_of_size)} != {count}"

    #     # write ids_of_size to buffer
    #     tmp = np.ndarray(
    #         ids_of_size.shape,
    #         dtype=ids_of_size.dtype,
    #         buffer=buffer,
    #         offset=shm_mem_offsets[size],
    #     )
    #     tmp[:] = ids_of_size[:]

    #     tmp = np.ndarray(
    #         ids_of_size.shape,
    #         dtype=ids_of_size.dtype,
    #         buffer=buffer,
    #         offset=shm_mem_offsets[size] + size - count * 4,
    #     )
    #     tmp[:] = ids_of_size[:]

    # pin_mem_size, pin_mem_offsets = compute_pin_offsets(param_meta_map)
    # name_id_map = {}
    # for name, meta in param_meta_map.items():
    #     name_id_map[name] = meta["id"]

    # checkpoint_handle = CheckpointHandle(args.ckpt_path)
    # checkpoint_handle.read_checkpoint(pin_mem_offsets, name_id_map)

    # pin_param_buffer = torch.zeros(pin_mem_size, dtype=torch.uint8).pin_memory()
    # pin_buffer = pin_param_buffer.data_ptr()

    # # read from file offset to pin_buffer offsets
    # for name in param_meta_map:
    #     param_meta = param_meta_map[name]
    #     file_offset = param_meta["file_offset"]
    #     size = param_meta["size"]
    #     pin_offset = pin_mem_offsets[name]
    #     with open(os.path.join(args.ckpt_path, "archer_param_0"), "rb") as f:
    #         f.seek(file_offset)
    #         pin_buffer[pin_offset:pin_offset+size] = torch.ByteTensor(list(f.read(size)))


if __name__ == "__main__":
    main()
