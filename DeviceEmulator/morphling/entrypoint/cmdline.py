import argparse
import gc
import json
import os
import shutil

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from tqdm import tqdm

from morphling._C import ArcherTensorHandle
from morphling.utils import get_checkpoint_paths


def save_model(model_name_or_path, output_path, dtype, force=False):
    if os.path.exists(model_name_or_path):
        checkpoint_paths = get_checkpoint_paths(model_name_or_path)
    else:
        checkpoint_paths = None
        # get the checkpoint download path from huggingface hub
        model_path = snapshot_download(
            model_name_or_path,
            cache_dir=os.environ.get("TRANSFORMERS_CACHE", None),
            ignore_patterns=["flax*", "tf*"],
        )
        if model_path is None:
            raise RuntimeError(
                f"The `snapshot_download` function could not find the checkpoint {model_name_or_path}. "
                f"Please provide a valid checkpoint."
            )
        checkpoint_paths = get_checkpoint_paths(model_path)

    print(f"Checkpoint paths: {checkpoint_paths}")

    if force and os.path.exists(output_path):
        shutil.rmtree(output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dtype_cls = torch.float32
    if dtype == "float16":
        dtype_cls = torch.float16

    param_meta_map_file = os.path.join(output_path, "param_meta_map.json")
    tensor_handle = ArcherTensorHandle(output_path)
    param_meta_map = {}
    param_id = 0

    if not tensor_handle.is_tensor_index_initialized() or not os.path.exists(
        param_meta_map_file
    ):
        print("Creating model from scratch ...", checkpoint_paths, flush=True)

        for ckpt in tqdm(
            checkpoint_paths, desc="Loading checkpoint files", smoothing=0
        ):
            state_dict = {}
            if "safetensors" in ckpt:
                with safe_open(ckpt, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        state_dict[k] = f.get_tensor(k)
            else:
                state_dict = torch.load(ckpt)

            # convert all tensors in state_dict to self.dtype
            for k, v in state_dict.items():
                state_dict[k] = v.to(dtype_cls).to("cpu")

            param_names = list(state_dict.keys())

            for param_name in param_names:
                if not param_name in param_meta_map:
                    param = state_dict[param_name]

                    file_offset = tensor_handle.offload_tensor(param, param_id)

                    param_meta_map[param_name] = {
                        "id": param_id,
                        "size": param.numel() * param.element_size(),
                        # "shm_offset": -1,
                        "file_offset": file_offset,
                        "shape": tuple(param.shape),
                        "stride": tuple(param.stride()),
                        "dtype": str(param.dtype),
                    }
                    param_id += 1

            del state_dict
            gc.collect()
            torch.cuda.empty_cache()

        with open(param_meta_map_file, "w") as f:
            json.dump(param_meta_map, f)

    else:
        print("Saved model exists", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Morphling Emulator Interface")

    subparsers = parser.add_subparsers(dest="command")

    # Subparser for `save` command
    save_parser = subparsers.add_parser("save", help="Save model checkpoint")
    save_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path from Huggingface",
    )
    save_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the model checkpoint",
    )
    save_parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite the output directory if it exists",
    )
    save_parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Data type to save the model",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "save":
        print(f"Saving model {args.model} to {args.output}")
        save_model(args.model, args.output, args.dtype, args.force)
    else:
        print("Unknown command")


if __name__ == "__main__":
    main()
