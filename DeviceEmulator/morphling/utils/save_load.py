import json
import os

import torch
import torch.nn as nn

from morphling._C import restore_tensors, save_tensors


def load_model(model: nn.Module, load_path: str):
    """
    Args:
        model: nn.Module
            a model to be loaded
        load_path: str
            a local path to load the converted model
    """
    model = model.cpu()
    model_state_dict = model.state_dict()
    tensor_index = {}
    with open(os.path.join(load_path, "tensor_index.json"), "r") as f:
        tensor_index = json.load(f)

    tensor_meta_index = {}
    tensor_data_index = {}
    for name, (offset, size, shape, stride, dtype) in tensor_index.items():
        tensor_meta_index[name] = (shape, stride, dtype)
        tensor_data_index[name] = (offset, size)

    state_dict = restore_tensors(
        tensor_meta_index, cuda_memory_ptrs, tensor_device_offsets
    )

    # This section of code was adopted from the Hugging Face Transformers project under Apache-2.0 License.
    # Source:


def save_model(model: nn.Module, save_path: str):
    """
    Args:
        model: nn.Module
            a model to be saved
        storage_path: str
            a local path to save the converted model
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    model = model.cpu()
    model_state_dict = model.state_dict()
    tensor_names = list(model_state_dict.keys())
    tensor_data_index = {}
    for name, param in model_state_dict.items():
        param_storage = param.untyped_storage()
        data_ptr = param_storage.data_ptr()
        size = param_storage.size()
        tensor_data_index[name] = (data_ptr, size)

    # save tensors
    tensor_offsets = save_tensors(tensor_names, tensor_data_index, save_path)

    # create tensor index
    tensor_index = {}
    for name, param in model_state_dict.items():
        # name: offset, size
        tensor_index[name] = (
            tensor_offsets[name],
            tensor_data_index[name][1],
            tuple(param.shape),
            tuple(param.stride()),
            str(param.dtype),
        )

    # This section of code was adopted from the Hugging Face Transformers project under Apache-2.0 License.
    # Source: https://github.com/huggingface/transformers/blob/9fe3f585bb4ea29f209dc705d269fbe292e1128f/src/transformers/modeling_utils.py#L2425-L2447
    # Modifications made: Removed the support for '_hf_peft_config_loaded'

    # Save the config
    model.config.save_pretrained(save_path)

    # save tensor index
    with open(os.path.join(save_path, "tensor_index.json"), "w") as f:
        json.dump(tensor_index, f)
