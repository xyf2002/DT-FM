# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #
# TODO: Add license
import concurrent.futures
import json
import os
import time
import uuid
from typing import Dict, Optional, Union

import torch
from accelerate import dispatch_model, init_empty_weights

# from accelerate.hooks import add_hook_to_module
from accelerate.utils import set_module_tensor_to_device
from serverless_llm_store._C import (
    allocate_cuda_memory,
    get_cuda_memory_handles,
    get_device_uuid_map,
    restore_tensors,
    save_tensors,
)
from serverless_llm_store.client import SllmStoreClient
from serverless_llm_store.device_map_utils import (
    DeviceMapType,
    _compute_device_placement_from_map,
    _compute_device_placement_from_map_fast,
    _expand_tensor_name,
    _transform_device_map_to_dict,
)
from serverless_llm_store.logger import init_logger
from serverless_llm_store.utils import (
    calculate_device_memory,
    calculate_tensor_device_offsets,
    dtype_byte_size,
    get_no_split_modules,
    get_tied_no_split_modules,
    send_module_buffers_to_device,
)
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

logger = init_logger(__name__)


def _get_uuid():
    return str(uuid.uuid4())


def save_dict(model_state_dict: Dict[str, torch.Tensor], model_path: str):
    tensor_names = list(model_state_dict.keys())
    tensor_data_index = {}
    for name, param in model_state_dict.items():
        param_storage = param.untyped_storage()
        data_ptr = param_storage.data_ptr()
        size = param_storage.size()
        tensor_data_index[name] = (data_ptr, size)

    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    # save tensors
    tensor_offsets = save_tensors(tensor_names, tensor_data_index, model_path)

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

    # save tensor index
    with open(os.path.join(model_path, "tensor_index.json"), "w") as f:
        json.dump(tensor_index, f)


def save_model(model: nn.Module, model_path: str):
    """
    Args:
        model: nn.Module
            a model to be saved
        storage_path: str
            a local path to save the converted model
    """
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

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
    tensor_offsets = save_tensors(tensor_names, tensor_data_index, model_path)

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
    #
    # Save the config
    model.config.save_pretrained(model_path)
    if model.can_generate():
        # generation config built from the model config + the model config holds generation kwargs -> generate
        # may revert to legacy behavior if the two don't match
        if (
            model.generation_config._from_model_config
            and model.config._has_non_default_generation_parameters()
        ):
            new_generation_config = GenerationConfig.from_model_config(
                model.config
            )
            if new_generation_config != model.generation_config:
                logger.warning(
                    "Your generation config was originally created from the model config, but the model "
                    "config has changed since then. Unless you pass the `generation_config` argument to this "
                    "model's `generate` calls, they will revert to the legacy behavior where the base "
                    "`generate` parameterization is loaded from the model config instead. "
                    "To avoid this behavior and this warning, we recommend you to overwrite the generation "
                    "config model attribute before calling the model's `save_pretrained`, preferably also "
                    "removing any generation kwargs from the model config. This warning will be raised to an "
                    "exception in v4.41."
                )
        model.generation_config.save_pretrained(model_path)

    # save tensor index
    with open(os.path.join(model_path, "tensor_index.json"), "w") as f:
        json.dump(tensor_index, f)

    # save module index
    no_split_modules = get_no_split_modules(model, model._no_split_modules)
    with open(os.path.join(model_path, "no_split_modules.json"), "w") as f:
        json.dump(no_split_modules, f)

    # save tied parameters
    tied_no_split_modules = get_tied_no_split_modules(model, no_split_modules)
    with open(os.path.join(model_path, "tied_no_split_modules.json"), "w") as f:
        json.dump(tied_no_split_modules, f)
