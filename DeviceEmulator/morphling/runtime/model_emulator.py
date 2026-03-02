import functools
import gc
import json
import math
import os
from collections import Counter
from typing import Callable, Dict, Tuple, Type, Union

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from tqdm import tqdm
from transformers import AutoConfig
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel

from morphling._C import ArcherTensorHandle, MemoryManagerClient, set_tensor_shm

# from morphling._intercept import MemoryManagerClient
from morphling.common import *
from morphling.utils import get_checkpoint_paths


def do_nothing_decorator(orig_func: Callable) -> Callable:
    @functools.wraps(orig_func)
    def do_nothing(*args, **kwargs):
        pass

    return do_nothing


def param_init_decorator(orig_param_init: Callable) -> Callable:
    @functools.wraps(orig_param_init)
    def empty_param_init(cls, *args, **kwargs):
        orig_param_init(cls, *args, **kwargs)

        # cls.param_real_shape = {}
        for name, param in cls.named_parameters(recurse=False):
            # cls.param_real_shape[name] = param.shape
            param.data = torch.empty(
                param.shape, dtype=param.dtype, device=param.device
            )

        for name, buf in cls.named_buffers(recurse=False):
            # cls.param_real_shape[name] = buf.shape
            buf.data = torch.empty(
                param.shape, dtype=buf.dtype, device=buf.device
            )

    return empty_param_init


def from_pretrained_decorator(orig_from_pretrained: Callable) -> Callable:
    @functools.wraps(orig_from_pretrained)
    def archer_from_pretrained(cls, *args, **kwargs):
        # print("Creating model from scratch ...")
        config = AutoConfig.from_pretrained(args[0])
        model = cls._from_config(config, torch_dtype=torch.float32)

        print(f"Model config: {config}")
        print(f"Model: {model}")

        client = MemoryManagerClient()
        param_shm_map = client.get_model_param()
        print(f"param_shm_map: {param_shm_map}")

        for name, param in model.named_parameters(recurse=True):
            if name not in param_shm_map:
                print(f"param {name} not found in param_shm_map")
                continue
            shm_name, shm_size = param_shm_map[name]
            tensor = torch.empty(param.data.shape, dtype=param.data.dtype)
            set_tensor_shm(tensor, shm_name, shm_size)
            param.data = tensor
            # print(f"set tensor {name} to shm {shm_name} with size {shm_size}", param.data.size())
            assert ~(
                torch.isclose(param.data, torch.zeros_like(param.data)).all()
                == True
            ), f"param {name} is zero {param}"
            # print(f"param {name} is not zero {param}")

        return model

    return archer_from_pretrained


class InitEmptyModel:
    def __init__(self, cls: Type[PreTrainedModel]):
        self.cls = cls

    def __enter__(self):
        # for all the modules in torch.nn, add post_init method
        # assert False, torch.nn.modules.__dict__
        for name, module in torch.nn.modules.__dict__.items():
            if not isinstance(module, type):
                continue
            if not issubclass(module, torch.nn.modules.module.Module):
                continue
            if name in [
                "Module",
                "Sequential",
                "ModuleDict",
                "ModuleList",
                "ParameterList",
                "ParameterDict",
            ]:
                continue
            module._old_init = module.__init__
            module.__init__ = param_init_decorator(module.__init__)

            if hasattr(module, "reset_parameters"):
                module._old_reset_parameters = module.reset_parameters
                module.reset_parameters = do_nothing_decorator(
                    module.reset_parameters
                )

        # self.cls._old_init = self.cls.__init__
        # self.cls.__init__ = do_nothing_decorator(self.cls.__init__)

        self.cls._old_post_init = self.cls.post_init
        PreTrainedModel._old_post_init = PreTrainedModel.post_init

        self.cls.post_init = do_nothing_decorator(self.cls.post_init)
        PreTrainedModel.post_init = do_nothing_decorator(
            PreTrainedModel.post_init
        )

        self.cls._old_from_pretrained = self.cls.from_pretrained
        self.cls.from_pretrained = classmethod(
            from_pretrained_decorator(self.cls.from_pretrained)
        )

    def __exit__(self, exc_type, exc_value, traceback):
        # self.cls.__init__ = self.cls._old_init
        self.cls.post_init = self.cls._old_post_init
        PreTrainedModel.post_init = PreTrainedModel._old_post_init
        self.cls.from_pretrained = self.cls._old_from_pretrained

        for name, module in torch.nn.modules.__dict__.items():
            if not isinstance(module, type):
                continue
            if not issubclass(module, torch.nn.modules.module.Module):
                continue
            if name in [
                "Module",
                "Sequential",
                "ModuleDict",
                "ModuleList",
                "ParameterList",
                "ParameterDict",
            ]:
                continue
            module.__init__ = module._old_init

            if hasattr(module, "reset_parameters"):
                module.reset_parameters = module._old_reset_parameters

    # @staticmethod
    # def set_model_param(model):
    #     client = MemoryManagerClient()
    #     param_shm_map = client.get_model_param()
    #     print(f"param_shm_map: {param_shm_map}")

    #     for name, param in model.named_parameters(recurse=True):
    #         if name not in param_shm_map:
    #             print(f"param {name} not found in param_shm_map")
    #             continue
    #         shm_name, shm_size = param_shm_map[name]
    #         tensor = torch.empty(param.data.shape, dtype=param.data.dtype)
    #         set_tensor_shm(tensor, shm_name, shm_size)
    #         param.data = tensor
    #         # print(f"set tensor {name} to shm {shm_name} with size {shm_size}", param.data.size())
    #         assert ~(
    #             torch.isclose(param.data, torch.zeros_like(param.data)).all()
    #             == True
    #         ), f"param {name} is zero {param}"
    #         # print(f"param {name} is not zero {param}")


class EmulationEngine(object):
    param_id = 0
    request_id = 0
    # request_id_flag = False
    config = {}

    def __init__(self, config: PretrainedConfig):
        self.config = config

        model_name_or_path = config._name_or_path
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
        self.ckpt_files = checkpoint_paths

    def init(
        self,
        cls: Type[PreTrainedModel],
        config: Union[str, Dict, EmulatorConfig],
    ):
        self.cls = cls
        self.param_meta_map = {}
        self.tensor_id_map = {}
        self.registered_tensors = set()
        self.forward_hooks = []
        self.backward_hooks = []

        self.offload_set = set()

        if isinstance(config, str):
            config = EmulatorConfig.load_from_file(config)
        elif isinstance(config, dict):
            config = EmulatorConfig.load_from_json(config)

        self.checkpoint = config.ckpt_path
        os.makedirs(self.checkpoint, exist_ok=True)

        self.emulator_config = config
        self.ArcherTensorHandle = ArcherTensorHandle(config.ckpt_path)

        return self

    def __enter__(self):
        def do_nothing_decorator(orig_func: Callable) -> Callable:
            @functools.wraps(orig_func)
            def do_nothing(*args, **kwargs):
                pass

            return do_nothing

        def post_init_decorator(orig_post_init: Callable) -> Callable:
            # FIXME: this is a hacky way to get rid of the write to weight in the post_init, need a better way to do this if we need to support model training
            @functools.wraps(orig_post_init)
            def archer_post_init(cls, *args, **kwargs):
                pass

            return archer_post_init

        def torch_index_select_decorator(orig_torch_index_select: Callable):
            @functools.wraps(orig_torch_index_select)
            def archer_torch_index_select(input, dim, index):
                return orig_torch_index_select(
                    input, dim, index.to(input.device)
                ).to("cuda:0")

            return archer_torch_index_select

        def apply_to_model_decorator(orig_apply_to_model: Callable) -> Callable:
            @functools.wraps(orig_apply_to_model)
            def archer_apply_to_model(cls, fn):
                for name, param in cls.named_parameters(recurse=True):
                    if name not in self.param_meta_map:
                        continue
                    param.data = torch.zeros(
                        1,
                        dtype=param.dtype,
                        device=param.device,
                        pin_memory=True,
                    )

                for name, buffer in cls.named_buffers(recurse=True):
                    if name not in self.param_meta_map:
                        continue
                    buffer.data = torch.zeros(
                        1,
                        dtype=buffer.dtype,
                        device=buffer.device,
                        pin_memory=True,
                    )

            return archer_apply_to_model

        def init_decorator(orig_init: Callable) -> Callable:
            @functools.wraps(orig_init)
            def archer_init(cls, config, *args, **kwargs):
                # self.config = config
                pass

            return archer_init

        def param_init_decorator(orig_param_init: Callable) -> Callable:
            @functools.wraps(orig_param_init)
            def archer_param_init(cls, *args, **kwargs):
                orig_param_init(cls, *args, **kwargs)

                cls.param_real_shape = {}
                for name, param in cls.named_parameters(recurse=False):
                    cls.param_real_shape[name] = param.shape
                    param.data = torch.empty(
                        param.shape, dtype=param.dtype, device=param.device
                    )
                    # self.model_create_counter.update(1)

                for name, buf in cls.named_buffers(recurse=False):
                    cls.param_real_shape[name] = buf.shape
                    buf.data = torch.empty(
                        param.shape, dtype=buf.dtype, device=buf.device
                    )
                    # self.model_create_counter.update(1)

            return archer_param_init

        self.cls._old_init = self.cls.__init__
        self.cls.__init__ = init_decorator(self.cls._old_init)

        torch.nn.modules.module.Module._old_apply = (
            torch.nn.modules.module.Module.apply
        )
        torch.nn.modules.module.Module.apply = apply_to_model_decorator(
            torch.nn.modules.module.Module._old_apply
        )

        torch._old_index_select = torch.index_select
        torch.index_select = torch_index_select_decorator(
            torch._old_index_select
        )
        torch.Tensor._old_index_select = torch.Tensor.index_select
        torch.Tensor.index_select = torch_index_select_decorator(
            torch.Tensor._old_index_select
        )

        self.cls._old_post_init = self.cls.post_init
        self.cls.post_init = post_init_decorator(self.cls._old_post_init)
        PreTrainedModel._old_post_init = PreTrainedModel.post_init
        PreTrainedModel.post_init = post_init_decorator(
            PreTrainedModel._old_post_init
        )

        # for all the modules in torch.nn, add post_init method
        # assert False, torch.nn.modules.__dict__
        for name, module in torch.nn.modules.__dict__.items():
            if not isinstance(module, type):
                continue
            if not issubclass(module, torch.nn.modules.module.Module):
                continue
            if name in [
                "Module",
                "Sequential",
                "ModuleDict",
                "ModuleList",
                "ParameterList",
                "ParameterDict",
            ]:
                continue
            module._old_init = module.__init__
            module.__init__ = param_init_decorator(module.__init__)

            if hasattr(module, "reset_parameters"):
                module._old_reset_parameters = module.reset_parameters
                module.reset_parameters = do_nothing_decorator(
                    module.reset_parameters
                )

        def from_pretrained_decorator(
            orig_from_pretrained: Callable,
        ) -> Callable:
            @functools.wraps(orig_from_pretrained)
            def archer_from_pretrained(cls, *args, **kwargs):
                # print("Creating model from scratch ...")

                param_meta_map_file = os.path.join(
                    self.checkpoint, "param_meta_map.json"
                )

                self.dtype_cls = self.config.torch_dtype

                if (
                    not self.ArcherTensorHandle.is_tensor_index_initialized()
                    or not os.path.exists(param_meta_map_file)
                ):
                    print(
                        "Creating model from scratch ...",
                        self.ckpt_files,
                        flush=True,
                    )

                    self.cls.__init__ = self.cls._old_init

                    empty_state_dict = {}
                    for ckpt in tqdm(
                        self.ckpt_files,
                        desc="Loading checkpoint files",
                        smoothing=0,
                    ):
                        state_dict = {}
                        if "safetensors" in ckpt:
                            with safe_open(
                                ckpt, framework="pt", device="cpu"
                            ) as f:
                                for k in f.keys():
                                    state_dict[k] = f.get_tensor(k)
                        else:
                            state_dict = torch.load(ckpt)

                        # convert all tensors in state_dict to self.dtype
                        for k, v in state_dict.items():
                            state_dict[k] = v.to(self.dtype_cls).to("cpu")

                        self._offload_state_dict(state_dict, empty_state_dict)

                        # print("Loading ckpt file", ckpt, flush=True)

                        del state_dict
                        gc.collect()
                        torch.cuda.empty_cache()

                    # update_shm_offsets(self.param_meta_map)

                    with open(param_meta_map_file, "w") as f:
                        json.dump(self.param_meta_map, f)

                else:
                    print("Loading model from offload_path ...", flush=True)
                    self.cls.__init__ = self.cls._old_init
                    # load the name_id_map
                    with open(param_meta_map_file, "r") as f:
                        self.param_meta_map = json.load(f)

                is_flash_attn_available = kwargs.get(
                    "is_flash_attn_available", False
                )

                if (
                    self.dtype_cls is torch.bfloat16
                    or self.dtype_cls is torch.float16
                ):
                    model = cls._from_config(
                        self.config,
                        torch_dtype=self.dtype_cls,
                        attn_implementation=(
                            "flash_attention_2"
                            if is_flash_attn_available
                            else "eager"
                        ),
                    )
                else:
                    model = cls._from_config(self.config)

                base_model_prefix = model.base_model_prefix

                for name, param in model.named_parameters(recurse=True):
                    # remove base_model_prefix from self.param_meta_map
                    if name.startswith(base_model_prefix):
                        name_without_prefix = name[
                            (len(base_model_prefix) + 1) :
                        ]
                        if name_without_prefix in self.param_meta_map:
                            self.param_meta_map[name] = self.param_meta_map[
                                name_without_prefix
                            ]
                            self.param_meta_map.pop(name_without_prefix)
                    param.ar_id = self.param_meta_map.get(name, None)

                self.client = MemoryManagerClient()
                param_shm_map = self.client.get_model_param()
                print(f"param_shm_map: {param_shm_map}")

                for name, param in model.named_parameters(recurse=True):
                    if name not in param_shm_map:
                        print(f"param {name} not found in param_shm_map")
                        continue
                    shm_name, shm_size = param_shm_map[name]
                    tensor = torch.empty(
                        param.data.shape, dtype=param.data.dtype
                    )
                    set_tensor_shm(tensor, shm_name, shm_size)
                    param.data = tensor
                    # print(f"set tensor {name} to shm {shm_name} with size {shm_size}", param.data.size())
                    assert ~(
                        torch.isclose(
                            param.data, torch.zeros_like(param.data)
                        ).all()
                        == True
                    ), f"param {name} is zero {param}"
                    # print(f"param {name} is not zero {param}")

                return model

            return archer_from_pretrained

        self.cls._old_from_pretrained = self.cls.from_pretrained
        self.cls.from_pretrained = classmethod(
            from_pretrained_decorator(self.cls.from_pretrained)
        )

        return self

    # clean up initialization hooks
    def __exit__(self, exc_type, exc_value, traceback):
        self.cls.__init__ = self.cls._old_init
        self.cls.from_pretrained = self.cls._old_from_pretrained
        torch.nn.modules.module.Module.apply = (
            torch.nn.modules.module.Module._old_apply
        )
        torch.index_select = torch._old_index_select
        torch.Tensor.index_select = torch.Tensor._old_index_select

        self.cls.post_init = self.cls._old_post_init
        PreTrainedModel.post_init = PreTrainedModel._old_post_init

        for name, module in torch.nn.modules.__dict__.items():
            if not isinstance(module, type):
                continue
            if not issubclass(module, torch.nn.modules.module.Module):
                continue
            if name in [
                "Module",
                "Sequential",
                "ModuleDict",
                "ModuleList",
                "ParameterList",
                "ParameterDict",
            ]:
                continue
            module.__init__ = module._old_init

            if hasattr(module, "reset_parameters"):
                module.reset_parameters = module._old_reset_parameters

    def _generate_param_id(self):
        param_id = self.param_id
        self.param_id += 1
        return param_id

    def _offload_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        empty_state_dict: Dict[str, torch.Tensor],
    ) -> None:
        param_names = list(state_dict.keys())

        for param_name in param_names:
            if not param_name in self.param_meta_map:
                param = state_dict[param_name]
                param_id = self._generate_param_id()

                file_offset = self.ArcherTensorHandle.offload_tensor(
                    param, param_id
                )

                self.param_meta_map[param_name] = {
                    "id": param_id,
                    "size": param.numel() * param.element_size(),
                    # "shm_offset": -1,
                    "file_offset": file_offset,
                    "shape": tuple(param.shape),
                    "stride": tuple(param.stride()),
                    "dtype": str(param.dtype),
                }

            # if not self.ArcherTensorHandle.is_tensor_offloaded(self.param_meta_map[param_name]["id"]):

        gc.collect()
        torch.cuda.empty_cache()
