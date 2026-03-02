# import os
# import morphling._intercept
import ctypes
import sys

import torch

from morphling.runtime import InitEmptyModel

# from morphling.common import EmulatorConfig


# Get the shared libraries loaded for the imported module (e.g., torch)
def get_loaded_so_libs(module):
    loaded_libs = set()

    # Get the file path of the module
    module_path = module.__file__
    print(f"Module path: {module_path}")

    # List all files in the sys.modules
    for name, mod in sys.modules.items():
        if mod and hasattr(mod, "__file__") and mod.__file__:
            lib_path = mod.__file__
            if lib_path.endswith(".so"):
                loaded_libs.add(lib_path)

    return loaded_libs


# # List all loaded shared libraries (.so files)
# so_libs = get_loaded_so_libs(morphling._C)
# for lib in so_libs:
#     print(f"Loaded .so file: {lib}")

# module_path = morphling._C.__file__
# new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
#     module_path, 'TorchAllocate', 'TorchFree')
# Swap the current allocator
# torch.cuda.memory.change_current_allocator(new_alloc)

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BertModel,
    OPTForCausalLM,
)

model_name = "facebook/opt-125m"
# model_name = "google-bert/bert-base-cased"
model_config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Model name: {model_name}")

initializer = InitEmptyModel(OPTForCausalLM)
with initializer:
    model = OPTForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    )
    model.eval()

print(f"Emulator model loaded", flush=True)

# engine = EmulationEngine(model_config)
# emulator_config = EmulatorConfig(ckpt_path="../checkpoints/opt-125m")

oringinal_model = OPTForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float32
)
oringinal_model.eval()
print(f"Original model loaded", flush=True)
for name, param in oringinal_model.named_parameters():
    print(f"Name: {name}, Param: {hex(param.data.data_ptr())}", flush=True)

# Load the model
# with engine.init(OPTForCausalLM, emulator_config):
#     model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
# model.eval()

input_text = "Hello, my dog is cute."
input_ids = tokenizer(input_text, return_tensors="pt")
print(input_ids)

print(f"Input ids: {hex(input_ids['input_ids'].data_ptr())}")
print(f"Attention mask: {hex(input_ids['attention_mask'].data_ptr())}")

# # Forward pass
outputs = model(**input_ids, return_dict=True)
print("Emulator forward pass done", flush=True)

oringinal_outputs = oringinal_model(**input_ids, return_dict=True)
print("Original forward pass done", flush=True)

print(outputs.logits.dtype)
print(oringinal_outputs.logits.dtype)

print(outputs.logits)
print(oringinal_outputs.logits)
assert torch.allclose(outputs.logits, oringinal_outputs.logits, atol=1e-4)

# for key, value in outputs.items():
#     print(f"Output key: {key}")
#     print(f"Output value: {value}")
#     print(f"Output shape: {value.shape}")

#     if key in oringinal_outputs:
#         print(f"Original output shape: {oringinal_outputs[key].shape}")

#     # check if all close
#     if key in oringinal_outputs:
#         assert torch.allclose(value, oringinal_outputs[key], atol=1e-4)
#     else:
#         print(f"Key {key} not in original output")

# a = torch.ones(2048, 1024, 1024)
# b = torch.ones(2048, 1024, 1024)

# print(a + b)
