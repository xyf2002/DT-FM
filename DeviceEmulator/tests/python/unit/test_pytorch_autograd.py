import torch
from transformers import AutoConfig
from transformers.models.opt.modeling_opt import OPTDecoderLayer

from morphling.hooks import *

model_name = "facebook/opt-125m"


config = AutoConfig.from_pretrained(model_name)
layer = OPTDecoderLayer(config)

torch.bmm = LinearFunction.apply


# decorator for a class method, add arguments for the function at position 0
def add_arguments(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


torch.Tensor.__add__ = add_arguments(AddFunction.apply)
# linear = LinearFunction.apply

# # replace pytorch layer norm with custom function
# torch.nn.LayerNorm = LayerNormFunction
torch.nn.functional.layer_norm = LayerNormFunction.apply

inputs = torch.randn(1, 1, 768)
output = layer(inputs)
hidden_states = output[0]

labels = torch.randn(hidden_states.shape)
loss = torch.nn.functional.mse_loss(hidden_states, labels)
loss.backward()
