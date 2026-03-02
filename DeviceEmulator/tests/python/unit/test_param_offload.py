import numpy as np
from transformers import AutoConfig, OPTForCausalLM

from morphling.common import EmulatorConfig
from morphling.runtime import EmulationEngine

model_name = "facebook/opt-125m"
config = AutoConfig.from_pretrained(model_name)

engine = EmulationEngine(config)
engine_config = EmulatorConfig(
    gpu_memory=0.5, cpu_memory=0.5, ckpt_path="../checkpoints"
)

with engine.init(OPTForCausalLM, engine_config):
    model = OPTForCausalLM.from_pretrained(model_name)

for name, param in model.named_parameters():
    assert np.prod(param.shape) == 1, f"Param {name} has shape {param.shape}"
