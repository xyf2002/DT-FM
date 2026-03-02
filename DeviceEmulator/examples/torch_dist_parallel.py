# example for torch DDP + DPP

import argparse
import os

import torch
import torch.distributed as dist
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.pipelining import (
    PipelineStage,
    ScheduleGPipe,
    SplitPoint,
    pipeline,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoModelForCausalLM, AutoTokenizer

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
# device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
device = torch.device("cpu")
torch.distributed.init_process_group(
    rank=rank, world_size=world_size, backend="gloo", init_method="env://"
)

print(f"rank = {rank}, world_size = {world_size}, device = {device}")

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)

args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model)

state_dict = model.state_dict()
print(f"model.state_dict() = {state_dict.keys()}")

tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.pad_token = tokenizer.eos_token
mb_prompts = (
    "How do you",
    "I like to",
)  # microbatch size = 2

# Cut model by equal number of layers per rank
layers_per_rank = model.config.num_hidden_layers // world_size
print(f"layers_per_rank = {layers_per_rank}")
split_spec = {
    f"model.decoder.layers.{i * layers_per_rank}": SplitPoint.BEGINNING
    for i in range(1, world_size)
}

# Create a pipeline representation from the model
mb_inputs = tokenizer(mb_prompts, return_tensors="pt", padding=True).to(device)
pipe = pipeline(model, mb_args=(mb_inputs["input_ids"],), split_spec=split_spec)

# Create pipeline stage for each rank
stage = pipe.build_stage(rank, device=device)
print("Pipeline stage created")

# Run time inputs
full_batch_prompts = (
    "How do you",
    "I like to",
    "Can I help",
    "You need to",
    "The weather is",
    "I found a",
    "What is your",
    "You are so",
)  # full batch size = 8
inputs = tokenizer(full_batch_prompts, return_tensors="pt", padding=True).to(
    device
)

# Attach to a schedule
# number of microbatches = 8 // 2 = 4
num_mbs = 4
schedule = ScheduleGPipe(stage, num_mbs)
print("Schedule created")

# Run
if rank == 0:
    args = inputs["input_ids"]
else:
    args = None

output = schedule.step(args)
print("Schedule step done")

# Decode
if output is not None:
    next_token_logits = output[0][:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)
    print(tokenizer.batch_decode(next_token))
