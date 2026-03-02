import sys

import torch
from transformers import AutoModelForCausalLM

sys.path.append("../../")

from morphling.common.decorators import func_access_decorator
from morphling.common.keywords import *

model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(model_name)

inputs = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
labels = torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]])


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
optimizer.zero_grad()

# for all functions in torch.Tensor, add decorator
functions = [f for f in dir(torch.Tensor) if callable(getattr(torch.Tensor, f))]
for f in functions:
    if f in pyobj_keywords or f in pytorch_keywords:
        continue
    func = getattr(torch.Tensor, f)
    setattr(torch.Tensor, f, func_access_decorator(func, f))

# for all functions in torch, add decorator
functions = [f for f in dir(torch) if callable(getattr(torch, f))]
for f in functions:
    if f in pyobj_keywords or f in pytorch_keywords:
        continue
    func = getattr(torch, f)
    setattr(torch, f, func_access_decorator(func, f))

model.train()
outputs = model(inputs, labels=labels)
loss = outputs.loss
print(outputs.loss)

loss.backward()
# optimizer.step()
