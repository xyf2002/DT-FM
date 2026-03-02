import random

import numpy as np
import torch
from transformers import AdamW, GPT2LMHeadModel, GPT2Tokenizer, set_seed

# Set the seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
set_seed(seed)

# Ensure deterministic operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.set_default_dtype(torch.float32)

print("Verified default dtype:", torch.get_default_dtype())

device = torch.device("cpu")

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)


def save_initial_weights(model, filename):
    initial_weights = (
        model.transformer.h[0].attn.c_attn.weight.data.cpu().numpy()
    )
    np.save(filename, initial_weights)
    print(f"Initial weights saved to {filename}")


if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))

input_text = "Hello, how are you? " * 100
inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)

inputs["input_ids"] = inputs["input_ids"].repeat(16, 1)
inputs["attention_mask"] = inputs["attention_mask"].repeat(16, 1)

batch_size, sequence_length = inputs["input_ids"].shape
print("Batch size:", batch_size)
print("Sequence length:", sequence_length)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

num_epochs = 1
steps_per_epoch = 10

for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1}/{num_epochs}")

    for step in range(steps_per_epoch):
        optimizer.zero_grad()

        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

        print(f"Step {step + 1}/{steps_per_epoch}, Loss: {loss.item()}")

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs} completed")

print(
    f"Training completed with Batch size: {batch_size}, Sequence length: {sequence_length}"
)
