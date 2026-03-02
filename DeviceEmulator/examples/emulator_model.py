from transformers import AutoModelForCausalLM, AutoTokenizer

from morphling import EmulatorEngine

model_name = "facebook/opt-125m"

engine = EmulatorEngine()

with engine.init():
    model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)

input_text = "Hello, my dog is cute."
input_ids = tokenizer(input_text, return_tensors="pt")
print(input_ids)

model = model.apply_parallelization()

# emulator and parallelism happen in background
outputs = model(**input_ids)
