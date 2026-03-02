import torch
from transformers import AutoConfig, AutoModelForCausalLM


def torch_decorator(func, funcname=None):
    def wrapper(*args, **kwargs):
        print("torch_decorator ", funcname)
        return func(*args, **kwargs)

    return wrapper


a = torch.ones((1, 3))
b = torch.ones((1, 3))


pyobj_keywords = [
    "__class__",
    "__delattr__",
    "__dict__",
    "__dir__",
    "__doc__",
    "__eq__",
    "__format__",
    "__ge__",
    "__getattribute__",
    "__gt__",
    "__hash__",
    "__init__",
    "__init_subclass__",
    "__le__",
    "__lt__",
    "__module__",
    "__ne__",
    "__new__",
    "__reduce__",
    "__reduce_ex__",
    "__repr__",
    "__setattr__",
    "__sizeof__",
    "__str__",
    "__subclasshook__",
    "__weakref__",
    "Tensor",
]


model_name = "facebook/opt-1.3b"
config = AutoConfig.from_pretrained(model_name)
print(config)
config.hidden_size = 1
config.num_attention_heads = 1
config.word_embed_proj_dim = 1
model = AutoModelForCausalLM.from_config(config)

for name, param in model.named_parameters():
    # print(name)
    if "embed" in name:
        print("param byte size: ", param.numel() * param.element_size())
        continue

    shape = param.shape

    # set shape size to all 1s
    shape = [1 for _ in shape]
    param.data = torch.rand(shape)


# for all functions in torch.Tensor, add decorator
functions = [f for f in dir(torch.Tensor) if callable(getattr(torch.Tensor, f))]
for f in functions:
    if f in pyobj_keywords:
        continue
    func = getattr(torch.Tensor, f)
    setattr(torch.Tensor, f, torch_decorator(func, f))

# for all functions in torch, add decorator
functions = [f for f in dir(torch) if callable(getattr(torch, f))]
for f in functions:
    if f in pyobj_keywords:
        continue
    func = getattr(torch, f)
    setattr(torch, f, torch_decorator(func, f))

inputs = torch.zeros((1, 1), dtype=torch.long)
output = model.generate(
    inputs, do_sample=True, max_length=50, pad_token_id=50256
)

# print(output)
