import torch
from ..core.config import AsteroidConfig

def prepare_sst2(cfg: AsteroidConfig):
    """Load SST-2 and pre-embed — reused from DT-FM."""
    try:
        from datasets import load_dataset
        from transformers import GPT2Tokenizer, GPT2Model
    except ImportError:
        print("Install: pip install datasets transformers")
        raise

    sst2 = load_dataset("stanfordnlp/sst2")
    train_raw, val_raw = sst2["train"], sst2["validation"]
    num_train = min(4096, len(train_raw))
    num_val = min(872, len(val_raw))

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    hf_gpt2 = GPT2Model.from_pretrained("gpt2")

    embed_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tok_emb = hf_gpt2.wte.to(embed_dev)
    pos_emb = hf_gpt2.wpe.to(embed_dev)

    def embed(texts, labels, max_len=cfg.max_seq_len, bs=256):
        all_e, all_l = [], []
        for i in range(0, len(texts), bs):
            enc = tokenizer(texts[i:i+bs], padding="max_length", truncation=True,
                           max_length=max_len, return_tensors="pt")
            ids = enc["input_ids"].to(embed_dev)
            with torch.no_grad():
                pos = torch.arange(max_len, device=embed_dev).unsqueeze(0)
                e = tok_emb(ids) + pos_emb(pos)
            all_e.append(e.cpu())
            all_l.append(torch.tensor(labels[i:i+bs], dtype=torch.long))
        return torch.cat(all_e), torch.cat(all_l)

    idx = torch.randperm(len(train_raw))[:num_train].tolist()
    tr_e, tr_l = embed([train_raw[i]["sentence"] for i in idx],
                       [train_raw[i]["label"] for i in idx])
    va_e, va_l = embed([val_raw[i]["sentence"] for i in range(num_val)],
                       [val_raw[i]["label"] for i in range(num_val)])

    del hf_gpt2, tok_emb, pos_emb
    torch.cuda.empty_cache()
    return (tr_e, tr_l), (va_e, va_l)