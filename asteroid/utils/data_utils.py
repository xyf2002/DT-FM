import torch
from ..core.config import AsteroidConfig


def prepare_data(cfg: AsteroidConfig):
    """Unified data preparation: dispatches based on task_type."""
    if cfg.task_type == "lm":
        return prepare_lm(cfg)
    else:
        return prepare_sst2(cfg)


def prepare_lm(cfg: AsteroidConfig):
    """
    Prepare causal-LM data from SST-2 sentences (or any text dataset).

    Returns token-ID tensors:
        train_data = (input_ids, target_ids)   shape (N, seq_len)  dtype long
        val_data   = (input_ids, target_ids)   shape (M, seq_len)  dtype long

    For causal LM the target is the *same* sequence — the LMHead does the
    internal shift (logits[:,:-1] vs targets[:,1:]).
    """
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError:
        print("Install: pip install datasets transformers")
        raise

    # Use GPT-2 tokenizer (vocab_size=50257) for all from-scratch training
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load raw text — SST-2 sentences concatenated for LM
    sst2 = load_dataset("stanfordnlp/sst2")
    train_texts = [row["sentence"] for row in sst2["train"]]
    val_texts   = [row["sentence"] for row in sst2["validation"]]

    def texts_to_chunks(texts, seq_len, max_chunks=0):
        """Tokenize texts, concatenate, split into fixed-length chunks."""
        all_ids = []
        for t in texts:
            ids = tokenizer.encode(t, add_special_tokens=False)
            all_ids.extend(ids)
            all_ids.append(tokenizer.eos_token_id)  # sentence separator
        # Trim to full chunks
        n = (len(all_ids) // seq_len) * seq_len
        all_ids = all_ids[:n]
        chunks = torch.tensor(all_ids, dtype=torch.long).view(-1, seq_len)
        if max_chunks > 0 and chunks.size(0) > max_chunks:
            chunks = chunks[:max_chunks]
        return chunks

    seq_len = cfg.max_seq_len
    train_ids = texts_to_chunks(train_texts, seq_len, max_chunks=8192)
    val_ids   = texts_to_chunks(val_texts,   seq_len, max_chunks=1024)

    print(f"  LM data: train={train_ids.shape}, val={val_ids.shape}, "
          f"vocab={tokenizer.vocab_size}")

    # For causal LM: input == target (head shifts internally)
    return (train_ids, train_ids.clone()), (val_ids, val_ids.clone())


def prepare_sst2(cfg: AsteroidConfig):
    """Load SST-2 and pre-embed — reused from DT-FM (classification task)."""
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