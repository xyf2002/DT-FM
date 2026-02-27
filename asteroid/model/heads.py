import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationHead(nn.Module):
    """Sequence classification: pool hidden states → linear.
    Matches DT-FM's SeqClassification but configurable pooling."""
    def __init__(self, d_model, num_classes, pool="mean"):
        super().__init__()
        self.pool = pool  # "mean" | "first" | "last"
        self.ln_f = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, targets=None):
        x = self.ln_f(x)
        if self.pool == "first":
            pooled = x[:, 0]           # CLS-style (BERT)
        elif self.pool == "last":
            pooled = x[:, -1]          # GPT-style last token
        else:
            pooled = x.mean(dim=1)     # mean pooling (default)
        logits = self.classifier(pooled)
        if targets is not None:
            return F.cross_entropy(logits, targets, reduction='mean')
        return logits

class LMHead(nn.Module):
    """Language modelling head: LayerNorm → linear projection to vocab.
    Matches DT-FM's Seq2SeqClassification with causal LM shift."""
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x, targets=None):
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is not None:
            # Causal LM shift: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_targets.view(-1), reduction='mean')
        return logits