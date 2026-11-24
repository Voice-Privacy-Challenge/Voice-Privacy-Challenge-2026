"""A tiny tokenizer and encoder, so that tests need neither a download nor a GPU."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    """One token per whitespace-separated word, id = 1 + (sum of code points mod 50); 0 is padding."""

    def __call__(self, texts, padding=True, truncation=True, max_length=96, return_tensors='pt'):
        rows = [[1 + sum(map(ord, word)) % 50 for word in text.split()][:max_length] for text in texts]
        width = max(len(r) for r in rows)
        ids = torch.tensor([r + [0] * (width - len(r)) for r in rows])
        return FakeBatch(input_ids=ids, attention_mask=(ids > 0).long())


class FakeModel(torch.nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        torch.manual_seed(0)
        self.embedding = torch.nn.Embedding(51, hidden)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def save_head(head_dir, head, meta):
    """Write ``head.safetensors`` and ``head.json`` (``meta`` plus the layer sizes) to ``head_dir``."""
    head_dir = Path(head_dir)
    head_dir.mkdir(parents=True, exist_ok=True)
    save_file(head.state_dict(), str(head_dir / 'head.safetensors'))
    sizes = {'hidden': head.project[0].in_features, 'bottleneck': head.project[0].out_features,
             'embedding': head.project[2].out_features}
    with open(head_dir / 'head.json', 'w', encoding='utf-8') as f:
        json.dump({**meta, **sizes}, f)
