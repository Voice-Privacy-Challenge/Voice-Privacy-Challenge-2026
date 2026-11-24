"""Speaker head of the text attacker.

The head maps a pooled sentence feature of the frozen text encoder to a
256-dimensional speaker embedding. A released head is a directory with
    head.safetensors   the projection
    head.json          architecture, text convention and provenance
"""
import json
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class Head(nn.Module):
    def __init__(self, hidden=1024, bottleneck=512, embedding=256):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(hidden, bottleneck), nn.ReLU(), nn.Linear(bottleneck, embedding))

    def embed(self, pooled):
        """Pooled encoder features [N, hidden] -> unit-length speaker embeddings [N, embedding]."""
        return F.normalize(self.project(pooled), dim=-1)


def load_head(head_dir, device='cpu'):
    """Load a head in evaluation mode.

    Returns:
        (Head, content of head.json)
    """
    head_dir = Path(head_dir)
    with open(head_dir / 'head.json', encoding='utf-8') as f:
        meta = json.load(f)
    head = Head(hidden=meta['hidden'], bottleneck=meta['bottleneck'], embedding=meta['embedding'])
    head.load_state_dict(load_file(str(head_dir / 'head.safetensors')))
    return head.to(device).eval(), meta
