"""Frozen text encoder of the text attacker.

A sentence is represented by the mean of the last-layer token states of the
encoder (padding excluded). The encoder is never trained and runs in float32:
half precision changes the EER of the released heads by several points.
"""
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

EMPTY_TEXT = '.'  # stands in for an empty transcript, as when the released heads were trained


class PooledEncoder:
    def __init__(self, backbone='BAAI/bge-m3', revision=None, device='cpu'):
        """
        Args:
            backbone: Hugging Face model id or local directory
            revision: commit of the model on the hub
            device: device the encoder runs on
        """
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(backbone), revision=revision)
        self.model = AutoModel.from_pretrained(str(backbone), revision=revision, torch_dtype=torch.float32)
        self.model = self.model.to(self.device).eval()

    @torch.inference_mode()
    def pool(self, texts, max_length=96, batch_size=32):
        """Mean-pooled last-layer states of each text, truncated to ``max_length`` tokens.

        Returns:
            float32 array [len(texts), hidden size]
        """
        texts = [t if t.strip() else EMPTY_TEXT for t in texts]
        features = [None] * len(texts)
        order = np.argsort([len(t) for t in texts], kind='stable')  # similar lengths together: little padding
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            batch = self.tokenizer([texts[i] for i in index], padding=True, truncation=True,
                                   max_length=max_length, return_tensors='pt').to(self.device)
            states = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).last_hidden_state
            mask = batch['attention_mask'].unsqueeze(-1).float()
            pooled = ((states * mask).sum(1) / mask.sum(1)).cpu().numpy()
            for i, row in zip(index, pooled):
                features[i] = row
        return np.stack(features)
