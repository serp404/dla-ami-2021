import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, maxlen: int = 1000) -> None:
        super().__init__()
        multiplier = math.log(10000) / emb_size
        den = torch.exp(-torch.arange(0, emb_size, 2) * multiplier)
        pos = torch.arange(0, maxlen).unsqueeze(dim=1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(dim=0)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        seq_len = token_embedding.shape[1]
        return token_embedding + self.pos_embedding[:, :seq_len, :]
