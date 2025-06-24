import torch
import torch.nn as nn


class LinearEmbedding(nn.Module):
    def __init__(self, d_input, d_emb):
        super().__init__()
        self.linear = nn.Linear(d_input, d_emb)
        self._init_weights()

    def forward(self, x):
        return self.linear(x)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)


class PositionEmbedding(nn.Module):
    def __init__(self, d_emb):
        super().__init__()
        self.d_emb = d_emb

    def forward(self, x):
        pass

    def _init_weights(self):
        pass
