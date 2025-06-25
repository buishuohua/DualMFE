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


class RelativePositionEmbedding(nn.Module):
    def __init__(self, d_emb, max_len=1440):
        super().__init__()
        self.d_emb = d_emb
        self.bias = nn.Parameter(torch.zeros(1, 1, d_emb))
        self._init_weights()

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device)
        pos_matrix = pos.unsqueeze(0) - pos.unsqueeze(1)
        shifted_pos = pos_matrix + seq_len - 1
        biased_pos = shifted_pos + self.bias

        pos_emb = pos / \
            (10000 ** (torch.arange(0, self.d_emb, 2, device=x.device) / self.d_emb))
        pos_emb = torch.cat([torch.sin(pos_emb), torch.cos(pos_emb)], dim=-1)

    def _init_weights(self):
        pass


class SinPositionEmbedding(nn.Module):
    def __init__(self, d_emb):
        super().__init__()
        self.d_emb = d_emb

    def forward(self, x):
        pass

    def _init_weights(self):
        pass
