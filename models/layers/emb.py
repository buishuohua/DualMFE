import torch
import torch.nn as nn
import math

class FeatureLinearEmbedding(nn.Module):
    def __init__(self, d_feature, d_emb, dropout):
        super().__init__()
        self.linear = nn.Linear(d_feature, d_emb)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def forward(self, x):
        return self.dropout(self.linear(x))

    def _init_weights(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)


class SinCosPositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=2880):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class SequenceLinearEmbedding(nn.Module):
    def __init__(self, seq_len, d_emb, dropout):
        super().__init__()
        self.linear = nn.Linear(seq_len, d_emb)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def forward(self, x):
        return self.dropout(self.linear(x))

    def _init_weights(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)
