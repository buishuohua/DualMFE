import torch
import torch.nn as nn


class MHAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.mha = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self._init_weights()

    def forward(self, x, mask=None):
        out, _ = self.mha(x, x, x, need_weights=False, mask=mask)
        return out

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
