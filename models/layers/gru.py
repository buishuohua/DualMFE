import torch.nn as nn


class GRUBlock(nn.Module):
    def __init__(self, d_model, n_layers, dropout, d_ff, activation, ln_eps):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.dropout = dropout
        self.gru = nn.GRU(d_model, d_model, n_layers,
                          batch_first=True, dropout=dropout, bidirectional=False)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            activation,
            nn.Linear(d_ff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model, eps=ln_eps)
        self.layernorm2 = nn.LayerNorm(d_model, eps=ln_eps)
        self._init_weights()

    def forward(self, x):
        residual_x = x
        x = self.layernorm1(x)
        gru_output, _ = self.gru(x)
        x = residual_x + gru_output
        residual_gru = x
        x = self.layernorm2(x)
        ff_out = self.ff(x)
        out = residual_gru + ff_out
        return out

    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        for name, param in self.ff.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
