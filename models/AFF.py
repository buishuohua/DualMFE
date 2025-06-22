import torch
import torch.nn as nn

class iTransformers(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

class demoGRU(nn.Module):
    def __init__(self, d_input, d_hidden, n_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(d_input, d_hidden, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(d_hidden, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        labels = []
        for t_idx in range(out.size(1)):
            lable = self.fc(out[:, t_idx, :])
            labels.append(lable)
        out = torch.stack(labels, dim=1)
        return out
