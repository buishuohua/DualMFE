import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.components.emb import LinearEmbedding
from models.components.attn import MHAttention
from models.components.gru import GRUBlock
from configs.modelConfig import ModelConfig


class iTransformer(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class baseGRU(nn.Module):
    def __init__(self, modelconfig: ModelConfig):
        super().__init__()
        d_input = modelconfig.d_input
        n_layers = modelconfig.n_layers
        dropout = modelconfig.dropout
        bidirection = modelconfig.bidirection
        activation = modelconfig.get_activation()
        d_model = modelconfig.d_model
        d_ff = modelconfig.d_ff
        ln_eps = modelconfig.ln_eps

        self.n_blocks = modelconfig.n_blocks
        self.gru_blocks = nn.ModuleList(
            [GRUBlock(d_model, n_layers, dropout, bidirection, d_ff, activation, ln_eps) for _ in range(self.n_blocks)])
        self.emb = LinearEmbedding(d_input, d_model)

        self.layernorm = nn.LayerNorm(d_model, ln_eps)
        self.fc = nn.Linear(d_model, 1)
        self._init_weights()

    def forward(self, x):
        x = self.emb(x)
        for i, gru_block in enumerate(self.gru_blocks):
            x = gru_block(x)
        x = self.layernorm(x)
        out = self.fc(x).squeeze(-1)
        return out

    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)


class vTransformer(nn.Module):
    def __init__(self, modelconfig: ModelConfig):
        super().__init__()
        d_input = modelconfig.d_input
        d_model = modelconfig.d_model
        n_blocks = modelconfig.n_blocks
        n_heads = modelconfig.n_heads
        dropout = modelconfig.dropout
        activation = modelconfig.get_activation()

        self.emb = LinearEmbedding(d_input, d_model)
        self.layernorm = nn.LayerNorm(d_model)
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model, dropout=dropout, batch_first=True, activation=activation, layer_norm_eps=1e-5)
        self.transformer = TransformerEncoder(encoder_layers, n_blocks)
        self.fc = nn.Linear(d_model, 1)
        self._init_weights()

    def forward(self, x):
        x = self.emb(x)
        x = self.layernorm(x)
        out = self.transformer(x)

    def _init_weights(self):
        for name, params in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(params)
            elif 'bias' in name:
                nn.init.constant_(params, 0)
