import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from models.components.emb import LinearEmbedding, RelativePositionEmbedding
from models.components.gru import GRUBlock
from configs.modelConfig import ModelConfig

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
        d_ff = modelconfig.d_ff
        n_blocks = modelconfig.n_blocks
        n_heads = modelconfig.n_heads
        dropout = modelconfig.dropout
        activation = modelconfig.get_activation()
        ln_eps = modelconfig.ln_eps

        self.emb = LinearEmbedding(d_input, d_model)
        # self.r_pos_emb = RelativePositionEmbedding(d_model)
        self.layernorm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation=activation,
            layer_norm_eps=ln_eps,
            norm_first=True
        )
        self.transformer = TransformerDecoder(decoder_layer, n_blocks)
        self.fc = nn.Linear(d_model, 1)
        self._init_weights()

    def forward(self, x):
        seq_len = x.size(1)
        casual_mask = torch.triu(torch.ones(
            (seq_len, seq_len), device=x.device) * float('-inf'), diagonal=1)

        l_emb_x = self.emb(x)
        pos_emd_x = torch.zeros_like(l_emb_x)
        embed_x = l_emb_x + pos_emd_x
        repr_x = self.transformer(
            tgt=embed_x, memory=embed_x, tgt_mask=casual_mask, memory_mask=casual_mask)

        out = self.fc(repr_x).squeeze(-1)
        return out

    def _init_weights(self):
        # TODO: More brilliant way
        for name, params in self.named_parameters():
            if 'weight' in name and len(params.shape) >= 2:
                nn.init.kaiming_normal_(params)
            elif 'bias' in name or len(params.shape) == 1:
                nn.init.constant_(params, 0)


class iTransformer(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass
