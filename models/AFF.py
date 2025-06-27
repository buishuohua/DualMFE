import torch
import torch.nn as nn
from models.layers.emb import FeatureLinearEmbedding, SinCosPositionEmbedding, SequenceLinearEmbedding
from models.layers.gru import GRUBlock
from models.layers.attn import FullAttention
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.layers.enc_dec import iTransformer_Encoder, iTransformer_EncoderLayer
from configs.modelConfig import ModelConfig

class baseGRU(nn.Module):
    def __init__(self, modelconfig: ModelConfig):
        super().__init__()
        d_feature = modelconfig.d_feature
        n_layers = modelconfig.n_blocks
        dropout = modelconfig.dropout
        activation = modelconfig.get_activation()
        d_model = modelconfig.d_model
        d_ff = modelconfig.d_ff
        ln_eps = modelconfig.ln_eps

        self.n_blocks = modelconfig.n_blocks
        self.emb = FeatureLinearEmbedding(d_feature, d_model, dropout)
        self.gru_blocks = nn.ModuleList(
            [GRUBlock(d_model, n_layers, dropout, d_ff, activation, ln_eps) for _ in range(self.n_blocks)])

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
        d_feature = modelconfig.d_feature
        d_model = modelconfig.d_model
        d_ff = modelconfig.d_ff
        n_blocks = modelconfig.n_blocks
        n_heads = modelconfig.n_heads
        dropout = modelconfig.dropout
        attn_dropout = modelconfig.attn_dropout
        activation = modelconfig.get_activation()
        ln_eps = modelconfig.ln_eps

        self.emb = FeatureLinearEmbedding(d_feature, d_model, dropout)
        self.sin_cos_pos_emb = SinCosPositionEmbedding(d_model)
        self.layernorm = nn.LayerNorm(d_model)
        self.encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=attn_dropout,
            batch_first=True,
            activation=activation,
            layer_norm_eps=ln_eps,
            norm_first=False
        )
        self.encoder = TransformerEncoder(self.encoder_layer, n_blocks)
        self.fc = nn.Linear(d_model, 1)
        self._init_weights()

    def forward(self, x):
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(
            (seq_len, seq_len), device=x.device) * float('-inf'), diagonal=1)

        l_emb_x = self.emb(x)
        pos_emd_x = self.sin_cos_pos_emb(l_emb_x)
        embed_x = l_emb_x + pos_emd_x
        repr_x = self.encoder(src=embed_x, mask=mask)

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
    def __init__(self, modelconfig: ModelConfig):
        super().__init__()
        seq_len = modelconfig.seq_len
        d_model = modelconfig.d_model
        d_feature = modelconfig.d_feature
        d_ff = modelconfig.d_ff
        n_blocks = modelconfig.n_blocks
        n_heads = modelconfig.n_heads
        dropout = modelconfig.dropout
        attn_dropout = modelconfig.attn_dropout
        activation = modelconfig.get_activation()
        mask_flag = modelconfig.mask_flag
        ln_eps = modelconfig.ln_eps

        self.emb = SequenceLinearEmbedding(seq_len, d_model, dropout)
        self.layernorm = nn.LayerNorm(d_model, ln_eps)
        self.encoder = iTransformer_Encoder(
            attn_layers=[
                iTransformer_EncoderLayer(
                    FullAttention(mask_flag, n_heads, attn_dropout),
                    d_model, d_ff, dropout, activation
                ) for _ in range(n_blocks)
            ],
            conv_layers=None,
            norm_layer=self.layernorm
        )
        self.fc1 = nn.Linear(d_model, seq_len)
        self.fc2 = nn.Linear(d_feature, 1)
        self._init_weights()


    def forward(self, x):
        x = x.permute(0, 2, 1)  # (Batch, Feature, SeqLen)
        embed_x = self.emb(x)  # (Batch, Feature, D_model)
        repr_x, _ = self.encoder(embed_x)

        out = self.fc1(repr_x).squeeze(-1)  # (Batch, Feature, SeqLen)
        out = self.fc2(out.permute(0, 2, 1)).squeeze(-1)
        return out

    def _init_weights(self):
        for _, params in self.named_parameters():
            if isinstance(params, nn.Linear):
                nn.init.kaiming_normal_(params.weight)
                nn.init.constant_(params.bias, 0)
