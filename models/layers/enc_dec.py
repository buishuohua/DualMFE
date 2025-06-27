import torch
import torch.nn as nn
import torch.functional as F


class iTransformer_EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, dropout, activation):
        super().__init__()
        d_ff = d_ff
        self.attention = attention
        self.activation = activation
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None, tau=None):
        residual = x
        x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau
        ) # (Batch, D_feature, D_model)
        x = residual + self.dropout(x)
        x = self.norm1(x)

        residual_ffn = x
        x = self.ffn(x)
        x = residual_ffn + self.dropout(x)
        x = self.norm2(x)
        return x, attn


class iTransformer_Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(
            conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None):
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                x, attn = attn_layer(
                    x, attn_mask=attn_mask, tau=tau)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(
                    x, attn_mask=attn_mask, tau=tau)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
