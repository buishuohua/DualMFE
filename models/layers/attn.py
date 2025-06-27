import torch
import torch.nn as nn


class FullAttention(nn.Module):
    def __init__(self, mask_flag, n_heads, attn_dropout, scale=None):
        super().__init__()
        self.mask_flag = mask_flag
        self.n_heads = n_heads
        self.attn_dropout = attn_dropout
        self.dropout = nn.Dropout(attn_dropout)
        self.scale = scale

    def forward(self, q, k, v, attn_mask=None, tau=None):
        batch, d_feature, d_model = q.size()
        d_head = d_model // self.n_heads

        q, k, v = q.view(batch, d_feature, self.n_heads, d_head), k.view(batch, d_feature, self.n_heads, d_head), v.view(batch, d_feature, self.n_heads, d_head)
        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)

        if self.scale is None:
            self.scale = torch.sqrt(torch.tensor(d_head, dtype=torch.float32, device=q.device))
        scores = torch.matmul(q, k.transpose(-1, -2))
        scores /= self.scale

        if self.mask_flag:
            if attn_mask is None:
                #TODO：Mask not compatible with temporal
                attn_mask = torch.triu(torch.ones((d_feature, d_feature), dtype=torch.bool, device=q.device), diagonal=1)
            scores = scores.masked_fill_(attn_mask, float('-inf'))

        if tau is not None:
            scores = scores / tau

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch, d_feature, -1)

        return out, attn
