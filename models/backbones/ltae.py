import copy

import numpy as np
import torch
import torch.nn as nn

from src.backbones.positional_encoding import PositionalEncoder


class LTAE2d(nn.Module):
    def __init__(self, in_channels=128, n_head=16, d_k=4, mlp=[256, 128], dropout=0.2, d_model=256, T=1000, return_att=False, positional_encoding=False):
        super(LTAE2d, self).__init__()
        self.in_channels = in_channels
        self.mlp = copy.deepcopy(mlp)
        self.return_att = return_att
        self.n_head = n_head

        self.in_norm = nn.LayerNorm(self.in_channels) # nn.GroupNorm(num_groups=n_head, num_channels=self.in_channels)

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
            self.innorm = nn.LayerNorm(d_model)
        else:
            self.d_model = in_channels
            self.inconv = None

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(self.d_model // n_head, T=T, repeat=n_head)
        else:
            self.positional_encoder = None

        self.attention_heads = MultiHeadAttention(n_head=n_head, d_k=d_k, d_in=self.d_model)
        # self.out_norm = nn.GroupNorm(num_groups=n_head, num_channels=mlp[-1])

        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend(
                [nn.Linear(self.mlp[i], self.mlp[i + 1]),
                 nn.BatchNorm1d(self.mlp[i + 1]),
                 nn.ReLU()]
            )
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(mlp[-1]) # nn.GroupNorm(num_groups=n_head, num_channels=self.in_channels)

    def forward(self, x, batch_positions=None):
        sz_b, d, seq_len, h, w = x.shape

        out = x.permute(0, 3, 4, 2, 1).contiguous().view(sz_b*h*w, seq_len, d)
        out = self.in_norm(out)

        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)
            out = self.innorm(out)

        if self.positional_encoder is not None:
            bp = (batch_positions.unsqueeze(-1).repeat((1, 1, h)).unsqueeze(-1).repeat((1, 1, 1, w)))  # BxTxHxW
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            out = out + self.positional_encoder(bp)
            out = self.innorm(out)

        out, attn = self.attention_heads(out)

        out = (out.permute(1, 0, 2).contiguous().view(sz_b*h*w, -1))  # Concatenate heads
        out = self.dropout(self.mlp(out))
        out = self.out_norm(out) if self.out_norm is not None else out
        out = out.view(sz_b, h, w, -1).permute(0, 3, 1, 2)

        attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(0, 1, 4, 2, 3)  # head x b x t x h x w

        if self.return_att:
            return out, attn
        else:
            return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        # self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        # nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))
        self.fc1_q = nn.Linear(d_in, n_head*d_k)
        nn.init.normal_(self.fc1_q.weight, mean=0, std=np.sqrt(2.0/(d_k)))
        self.fc2 = nn.Sequential(nn.BatchNorm1d(n_head*d_k), nn.Linear(n_head*d_k, n_head*d_k))

        self.fc1_k = nn.Linear(d_in, n_head*d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0/(d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, v):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = v.size()

        # q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(-1, d_k)  # (n*b) x d_k
        q = self.fc1_q(v).view(sz_b, seq_len, n_head, d_k)
        q = q.mean(dim=1).squeeze()  # MEAN query
        q = self.fc2(q.view(sz_b, n_head * d_k)).view(sz_b, n_head, d_k)
        q = q.permute(1, 0, 2).contiguous().view(n_head * sz_b, d_k)

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        # v = torch.stack(v.split(v.shape[-1]//n_head, dim=-1)).view(n_head*sz_b, seq_len, -1)
        v = v.repeat(n_head, 1, 1)

        output, attn = self.attention(q, k, v)

        attn = attn.view(n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2)

        output = output.view(n_head, sz_b, 1, d_in).contiguous()
        # output = output.view(n_head, sz_b, 1, d_in)
        output = output.squeeze(dim=2)
        return output, attn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        return output, attn
