import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=768, num_head=12, head_dim=64, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_head = num_head
        self.head_dim = head_dim
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.final = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = 1 / math.sqrt(head_dim)
        self.dropout = nn.Dropout(dropout)

    def sdpa(self, query, key, value):
        B, H, T, D = query.shape
        attn_weight = query @ key.transpose(-2, -1) * self.scale

        causal_mask = torch.tril(torch.ones(T, T, device=query.device)).bool()
        attn_weight.masked_fill_(~causal_mask, float("-inf"))

        attn_weight = self.dropout(torch.softmax(attn_weight, dim=-1))
        return attn_weight @ value

    def forward(self, x):
        B, T = x.shape[:-1]
        linear = self.qkv(x)
        q, k, v = linear.chunk(3, dim=-1)

        q = q.view(*q.shape[:-1], self.num_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.view(*k.shape[:-1], self.num_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.view(*v.shape[:-1], self.num_head, self.head_dim).transpose(0, 2, 1, 3)

        attn_out = self.sdpa(q, k, v)
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(B, T, self.num_head * self.head_dim)
        )

        return self.final(attn_out)
