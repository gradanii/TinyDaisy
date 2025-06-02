import torch.nn as nn
from .attention import MultiHeadAttention
from .embeddings import TokenEmbedding


class FeedForward(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim
        self.l1 = nn.Linear(embed_dim, 4 * embed_dim, bias=True)
        self.l2 = nn.Linear(4 * embed_dim, embed_dim, bias=True)
        self.gelu = nn.GELU()

    def forward(self, x):
        l1 = self.gelu(self.l1(x))
        return self.l2(l1)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim=embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim=embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, num_blocks=12, embed_dim=768, vocab_size=4096, max_len=512):
        super().__init__()
        self.embed = TokenEmbedding(
            num_embed=vocab_size, embed_dim=embed_dim, max_len=max_len
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim=embed_dim) for _ in range(num_blocks)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)
