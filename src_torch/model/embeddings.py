import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, num_embed=4096, embed_dim=768, max_len=512, dropout=0.01):
        super().__init__()

        self.vec_matrix = nn.Embedding(num_embed, embed_dim)
        self.pos_matrix = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        b, t = inputs.shape
        vec_embed = self.vec_matrix(inputs)
        x = vec_embed + self.pos_matrix[:, :t, :]

        return self.dropout(x)
