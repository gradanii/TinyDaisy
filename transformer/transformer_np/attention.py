import numpy as np

from embeddings import embeddings


class attention:
    def __init__(self, embed_dim=256, head_dim=64):
        self.embed_dim = embed_dim
        self.head_dim = head_dim

    def self_attention(self, string):
        w_q = np.random.randn(self.embed_dim, self.head_dim)
        w_k = np.random.randn(self.embed_dim, self.head_dim)
        w_v = np.random.randn(self.embed_dim, self.head_dim)

        embed = embeddings()
        embedding = embed.forward(string)

        query = embedding @ w_q
        key = embedding @ w_k
        value = embedding @ w_v
