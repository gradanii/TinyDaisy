import numpy as np

from embeddings import embeddings


def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


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

        scores = query @ key.T

        scaled_scores = scores / np.sqrt(self.head_dim)

        a = softmax(scaled_scores)

        z = a @ value
