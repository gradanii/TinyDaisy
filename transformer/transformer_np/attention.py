import numpy as np

from embeddings import embeddings


class SelfAttention:
    def __init__(self, embed_dim=256, head_dim=64):
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.embed = embeddings()
        self.w_q = np.random.randn(self.embed_dim, self.head_dim)
        self.w_k = np.random.randn(self.embed_dim, self.head_dim)
        self.w_v = np.random.randn(self.embed_dim, self.head_dim)

    @staticmethod
    def softmax(x, axis=-1):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def head(self, string):
        self.embedding = self.embed.forward(string)

        query = self.embedding @ self.w_q
        key = self.embedding @ self.w_k
        value = self.embedding @ self.w_v

        scores = query @ key.T

        scaled_scores = scores / np.sqrt(self.head_dim)

        a = self.softmax(scaled_scores)

        z = a @ value

        return z, a
