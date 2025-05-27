import numpy as np
from tokenizer import BPETokenizer


class Config:
    def __init__(self):
        self.tokenizer = BPETokenizer()
        self.num_embed = 512
        self.embed_dim = 768
        self.head_dim = 64
        self.num_head = 12

        # Embedding params
        self.vec_matrix = self.he_init(self.num_embed, self.embed_dim)
        self.pos_matrix = self.he_init(self.num_embed, self.embed_dim)

        # Attention params
        self.w_q = [
            self.he_init(self.embed_dim, self.head_dim) for _ in range(self.num_head)
        ]
        self.w_k = [
            self.he_init(self.embed_dim, self.head_dim) for _ in range(self.num_head)
        ]
        self.w_v = [
            self.he_init(self.embed_dim, self.head_dim) for _ in range(self.num_head)
        ]
        self.w_o = self.he_init(self.head_dim * self.num_head, self.embed_dim)

        self.b_q = np.zeros(
            self.head_dim,
        )
        self.b_k = np.zeros(
            self.head_dim,
        )
        self.b_v = np.zeros(
            self.head_dim,
        )

        # LayerNorm params
        self.gamma = np.ones(self.embed_dim)
        self.beta = np.zeros(self.embed_dim)

    @staticmethod
    def he_init(dim_in, dim_out):
        std = np.sqrt(1 / dim_in)
        return np.random.randn(dim_in, dim_out) * std
