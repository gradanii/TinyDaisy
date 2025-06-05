import numpy as np
from .tokenizer import BPETokenizer


class Config:
    def __init__(self):
        self.num_embed = 512
        self.embed_dim = 768
        self.head_dim = 64
        self.num_head = 12
        self.num_blocks = 12
        self.vocab_size = 4096
        self.tokenizer = BPETokenizer(vocab_size=self.vocab_size)

        # Embedding params
        self.vec_matrix = self.he_init(self.num_embed, self.embed_dim)
        self.pos_matrix = self.he_init(self.num_embed, self.embed_dim)

        # Attention params
        self.w_q = self.he_init(self.embed_dim, self.embed_dim)
        self.w_k = self.he_init(self.embed_dim, self.embed_dim)
        self.w_v = self.he_init(self.embed_dim, self.embed_dim)
        self.w_o = self.he_init(self.head_dim * self.num_head, self.embed_dim)

        self.b_q = np.zeros(
            self.embed_dim,
        )
        self.b_k = np.zeros(
            self.embed_dim,
        )
        self.b_v = np.zeros(
            self.embed_dim,
        )

        # LayerNorm params
        self.gamma = np.ones(self.embed_dim)
        self.beta = np.zeros(self.embed_dim)

        # MLP params
        self.w_l1 = self.he_init(self.embed_dim, 4 * self.embed_dim)
        self.w_l2 = self.he_init(4 * self.embed_dim, self.embed_dim)

        self.b_l1 = np.zeros(
            4 * self.embed_dim,
        )
        self.b_l2 = np.zeros(
            self.embed_dim,
        )

        # Forward params
        self.w_voc = self.he_init(self.embed_dim, self.vocab_size)
        self.b_voc = np.zeros(
            self.vocab_size,
        )

    @staticmethod
    def he_init(dim_in, dim_out):
        std = np.sqrt(1 / dim_in)
        return np.random.randn(dim_in, dim_out) * std
