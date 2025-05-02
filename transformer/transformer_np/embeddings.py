import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizer.tokenizer import BPETokenizer


class embeddings:
    def __init__(self, num_embed=512, embed_dim=26, max_seq_len=256):
        self.tokenizer = BPETokenizer(vocab_size=num_embed)
        self.vec_matrix = np.random.randn(num_embed, embed_dim)
        self.pos_matrix = np.random.randn(max_seq_len, embed_dim)
        self.num_embed = num_embed
        self.embed_dim = embed_dim

    def vector(self, string):
        tokens = np.array(self.tokenizer.encode(string), dtype=np.int64)
        seq_len = len(tokens)
        vec_embed = self.vec_matrix[tokens]

        return vec_embed, seq_len

    def position(self, seq_len):
        pos_array = np.array([i for i in range(seq_len)], dtype=np.int64)
        pos_embed = self.pos_matrix[pos_array]

        return pos_embed

    def forward(self, string):
        vec_embed, seq_len = self.vector(string)
        pos_embed = self.position(seq_len)

        embedding = vec_embed + pos_embed

        return embedding
