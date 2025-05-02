import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizer.tokenizer import BPETokenizer


class embeddings:
    def __init__(self, num_embed=512, embed_dim=26):
        self.tokenizer = BPETokenizer(vocab_size=num_embed)
        self.embeddings = np.random.randn(num_embed, embed_dim)
        self.num_embed = num_embed
        self.embed_dim = embed_dim

    def vector(self, string):
        tokens = np.array(self.tokenizer.encode(string))
        embedding = self.embeddings[tokens]

        return embedding
