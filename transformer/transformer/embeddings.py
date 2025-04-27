import sys
import os
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from tokenizer.tokenizer import BPETokenizer


class embeddings:
    def __init__(self, num_embed=512, embed_dim=26):
        self.tokenizer = BPETokenizer(vocab_size=num_embed)
        self.embeddings = nn.Embedding(num_embed, num_embed)
        self.num_embed = num_embed
        self.embed_dim = embed_dim

    def vector(self, string):
        tokens = torch.tensor(self.tokenizer.encode(string), dtype=torch.long)
        vec_embed = self.embeddings(tokens)

        return vec_embed
