import numpy as np
from tokenizer import BPETokenizer


class Embedding:
    def __init__(self, num_embed=512, embed_dim=768):
        self.tokenizer = BPETokenizer()
        self.vec_matrix = np.random.randn(num_embed, embed_dim)
        self.pos_matrix = np.random.randn(num_embed, embed_dim)

    def vector(self, list_of_strings):
        batch = []
        lengths = []
        tokens = [self.tokenizer.encode(s) for s in list_of_strings]
        max_len = max(len(t) for t in tokens)
        for token in tokens:
            padded = np.pad(token, (0, max_len - len(token)), constant_values=0)
            embedding = self.vec_matrix[padded]
            batch.append(embedding)
            lengths.append(len(token))

        return np.stack(batch), lengths

    def positional(self, lengths):
        batch = []
        max_len = max(lengths)
        for seq_len in lengths:
            pos_array = np.array([i for i in range(seq_len)], dtype=np.int64)
            padded = np.pad(pos_array, (0, max_len - seq_len), constant_values=0)
            pos_embed = self.pos_matrix[padded]
            batch.append(pos_embed)

        return np.stack(batch)

    def forward(self, list_of_strings):
        vec_embed, seq_len = self.vector(list_of_strings)
        pos_embed = self.positional(seq_len)
        embedding = vec_embed + pos_embed

        return embedding


class Attention:
    def __init__(self, embed_dim=768, head_dim=64, num_head=12):
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.num_head = num_head
        self.w_q = [
            np.random.randn(self.embed_dim, self.head_dim) for _ in range(self.num_head)
        ]
        self.w_k = [
            np.random.randn(self.embed_dim, self.head_dim) for _ in range(self.num_head)
        ]
        self.w_v = [
            np.random.randn(self.embed_dim, self.head_dim) for _ in range(self.num_head)
        ]
        self.w_o = np.random.randn(self.head_dim * self.num_head, self.embed_dim)
        self.embed = Embedding()

    @staticmethod
    def softmax(x, axis=-1):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def sdpa(self, q, k, v):
        scores = np.matmul(q, np.swapaxes(k, -2, -1))
        scaled_scores = scores / np.sqrt(self.head_dim)

        d_m = scaled_scores.shape[1]
        mask = np.triu(np.ones((d_m, d_m)) * -np.inf, k=1)
        mask = mask[np.newaxis, :, :]
        masked_scores = scaled_scores + mask

        a = self.softmax(masked_scores)
        z = np.matmul(a, v)

        return z

    def MultiHead(self, list_of_strings):
        x = self.embed.forward(list_of_strings)

        heads = []
        for i in range(self.num_head):
            query = np.matmul(x, self.w_q[i])
            key = np.matmul(x, self.w_k[i])
            value = np.matmul(x, self.w_v[i])

            z_i = self.sdpa(query, key, value)
            heads.append(z_i)

        concat = np.concatenate(heads, axis=-1)
        output = np.matmul(concat, self.w_o)

        return output


def GELU(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.pow(x, 3))))
