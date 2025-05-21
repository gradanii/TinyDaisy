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
        self.vec_matrix = np.random.randn(self.num_embed, self.embed_dim)
        self.pos_matrix = np.random.randn(self.num_embed, self.embed_dim)


def embedding(cfg, strings):
    def vector(strings):
        batch = []
        lengths = []
        tokens = [cfg.tokenizer.encode(s) for s in strings]
        max_len = max(len(t) for t in tokens)
        for token in tokens:
            padded = np.pad(token, (0, max_len - len(token)), constant_values=0)
            embedding = cfg.vec_matrix[padded]
            batch.append(embedding)
            lengths.append(len(token))

        return np.stack(batch), lengths

    def position(lengths):
        batch = []
        max_len = max(lengths)
        for seq_len in lengths:
            pos_array = np.array([i for i in range(seq_len)], dtype=np.int64)
            padded = np.pad(pos_array, (0, max_len - seq_len), constant_values=0)
            pos_embed = cfg.pos_matrix[padded]
            batch.append(pos_embed)

        return np.stack(batch)

    vec_embed, lengths = vector(strings)
    pos_embed = position(lengths)

    return vec_embed + pos_embed


def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def sdpa(cfg, q, k, v):
    scores = np.matmul(q, np.swapaxes(k, -2, -1))
    scaled_scores = scores / np.sqrt(cfg.head_dim)

    d_m = scaled_scores.shape[1]
    mask = np.triu(np.ones((d_m, d_m)) * -np.inf, k=1)
    mask = mask[np.newaxis, :, :]
    masked_scores = scaled_scores + mask

    a = softmax(masked_scores)
    z = np.matmul(a, v)

    return z


def multihead(cfg, x):
    w_q = [np.random.randn(cfg.embed_dim, cfg.head_dim) for _ in range(cfg.num_head)]
    w_k = [np.random.randn(cfg.embed_dim, cfg.head_dim) for _ in range(cfg.num_head)]
    w_v = [np.random.randn(cfg.embed_dim, cfg.head_dim) for _ in range(cfg.num_head)]
    w_o = np.random.randn(cfg.head_dim * cfg.num_head, cfg.embed_dim)

    heads = []
    for i in range(cfg.num_head):
        query = np.matmul(x, w_q[i])
        key = np.matmul(x, w_k[i])
        value = np.matmul(x, w_v[i])

        z_i = sdpa(cfg, query, key, value)
        heads.append(z_i)

        concat = np.concatenate(heads, axis=-1)
        output = np.matmul(concat, w_o)

        return output


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.pow(x, 3))))


def layernorm(cfg, x):
    gamma = np.random.randn(cfg.embed_dim)
    beta = np.random.randn(cfg.embed_dim)
    eps = 1e-5

    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_normed = (x - mean) / np.sqrt(var + eps)

    return gamma * x_normed + beta
