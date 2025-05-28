import numpy as np


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

        vec_embed = np.stack(batch)

        assert vec_embed.ndim == 3, (
            f"Embedding output must be 3D, got {vec_embed.shape}"
        )
        assert vec_embed.shape[2] == cfg.embed_dim, (
            f"Embedding dim mismatch: expected {cfg.embed_dim}, got {vec_embed.shape[2]}"
        )
        return vec_embed, lengths

    def position(lengths):
        batch = []
        max_len = max(lengths)
        for seq_len in lengths:
            pos_array = np.array([i for i in range(seq_len)], dtype=np.int64)
            padded = np.pad(pos_array, (0, max_len - seq_len), constant_values=0)
            embedding = cfg.pos_matrix[padded]
            batch.append(embedding)

        pos_embed = np.stack(batch)

        assert pos_embed.shape == (1, max_len, cfg.embed_dim), (
            "Positional Embedding shape mismatch"
        )
        return pos_embed

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
    heads = []

    x = np.transpose(
        np.reshape(x, (x.shape[0], x.shape[1], cfg.num_head, cfg.head_dim)),
        (0, 2, 1, 3),
    )
    for i in range(cfg.num_head):
        query = np.matmul(x, cfg.w_q[i]) + cfg.b_q[i]
        key = np.matmul(x, cfg.w_k[i]) + cfg.b_k[i]
        value = np.matmul(x, cfg.w_v[i]) + cfg.b_v[i]

        assert query.shape == key.shape == value.shape
        assert query.shape[-1] == cfg.embed_dim

        z_i = sdpa(cfg, query, key, value)
        heads.append(z_i)

    concat = np.concatenate(heads, axis=-1)
    output = np.matmul(concat, cfg.w_o)

    assert output.shape == x.shape, "Attention must preserve input shape"
    return output


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.pow(x, 3))))


def layernorm(cfg, x):
    eps = 1e-5

    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_normed = (x - mean) / np.sqrt(var + eps)

    normed = cfg.gamma * x_normed + cfg.beta

    assert normed.shape == x.shape, "Layernorm must preserve input shape"


def mlp(cfg, x):
    l1 = np.matmul(x, cfg.w_l1) + cfg.b_l1
    assert l1.shape[-1] == 4 * cfg.embed_dim

    l2 = np.matmul(gelu(l1), cfg.w_l2) + cfg.b_l2
    assert l2.shape == x.shape, "Final MLP projection must preserve input shape"


def forward(cfg, strings):
    x = embedding(cfg, strings)

    for _ in range(cfg.num_blocks):
        x1 = layernorm(cfg, x)
        x2 = x + multihead(cfg, x1)
        assert x2.shape == x.shape, "Residual addition shape mismatch"

        x3 = layernorm(cfg, x2)
        x = x2 + mlp(cfg, x3)

    return layernorm(cfg, x)
