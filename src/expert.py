import numpy as np


def embedding(cfg, inputs):
    vec_embed = cfg.vec_matrix[inputs]

    assert vec_embed.ndim == 3, f"Embedding output must be 3D, got {vec_embed.shape}"
    assert vec_embed.shape[2] == cfg.embed_dim, (
        f"Embedding dim mismatch: expected {cfg.embed_dim}, got {vec_embed.shape[2]}"
    )

    pos_embed = cfg.pos_matrix[np.arange(inputs.shape[-1])][np.newaxis, :, :]
    assert pos_embed.shape == (1, inputs.shape[-1], cfg.embed_dim), (
        f"Positional Embedding shape mismatch, expected (1, max_len, embed_dim), got {pos_embed.shape}"
    )

    return vec_embed + pos_embed


def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def sdpa(cfg, q, k, v):
    scores = np.matmul(q, np.swapaxes(k, -2, -1))
    scaled_scores = scores / np.sqrt(cfg.head_dim)

    d_m = scaled_scores.shape[-1]
    mask = np.full((d_m, d_m), 0.0)
    mask[np.triu_indices(d_m, 1)] = -np.inf
    mask = mask[np.newaxis, np.newaxis, :, :]
    masked_scores = scaled_scores + mask

    a = softmax(masked_scores)
    z = np.matmul(a, v)

    return z


def multihead(cfg, x):
    query = np.matmul(x, cfg.w_q) + cfg.b_q
    key = np.matmul(x, cfg.w_k) + cfg.b_k
    value = np.matmul(x, cfg.w_v) + cfg.b_v

    q = np.transpose(
        np.reshape(query, (x.shape[0], x.shape[1], cfg.num_head, cfg.head_dim)),
        (0, 2, 1, 3),
    )

    k = np.transpose(
        np.reshape(key, (x.shape[0], x.shape[1], cfg.num_head, cfg.head_dim)),
        (0, 2, 1, 3),
    )

    v = np.transpose(
        np.reshape(value, (x.shape[0], x.shape[1], cfg.num_head, cfg.head_dim)),
        (0, 2, 1, 3),
    )

    assert query.shape == key.shape == value.shape
    assert query.shape[-1] == cfg.embed_dim

    z = sdpa(cfg, q, k, v)
    z = z.transpose(0, 2, 1, 3).reshape(x.shape)
    output = np.matmul(z, cfg.w_o)

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
    return normed


def mlp(cfg, x):
    l1 = np.matmul(x, cfg.w_l1) + cfg.b_l1
    assert l1.shape[-1] == 4 * cfg.embed_dim

    l2 = np.matmul(gelu(l1), cfg.w_l2) + cfg.b_l2
    assert l2.shape == x.shape, "Final MLP projection must preserve input shape"
    return l2


def forward(cfg, inputs):
    x = embedding(cfg, inputs)

    for _ in range(cfg.num_blocks):
        x1 = layernorm(cfg, x)
        x2 = x + multihead(cfg, x1)
        x3 = layernorm(cfg, x2)
        x = x2 + mlp(cfg, x3)

    logits = layernorm(cfg, x)
    logits = np.matmul(x, cfg.w_voc) + cfg.b_voc

    return logits
