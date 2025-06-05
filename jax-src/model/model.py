import jax.numpy as jnp
import jax.nn as nn
from jax import random

DAISY_CONFIG = {
    "embed_dim": 768,
    "num_embed": 4096,
    "head_dim": 64,
    "num_head": 12,
    "vocab_size": 4096,
    "num_blocks": 12,
}


def he_init(d_in, d_out):
    scale = jnp.sqrt(1 / d_in)
    return random.normal(random.key(0), shape=(d_in, d_out)) * scale


def embedding(params, inputs):
    vec_embed = jnp.take(params["embedding"]["vec_matrix"], inputs)
    pos_embed = jnp.take(
        params["embedding"]["pos_matrix"], jnp.arange(inputs.shape[-1])
    )[jnp.newaxis, :, :]

    return vec_embed + pos_embed


def softmax(x, axis=-1):
    x_exp = jnp.exp(x - jnp.max(x, axis=axis, keepdims=True))
    return x_exp / jnp.sum(x_exp, axis=axis, keepdims=True)


def sdpa(cfg, q, k, v):
    scores = jnp.matmul(q, k.swapaxes(-2, -1))
    scaled_scores = scores / jnp.sqrt(cfg["head_dim"])

    d_m = scaled_scores.shape[-1]
    mask = jnp.full((d_m, d_m), 0.0)
    mask = mask.at[jnp.triu_indices(d_m, 1)].set(-jnp.inf)
    masked_scores = scaled_scores + mask

    a = softmax(masked_scores)
    z = jnp.matmul(a, v)

    return z


def multihead(cfg, x, block):
    B, T, E = x.shape
    q = jnp.matmul(x, block["w_q"]) + block["b_q"]
    q = q.reshape(B, T, cfg["num_head"], cfg["head_dim"]).transpose(0, 2, 1, 3)

    k = jnp.matmul(x, block["w_k"]) + block["b_k"]
    k = k.reshape(B, T, cfg["num_head"], cfg["head_dim"]).transpose(0, 2, 1, 3)

    v = jnp.matmul(x, block["w_v"]) + block["b_v"]
    v = v.reshape(B, T, cfg["num_head"], cfg["head_dim"]).transpose(0, 2, 1, 3)

    z = sdpa(cfg, q, k, v)
    B, H, T, D = z.shape
    z = z.transpose(0, 2, 1, 3).reshape(B, T, H * D)

    return jnp.matmul(z, block["w_o"]) + block["b_o"]


def layernorm(x, gamma, beta):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x_normed = (x - mean) / jnp.sqrt(var + 1e-5)

    return gamma * x_normed + beta


def mlp(x, block):
    return (
        jnp.matmul(nn.gelu(jnp.matmul(x, block["w_l1"]) + block["b_l1"]), block["w_l2"])
        + block["b_l2"]
    )


def forward(cfg, params, inputs):
    x = embedding(params, inputs)

    for block in params["blocks"]:
        x1 = layernorm(x, block["gamma1"], block["beta1"])
        x2 = x + multihead(cfg, x1, block)
        x3 = layernorm(x2, block["gamma2"], block["beta2"])
        x = x2 + mlp(x3, block)

    logits = layernorm(x, params["final_norm"]["gamma"], params["final_norm"]["beta"])
    logits = jnp.matmul(logits, params["final_proj"]["w"]) + params["final_proj"]["b"]

    return logits
