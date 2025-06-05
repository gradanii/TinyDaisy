import jax.numpy as jnp
from jax import random


def he_init(key, d_in, d_out):
    scale = jnp.sqrt(1 / d_in)
    return random.normal(key, shape=(d_in, d_out)) * scale


DAISY_CONFIG = {
    "embed_dim": 768,
    "num_embed": 4096,
    "head_dim": 64,
    "num_head": 12,
    "vocab_size": 4096,
    "num_blocks": 12,
}


def init_params(key, cfg=DAISY_CONFIG):
    params = {}
    keys = random.split(key, 100)  # split enough ahead of time
    ki = 0

    def next_key():
        nonlocal ki
        k = keys[ki]
        ki += 1
        return k

    # Embedding
    params["embedding"] = {
        "vec_matrix": he_init(next_key(), cfg["num_embed"], cfg["embed_dim"]),
        "pos_matrix": he_init(next_key(), cfg["num_embed"], cfg["embed_dim"]),
    }

    # Blocks
    params["blocks"] = []
    for _ in range(cfg["num_blocks"]):
        block = {
            "w_q": he_init(next_key(), cfg["embed_dim"], cfg["embed_dim"]),
            "w_k": he_init(next_key(), cfg["embed_dim"], cfg["embed_dim"]),
            "w_v": he_init(next_key(), cfg["embed_dim"], cfg["embed_dim"]),
            "w_o": he_init(next_key(), cfg["embed_dim"], cfg["embed_dim"]),
            "b_q": jnp.zeros(cfg["embed_dim"]),
            "b_k": jnp.zeros(cfg["embed_dim"]),
            "b_v": jnp.zeros(cfg["embed_dim"]),
            "b_o": jnp.zeros(cfg["embed_dim"]),
            "gamma1": jnp.ones(cfg["embed_dim"]),
            "beta1": jnp.zeros(cfg["embed_dim"]),
            "gamma2": jnp.ones(cfg["embed_dim"]),
            "beta2": jnp.zeros(cfg["embed_dim"]),
            "w_l1": he_init(next_key(), cfg["embed_dim"], 4 * cfg["embed_dim"]),
            "b_l1": jnp.zeros(4 * cfg["embed_dim"]),
            "w_l2": he_init(next_key(), 4 * cfg["embed_dim"], cfg["embed_dim"]),
            "b_l2": jnp.zeros(cfg["embed_dim"]),
        }
        params["blocks"].append(block)

    # Final layer norm + vocab projection
    params["final_norm"] = {
        "gamma": jnp.ones(cfg["embed_dim"]),
        "beta": jnp.zeros(cfg["embed_dim"]),
    }
    params["vocab_proj"] = {
        "w": he_init(next_key(), cfg["embed_dim"], cfg["vocab_size"]),
        "b": jnp.zeros(cfg["vocab_size"]),
    }

    return params
