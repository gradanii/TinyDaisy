import flax.nnx as nnx
import jax.numpy as jnp
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
    keys = random.split(random.PRNGKey(42), 1000)
    ki = 0

    def next_key():
        nonlocal ki
        k = keys[ki]
        ki += 1
        return k

    scale = jnp.sqrt(1 / d_in)
    return random.normal(next_key(), shape=(d_in, d_out)) * scale


class Embedding(nnx.Module):
    def __init__(self, cfg, rngs):
        super().__init__()

        self.vec_matrix = nnx.Embed(cfg["num_embed"], cfg["embed_dim"], rngs=rngs)
        self.pos_matrix = nnx.Param(random.uniform)

    def forward(self, inputs):
        vec_embed = self.vec_matrix(inputs)
