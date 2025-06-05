import jax.numpy as jnp
from jax import random


class KeyGen:
    def __init__(self, seed=42):
        self.key = random.PRNGKey(seed)

    def next(self):
        self.key, subkey = random.split(self.key)
        return subkey


def he_init(key, d_in, d_out):
    scale = jnp.sqrt(1 / d_in)
    return random.normal(key, shape=(d_in, d_out)) * scale
