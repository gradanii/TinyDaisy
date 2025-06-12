from daisygrad.neural.layers import Parameter, Module, Linear, Embedding, Dropout
import jax.numpy as jnp

class RMSNorm(Module):
    def __init__(self, features):

