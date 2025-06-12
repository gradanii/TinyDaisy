from daisygrad.neural.layers import Parameter, Module, Embedding, Linear, LayerNorm, Dropout
from dataclasses import dataclass
import jax.numpy as jnp
from jax import random

class Embed(Module):
    def __init__(self, key, config):
        super().__init__()
        vec_key, pos_key = random.split(key)
        scale = 1 / jnp.sqrt(config.n_embed)
        self.vec_matrix = Embedding(vec_key, config.vocab_size, config.n_embed)
        self.pos_matrix = Parameter(random.normal(pos_key, shape=(config.block_size, config.n_embed)) * scale)

    def __call__(self, inputs):
        vec_embed = self.vec_matrix(inputs)
        indices = jnp.arange(inputs.shape[-1])
        pos_embed = self.pos_matrix.take(indices)
        return vec_embed + pos_embed

    @property
    def weight(self):
        return self.vec_matrix.embed

class CausalSelfAttention(Module):
    def __init__(self, key, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        qkv_key, proj_key, drop_key = random.split(key, 3)
        self.qkv = Linear(qkv_key, config.n_embed, 3 * config.n_embed, bias=True)
        self.proj = Linear(proj_key, config.n_embed, config.n_embed, bias=True)
        self.n_head = config.n_head
        self.dropout = Dropout(drop_key, config.dropout)

    def __call__(self, x, train=True):
        B, T, E = x.shape
        qkv_projected = self.qkv(x)
        q, k, v = qkv_projected.split(3, axis=-1)
        q = q.reshape(B, T, self.n_head, E // self.n_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, E // self.n_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, E // self.n_head).transpose(0, 2, 1, 3)

        D = q.shape[-1]
        scores = q @ k.transpose(0, 1, 3, 2)
        scaled_scores = scores / jnp.sqrt(D) 

        d_m = scaled_scores.shape[-1]
        mask = jnp.full((d_m, d_m), 0.0)
        mask = mask.at[jnp.triu_indices(d_m, 1)].set(-jnp.inf)[None, None, :, :]
        masked_scores = scaled_scores + mask

        a = masked_scores.softmax()
        z = a @ v
        
        z = z.transpose(0, 2, 1, 3).reshape((B, T, E))
        return self.dropout(self.proj(z), train=train)

class MLP(Module):
    def __init__(self, key, config):
        super().__init__()
        l1_key, l2_key, drop_key = random.split(key, 3)
        self.l1 = Linear(l1_key, config.n_embed, 4 * config.n_embed, bias=True)
        self.l2 = Linear(l2_key, 4 * config.n_embed, config.n_embed, bias=True)
        self.dropout = Dropout(drop_key, config.dropout)

    def __call__(self, x, train=True):
        return self.dropout(self.l2(self.l1(x).gelu()), train=train)

@dataclass
class DaisyConfig:
    block_size: int = 512
    vocab_size: int = 4096
    n_embed = 768
    n_head = 12
    n_blocks = 12
    dropout: float = 0.0

class Transformer(Module):
    def __init__(self, key, config):
        super().__init__()
        attn_key, mlp_key, drop1_key, drop2_key = random.split(key, 4)
        self.attn = CausalSelfAttention(attn_key, config)
        self.ln_1 = LayerNorm(config.n_embed)
        self.ln_2 = LayerNorm(config.n_embed)
        self.mlp = MLP(mlp_key, config)
        self.drop1 = Dropout(drop1_key, config.dropout)
        self.drop2 = Dropout(drop2_key, config.dropout)

    def __call__(self, x, train=True):
        x = x + self.drop1(self.attn(self.ln_1(x)), train=train)
        x = x + self.drop2(self.mlp(self.ln_2(x)), train=train)
        return x

class Model(Module):
    def __init__(self, key, config=DaisyConfig()):
        super().__init__()
        keys = random.split(key, config.n_blocks + 1)
        key, embed_key, final_key = random.split(keys[0], 3)
        self.embedding = Embed(embed_key, config)
        self.blocks = [Transformer(keys[i], config) for i in range(config.n_blocks)]
        self.ln_f = LayerNorm(config.n_embed)
        final_weight = self.embedding.weight.transpose(-1, -2)
        self.final = Linear(final_key, config.n_embed, config.vocab_size, bias=True, weight=final_weight)

    def __call__(self, inputs, train=True):
        x = self.embedding(inputs)

        for block in self.blocks:
            x = block(x, train=train)

        logits = self.final(self.ln_f(x))
        return logits


                                                                                                                                                                                                                            
