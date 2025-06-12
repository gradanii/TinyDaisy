from daisygrad.tensor import DaisyTensor
import jax.numpy as jnp
from jax import random

class Parameter(DaisyTensor):
    def __init__(self, data):
        super().__init__(data, requires_grad=True)

class Module:
    def parameters(self):
        params = []

        for attr in vars(self).values():
            if isinstance(attr, Parameter):
                params.append(attr)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
            elif isinstance(attr, dict):
                for item in attr.values():
                    if isinstance(item, Module):
                        params.extend(item.parameters())

        return params

class Linear(Module):
    def __init__(self, key, in_features, out_features, bias: bool, weight=None):
        if weight is not None:
            self.weight = weight
        else:
            scale = 1 / jnp.sqrt(in_features)
            self.weight = Parameter(random.normal(key, shape=(in_features, out_features)) * scale)
        self.bias = Parameter(jnp.zeros(out_features,)) if bias else None

    def __call__(self, x):
        out = x @ self.weight
        return out + self.bias if self.bias is not None else out


class Embedding(Module):
    def __init__(self, key, in_features, out_features):
        scale = 1 / jnp.sqrt(in_features)
        self.embed = Parameter(random.normal(key, shape=(in_features, out_features)) * scale)

    def __call__(self, x):
        out = self.embed.take(x)
        return out


class LayerNorm(Module):
    def __init__(self, features):
        self.gamma = Parameter(jnp.ones(features))
        self.beta = Parameter(jnp.zeros(features))

    def __call__(self, x):
        x = x if isinstance(x, DaisyTensor) else DaisyTensor(x, requires_grad=False)
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean).pow(2)).mean(axis=-1, keepdims=True)
        x_normed = (x - mean) / (var + 1e-5).sqrt()

        return self.gamma * x_normed + self.beta


class Dropout:
    def __init__(self, key, p=0.5):
        self.p = p
        self.key = key

    def __call__(self, x, train=True):
        if not x.requires_grad or self.p == 0.0 or not train:
            return x
        keep_prob = 1.0 - self.p
        self.key, subkey = random.split(self.key)
        mask = random.bernoulli(subkey, keep_prob, x.data.shape)
        out = DaisyTensor(x.data * mask / keep_prob, requires_grad=x.requires_grad)
        out._prev = {x}
        out._op = 'dropout'
        return out



