import jax.numpy as jnp
from jax import random
import numpy as np

class DaisyTensor:
    def __init__(self, data, requires_grad: bool, _children=(), _op='', _meta=None, name=None):
        self.data = jnp.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self._meta = _meta
        self.shape = self.data.shape
        self.name = name

    def __add__(self, other):
        other_dt = self._unwrap(other)
        out = DaisyTensor(self.data + other_dt.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other_dt} if other_dt.requires_grad else {self}
        out._op = 'add'

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                self.grad = jnp.add(self.grad, out.grad) if self.grad is not None else out.grad
            if other_dt.requires_grad:
                other_dt.grad = jnp.add(other_dt.grad, out.grad) if other_dt.grad is not None else out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        other_dt = self._unwrap(other)
        out = DaisyTensor(self.data - other_dt.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other_dt} if other_dt.requires_grad else {self}
        out._op = 'sub'

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                self.grad = jnp.add(self.grad, out.grad) if self.grad is not None else out.grad
            if other_dt.requires_grad:
                other_dt.grad = jnp.add(other_dt.grad, -out.grad) if other_dt.grad is not None else -out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other_dt = self._unwrap(other)
        out = DaisyTensor(self.data * other_dt.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other_dt} if other_dt.requires_grad else {self}
        out._op = 'mul'

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                self.grad = jnp.add(self.grad, other.data * out.grad) if self.grad is not None else (other.data * out.grad)
            if other_dt.requires_grad:
                other_dt.grad = jnp.add(other_dt.grad, self.data * out.grad) if other_dt.grad is not None else (self.data * out.grad)

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other_dt = self._unwrap(other)
        out = DaisyTensor(self.data / other_dt.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other_dt} if other_dt.requires_grad else {self}
        out._op = 'truediv'

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                self.grad = jnp.add(self.grad, (out.grad / other_dt.data)) if self.grad is not None else (out.grad / other.data)
            if other_dt.requires_grad:
                other_dt.grad = jnp.add(other_dt.grad, -1 * out.grad * self.data / (other_dt.data**2)) if other_dt.grad is not None else (-1 * out.grad * self.data / (other_dt.data**2))
                                                                                                                                                                                                                                                                                                        
        out._backward = _backward
        return out

    def __matmul__(self, other):
        other_dt = self._unwrap(other)
        out = DaisyTensor(self.data @ other_dt.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other_dt} if other_dt.requires_grad else {self}
        out._op = 'matmul'

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                if out.grad.ndim < 2:
                    out.grad = jnp.expand_dims(out.grad, axis=0)
                self.grad = jnp.add(self.grad, (out.grad @ other_dt.data.T)) if self.grad is not None else (out.grad @ other.data.T)
            if other_dt.requires_grad:
                if out.grad.ndim < 2:
                    out.grad = jnp.expand_dims(out.grad, axis=0)
                other_dt.grad = jnp.add(other_dt.grad, (self.data.T @ out.grad)) if other_dt.grad is not None else (self.data.T @ out.grad)

        out._backward = _backward
        return out


    def __neg__(self):
        out = DaisyTensor(-1 * self.data, requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = 'neg'

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                self.grad = jnp.add(self.grad, -out.grad) if self.grad is not None else (-out.grad)

        out._backward = _backward
        return out

    def _unwrap(self, other):
        return other if isinstance(other, DaisyTensor) else DaisyTensor(other, requires_grad=False)

    def backward(self):
        def build_tree(x):
            parents = []
            visited = set()
            def visit(node):
                if node not in visited:
                    visited.add(node)
                    for p in node._prev:
                        visit(p)
                    parents.append(node)
                return parents

            visit(x)
            return parents

        self.grad = jnp.array(1.0) if self.data.shape == () else jnp.ones_like(self.data)
        tree = build_tree(self)
        for t in tree:
            t._backward()

    def zero_grad(self, visited=None):
        if visited is None:
            visited = set()
        if self in visited:
            return
        visited.add(self)

        self.grad = None

        for p in self._prev:
            p.zero_grad(visited)

    def sum(self, axis: int, keepdims: bool):
        out = DaisyTensor(jnp.sum(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = 'sum'

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                if keepdims:
                    self.grad = jnp.add(self.grad, out.grad * jnp.ones_like(self.data)) if self.grad is not None else (out.grad * jnp.ones_like(self.data))
                else:
                    expand_axes = tuple(a for a in range(self.data.ndim) if a not in (axis if isinstance(axis, tuple) else (axis,)))
                    broadcasted_grad = jnp.broadcast_to(jnp.expand_dims(out.grad, axis=expand_axes), self.data.shape)
                    self.grad = jnp.add(self.grad, broadcasted_grad) if self.grad is not None else broadcasted_grad

        out._backward = _backward
        return out

    def mean(self, axis: int, keepdims: bool):
        out = self.sum(axis=axis, keepdims=keepdims) / (self.data.shape[axis] if axis is not None else self.data.size)
        out._op = 'mean'

        return out

    def exp(self):
        out = DaisyTensor(jnp.exp(self.data), requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = 'exp'

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                self.grad = jnp.add(self.grad, out.grad * jnp.exp(self.data)) if self.grad is not None else (out.grad * jnp.exp(self.data))
                
        out._backward = _backward
        return out

    def log(self):
        out = DaisyTensor(jnp.log(self.data), requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = 'log'

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                self.grad = jnp.add(self.grad, out.grad * (1 / self.data)) if self.grad is not None else (out.grad * (1 / self.data))

        out._backward = _backward
        return out

    def max(self, axis: int, keepdims: bool):
        out = DaisyTensor(jnp.max(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = 'max'

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                if keepdims:
                    self.grad = jnp.add(self.grad, out.grad * (self.data == out.data)) if self.grad is not None else (out.grad * (self.data == out.data))
                else:
                    expand_axes = tuple(a for a in range(self.data.ndim) if a not in (axis if isinstance(axis, tuple) else (axis,)))
                    broadcasted_grad = jnp.broadcast_to(jnp.expand_dims(out.grad, axis=expand_axes), self.data.shape)
                    broadcasted_data = jnp.broadcast_to(jnp.expand_dims(out.data, axis=expand_axes), self.data.shape)
                    self.grad = jnp.add(self.grad, broadcasted_grad * (self.data == broadcasted_data)) if self.grad is not None else (broadcasted_grad * (self.data == broadcasted_data))

        out._backward = _backward
        return out
    
    def min(self, axis: int, keepdims: bool):
        out = DaisyTensor(jnp.min(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = 'min'

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                if keepdims:
                    self.grad = jnp.add(self.grad, out.grad * (self.data == out.data)) if self.grad is not None else (out.grad * (self.data == out.data))
                else:
                    expand_axes = tuple(a for a in range(self.data.ndim) if a not in (axis if isinstance(axis, tuple) else (axis,)))
                    broadcasted_grad = jnp.broadcast_to(jnp.expand_dims(out.grad, axis=expand_axes), self.data.shape)
                    broadcasted_data = jnp.broadcast_to(jnp.expand_dims(out.data, axis=expand_axes), self.data.shape)
                    self.grad = jnp.add(self.grad, broadcasted_grad * (self.data == broadcasted_data)) if self.grad is not None else (broadcasted_grad * (self.data == broadcasted_data))

        out._backward = _backward
        return out

    def clip(self, x_min, x_max):
        out = DaisyTensor(jnp.clip(self.data, a_min=x_min, a_max=x_max), requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = 'clip'

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                self.grad = jnp.add(self.grad, out.grad * ((self.data > x_min) & (self.data < x_max)).astype(float)) if self.grad is not None else (out.grad * ((self.data > x_min) & (self.data < x_max)).astype(float))

        out._backward = _backward
        return out

    def pow(self, other):
        other_dt = self._unwrap(other)
        out = DaisyTensor(jnp.pow(self.data, other_dt.data), requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other_dt} if other_dt.requires_grad else {self}
        out._op = 'pow'

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                self.grad = jnp.add(self.grad, out.grad * other_dt.data * jnp.pow(self.data, other_dt.data - 1)) if self.grad is not None else (out.grad * (other_dt.data * jnp.pow(self.data, other_dt.data - 1)))
            if other_dt.requires_grad:
                other_dt.grad = jnp.add(self.grad, (out.grad * jnp.pow(self.data, other_dt.data) * jnp.log(self.data))) if self.grad is not None else (out.grad * jnp.pow(self.data, other_dt.data) * jnp.log(self.data))

        out._backward = _backward
        return out

    def sqrt(self):
        out = DaisyTensor(jnp.sqrt(self.data), requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = 'sqrt'
    
        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                self.grad = jnp.add(self.grad, out.grad * 0.5 * jnp.sqrt(self.data)) if self.grad is not None else (out.grad * 0.5 * jnp.sqrt(self.data))

        out._backward = _backward
        return out

    def abs(self):
        out = DaisyTensor(jnp.absolute(self.data), requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = 'abs'

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                self.grad = jnp.add(self.grad, out.grad * jnp.where(self.data == 0, 0.0, jnp.sign(self.data))) if self.grad is not None else (out.grad * jnp.where(self.data == 0, 0.0, jnp.sign(self.data)))

        out._backward = _backward
        return out

    def reshape(self, *args):
        new_shape = tuple(args)
        out = DaisyTensor(jnp.reshape(self.data, new_shape), requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = 'reshape'

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                self.grad = jnp.add(self.grad, jnp.reshape(out.grad, self.data.shape)) if self.grad is not None else jnp.reshape(out.grad, self.data.shape)

        out._backward = _backward
        return out

    def transpose(self, *axes):
        out = DaisyTensor(jnp.transpose(self.data, axes), requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = 'transpose'

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                grad_back = jnp.transpose(out.grad, tuple(np.argsort(axes))) if axes else jnp.transpose(out.grad)
                self.grad = jnp.add(self.grad, grad_back) if self.grad is not None else grad_back

        out._backward = _backward
        return out

    def take(self, indices, axis=0):
        out = DaisyTensor(jnp.take(self.data, indices=indices, axis=axis), requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = 'take'

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                self.grad = jnp.add(self.grad, jnp.zeros_like(self.data).at[indices].add(out.grad)) if self.grad is not None else jnp.zeros_like(self.data).at[indices].add(out.grad)

        out._backward = _backward
        return out

    def __getitem__(self, idx):
        return self.take(idx)

    def split(self, chunks: int, axis: int):
        splits = jnp.split(self.data, chunks, axis=axis)
        outputs = []

        for i, chunk in enumerate(splits):
            out = DaisyTensor(chunk, requires_grad=self.requires_grad)
            out._prev = {self}
            out._op = 'split'
            out._meta = {
                'axis': axis,
                'index': i,
                'num_splits': len(splits)
            }
            outputs.append(out)

        def _backward():
            for out in outputs:
                if out.grad is None:
                    return

            if self.requires_grad:
                sorted_grads = [out.grad for out in sorted(outputs, key=lambda x: x._meta['index'])]
                grad_back = jnp.concatenate(sorted_grads, axis=axis)
                self.grad = jnp.add(self.grad, grad_back) if self.grad is not None else grad_back

        for out in outputs:
            out._backward = _backward
        return outputs

    def softmax(self, axis: int=-1):
        x_exp = DaisyTensor(self.data - (self.max(axis=axis, keepdims=True)).data, requires_grad=False).exp()
        out = x_exp / x_exp.sum(axis=axis, keepdims=True)
        out._prev = {self}
        out._op = 'softmax'

        return out

    def tanh(self):
        out = DaisyTensor(jnp.tanh(self.data), requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = 'tanh'

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                self.grad = jnp.add(self.grad, out.grad * (1 - (jnp.tanh(self.data) ** 2))) if self.grad is not None else (out.grad * (1 - (jnp.tanh(self.data) ** 2)))

        out._backward = _backward
        return out
                                                                                                                                                                                                                                                                    
    def gelu(self):
        phi_x = jnp.tanh(jnp.sqrt(2 / jnp.pi) * (self.data + 0.044715 * jnp.pow(self.data, 3)))
        out = DaisyTensor(0.5 * self.data * (1 + phi_x), requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = 'gelu'

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                grad_back = (0.5 * (1 + phi_x)) + (0.5 * self.data * (1 - phi_x**2)) * jnp.sqrt(2 / jnp.pi) * (1 + (3 * (0.044715 * self.data**2)))
                self.grad = jnp.add(self.grad, out.grad * grad_back) if self.grad is not None else (out.grad * grad_back)

        out._backward = _backward
        return out

    def dropout(self, key, p=0.5):
        if not self.requires_grad or p == 0.0:
            return self
        keep_prob = 1.0 - p
        mask = random.bernoulli(key, keep_prob, shape=self.data.shape)
        out = DaisyTensor(self.data * mask / keep_prob, requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = 'dropout'

        return out



