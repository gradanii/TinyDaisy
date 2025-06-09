import jax.numpy as jnp
import numpy as np

class DaisyTensor:
    def __init__(self, data, requires_grad: bool, _children=(), _op='', name=None):
        self.data = jnp.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.name = name

    def __add__(self, other):
        out = DaisyTensor(self.data + self._unwrap(other), requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}
        out._op = 'add'

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                self.grad = jnp.add(self.grad, out.grad) if self.grad is not None else out.grad
            if other.requires_grad:
                other.grad = jnp.add(other.grad, out.grad) if other.grad is not None else out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        out = DaisyTensor(self.data - self._unwrap(other), requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}
        out._op = 'sub'

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                self.grad = jnp.add(self.grad, out.grad) if self.grad is not None else out.grad
            if other.requires_grad:
                other.grad = jnp.add(other.grad, -out.grad) if other.grad is not None else -out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        out = DaisyTensor(self.data * self._unwrap(other), requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}
        out._op = 'mul'

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                self.grad = jnp.add(self.grad, other.data * out.grad) if self.grad is not None else (other.data * out.grad)
            if other.requires_grad:
                other.grad = jnp.add(other.grad, self.data * out.grad) if other.grad is not None else (self.data * out.grad)

        out._backward = _backward
        return out

    def __truediv__(self, other):
        out = DaisyTensor(self.data / self._unwrap(other), requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}
        out._op = 'truediv'

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                self.grad = jnp.add(self.grad, (out.grad / other.data)) if self.grad is not None else (out.grad / other.data)
            if other.requires_grad:
                other.grad = jnp.add(other.grad, (-1 * out.grad * self.data / (other.data**2)) if other.grad is not None else (-1 * out.grad * self.data / (other.data**2)))

        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = DaisyTensor(self.data @ self._unwrap(other), requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}
        out._op = 'matmul'

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                if out.grad.ndim < 2:
                    out.grad = jnp.expand_dims(out.grad, axis=0)
                self.grad = jnp.add(self.grad, (out.grad @ other.data.T)) if self.grad is not None else (out.grad @ other.data.T)
            if other.requires_grad:
                if out.grad.ndim < 2:
                    out.grad = jnp.expand_dims(out.grad, axis=0)
                other.grad = jnp.add(other.grad, (self.data.T @ out.grad)) if other.grad is not None else (self.data.T @ out.grad)

        out._backward = _backward
        return out


    def __neg__(self, other):
        out = DaisyTensor(-1 * self.data, requires_grad=self.requires_grad)
        out._prev = {self, other}
        out._op = 'neg'

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                self.grad = jnp.add(self.grad, -out.grad) if self.grad is not None else (-out.grad)

        out._backward = _backward
        return out

    def _unwrap(self, other):
        return other.data if isinstance(other, DaisyTensor) else other

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
        out = DaisyTensor(jnp.pow(self.data, self._unwrap(other)), requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}
        out._op = 'pow'

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                self.grad = jnp.add(self.grad, out.grad * (self._unwrap(other) * jnp.pow(self.data, self._unwrap(other) - 1))) if self.grad is not None else (out.grad * (self._unwrap(other) * jnp.pow(self.data, self._unwrap(other) - 1)))
            if other.requires_grad:
                other.grad = jnp.add(self.grad, (out.grad * jnp.pow(self.data, other.data) * jnp.log(self.data))) if self.grad is not None else (out.grad * jnp.pow(self.data, other.data) * jnp.log(self.data))

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

    def reshape(self, new_shape: tuple):
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




