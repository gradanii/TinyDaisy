import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op="", name=None,):
        self.data = np.ndarray(data)
        self.requires_grad = requires_grad
        self.grad = None

        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.name = name

    def __repr__(self):
        return (
            f"Tensor(name={self.name!r}, shape={self.data.shape}, "
            f"requires_grad={self.requires_grad}, op={self._op!r})"
        )

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        out._op = "add"
        out._prev = {self, other}

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                self.grad = self.grad + out.grad if self.grad is not None else out.grad
            if other.requires_grad:
                other.grad = (other.grad + out.grad if other.grad is not None else out.grad)

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data - other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        out._op = "sub"
        out._prev = {self, other}

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                self.grad = self.grad + out.grad if self.grad is not None else out.grad
            if other.requires_grad:
                other.grad = (other.grad + (-1 * out.grad) if other.grad is not None else (-1 * out.grad))

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        out._op = "mul"
        out._prev = {self, other}

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                self.grad = (self.grad + (out.grad * other.data) if self.grad is not None else (out.grad * other.data))
            if other.requires_grad:
                other.grad = (other.grad + (out.grad * self.data) if other.grad is not None else (out.grad * self.data))

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        out._op = "truediv"
        out._prev = {self, other}

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                self.grad = (
                    self.grad + (out.grad / other.data) if self.grad is not None else (out.grad / other.data)
                )
            if other.requires_grad:
                other.grad = (
                    other.grad + (-1 * out.grad * self.data / (other.data**2)) if other.grad is not None else (-1 * out.grad * self.data / (other.data**2)))

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        out._op = "matmul"
        out._prev = {self, other}

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                if out.grad.ndim < 2:
                    out.grad = np.expand_dims(out.grad, axis=0)
                self.grad = (self.grad + (out.grad @ other.data.T) if self.grad is not None else (out.grad @ other.data.T))
            if other.requires_grad:
                if out.grad.ndim < 2:
                    out.grad = np.expand_dims(out.grad, axis=0)
                other.grad = (other.grad + (self.data.T @ out.grad) if other.grad is not None else (self.data.T @ out.grad))

        out._backward = _backward
        return out

    def __neg__(self):
        out = Tensor(-1 * self.data, requires_grad=self.requires_grad)

        out._op = "neg"
        out._prev = {self}

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                self.grad = (self.grad + (-1 * out.grad) if self.grad is not None else (-1 * out.grad))

        out._backward = _backward
        return out

    def backward(self):
        def find_parents(x):
            parents = []
            visited = []
            def visit(node):
                if node not in visited:
                    visited.append(node)
                    for p in node._prev:
                        visit(p)
                    parents.append(node)
                return parents

            visit(x)
            return parents

        self.grad = np.array(1.0) if self.data.shape == () else np.ones_like(self.data)
        parents = find_parents(self)
        for p in parents:
            p._backward()

    def sum(self, axis=-1, keepdims=True):
        self = self if isinstance(self, Tensor) else Tensor(self)
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)

        out._op = "sum"
        out._prev = {self}
        
        def _backward():
            if out.grad is None:
                return
        
            if self.requires_grad:
                if keepdims:
                    self.grad = self.grad + (out.grad * np.ones_like(self.data)) if self.grad is not None else (out.grad * np.ones_like(self.data))
                else:
                    expand_axes = tuple(a for a in range(self.data.ndim) if a not in (axis if isinstance(axis, tuple) else (axis,)))
                    broadcasted_grad = np.broadcast_to(np.expand_dims(out.grad, axis=expand_axes), self.data.shape)
                    self.grad = self.grad + broadcasted_grad if self.grad is not None else broadcasted_grad

        out._backward = _backward
        return out

    def mean(self, axis=-1, keepdims=True):
        self = self if isinstance(self, Tensor) else Tensor(self)
        out = self.sum(axis=axis, keepdims=keepdims) / (self.data.shape[axis] if axis is not None else self.data.size)

        out._op = "mean"
        return out

    def exp(self):
        self = self if isinstance(self, Tensor) else Tensor(self)
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad)

        out._op = "exp"
        out._prev = {self}

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                self.grad = self.grad + out.grad * np.exp(self.data) if self.grad is not None else (out.grad * np.exp(self.data))

        out._backward = _backward
        return out

    def log(self):
        self = self if isinstance(self, Tensor) else Tensor(self)
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad)

        out._op = "log"
        out._prev = {self}

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                self.grad = self.grad + (out.grad * (1 / self.data)) if self.grad is not None else (out.grad * (1 / self.data))

        out._backward = _backward
        return out

    def max(self, axis=-1, keepdims=True):
        self = self if isinstance(self, Tensor) else Tensor(self)
        out = Tensor(np.max(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)

        out._op = "max"
        out._prev = {self}

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                if keepdims:
                    self.grad = self.grad + (out.grad * (self.data == out.data)) if self.grad is not None else (out.grad * (self.data == out.data))
                else:
                    expand_axes = tuple(a for a in range(self.data.ndim) if a not in (axis if isinstance(axis, tuple) else (axis,)))
                    broadcasted_grad = np.broadcast_to(np.expand_dims(out.grad, axis=expand_axes), self.data.shape)
                    broadcasted_data = np.broadcast_to(np.expand_dims(out.data, axis=expand_axes), self.data.shape)
                    self.grad = self.grad + (broadcasted_grad * (self.data == broadcasted_data)) if self.grad is not None else (broadcasted_grad * (self.data == broadcasted_data))

        out._backward = _backward
        return out

    def min(self, axis=-1, keepdims=True):
        self = self if isinstance(self, Tensor) else Tensor(self)
        out = Tensor(np.min(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)

        out._op = "max"
        out._prev = {self}

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                if keepdims:
                    self.grad = self.grad + (out.grad * (self.data == out.data)) if self.grad is not None else (out.grad * (self.data == out.data))
                else:
                    expand_axes = tuple(a for a in range(self.data.ndim) if a not in (axis if isinstance(axis, tuple) else (axis,)))
                    broadcasted_grad = np.broadcast_to(np.expand_dims(out.grad, axis=expand_axes), self.data.shape)
                    broadcasted_data = np.broadcast_to(np.expand_dims(out.data, axis=expand_axes), self.data.shape)
                    self.grad = self.grad + (broadcasted_grad * (self.data == broadcasted_data)) if self.grad is not None else (broadcasted_grad * (self.data == broadcasted_data))

        out._backward = _backward
        return out

    def clip(self, x_min, x_max):
        self = self if isinstance(self, Tensor) else Tensor(self)
        out = Tensor(np.clip(self.data, a_min=x_min, a_max=x_max), requires_grad=self.requires_grad)

        out._op = "clip"
        out._prev = {self}

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                self.grad = self.grad + (out.grad * ((self.data > x_min) & (self.data < x_max)).astype(float)) if self.grad is not None else (out.grad * ((self.data < x_min) & (self.data > x_max)).astype(float))

        out._backward = _backward
        return out

    def pow(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.pow(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad)

        out._op = "pow"
        out._prev = {self, other}

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                self.grad = self.grad + (out.grad * other.data * np.pow(self.data, other.data - 1)) if self.grad is not None else (out.grad * other.data * np.pow(self.data, other.data - 1))

            if other.requires_grad:
                other.grad = other.grad + (out.grad * np.pow(self.data, other.data) * np.log(self.data)) if other.grad is not None else (out.grad * np.pow(self.data, other.data) * np.log(self.data))

        out._backward = _backward
        return out

    def sqrt(self):
        self = self if isinstance(self, Tensor) else Tensor(self)
        out = Tensor(np.sqrt(self.data), requires_grad=self.requires_grad)

        out._op = "sqrt"
        out._prev = {self}

        def _backward():
            if out.grad is None:
                return
    
            if self.requires_grad:
                self.grad = self.grad + (out.grad * (0.5 / np.sqrt(self.data))) if self.grad is not None else (out.grad * (0.5 / np.sqrt(self.data)))

        out._backward = _backward
        return out
