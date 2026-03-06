"""Minimal automatic-differentiation engine backed by NumPy/CuPy.

Every mathematical operation on a :class:`Tensor` records a *backward*
function so that :meth:`Tensor.backward` can later propagate gradients
from the loss all the way back to the leaf parameters.

The design mirrors the essentials of PyTorch / Autograd, but is implemented
entirely in plain Python + NumPy — no ML framework is used.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from llm.backend import array_module

xp = array_module()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _ensure_tensor(value: "Tensor | Any | float | int") -> "Tensor":
    """Wrap a raw value in a Tensor if it is not already one."""
    if isinstance(value, Tensor):
        return value
    return Tensor(value)


def _unbroadcast(grad: Any, target_shape: tuple[int, ...]) -> Any:
    """Reduce *grad* so its shape matches *target_shape*.

    NumPy broadcasting can implicitly expand dimensions; when computing
    gradients we must sum over those expanded axes to match the original
    tensor's shape.
    """
    # 1) Sum over any extra leading dimensions.
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)

    # 2) Sum over axes that were size-1 in the original (broadcast) shape.
    for axis, (orig_size, grad_size) in enumerate(zip(target_shape, grad.shape)):
        if orig_size == 1 and grad_size != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad


# ---------------------------------------------------------------------------
# Tensor class
# ---------------------------------------------------------------------------

class Tensor:
    """A multi-dimensional array that tracks operations for autodiff.

    Leaf tensors (model parameters) are created with ``requires_grad=True``.
    All other tensors are created by operations and carry a ``_backward``
    closure that knows how to send gradients to its parents.

    Example usage::

        x = Tensor([[1.0, 2.0]], requires_grad=True)
        w = Tensor([[3.0], [4.0]], requires_grad=True)
        loss = (x @ w).sum()
        loss.backward()
        print(x.grad)  # dL/dx
    """

    def __init__(
        self,
        data: Any | list | float | int,
        requires_grad: bool = False,
        _children: tuple["Tensor", ...] = (),
        _op: str = "",
    ) -> None:
        self.data = xp.asarray(data, dtype=xp.float32)
        # Gradient accumulates here during backprop (same shape as data).
        self.grad = xp.zeros_like(self.data)
        # Automatically propagate requires_grad: if any parent needs a gradient,
        # this intermediate tensor also needs one so the chain rule can flow through.
        self.requires_grad: bool = requires_grad or any(
            c.requires_grad for c in _children
        )
        # Closure that back-propagates the gradient from this node to parents.
        self._backward: Callable[[], None] = lambda: None
        # Parent tensors in the computation graph.
        self._prev: tuple["Tensor", ...] = _children
        # Name of the operation that created this tensor (for debugging).
        self._op: str = _op

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def T(self) -> "Tensor":
        """Transpose (last two axes for batched matrices)."""
        return self.transpose()

    # ------------------------------------------------------------------
    # Gradient helpers
    # ------------------------------------------------------------------

    def zero_grad(self) -> None:
        """Set the gradient buffer to zero (called before each forward pass)."""
        self.grad = xp.zeros_like(self.data)

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def __add__(self, other: "Tensor | float | int") -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other), _op="+")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other: "float | int") -> "Tensor":
        return self + other

    def __neg__(self) -> "Tensor":
        return self * (-1.0)

    def __sub__(self, other: "Tensor | float | int") -> "Tensor":
        return self + (-_ensure_tensor(other))

    def __rsub__(self, other: "float | int") -> "Tensor":
        return _ensure_tensor(other) + (-self)

    def __mul__(self, other: "Tensor | float | int") -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(self.data * other.data, _children=(self, other), _op="*")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other: "float | int") -> "Tensor":
        return self * other

    def __truediv__(self, other: "Tensor | float | int") -> "Tensor":
        return self * (_ensure_tensor(other) ** -1.0)

    def __rtruediv__(self, other: "float | int") -> "Tensor":
        return _ensure_tensor(other) * (self ** -1.0)

    def __pow__(self, exponent: float | int) -> "Tensor":
        assert isinstance(exponent, (int, float)), "Exponent must be a scalar."
        out = Tensor(self.data**exponent, _children=(self,), _op=f"**{exponent}")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Matrix operations
    # ------------------------------------------------------------------

    def __matmul__(self, other: "Tensor") -> "Tensor":
        """Batched matrix multiplication (supports any number of batch dims)."""
        other = _ensure_tensor(other)
        out = Tensor(self.data @ other.data, _children=(self, other), _op="@")

        def _backward() -> None:
            if self.requires_grad:
                # dL/dA = dL/dC @ B^T
                grad_a = out.grad @ xp.swapaxes(other.data, -1, -2)
                self.grad += _unbroadcast(grad_a, self.data.shape)
            if other.requires_grad:
                # dL/dB = A^T @ dL/dC  — sum over any batch dims not in B.
                grad_b = xp.swapaxes(self.data, -1, -2) @ out.grad
                other.grad += _unbroadcast(grad_b, other.data.shape)

        out._backward = _backward
        return out

    def transpose(self, *axes: int) -> "Tensor":
        """Permute axes.  With no arguments, reverses last two axes."""
        if not axes:
            # Default: swap last two axes (standard matrix transpose).
            perm = list(range(self.ndim))
            perm[-2], perm[-1] = perm[-1], perm[-2]
        else:
            perm = list(axes)
        out = Tensor(xp.transpose(self.data, perm), _children=(self,), _op="T")

        def _backward() -> None:
            if self.requires_grad:
                # Inverse permutation to route gradient back.
                inv_perm = [0] * len(perm)
                for i, p in enumerate(perm):
                    inv_perm[p] = i
                self.grad += xp.transpose(out.grad, inv_perm)

        out._backward = _backward
        return out

    def reshape(self, *shape: int) -> "Tensor":
        out = Tensor(self.data.reshape(*shape), _children=(self,), _op="reshape")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def __getitem__(self, idx: object) -> "Tensor":
        """Index / slice the tensor."""
        out = Tensor(self.data[idx], _children=(self,), _op="index")

        def _backward() -> None:
            if self.requires_grad:
                # Scatter the gradient back to the indexed positions.
                xp.add.at(self.grad, idx, out.grad)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Reduction operations
    # ------------------------------------------------------------------

    def sum(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> "Tensor":
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            _children=(self,),
            _op="sum",
        )

        def _backward() -> None:
            if self.requires_grad:
                grad = out.grad
                if not keepdims and axis is not None:
                    # Re-insert the summed-away axis so broadcasting works.
                    grad = xp.expand_dims(grad, axis=axis)
                self.grad += grad * xp.ones_like(self.data)

        out._backward = _backward
        return out

    def mean(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> "Tensor":
        n = self.data.size if axis is None else self.data.shape[axis]  # type: ignore[index]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ------------------------------------------------------------------
    # Element-wise math
    # ------------------------------------------------------------------

    def exp(self) -> "Tensor":
        out = Tensor(xp.exp(self.data), _children=(self,), _op="exp")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.data * out.grad  # d/dx e^x = e^x

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        """Natural logarithm (numerically stabilised with a small epsilon)."""
        eps = 1e-8
        out = Tensor(xp.log(self.data + eps), _children=(self,), _op="log")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += (1.0 / (self.data + eps)) * out.grad

        out._backward = _backward
        return out

    def sqrt(self) -> "Tensor":
        return self ** 0.5

    def relu(self) -> "Tensor":
        out = Tensor(xp.maximum(0.0, self.data), _children=(self,), _op="relu")

        def _backward() -> None:
            if self.requires_grad:
                self.grad += (self.data > 0).astype(xp.float32) * out.grad

        out._backward = _backward
        return out

    def gelu(self) -> "Tensor":
        """Gaussian Error Linear Unit — fast sigmoid approximation.

        gelu(x) ≈ x · σ(1.702 x)   where σ is the logistic sigmoid.
        """
        sigmoid_val = 1.0 / (1.0 + xp.exp(-1.702 * self.data))
        out = Tensor(self.data * sigmoid_val, _children=(self,), _op="gelu")

        def _backward() -> None:
            if self.requires_grad:
                # d/dx [x · σ(1.702x)] = σ(1.702x) + x · 1.702 · σ(1.702x)(1 − σ(1.702x))
                dsigmoid = 1.702 * sigmoid_val * (1.0 - sigmoid_val)
                dgelu = sigmoid_val + self.data * dsigmoid
                self.grad += dgelu * out.grad

        out._backward = _backward
        return out

    def softmax(self, axis: int = -1) -> "Tensor":
        """Numerically stable softmax along *axis*."""
        # Subtract max for numerical stability (doesn't change the result).
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        e_x = xp.exp(shifted)
        s = e_x / e_x.sum(axis=axis, keepdims=True)
        out = Tensor(s, _children=(self,), _op="softmax")

        def _backward() -> None:
            if self.requires_grad:
                # Jacobian-vector product: dL/dx = s · (dL/ds − ∑ dL/ds·s)
                dot = (out.grad * s).sum(axis=axis, keepdims=True)
                self.grad += s * (out.grad - dot)

        out._backward = _backward
        return out

    def log_softmax(self, axis: int = -1) -> "Tensor":
        """Numerically stable log-softmax (preferred over log(softmax(x)))."""
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        log_sum_exp = xp.log(xp.exp(shifted).sum(axis=axis, keepdims=True))
        out_data = shifted - log_sum_exp
        out = Tensor(out_data, _children=(self,), _op="log_softmax")

        def _backward() -> None:
            if self.requires_grad:
                # d log_softmax / dx = I − softmax(x)
                sm = xp.exp(out_data)
                self.grad += out.grad - sm * out.grad.sum(axis=axis, keepdims=True)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Concatenation / stacking (static helpers)
    # ------------------------------------------------------------------

    @staticmethod
    def cat(tensors: Sequence["Tensor"], axis: int = 0) -> "Tensor":
        """Concatenate tensors along *axis* (like np.concatenate)."""
        out = Tensor(
            xp.concatenate([t.data for t in tensors], axis=axis),
            _children=tuple(tensors),
            _op="cat",
        )
        # Pre-compute split indices for the backward pass.
        sizes = [t.data.shape[axis] for t in tensors]
        splits = xp.cumsum(sizes[:-1]).tolist()

        def _backward() -> None:
            grads = xp.split(out.grad, splits, axis=axis)
            for t, g in zip(tensors, grads):
                if t.requires_grad:
                    t.grad += g

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Backward pass (reverse-mode autodiff)
    # ------------------------------------------------------------------

    def backward(self) -> None:
        """Propagate gradients from this tensor back to all leaf tensors.

        Performs a topological sort of the computation graph and calls each
        node's ``_backward`` closure in reverse order (from output → inputs).
        """
        # Build a topological ordering of all ancestor tensors.
        topo: list[Tensor] = []
        visited: set[int] = set()

        def _build_topo(v: Tensor) -> None:
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    _build_topo(child)
                topo.append(v)

        _build_topo(self)

        # Seed gradient: dL/dL = 1.
        self.grad = xp.ones_like(self.data)

        # Walk the graph in reverse, calling each node's backward closure.
        for v in reversed(topo):
            v._backward()

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Tensor(shape={self.shape}, op='{self._op}', "
            f"requires_grad={self.requires_grad})"
        )
