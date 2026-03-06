"""Neural-network modules built on top of the autograd Tensor.

Each module exposes:
* ``forward(*inputs)`` — the forward computation.
* ``parameters()``     — an iterator over trainable Tensors.
* ``zero_grad()``      — resets all parameter gradients to zero.
* ``train() / eval()`` — toggle training vs. inference mode.

No ML framework is used.  All maths is expressed in terms of Tensor
operations whose gradients are tracked automatically by ``autograd.py``.
"""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np

from llm.autograd import Tensor


# ---------------------------------------------------------------------------
# Base module
# ---------------------------------------------------------------------------

class Module:
    """Abstract base class for all neural-network layers."""

    def __init__(self) -> None:
        self._training: bool = True

    # Subclasses must implement this.
    def forward(self, *args: Tensor) -> Tensor:  # type: ignore[return]
        raise NotImplementedError

    def __call__(self, *args: Tensor) -> Tensor:
        return self.forward(*args)

    def parameters(self) -> Iterator[Tensor]:
        """Yield all trainable parameters in this module (and sub-modules)."""
        for value in self.__dict__.values():
            if isinstance(value, Tensor) and value.requires_grad:
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Module):
                        yield from item.parameters()

    def zero_grad(self) -> None:
        """Reset gradients for every parameter to zero."""
        for p in self.parameters():
            p.zero_grad()

    def train(self) -> "Module":
        """Switch this module (and all sub-modules) to training mode."""
        self._training = True
        for value in self.__dict__.values():
            if isinstance(value, Module):
                value.train()
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Module):
                        item.train()
        return self

    def eval(self) -> "Module":
        """Switch this module (and all sub-modules) to evaluation mode."""
        self._training = False
        for value in self.__dict__.values():
            if isinstance(value, Module):
                value.eval()
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Module):
                        item.eval()
        return self

    # ------------------------------------------------------------------
    # Weight persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, np.ndarray]:
        """Return a flat dict of parameter name → numpy array."""
        params: dict[str, np.ndarray] = {}
        self._collect_state(self, "", params)
        return params

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        """Load weights from a flat dict (as returned by :meth:`state_dict`)."""
        self._apply_state(self, "", state)

    # Internal helpers for recursive state dict traversal.
    def _collect_state(
        self, module: "Module", prefix: str, out: dict[str, np.ndarray]
    ) -> None:
        for name, value in module.__dict__.items():
            key = f"{prefix}{name}" if prefix else name
            if isinstance(value, Tensor) and value.requires_grad:
                out[key] = value.data
            elif isinstance(value, Module):
                self._collect_state(value, key + ".", out)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        self._collect_state(item, f"{key}.{i}.", out)

    def _apply_state(
        self, module: "Module", prefix: str, state: dict[str, np.ndarray]
    ) -> None:
        for name, value in module.__dict__.items():
            key = f"{prefix}{name}" if prefix else name
            if isinstance(value, Tensor) and value.requires_grad:
                if key in state:
                    value.data = state[key].astype(np.float32)
            elif isinstance(value, Module):
                self._apply_state(value, key + ".", state)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        self._apply_state(item, f"{key}.{i}.", state)


# ---------------------------------------------------------------------------
# Fundamental layers
# ---------------------------------------------------------------------------

class Embedding(Module):
    """A lookup table that maps integer token IDs to dense embedding vectors.

    Parameters
    ----------
    num_embeddings:
        Vocabulary size (number of distinct tokens).
    embedding_dim:
        Dimensionality of each embedding vector.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        # Initialise with small random values (scaled by 1/sqrt(embedding_dim)).
        scale = 1.0 / math.sqrt(embedding_dim)
        self.weight = Tensor(
            np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * scale,
            requires_grad=True,
        )

    def forward(self, indices: Tensor) -> Tensor:  # type: ignore[override]
        """Look up embedding vectors for each index in *indices*.

        Args:
            indices: Integer tensor of shape (batch, seq_len).

        Returns:
            Float tensor of shape (batch, seq_len, embedding_dim).
        """
        return self.weight[indices.data.astype(int)]


class Linear(Module):
    """Fully-connected (affine) layer: y = x W^T + b.

    Parameters
    ----------
    in_features:
        Number of input features.
    out_features:
        Number of output features.
    bias:
        Whether to include a learnable bias vector.
    """

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True
    ) -> None:
        super().__init__()
        # Kaiming uniform initialisation (good default for layers before ReLU/GELU).
        scale = math.sqrt(2.0 / in_features)
        self.weight = Tensor(
            np.random.randn(in_features, out_features).astype(np.float32) * scale,
            requires_grad=True,
        )
        if bias:
            self.bias: Tensor | None = Tensor(
                np.zeros(out_features, dtype=np.float32), requires_grad=True
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class LayerNorm(Module):
    """Layer normalisation over the last dimension.

    Normalises each feature vector to have zero mean and unit variance,
    then rescales with learnable parameters γ (scale) and β (shift).

    y = γ · (x − μ) / √(σ² + ε) + β
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        # Learnable scale and shift (initialised to 1 and 0 respectively).
        self.gamma = Tensor(np.ones(d_model, dtype=np.float32), requires_grad=True)
        self.beta = Tensor(np.zeros(d_model, dtype=np.float32), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # Compute mean and variance along the last axis (the feature dimension).
        mean = x.mean(axis=-1, keepdims=True)
        # Variance: E[(x - μ)²]
        diff = x - mean
        var = (diff * diff).mean(axis=-1, keepdims=True)
        # Normalise.
        x_hat = diff * ((var + self.eps) ** -0.5)
        # Scale and shift.
        return x_hat * self.gamma + self.beta


class Dropout(Module):
    """Randomly zeroes elements during training (for regularisation).

    During evaluation mode the layer is an identity function.
    """

    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        if not self._training or self.p == 0.0:
            return x
        # Bernoulli mask scaled by 1/(1-p) so the expected value is unchanged.
        keep_prob = 1.0 - self.p
        mask = (np.random.rand(*x.shape) < keep_prob).astype(np.float32)
        # We build a new Tensor so the mask is baked into the computation graph.
        mask_t = Tensor(mask / keep_prob)
        return x * mask_t


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(Module):
    """Causal multi-head self-attention (the core of the transformer).

    Uses a causal mask so each token can only attend to *previous* tokens
    (auto-regressive / decoder-style attention).

    Parameters
    ----------
    d_model:
        Total model dimension.
    n_heads:
        Number of parallel attention heads.  ``d_model`` must be divisible
        by ``n_heads``.
    dropout:
        Dropout probability applied to attention weights.
    context_length:
        Maximum sequence length (used to pre-compute the causal mask).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        context_length: int = 128,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # dimension per head
        self.scale = math.sqrt(self.d_head)

        # Single fused projection for Q, K, V (3 × d_model outputs).
        self.qkv_proj = Linear(d_model, 3 * d_model, bias=False)
        # Output projection that mixes the heads back together.
        self.out_proj = Linear(d_model, d_model)
        self.attn_dropout = Dropout(dropout)
        self.resid_dropout = Dropout(dropout)

        # Pre-computed causal mask: lower-triangular of ones.
        # Shape: (1, 1, context_length, context_length) for broadcasting.
        causal = np.tril(np.ones((context_length, context_length), dtype=np.float32))
        self._causal_mask = causal  # plain numpy, not a Tensor parameter

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """Apply causal multi-head attention.

        Args:
            x: Input of shape (batch, seq_len, d_model).

        Returns:
            Output of shape (batch, seq_len, d_model).
        """
        B, T, C = x.shape

        # ---- 1. Project to Q, K, V ----------------------------------------
        qkv = self.qkv_proj(x)          # (B, T, 3·C)
        # Split into Q, K, V along the last dimension.
        q = qkv[:, :, : C]             # (B, T, C)
        k = qkv[:, :, C : 2 * C]       # (B, T, C)
        v = qkv[:, :, 2 * C :]         # (B, T, C)

        # ---- 2. Reshape into heads -----------------------------------------
        # (B, T, C) → (B, n_heads, T, d_head)
        def split_heads(t: Tensor) -> Tensor:
            # Reshape to (B, T, n_heads, d_head), then transpose to (B, n_heads, T, d_head).
            t = t.reshape(B, T, self.n_heads, self.d_head)
            return t.transpose(0, 2, 1, 3)  # (B, nh, T, dh)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # ---- 3. Scaled dot-product attention --------------------------------
        # scores: (B, nh, T, T)
        scores = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / self.scale)

        # Apply causal mask: positions that a token cannot attend to get −∞.
        causal_mask = self._causal_mask[:T, :T]  # (T, T)
        # Convert 0s to large negative numbers (−1e9 ≈ −∞ before softmax).
        neg_inf_mask = Tensor((1.0 - causal_mask) * -1e9)
        scores = scores + neg_inf_mask  # broadcast over (B, nh) dims

        attn = scores.softmax(axis=-1)          # (B, nh, T, T)
        attn = self.attn_dropout(attn)

        # Weighted sum of value vectors.
        out = attn @ v                          # (B, nh, T, dh)

        # ---- 4. Merge heads back -------------------------------------------
        # (B, nh, T, dh) → (B, T, nh, dh) → (B, T, C)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)

        # ---- 5. Output projection ------------------------------------------
        return self.resid_dropout(self.out_proj(out))


class FeedForward(Module):
    """Position-wise feed-forward network (applied independently to each token).

    Architecture: Linear → GELU → Linear → Dropout
    The hidden dimension is typically 4× the model dimension.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.dropout(self.fc2(self.fc1(x).gelu()))


class TransformerBlock(Module):
    """One transformer decoder block.

    Consists of:
    1. Pre-norm multi-head causal self-attention + residual connection.
    2. Pre-norm feed-forward network + residual connection.

    Pre-norm (LayerNorm *before* the sub-layer) tends to train more
    stably than post-norm.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        context_length: int = 128,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, context_length)
        self.norm2 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # Self-attention with residual connection.
        x = x + self.attn(self.norm1(x))
        # Feed-forward network with residual connection.
        x = x + self.ff(self.norm2(x))
        return x
