"""Tests for neural network modules (nn.py)."""

import numpy as np
import pytest

from llm.autograd import Tensor
from llm.nn import (
    Dropout,
    Embedding,
    FeedForward,
    LayerNorm,
    Linear,
    MultiHeadAttention,
    TransformerBlock,
)


class TestLinear:
    def test_output_shape(self):
        layer = Linear(4, 8)
        x = Tensor(np.random.randn(2, 4).astype(np.float32))
        out = layer(x)
        assert out.shape == (2, 8)

    def test_batched_output_shape(self):
        layer = Linear(16, 32)
        x = Tensor(np.random.randn(3, 10, 16).astype(np.float32))
        out = layer(x)
        assert out.shape == (3, 10, 32)

    def test_gradient_flows(self):
        layer = Linear(4, 2)
        x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad.shape == x.data.shape
        assert layer.weight.grad.shape == layer.weight.data.shape

    def test_no_bias(self):
        layer = Linear(4, 8, bias=False)
        assert layer.bias is None
        params = list(layer.parameters())
        assert len(params) == 1  # only weight

    def test_parameters(self):
        layer = Linear(4, 8)
        params = list(layer.parameters())
        assert len(params) == 2  # weight + bias


class TestLayerNorm:
    def test_output_shape(self):
        ln = LayerNorm(8)
        x = Tensor(np.random.randn(2, 5, 8).astype(np.float32))
        out = ln(x)
        assert out.shape == (2, 5, 8)

    def test_normalises_mean_and_variance(self):
        """After LayerNorm, each feature vector should have ≈0 mean, ≈1 var."""
        ln = LayerNorm(64)
        x = Tensor(np.random.randn(8, 64).astype(np.float32) * 10 + 5)
        out = ln(x)
        means = out.data.mean(axis=-1)
        vars_ = out.data.var(axis=-1)
        np.testing.assert_array_almost_equal(means, np.zeros(8), decimal=5)
        np.testing.assert_array_almost_equal(vars_, np.ones(8), decimal=4)

    def test_gradient_flows(self):
        ln = LayerNorm(8)
        x = Tensor(np.random.randn(3, 8).astype(np.float32), requires_grad=True)
        out = ln(x)
        out.sum().backward()
        assert x.grad.shape == x.data.shape


class TestEmbedding:
    def test_output_shape(self):
        emb = Embedding(10, 16)
        idx = Tensor(np.array([[0, 1, 2]], dtype=np.float32))
        out = emb(idx)
        assert out.shape == (1, 3, 16)

    def test_lookup_values(self):
        emb = Embedding(5, 4)
        emb.weight.data = np.eye(5, 4, dtype=np.float32)
        idx = Tensor(np.array([[0, 2]], dtype=np.float32))
        out = emb(idx)
        np.testing.assert_array_almost_equal(out.data[0, 0], emb.weight.data[0])
        np.testing.assert_array_almost_equal(out.data[0, 1], emb.weight.data[2])

    def test_gradient_flows(self):
        emb = Embedding(10, 8)
        idx = Tensor(np.array([[1, 3, 5]], dtype=np.float32))
        out = emb(idx)
        out.sum().backward()
        assert emb.weight.grad.shape == emb.weight.data.shape


class TestDropout:
    def test_identity_in_eval_mode(self):
        drop = Dropout(p=0.9)
        drop.eval()
        x = Tensor(np.ones((100, 100), dtype=np.float32))
        out = drop(x)
        np.testing.assert_array_equal(out.data, x.data)

    def test_zeroes_elements_in_train_mode(self):
        np.random.seed(0)
        drop = Dropout(p=0.5)
        drop.train()
        x = Tensor(np.ones((1000,), dtype=np.float32))
        out = drop(x)
        # Roughly half the elements should be zero.
        zero_fraction = np.mean(out.data == 0.0)
        assert 0.3 < zero_fraction < 0.7

    def test_scaled_correctly_in_train_mode(self):
        np.random.seed(0)
        drop = Dropout(p=0.5)
        drop.train()
        x = Tensor(np.ones((10000,), dtype=np.float32))
        out = drop(x)
        # Non-zero elements should be scaled up by 1/(1-p) = 2.
        non_zero = out.data[out.data != 0.0]
        np.testing.assert_array_almost_equal(non_zero, np.full_like(non_zero, 2.0))


class TestMultiHeadAttention:
    def test_output_shape(self):
        attn = MultiHeadAttention(d_model=16, n_heads=2, context_length=10)
        attn.eval()
        x = Tensor(np.random.randn(2, 5, 16).astype(np.float32))
        out = attn(x)
        assert out.shape == (2, 5, 16)

    def test_causal_mask(self):
        """Future tokens should not influence past token representations."""
        np.random.seed(7)
        attn = MultiHeadAttention(d_model=8, n_heads=2, context_length=4)
        attn.eval()

        x1 = Tensor(np.random.randn(1, 4, 8).astype(np.float32))
        out1 = attn(x1)

        # Change the last token; the output at position 0 must not change.
        x2_data = x1.data.copy()
        x2_data[0, -1, :] += 99.0
        x2 = Tensor(x2_data)
        out2 = attn(x2)

        np.testing.assert_array_almost_equal(
            out1.data[0, 0, :], out2.data[0, 0, :], decimal=5
        )

    def test_gradient_flows(self):
        attn = MultiHeadAttention(d_model=8, n_heads=2, context_length=6)
        x = Tensor(np.random.randn(1, 4, 8).astype(np.float32), requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad.shape == x.data.shape


class TestFeedForward:
    def test_output_shape(self):
        ff = FeedForward(d_model=16, d_ff=64)
        ff.eval()
        x = Tensor(np.random.randn(2, 5, 16).astype(np.float32))
        out = ff(x)
        assert out.shape == (2, 5, 16)

    def test_gradient_flows(self):
        ff = FeedForward(d_model=8, d_ff=32)
        x = Tensor(np.random.randn(2, 3, 8).astype(np.float32), requires_grad=True)
        out = ff(x)
        out.sum().backward()
        assert x.grad.shape == x.data.shape


class TestTransformerBlock:
    def test_output_shape(self):
        block = TransformerBlock(d_model=16, n_heads=2, d_ff=64, context_length=10)
        block.eval()
        x = Tensor(np.random.randn(2, 5, 16).astype(np.float32))
        out = block(x)
        assert out.shape == (2, 5, 16)

    def test_residual_connections(self):
        """Output must be close to input for a freshly initialised block (small weights)."""
        block = TransformerBlock(d_model=8, n_heads=2, d_ff=32, context_length=6)
        block.eval()
        # Zero out the weights so the block acts as an identity.
        for p in block.parameters():
            p.data[:] = 0.0
        block.norm1.gamma.data[:] = 1.0
        block.norm2.gamma.data[:] = 1.0

        x = Tensor(np.random.randn(1, 4, 8).astype(np.float32))
        out = block(x)
        # With zeroed projections, output ≈ input (residual path dominates).
        np.testing.assert_array_almost_equal(out.data, x.data, decimal=4)

    def test_parameter_count(self):
        block = TransformerBlock(d_model=32, n_heads=4, d_ff=128, context_length=16)
        n = sum(p.data.size for p in block.parameters())
        assert n > 0
