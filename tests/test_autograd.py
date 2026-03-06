"""Tests for the autograd engine (Tensor class)."""

import math

import numpy as np
import pytest

from llm.autograd import Tensor


class TestTensorCreation:
    def test_basic_creation(self):
        t = Tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)
        assert t.ndim == 1
        np.testing.assert_array_almost_equal(t.data, [1.0, 2.0, 3.0])

    def test_requires_grad_false_by_default(self):
        t = Tensor([1.0])
        assert not t.requires_grad

    def test_requires_grad_true(self):
        t = Tensor([1.0], requires_grad=True)
        assert t.requires_grad

    def test_grad_initialised_to_zero(self):
        t = Tensor([[1.0, 2.0]], requires_grad=True)
        np.testing.assert_array_equal(t.grad, np.zeros((1, 2)))


class TestArithmetic:
    def test_add_scalars(self):
        a = Tensor(2.0, requires_grad=True)
        b = Tensor(3.0, requires_grad=True)
        c = a + b
        c.backward()
        assert float(c.data) == pytest.approx(5.0)
        assert float(a.grad) == pytest.approx(1.0)
        assert float(b.grad) == pytest.approx(1.0)

    def test_mul_scalars(self):
        a = Tensor(2.0, requires_grad=True)
        b = Tensor(4.0, requires_grad=True)
        c = a * b
        c.backward()
        assert float(c.data) == pytest.approx(8.0)
        assert float(a.grad) == pytest.approx(4.0)  # dL/da = b
        assert float(b.grad) == pytest.approx(2.0)  # dL/db = a

    def test_sub(self):
        a = Tensor(5.0, requires_grad=True)
        b = Tensor(3.0, requires_grad=True)
        c = a - b
        c.backward()
        assert float(c.data) == pytest.approx(2.0)
        assert float(a.grad) == pytest.approx(1.0)
        assert float(b.grad) == pytest.approx(-1.0)

    def test_div(self):
        a = Tensor(6.0, requires_grad=True)
        b = Tensor(2.0, requires_grad=True)
        c = a / b
        c.backward()
        assert float(c.data) == pytest.approx(3.0)
        # dL/da = 1/b = 0.5
        assert float(a.grad) == pytest.approx(0.5)

    def test_pow(self):
        a = Tensor(3.0, requires_grad=True)
        c = a ** 2
        c.backward()
        assert float(c.data) == pytest.approx(9.0)
        assert float(a.grad) == pytest.approx(6.0)  # 2 * 3^1

    def test_neg(self):
        a = Tensor(5.0, requires_grad=True)
        b = -a
        b.backward()
        assert float(b.data) == pytest.approx(-5.0)
        assert float(a.grad) == pytest.approx(-1.0)

    def test_radd(self):
        a = Tensor(3.0, requires_grad=True)
        b = 1.0 + a
        b.backward()
        assert float(b.data) == pytest.approx(4.0)

    def test_rmul(self):
        a = Tensor(3.0, requires_grad=True)
        b = 2.0 * a
        b.backward()
        assert float(b.data) == pytest.approx(6.0)
        assert float(a.grad) == pytest.approx(2.0)

    def test_chain_rule(self):
        """d/dx [(x+2)*(x+3)] at x=1 = (x+3) + (x+2) = 4+3 = 7."""
        x = Tensor(1.0, requires_grad=True)
        y = (x + 2.0) * (x + 3.0)
        y.backward()
        assert float(y.data) == pytest.approx(12.0)
        assert float(x.grad) == pytest.approx(7.0)


class TestMatMul:
    def test_matmul_2d(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        c = a @ b
        s = c.sum()
        s.backward()
        np.testing.assert_array_almost_equal(c.data, [[1.0, 2.0], [3.0, 4.0]])

    def test_matmul_batched_gradient(self):
        """Gradient of a weight matrix used in a batched matmul."""
        # Simulates a Linear layer: (batch=2, seq=3, in=4) @ (in=4, out=5)
        x = Tensor(np.random.randn(2, 3, 4).astype(np.float32), requires_grad=True)
        w = Tensor(np.random.randn(4, 5).astype(np.float32), requires_grad=True)
        out = x @ w
        out.sum().backward()
        # Weight gradient must have the same shape as w.
        assert w.grad.shape == w.data.shape


class TestReductions:
    def test_sum_all(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        s = a.sum()
        s.backward()
        assert float(s.data) == pytest.approx(10.0)
        np.testing.assert_array_almost_equal(a.grad, np.ones((2, 2)))

    def test_sum_axis(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        s = a.sum(axis=0)
        s.sum().backward()
        assert s.shape == (2,)

    def test_mean(self):
        a = Tensor([2.0, 4.0, 6.0], requires_grad=True)
        m = a.mean()
        m.backward()
        assert float(m.data) == pytest.approx(4.0)
        np.testing.assert_array_almost_equal(a.grad, [1 / 3, 1 / 3, 1 / 3])


class TestActivations:
    def test_relu(self):
        a = Tensor([-1.0, 0.0, 2.0], requires_grad=True)
        b = a.relu()
        b.sum().backward()
        np.testing.assert_array_almost_equal(b.data, [0.0, 0.0, 2.0])
        np.testing.assert_array_almost_equal(a.grad, [0.0, 0.0, 1.0])

    def test_gelu_output_shape(self):
        a = Tensor(np.random.randn(4, 8).astype(np.float32), requires_grad=True)
        b = a.gelu()
        assert b.shape == a.shape
        b.sum().backward()
        assert a.grad.shape == a.shape

    def test_exp(self):
        a = Tensor(0.0, requires_grad=True)
        b = a.exp()
        b.backward()
        assert float(b.data) == pytest.approx(1.0)
        assert float(a.grad) == pytest.approx(1.0)  # d/dx e^x at x=0 = 1

    def test_log(self):
        a = Tensor(math.e, requires_grad=True)
        b = a.log()
        b.backward()
        assert float(b.data) == pytest.approx(1.0)
        assert float(a.grad) == pytest.approx(1.0 / math.e, rel=1e-4)

    def test_softmax_sums_to_one(self):
        a = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        p = a.softmax(axis=-1)
        assert np.sum(p.data) == pytest.approx(1.0)

    def test_softmax_gradient(self):
        a = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        p = a.softmax(axis=-1)
        p.sum().backward()
        # The gradient of sum(softmax(x)) w.r.t. x should be all zeros
        # because sum(softmax(x)) = 1 is constant.
        np.testing.assert_array_almost_equal(a.grad, np.zeros_like(a.data), decimal=5)

    def test_log_softmax(self):
        a = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        log_p = a.log_softmax(axis=-1)
        # log_softmax should equal log(softmax)
        expected = np.log(
            np.exp([1.0, 2.0, 3.0]) / np.exp([1.0, 2.0, 3.0]).sum()
        )
        np.testing.assert_array_almost_equal(log_p.data[0], expected, decimal=5)


class TestIndexing:
    def test_getitem(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = a[0]
        b.sum().backward()
        # Gradient only flows back to the indexed row.
        np.testing.assert_array_almost_equal(a.grad, [[1.0, 1.0], [0.0, 0.0]])

    def test_embedding_style_indexing(self):
        """Simulate an embedding lookup."""
        emb = Tensor(np.eye(5, dtype=np.float32), requires_grad=True)
        idx = np.array([0, 2, 4])
        out = emb[idx]
        out.sum().backward()
        # Rows 0, 2, 4 should have grad 1; others 0.
        expected = np.zeros((5, 5), dtype=np.float32)
        expected[[0, 2, 4]] = 1.0
        np.testing.assert_array_almost_equal(emb.grad, expected)


class TestReshapeTranspose:
    def test_reshape(self):
        a = Tensor(np.arange(6, dtype=np.float32), requires_grad=True)
        b = a.reshape(2, 3)
        b.sum().backward()
        assert a.grad.shape == (6,)
        np.testing.assert_array_almost_equal(a.grad, np.ones(6))

    def test_transpose(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = a.T
        assert b.shape == (2, 2)
        b.sum().backward()
        np.testing.assert_array_almost_equal(a.grad, np.ones((2, 2)))

    def test_transpose_batched(self):
        a = Tensor(np.random.randn(2, 3, 4).astype(np.float32), requires_grad=True)
        b = a.transpose(0, 2, 1)
        assert b.shape == (2, 4, 3)
        b.sum().backward()
        assert a.grad.shape == (2, 3, 4)


class TestCat:
    def test_cat_axis0(self):
        a = Tensor([[1.0, 2.0]], requires_grad=True)
        b = Tensor([[3.0, 4.0]], requires_grad=True)
        c = Tensor.cat([a, b], axis=0)
        assert c.shape == (2, 2)
        c.sum().backward()
        np.testing.assert_array_almost_equal(a.grad, np.ones((1, 2)))
        np.testing.assert_array_almost_equal(b.grad, np.ones((1, 2)))

    def test_cat_axis1(self):
        a = Tensor([[1.0, 2.0]], requires_grad=True)
        b = Tensor([[3.0, 4.0, 5.0]], requires_grad=True)
        c = Tensor.cat([a, b], axis=1)
        assert c.shape == (1, 5)
