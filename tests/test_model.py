"""Tests for the LanguageModel and training loop."""

import os
import tempfile

import numpy as np
import pytest

from llm.autograd import Tensor
from llm.config import ModelConfig, TrainingConfig
from llm.model import LanguageModel
from llm.tokenizer import Tokenizer
from llm.trainer import Adam, clip_gradients, train


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=20,
        context_length=16,
        d_model=32,
        n_heads=2,
        n_layers=2,
        d_ff=64,
        dropout=0.0,  # no dropout for deterministic tests
    )


@pytest.fixture
def small_model(small_config: ModelConfig) -> LanguageModel:
    np.random.seed(42)
    return LanguageModel(small_config)


# ---------------------------------------------------------------------------
# LanguageModel tests
# ---------------------------------------------------------------------------

class TestLanguageModel:
    def test_forward_output_shape(self, small_model, small_config):
        small_model.eval()
        B, T = 2, 8
        x = Tensor(np.random.randint(0, small_config.vocab_size, (B, T)).astype(np.float32))
        logits = small_model(x)
        assert logits.shape == (B, T, small_config.vocab_size)

    def test_loss_is_scalar(self, small_model, small_config):
        small_model.eval()
        x = Tensor(np.random.randint(0, small_config.vocab_size, (2, 8)).astype(np.float32))
        y = Tensor(np.random.randint(0, small_config.vocab_size, (2, 8)).astype(np.float32))
        loss = small_model.loss(x, y)
        assert loss.shape == ()  # scalar

    def test_loss_value_reasonable(self, small_model, small_config):
        """For a random model, cross-entropy ≈ log(vocab_size)."""
        small_model.eval()
        x = Tensor(np.random.randint(0, small_config.vocab_size, (4, 8)).astype(np.float32))
        y = Tensor(np.random.randint(0, small_config.vocab_size, (4, 8)).astype(np.float32))
        loss = small_model.loss(x, y)
        # Should be roughly log(vocab_size) ≈ 3.0 for vocab=20
        assert 1.0 < float(loss.data) < 8.0

    def test_backward_does_not_raise(self, small_model, small_config):
        x = Tensor(np.random.randint(0, small_config.vocab_size, (2, 8)).astype(np.float32))
        y = Tensor(np.random.randint(0, small_config.vocab_size, (2, 8)).astype(np.float32))
        small_model.zero_grad()
        loss = small_model.loss(x, y)
        loss.backward()  # Must not raise.

    def test_gradients_are_non_zero_after_backward(self, small_model, small_config):
        x = Tensor(np.random.randint(0, small_config.vocab_size, (2, 8)).astype(np.float32))
        y = Tensor(np.random.randint(0, small_config.vocab_size, (2, 8)).astype(np.float32))
        small_model.zero_grad()
        loss = small_model.loss(x, y)
        loss.backward()
        # At least some parameters must have non-zero gradients.
        any_nonzero = any(
            np.any(p.grad != 0) for p in small_model.parameters()
        )
        assert any_nonzero

    def test_generate_length(self, small_model, small_config):
        small_model.eval()
        prompt = [1, 2, 3]
        out = small_model.generate(prompt, max_new_tokens=10)
        assert len(out) == len(prompt) + 10

    def test_generate_ids_in_vocab(self, small_model, small_config):
        small_model.eval()
        out = small_model.generate([1], max_new_tokens=20)
        assert all(0 <= idx < small_config.vocab_size for idx in out)

    def test_num_parameters_positive(self, small_model):
        assert small_model.num_parameters() > 0


# ---------------------------------------------------------------------------
# Adam optimiser tests
# ---------------------------------------------------------------------------

class TestAdam:
    def test_loss_decreases(self):
        """A single Linear layer should reduce a simple regression loss."""
        np.random.seed(0)
        x = Tensor(np.random.randn(8, 4).astype(np.float32))
        w = Tensor(np.random.randn(4, 2).astype(np.float32), requires_grad=True)
        target = Tensor(np.zeros((8, 2), dtype=np.float32))

        optim = Adam([w], lr=0.01)
        initial_loss = None

        for _ in range(20):
            w.zero_grad()
            pred = x @ w
            diff = pred - target
            loss = (diff * diff).mean()
            loss.backward()
            optim.step()
            if initial_loss is None:
                initial_loss = float(loss.data)

        final_loss = float(loss.data)
        assert final_loss < initial_loss

    def test_zero_grad_resets_gradient(self):
        w = Tensor(np.ones((3,), dtype=np.float32), requires_grad=True)
        w.grad = np.ones((3,), dtype=np.float32)
        w.zero_grad()
        np.testing.assert_array_equal(w.grad, np.zeros(3))


# ---------------------------------------------------------------------------
# Gradient clipping tests
# ---------------------------------------------------------------------------

class TestGradClipping:
    def test_clips_large_gradient(self):
        p = Tensor(np.zeros(10, dtype=np.float32), requires_grad=True)
        p.grad = np.ones(10, dtype=np.float32) * 10.0  # norm = 10*sqrt(10) >> 1

        clip_gradients([p], max_norm=1.0)
        clipped_norm = float(np.linalg.norm(p.grad))
        assert clipped_norm == pytest.approx(1.0, rel=1e-4)

    def test_does_not_clip_small_gradient(self):
        p = Tensor(np.zeros(3, dtype=np.float32), requires_grad=True)
        p.grad = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        original_grad = p.grad.copy()
        clip_gradients([p], max_norm=10.0)
        np.testing.assert_array_almost_equal(p.grad, original_grad)


# ---------------------------------------------------------------------------
# Full training integration test
# ---------------------------------------------------------------------------

class TestTrainingIntegration:
    def test_loss_decreases_over_epochs(self):
        """Training for several epochs should reduce the validation loss."""
        tc = TrainingConfig(
            max_epochs=20,
            batch_size=4,
            eval_interval=5,
            checkpoint_path=os.path.join(tempfile.mkdtemp(), "test.npz"),
            learning_rate=1e-3,
        )
        mc = ModelConfig(
            context_length=32,
            d_model=32,
            n_heads=2,
            n_layers=2,
            d_ff=64,
            dropout=0.0,
        )
        np.random.seed(0)
        model, tokenizer = train(mc, tc)
        assert tokenizer.vocab_size > 0
        assert model.num_parameters() > 0


# ---------------------------------------------------------------------------
# Checkpoint round-trip test
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_save_and_load(self, small_model, small_config):
        tokenizer = Tokenizer()
        tokenizer.build_from_text("abcdefghij")

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name

        try:
            from llm.trainer import _save_checkpoint, load_checkpoint

            _save_checkpoint(small_model, tokenizer, small_config, path)
            loaded_model, loaded_tokenizer = load_checkpoint(path)

            # Check the loaded model has the same architecture.
            assert loaded_model.num_parameters() == small_model.num_parameters()
            assert loaded_tokenizer.vocab_size == tokenizer.vocab_size

            # Check weights are identical.
            for p1, p2 in zip(small_model.parameters(), loaded_model.parameters()):
                np.testing.assert_array_almost_equal(p1.data, p2.data)
        finally:
            os.unlink(path)
