"""Training loop for the LLM.

Implements:
* Adam optimiser (from scratch — no ML framework).
* Gradient clipping.
* Weight decay (L2 regularisation).
* Periodic evaluation on a held-out validation split.
* Model checkpointing (NumPy .npz files).

Running this module directly (``python -m llm.trainer`` or ``llm-train``)
trains the model on the bundled conversation dataset and saves the weights.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from llm.autograd import Tensor
from llm.config import ModelConfig, TrainingConfig
from llm.data import Dataset, load_text, train_val_split
from llm.model import LanguageModel
from llm.tokenizer import Tokenizer


# ---------------------------------------------------------------------------
# Adam optimiser (from scratch)
# ---------------------------------------------------------------------------

class Adam:
    """Adaptive Moment Estimation (Adam) optimiser.

    Adam maintains a running average of gradients (first moment) and squared
    gradients (second moment) to adaptively scale the learning rate for each
    parameter.

    Reference: Kingma & Ba, 2015 — https://arxiv.org/abs/1412.6980
    """

    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 3e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

        self.t: int = 0  # timestep counter

        # Initialise moment estimates to zero for each parameter.
        self.m: list[np.ndarray] = [np.zeros_like(p.data) for p in parameters]
        self.v: list[np.ndarray] = [np.zeros_like(p.data) for p in parameters]

    def step(self) -> None:
        """Perform one optimisation step using the current gradients."""
        self.t += 1
        # Bias-correction factors compensate for the zero initialisation of moments.
        bc1 = 1.0 - self.beta1**self.t
        bc2 = 1.0 - self.beta2**self.t

        for i, p in enumerate(self.parameters):
            if not p.requires_grad or p.grad is None:
                continue

            g = p.grad.copy()

            # L2 weight decay: add a small gradient proportional to the weight.
            if self.weight_decay != 0.0:
                g += self.weight_decay * p.data

            # Update biased first moment (momentum).
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            # Update biased second moment (RMSprop-like).
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g**2)

            # Compute bias-corrected moments.
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2

            # Parameter update.
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------

def clip_gradients(parameters: list[Tensor], max_norm: float) -> float:
    """Clip parameter gradients so their global L2 norm ≤ *max_norm*.

    Clips in-place and returns the pre-clipping norm (useful for monitoring).
    """
    total_norm = 0.0
    for p in parameters:
        if p.requires_grad and p.grad is not None:
            total_norm += float(np.sum(p.grad**2))
    total_norm = math.sqrt(total_norm)

    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-8)
        for p in parameters:
            if p.requires_grad and p.grad is not None:
                p.grad *= scale

    return total_norm


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def _compute_loss(
    model: LanguageModel,
    dataset: Dataset,
    batch_size: int,
    n_batches: int = 5,
) -> float:
    """Estimate the mean loss over *n_batches* random mini-batches.

    The model is switched to eval mode (dropout disabled) during this call.
    """
    model.eval()
    losses = []
    for _ in range(n_batches):
        x_list, y_list = dataset.get_batch(batch_size)
        x_arr = np.array(x_list, dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32)
        x_t = Tensor(x_arr)
        y_t = Tensor(y_arr)
        loss = model.loss(x_t, y_t)
        losses.append(float(loss.data))
    model.train()
    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

import math  # noqa: E402  (placed here to keep the module header clean)


def train(
    model_config: ModelConfig | None = None,
    train_config: TrainingConfig | None = None,
    data_path: str | None = None,
) -> tuple[LanguageModel, Tokenizer]:
    """Train the LLM on the conversation dataset.

    Returns the trained model and the tokeniser (needed for the chat interface).
    """
    mc = model_config or ModelConfig()
    tc = train_config or TrainingConfig()

    # ---- Load data ---------------------------------------------------------
    print("Loading training data…")
    text = load_text(data_path)
    print(f"  Corpus size: {len(text):,} characters")

    # ---- Build vocabulary --------------------------------------------------
    tokenizer = Tokenizer()
    tokenizer.build_from_text(text)
    mc.vocab_size = tokenizer.vocab_size
    print(f"  Vocabulary size: {tokenizer.vocab_size} characters")

    # ---- Tokenise corpus ---------------------------------------------------
    token_ids = tokenizer.encode(text)
    train_ids, val_ids = train_val_split(token_ids, val_fraction=0.1)

    train_data = Dataset(train_ids, mc.context_length)
    val_data = Dataset(val_ids, mc.context_length)
    print(f"  Train tokens: {len(train_ids):,} | Val tokens: {len(val_ids):,}")

    # ---- Build model -------------------------------------------------------
    np.random.seed(42)  # reproducibility
    model = LanguageModel(mc)
    model.train()
    params = list(model.parameters())
    print(f"  Model parameters: {model.num_parameters():,}")

    # ---- Optimiser ---------------------------------------------------------
    optimiser = Adam(
        params,
        lr=tc.learning_rate,
        beta1=tc.beta1,
        beta2=tc.beta2,
        epsilon=tc.epsilon,
        weight_decay=tc.weight_decay,
    )

    # ---- Training loop -----------------------------------------------------
    print(f"\nTraining for {tc.max_epochs} epochs…")
    start_time = time.time()

    for epoch in range(1, tc.max_epochs + 1):
        # Sample a mini-batch.
        x_list, y_list = train_data.get_batch(tc.batch_size)
        x_arr = np.array(x_list, dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32)
        x_t = Tensor(x_arr)
        y_t = Tensor(y_arr)

        # Zero gradients, forward pass, backward pass.
        model.zero_grad()
        loss = model.loss(x_t, y_t)
        loss.backward()

        # Clip gradients and update parameters.
        clip_gradients(params, tc.grad_clip)
        optimiser.step()

        # ---- Periodic logging ----------------------------------------------
        if epoch % tc.eval_interval == 0 or epoch == tc.max_epochs:
            elapsed = time.time() - start_time
            val_loss = _compute_loss(model, val_data, tc.batch_size)
            train_loss = float(loss.data)
            print(
                f"  Epoch {epoch:>5}/{tc.max_epochs} | "
                f"train loss: {train_loss:.4f} | "
                f"val loss: {val_loss:.4f} | "
                f"elapsed: {elapsed:.1f}s"
            )

    # ---- Save checkpoint ---------------------------------------------------
    print(f"\nSaving model to '{tc.checkpoint_path}'…")
    _save_checkpoint(model, tokenizer, mc, tc.checkpoint_path)
    print("Done.")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    model: LanguageModel,
    tokenizer: Tokenizer,
    config: ModelConfig,
    path: str,
) -> None:
    """Save model weights and vocabulary to a single .npz file."""
    state = model.state_dict()
    # Store config fields as 0-d arrays so they survive the npz round-trip.
    meta = {
        "__vocab_size": np.array(config.vocab_size),
        "__context_length": np.array(config.context_length),
        "__d_model": np.array(config.d_model),
        "__n_heads": np.array(config.n_heads),
        "__n_layers": np.array(config.n_layers),
        "__d_ff": np.array(config.d_ff),
        # Encode the vocabulary JSON as a bytes array.
        "__vocab_json": np.frombuffer(
            _vocab_to_json(tokenizer).encode("utf-8"), dtype=np.uint8
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **state, **meta)


def load_checkpoint(
    path: str,
) -> tuple[LanguageModel, Tokenizer]:
    """Load a model and tokeniser from a .npz checkpoint."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: '{path}'")

    data = np.load(path, allow_pickle=False)

    # Restore config.
    config = ModelConfig(
        vocab_size=int(data["__vocab_size"]),
        context_length=int(data["__context_length"]),
        d_model=int(data["__d_model"]),
        n_heads=int(data["__n_heads"]),
        n_layers=int(data["__n_layers"]),
        d_ff=int(data["__d_ff"]),
    )

    # Restore tokeniser.
    import json
    vocab_json = bytes(data["__vocab_json"].tolist()).decode("utf-8")
    tokenizer = Tokenizer()
    tokenizer._char_to_idx = json.loads(vocab_json)
    tokenizer._idx_to_char = {v: k for k, v in tokenizer._char_to_idx.items()}

    # Rebuild model.
    np.random.seed(0)
    model = LanguageModel(config)

    # Load weights (skip the metadata keys that start with "__").
    state = {k: data[k] for k in data.files if not k.startswith("__")}
    model.load_state_dict(state)

    return model, tokenizer


def _vocab_to_json(tokenizer: Tokenizer) -> str:
    import json
    return json.dumps(tokenizer._char_to_idx, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line entry point for ``llm-train``."""
    parser = argparse.ArgumentParser(
        description="Train the LLM on a conversation corpus."
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to the plain-text training file (default: data/conversations.txt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini-batch size (default: 16).",
    )
    parser.add_argument(
        "--checkpoint",
        default="model.npz",
        help="Where to save the trained weights (default: model.npz).",
    )
    args = parser.parse_args()

    tc = TrainingConfig(
        max_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint,
    )

    try:
        train(train_config=tc, data_path=args.data)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
