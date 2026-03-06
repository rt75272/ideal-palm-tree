"""Configuration dataclasses for the LLM model and training loop.

All hyper-parameters live here so they are easy to locate and change.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Hyper-parameters that define the transformer architecture.

    Keep the defaults small so the model can be trained on a CPU in a
    reasonable amount of time while still demonstrating all the key ideas.
    """

    # ---- Vocabulary / tokeniser ----------------------------------------
    # Filled in automatically by the Tokenizer after it reads the data.
    vocab_size: int = 512

    # ---- Sequence length -----------------------------------------------
    # Maximum number of tokens the model can see at once (context window).
    context_length: int = 256

    # ---- Transformer dimensions ----------------------------------------
    # Width of every embedding vector and every hidden state.
    d_model: int = 256
    # Number of parallel attention heads (d_model must be divisible by n_heads).
    n_heads: int = 8
    # Number of stacked transformer blocks.
    n_layers: int = 6
    # Hidden dimension of the position-wise feed-forward network (typically 4×d_model).
    d_ff: int = 1024

    # ---- Regularisation ------------------------------------------------
    # Dropout probability — applied during training, disabled at inference.
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Hyper-parameters for the Adam optimiser and the training loop."""

    # ---- Optimiser -----------------------------------------------------
    learning_rate: float = 3e-4
    beta1: float = 0.9        # Exponential decay for the first moment estimate.
    beta2: float = 0.999      # Exponential decay for the second moment estimate.
    epsilon: float = 1e-8     # Numerical stability constant for Adam.
    weight_decay: float = 0.01  # L2 penalty on parameter weights.
    grad_clip: float = 1.0    # Maximum allowed gradient norm (gradient clipping).

    # ---- Training loop -------------------------------------------------
    batch_size: int = 32
    max_epochs: int = 100
    # Evaluate loss on a validation split every this many epochs.
    eval_interval: int = 10
    # Number of mini-batches used when estimating validation loss.
    eval_batches: int = 5

    # ---- Throughput -----------------------------------------------------
    # If True, apply aggressive throughput-oriented tuning when on GPU.
    gpu_boost: bool = True

    # ---- Checkpointing -------------------------------------------------
    # File where trained weights are saved (NumPy .npz format).
    checkpoint_path: str = "model.npz"
