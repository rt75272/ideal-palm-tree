"""The full transformer language model.

Puts together all the pieces from ``nn.py`` into a GPT-style decoder-only
transformer that predicts the next character at each position.

Architecture overview
---------------------
1. Token embedding     — maps each token ID to a dense vector.
2. Positional embedding — adds learned position information.
3. N transformer blocks — each with self-attention + FFN (see nn.py).
4. Final LayerNorm     — stabilises outputs before the prediction head.
5. Language-model head — linear projection to logits over the vocabulary.
"""

from __future__ import annotations

import math

import numpy as np

from llm.autograd import Tensor
from llm.backend import array_module, to_numpy
from llm.config import ModelConfig
from llm.nn import (
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    Module,
    TransformerBlock,
)

xp = array_module()


class LanguageModel(Module):
    """Decoder-only transformer language model.

    Parameters
    ----------
    config:
        A :class:`~llm.config.ModelConfig` instance that controls the
        architecture (vocabulary size, context length, layers, etc.).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # ---- Embeddings ---------------------------------------------------
        # Token embeddings: each token ID → d_model-dimensional vector.
        self.token_emb = Embedding(config.vocab_size, config.d_model)
        # Positional embeddings: each position 0…context_length-1 → d_model.
        # (Learned, not sinusoidal — simpler and works just as well for small models.)
        self.pos_emb = Embedding(config.context_length, config.d_model)
        self.emb_dropout = Dropout(config.dropout)

        # ---- Transformer blocks -------------------------------------------
        self.blocks: list[Module] = [
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                context_length=config.context_length,
            )
            for _ in range(config.n_layers)
        ]

        # ---- Output head --------------------------------------------------
        self.norm = LayerNorm(config.d_model)
        # Project from d_model to vocab_size to get per-token logits.
        self.lm_head = Linear(config.d_model, config.vocab_size, bias=False)

        # Note: weight tying between the token embedding and the lm_head output
        # projection is a common technique that reduces parameters.  We omit it
        # here because our embedding weight has shape (vocab_size, d_model) while
        # the Linear layer stores weights as (d_model, vocab_size) — a direct
        # alias would require transposition which complicates gradient bookkeeping
        # in our custom autograd engine.

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, idx: Tensor) -> Tensor:  # type: ignore[override]
        """Compute next-token logits for every position in *idx*.

        Args:
            idx: Integer tensor of shape (batch, seq_len) containing token IDs.
                 ``seq_len`` must be ≤ ``config.context_length``.

        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size).
        """
        B, T = idx.shape[0], idx.shape[1]
        assert T <= self.config.context_length, (
            f"Sequence length {T} exceeds context_length {self.config.context_length}"
        )

        # ---- Embeddings ----------------------------------------------------
        tok_emb = self.token_emb(idx)  # (B, T, d_model)

        # Build a position index tensor [0, 1, …, T-1].
        positions = Tensor(xp.arange(T, dtype=xp.float32).reshape(1, T))
        pos_emb = self.pos_emb(positions)  # (1, T, d_model)

        x = self.emb_dropout(tok_emb + pos_emb)  # (B, T, d_model)

        # ---- Transformer blocks --------------------------------------------
        for block in self.blocks:
            x = block(x)

        # ---- Output head ---------------------------------------------------
        x = self.norm(x)          # (B, T, d_model)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(self, idx: Tensor, targets: Tensor) -> Tensor:
        """Cross-entropy loss over all positions in the batch.

        Args:
            idx:     Input token IDs, shape (batch, seq_len).
            targets: Target token IDs, shape (batch, seq_len).
                     Each target is the token *after* the corresponding input.

        Returns:
            Scalar loss tensor.
        """
        logits = self.forward(idx)  # (B, T, V)
        B, T, V = logits.shape

        # Flatten to (B·T, V) so we can apply cross-entropy across all tokens.
        logits_flat = logits.reshape(B * T, V)
        targets_flat = targets.reshape(B * T)

        # Log-probabilities via numerically stable log-softmax.
        log_probs = logits_flat.log_softmax(axis=-1)  # (B·T, V)

        # Cross-entropy: −log p(target).  We gather the log-prob for the true
        # token at each position using integer indexing.
        target_ids = targets_flat.data.astype(int)
        # Build a Tensor view of just the correct class log-probs.
        correct_log_probs = log_probs[xp.arange(B * T), target_ids]  # (B·T,)
        loss = -correct_log_probs.mean()
        return loss

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    @staticmethod
    def _top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
        """Zero out all logits except the top-k (for top-k sampling)."""
        if k <= 0:
            return logits
        # Clamp k to the vocabulary size to avoid out-of-bounds partition.
        k = min(k, len(logits))
        # Find the k-th largest value.
        threshold = xp.partition(logits, -k)[-k]
        # Mask values below the threshold with a large negative number.
        filtered = xp.where(logits >= threshold, logits, -1e9)
        return filtered

    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 40,
        stop_token_ids: set[int] | None = None,
    ) -> list[int]:
        """Generate token IDs auto-regressively starting from *prompt_ids*.

        Args:
            prompt_ids:     Starting token sequence.
            max_new_tokens: Maximum number of tokens to generate.
            temperature:    Sampling temperature (higher → more random).
            top_k:          If > 0, restrict sampling to the top-k tokens.
            stop_token_ids: Optional set of token IDs that end generation
                            once produced.

        Returns:
            The prompt IDs followed by the generated IDs.
        """
        self.eval()  # Disable dropout during generation.
        ctx = self.config.context_length
        ids = list(prompt_ids)

        for _ in range(max_new_tokens):
            # Trim the context window to the last *ctx* tokens.
            window = ids[-ctx:]
            idx_t = Tensor(xp.asarray(window, dtype=xp.float32).reshape(1, -1))

            # Forward pass — we only need the logits for the *last* position.
            logits = self.forward(idx_t)              # (1, T, V)
            last_logits = logits.data[0, -1, :]       # (V,)

            # Apply temperature scaling.
            last_logits = last_logits / max(temperature, 1e-8)

            # Apply top-k filtering.
            last_logits = self._top_k_filter(last_logits, top_k)

            # Convert to probabilities via softmax.
            shifted = last_logits - last_logits.max()
            probs = xp.exp(shifted) / xp.exp(shifted).sum()

            # Sample the next token.
            probs_np = to_numpy(probs)
            next_id = int(np.random.choice(len(probs_np), p=probs_np))
            ids.append(next_id)

            # Stop early on configured boundary tokens (for example newline).
            if stop_token_ids is not None and len(ids) > len(prompt_ids) + 10 and next_id in stop_token_ids:
                break

        return ids

    # ------------------------------------------------------------------
    # Parameter count (handy for diagnostics)
    # ------------------------------------------------------------------

    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.data.size for p in self.parameters())
