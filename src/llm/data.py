"""Dataset loader and mini-batch sampler.

Reads the raw conversation text file and turns it into (input, target) pairs
for language-model training.  The targets are just the inputs shifted by one
position: training the model to predict the *next* token at every step.
"""

from __future__ import annotations

import os

from llm.backend import array_module, to_numpy

xp = array_module()


# Default path to the bundled conversation corpus (relative to repo root).
_DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "conversations.txt"
)


class Dataset:
    """Holds the full token-ID corpus and serves random mini-batches.

    Parameters
    ----------
    token_ids:
        The entire training corpus encoded as a flat list of integer IDs.
    context_length:
        Number of tokens in each training example (the model's context window).
    """

    def __init__(self, token_ids: list[int], context_length: int) -> None:
        self.token_ids = token_ids
        self.context_length = context_length
        # We need at least context_length + 1 tokens to form one (x, y) pair.
        self._max_start = len(token_ids) - context_length - 1
        self._token_ids_array = xp.asarray(token_ids, dtype=xp.int32)
        self._offsets = xp.arange(context_length, dtype=xp.int32)

    def __len__(self) -> int:
        """Number of valid starting positions in the corpus."""
        return max(0, self._max_start)

    def get_batch(
        self, batch_size: int
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Sample a random mini-batch of (inputs, targets) pairs.

        Each pair is a slice of length *context_length*.  The target is the
        input shifted right by one position (next-token prediction).

        Returns:
            A tuple ``(inputs, targets)`` where each element is a list of
            ``batch_size`` sequences (each a list of ``context_length`` ints).
        """
        if self._max_start <= 0:
            raise ValueError(
                f"Corpus too short ({len(self.token_ids)} tokens) for "
                f"context_length={self.context_length}."
            )
        x_arr, y_arr = self.get_batch_arrays(batch_size)
        return to_numpy(x_arr).astype(int).tolist(), to_numpy(y_arr).astype(int).tolist()

    def get_batch_arrays(self, batch_size: int):
        """Sample a random mini-batch and return backend arrays.

        This path is vectorised and avoids Python loops, which is substantially
        faster and keeps the GPU busier during training.
        """
        if self._max_start <= 0:
            raise ValueError(
                f"Corpus too short ({len(self.token_ids)} tokens) for "
                f"context_length={self.context_length}."
            )
        starts = xp.random.randint(0, self._max_start + 1, size=(batch_size, 1))
        idx = starts + self._offsets.reshape(1, -1)
        x = self._token_ids_array[idx].astype(xp.float32)
        y = self._token_ids_array[idx + 1].astype(xp.float32)
        return x, y


def load_text(path: str | None = None) -> str:
    """Read the conversation corpus from *path* (or the default location).

    Args:
        path: Explicit file path.  If ``None``, the bundled
              ``data/conversations.txt`` is used.

    Returns:
        The entire file contents as a single string.

    Raises:
        FileNotFoundError: If neither *path* nor the default file exists.
    """
    resolved = path or _DEFAULT_DATA_PATH
    resolved = os.path.abspath(resolved)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            f"Training data not found at '{resolved}'.  "
            "Please provide a text file via --data."
        )
    with open(resolved, encoding="utf-8") as fh:
        return fh.read()


def train_val_split(
    token_ids: list[int], val_fraction: float = 0.1
) -> tuple[list[int], list[int]]:
    """Split *token_ids* into training and validation sets.

    The split is done at a fixed boundary (no shuffling) so that token
    sequences remain contiguous within each split.

    Args:
        token_ids:    Full tokenised corpus.
        val_fraction: Fraction of tokens to reserve for validation.

    Returns:
        ``(train_ids, val_ids)`` tuple.
    """
    n = len(token_ids)
    split = int(n * (1.0 - val_fraction))
    return token_ids[:split], token_ids[split:]
