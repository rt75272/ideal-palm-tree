"""Character-level tokeniser.

Converts raw text ↔ sequences of integer token IDs.  No external
tokenisation libraries are used — the vocabulary is built directly from
the characters present in the training corpus.
"""

from __future__ import annotations

import json
import os
from typing import Iterator


class Tokenizer:
    """Maps each unique character in a corpus to an integer index.

    Special tokens
    --------------
    ``<pad>`` (index 0) — used to pad sequences to a fixed length.
    ``<unk>`` (index 1) — represents any character not seen during training.
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self) -> None:
        # char → index
        self._char_to_idx: dict[str, int] = {}
        # index → char
        self._idx_to_char: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build_from_text(self, text: str) -> None:
        """Create the vocabulary from *text*.

        Reserves index 0 for ``<pad>`` and 1 for ``<unk>``, then assigns
        consecutive indices to every unique character found in *text*.
        """
        self._char_to_idx = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self._idx_to_char = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}

        # Sort characters so the mapping is deterministic across runs.
        for ch in sorted(set(text)):
            if ch not in self._char_to_idx:
                idx = len(self._char_to_idx)
                self._char_to_idx[ch] = idx
                self._idx_to_char[idx] = ch

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Convert *text* to a list of integer token IDs."""
        unk = self._char_to_idx[self.UNK_TOKEN]
        return [self._char_to_idx.get(ch, unk) for ch in text]

    def decode(self, ids: list[int] | Iterator[int]) -> str:
        """Convert a sequence of token IDs back to a string."""
        pad = self._char_to_idx[self.PAD_TOKEN]
        return "".join(
            self._idx_to_char.get(i, self.UNK_TOKEN)
            for i in ids
            if i != pad  # skip padding tokens in output
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise the vocabulary to a JSON file at *path*."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self._char_to_idx, fh, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """Load a vocabulary previously saved with :meth:`save`."""
        with open(path, encoding="utf-8") as fh:
            self._char_to_idx = json.load(fh)
        self._idx_to_char = {v: k for k, v in self._char_to_idx.items()}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Number of unique tokens in the vocabulary."""
        return len(self._char_to_idx)

    def __len__(self) -> int:
        return self.vocab_size

    def __contains__(self, ch: str) -> bool:
        return ch in self._char_to_idx
