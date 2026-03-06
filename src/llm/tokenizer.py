"""Simple byte-level subword tokenizer.

The tokenizer starts with raw UTF-8 bytes and learns frequent byte-pair
merges directly from the training corpus. This keeps round-trip decoding
exact while using fewer tokens than a pure character-level approach.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Iterator


class Tokenizer:
    """Maps text to token IDs using a small byte-pair subword vocabulary.

    Special tokens
    --------------
    ``<pad>`` (index 0) — used to pad sequences to a fixed length.
    ``<unk>`` (index 1) — reserved for legacy checkpoints and invalid IDs.
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    DEFAULT_VOCAB_SIZE = 512
    DEFAULT_MIN_MERGE_FREQUENCY = 2

    def __init__(
        self,
        target_vocab_size: int = DEFAULT_VOCAB_SIZE,
        min_merge_frequency: int = DEFAULT_MIN_MERGE_FREQUENCY,
    ) -> None:
        self.target_vocab_size = max(target_vocab_size, 258)
        self.min_merge_frequency = max(min_merge_frequency, 2)
        self._mode = "bpe"
        self._token_to_idx: dict[bytes, int] = {}
        self._idx_to_token: dict[int, bytes] = {}
        self._char_to_idx: dict[str, int] = {}
        self._idx_to_char: dict[int, str] = {}
        self._max_token_bytes = 1
        self._reset_bpe_vocab()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build_from_text(self, text: str) -> None:
        """Build a byte-pair vocabulary from *text*."""
        self._mode = "bpe"
        self._char_to_idx = {}
        self._idx_to_char = {}
        self._reset_bpe_vocab()

        sequence = [bytes([value]) for value in text.encode("utf-8")]
        while len(self._token_to_idx) + 2 < self.target_vocab_size:
            pair_counts = Counter(zip(sequence, sequence[1:]))
            if not pair_counts:
                break

            best_frequency = max(pair_counts.values())
            if best_frequency < self.min_merge_frequency:
                break

            best_pair = min(
                pair for pair, freq in pair_counts.items() if freq == best_frequency
            )
            merged_token = best_pair[0] + best_pair[1]
            if merged_token in self._token_to_idx:
                break

            next_idx = len(self._token_to_idx) + 2
            self._token_to_idx[merged_token] = next_idx
            self._idx_to_token[next_idx] = merged_token
            self._max_token_bytes = max(self._max_token_bytes, len(merged_token))
            sequence = self._merge_sequence(sequence, best_pair, merged_token)

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Convert *text* to a list of integer token IDs."""
        if self._mode == "char":
            unk = self._char_to_idx[self.UNK_TOKEN]
            return [self._char_to_idx.get(ch, unk) for ch in text]

        payload = text.encode("utf-8")
        token_ids: list[int] = []
        index = 0
        while index < len(payload):
            max_len = min(self._max_token_bytes, len(payload) - index)
            matched = False
            for length in range(max_len, 0, -1):
                token = payload[index : index + length]
                token_id = self._token_to_idx.get(token)
                if token_id is not None:
                    token_ids.append(token_id)
                    index += length
                    matched = True
                    break
            if not matched:
                token_ids.append(1)
                index += 1
        return token_ids

    def decode(self, ids: list[int] | Iterator[int]) -> str:
        """Convert a sequence of token IDs back to a string."""
        if self._mode == "char":
            pad = self._char_to_idx[self.PAD_TOKEN]
            return "".join(
                self._idx_to_char.get(i, self.UNK_TOKEN)
                for i in ids
                if i != pad
            )

        payload = bytearray()
        for token_id in ids:
            if token_id == 0:
                continue
            token = self._idx_to_token.get(token_id)
            if token is None:
                continue
            payload.extend(token)
        return payload.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise the tokenizer state to a JSON file at *path*."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_state(), fh, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """Load tokenizer state previously saved with :meth:`save`."""
        with open(path, encoding="utf-8") as fh:
            self.load_state(json.load(fh))

    def to_state(self) -> dict[str, object]:
        """Return a serializable representation of the tokenizer."""
        if self._mode == "char":
            return {
                "version": 2,
                "mode": "char",
                "char_to_idx": self._char_to_idx,
            }

        return {
            "version": 2,
            "mode": "bpe",
            "target_vocab_size": self.target_vocab_size,
            "min_merge_frequency": self.min_merge_frequency,
            "token_bytes": [
                list(self._idx_to_token[index])
                for index in range(2, len(self._idx_to_token) + 2)
            ],
        }

    def load_state(self, state: object) -> None:
        """Restore tokenizer state from a dict or legacy raw vocab mapping."""
        if isinstance(state, dict) and state.get("version") == 2:
            mode = state.get("mode")
            if mode == "char":
                char_to_idx = state["char_to_idx"]
                assert isinstance(char_to_idx, dict)
                self._mode = "char"
                self._char_to_idx = {str(k): int(v) for k, v in char_to_idx.items()}
                self._idx_to_char = {v: k for k, v in self._char_to_idx.items()}
                self._token_to_idx = {}
                self._idx_to_token = {}
                self._max_token_bytes = 1
                return

            token_bytes = state["token_bytes"]
            assert isinstance(token_bytes, list)
            self.target_vocab_size = int(state.get("target_vocab_size", self.DEFAULT_VOCAB_SIZE))
            self.min_merge_frequency = int(
                state.get("min_merge_frequency", self.DEFAULT_MIN_MERGE_FREQUENCY)
            )
            self._mode = "bpe"
            self._char_to_idx = {}
            self._idx_to_char = {}
            self._token_to_idx = {}
            self._idx_to_token = {}
            self._max_token_bytes = 1
            for byte_value in range(256):
                token = bytes([byte_value])
                token_id = byte_value + 2
                self._token_to_idx[token] = token_id
                self._idx_to_token[token_id] = token
            for offset, raw_token in enumerate(token_bytes, start=2):
                token = bytes(raw_token)
                self._token_to_idx[token] = offset
                self._idx_to_token[offset] = token
                self._max_token_bytes = max(self._max_token_bytes, len(token))
            return

        # Backward compatibility for old checkpoints that stored raw char vocab.
        assert isinstance(state, dict)
        self._mode = "char"
        self._char_to_idx = {str(k): int(v) for k, v in state.items()}
        self._idx_to_char = {v: k for k, v in self._char_to_idx.items()}
        self._token_to_idx = {}
        self._idx_to_token = {}
        self._max_token_bytes = 1

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Number of unique tokens in the vocabulary."""
        if self._mode == "char":
            return len(self._char_to_idx)
        return len(self._token_to_idx) + 2

    def __len__(self) -> int:
        return self.vocab_size

    def __contains__(self, ch: str) -> bool:
        if self._mode == "char":
            return ch in self._char_to_idx
        return ch.encode("utf-8") in self._token_to_idx

    def _reset_bpe_vocab(self) -> None:
        self._token_to_idx = {}
        self._idx_to_token = {}
        for byte_value in range(256):
            token = bytes([byte_value])
            token_id = byte_value + 2
            self._token_to_idx[token] = token_id
            self._idx_to_token[token_id] = token
        self._max_token_bytes = 1

    @staticmethod
    def _merge_sequence(
        sequence: list[bytes],
        pair: tuple[bytes, bytes],
        merged: bytes,
    ) -> list[bytes]:
        result: list[bytes] = []
        index = 0
        while index < len(sequence):
            if (
                index < len(sequence) - 1
                and sequence[index] == pair[0]
                and sequence[index + 1] == pair[1]
            ):
                result.append(merged)
                index += 2
                continue
            result.append(sequence[index])
            index += 1
        return result
