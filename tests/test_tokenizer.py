"""Tests for the Tokenizer."""

import json
import os
import tempfile

import pytest

from llm.tokenizer import Tokenizer


class TestTokenizer:
    def test_build_from_text(self):
        tok = Tokenizer()
        tok.build_from_text("Hello!")
        # PAD + UNK + unique chars.
        assert tok.vocab_size == 2 + len(set("Hello!"))

    def test_encode_decode_roundtrip(self):
        tok = Tokenizer()
        text = "Hello, world!"
        tok.build_from_text(text)
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_unknown_char_maps_to_unk(self):
        tok = Tokenizer()
        tok.build_from_text("abc")
        ids = tok.encode("z")  # 'z' not in vocab
        assert ids == [tok._char_to_idx[Tokenizer.UNK_TOKEN]]

    def test_pad_token_filtered_from_decode(self):
        tok = Tokenizer()
        tok.build_from_text("abc")
        pad_id = tok._char_to_idx[Tokenizer.PAD_TOKEN]
        a_id = tok._char_to_idx["a"]
        decoded = tok.decode([pad_id, a_id, pad_id])
        assert decoded == "a"

    def test_contains(self):
        tok = Tokenizer()
        tok.build_from_text("abc")
        assert "a" in tok
        assert "z" not in tok

    def test_len(self):
        tok = Tokenizer()
        tok.build_from_text("xyz")
        assert len(tok) == tok.vocab_size

    def test_deterministic_vocab_order(self):
        tok1 = Tokenizer()
        tok2 = Tokenizer()
        tok1.build_from_text("bac")
        tok2.build_from_text("abc")
        # Vocabulary should be the same regardless of character order in input.
        assert tok1._char_to_idx == tok2._char_to_idx

    def test_save_and_load(self):
        tok = Tokenizer()
        tok.build_from_text("Hello world")
        original_ids = tok.encode("Hello")

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            path = f.name

        try:
            tok.save(path)
            tok2 = Tokenizer()
            tok2.load(path)
            loaded_ids = tok2.encode("Hello")
            assert loaded_ids == original_ids
            assert tok2.vocab_size == tok.vocab_size
        finally:
            os.unlink(path)
