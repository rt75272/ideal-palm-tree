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
        assert tok.vocab_size >= 258

    def test_encode_decode_roundtrip(self):
        tok = Tokenizer()
        text = "Hello, world!"
        tok.build_from_text(text)
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_unseen_text_roundtrip_still_works(self):
        tok = Tokenizer()
        tok.build_from_text("abc")
        ids = tok.encode("zebra")
        assert tok.decode(ids) == "zebra"

    def test_pad_token_filtered_from_decode(self):
        tok = Tokenizer()
        tok.build_from_text("abc")
        pad_id = 0
        a_id = tok.encode("a")[0]
        decoded = tok.decode([pad_id, a_id, pad_id])
        assert decoded == "a"

    def test_contains(self):
        tok = Tokenizer()
        tok.build_from_text("abc")
        assert "a" in tok
        assert "z" in tok

    def test_len(self):
        tok = Tokenizer()
        tok.build_from_text("xyz")
        assert len(tok) == tok.vocab_size

    def test_deterministic_vocab_order(self):
        tok1 = Tokenizer()
        tok2 = Tokenizer()
        sample = "banana bandana"
        tok1.build_from_text(sample)
        tok2.build_from_text(sample)
        assert tok1.to_state() == tok2.to_state()

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

    def test_multiline_roundtrip(self):
        tok = Tokenizer()
        text = "def add(a, b):\n    return a + b\n"
        tok.build_from_text(text)
        assert tok.decode(tok.encode(text)) == text
