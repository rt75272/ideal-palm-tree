from pathlib import Path

from llm.retrieval import Pair, best_match_reply, load_pairs, should_skip_retrieval


def test_best_match_reply_returns_greeting_match() -> None:
    pairs = [Pair(human="hello", assistant="Hi there")]

    assert best_match_reply("hello", pairs) == "Hi there"


def test_should_skip_retrieval_for_code_prompt() -> None:
    assert should_skip_retrieval("Write a Python function to add two numbers")


def test_should_skip_retrieval_for_multiline_code_snippet() -> None:
    prompt = "Fix this code:\ndef add(a, b):\n    return a - b"

    assert should_skip_retrieval(prompt)


def test_should_skip_retrieval_when_coding_assistant_enabled() -> None:
    assert should_skip_retrieval("hello", coding_assistant=True)


def test_should_not_skip_retrieval_for_simple_greeting() -> None:
    assert not should_skip_retrieval("hello")


def test_load_pairs_supports_multiline_blocks(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "<human>: Edit this code\n"
        "def add(a, b):\n"
        "    return a - b\n"
        "<assistant>: def add(a, b):\n"
        "    return a + b\n",
        encoding="utf-8",
    )

    pairs = load_pairs(str(corpus))

    assert len(pairs) == 1
    assert "return a - b" in pairs[0].human
    assert "return a + b" in pairs[0].assistant