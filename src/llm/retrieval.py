"""Simple retrieval fallback over the bundled conversation pairs."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass(frozen=True)
class Pair:
    human: str
    assistant: str


_DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "conversations.txt"
)

_GENERATION_FIRST_PATTERNS = (
    "write code",
    "write a python",
    "write a function",
    "edit this code",
    "fix this code",
    "refactor",
    "debug this",
    "traceback",
    "stack trace",
    "terminal transcript",
    "pytest",
    "fastapi",
    "pandas",
    "sqlalchemy",
    "function",
    "class ",
    "def ",
    "import ",
)


def load_pairs(path: str | None = None) -> list[Pair]:
    """Parse <human>/<assistant> pairs from the training corpus."""
    resolved = os.path.abspath(path or _DEFAULT_DATA_PATH)
    pairs: list[Pair] = []

    pending_human: list[str] | None = None
    pending_assistant: list[str] | None = None

    def flush_pair() -> None:
        nonlocal pending_human, pending_assistant
        if pending_human and pending_assistant:
            human = "\n".join(pending_human).strip()
            assistant = "\n".join(pending_assistant).strip()
            if human and assistant:
                pairs.append(Pair(human=human, assistant=assistant))
        pending_human = None
        pending_assistant = None

    with open(resolved, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if line.startswith("<human>:"):
                flush_pair()
                pending_human = [line[len("<human>:") :].strip()]
                pending_assistant = None
            elif line.startswith("<assistant>:") and pending_human is not None:
                pending_assistant = [line[len("<assistant>:") :].strip()]
            elif pending_assistant is not None:
                pending_assistant.append(line)
            elif pending_human is not None:
                pending_human.append(line)

    flush_pair()

    return pairs


def best_match_reply(
    user_input: str,
    pairs: list[Pair],
    threshold: float = 0.4,
) -> str | None:
    """Return retrieved assistant reply for similar user input, if confident."""
    query = _normalize(user_input)
    if not query or not pairs:
        return None

    # Short greeting aliases benefit from direct retrieval.
    if query in {"hi", "hey", "hello", "yo"}:
        for pair in pairs:
            if _normalize(pair.human) in {"hello", "hello!", "hi", "hi!"}:
                return pair.assistant

    best_score = -1.0
    best_reply: str | None = None

    for pair in pairs:
        candidate = _normalize(pair.human)
        score = SequenceMatcher(None, query, candidate).ratio()

        if query in candidate or candidate in query:
            score = max(score, 0.85)

        if score > best_score:
            best_score = score
            best_reply = pair.assistant

    if best_score >= threshold:
        return best_reply
    return None


def should_skip_retrieval(user_input: str, coding_assistant: bool = False) -> bool:
    """Return True when a prompt should prefer fresh generation over retrieval.

    Retrieval works well for short common prompts, but code-oriented requests are
    usually better served by generating a tailored answer from the model.
    """
    if coding_assistant:
        return True

    query = user_input.strip().lower()
    if not query:
        return False

    if "\n" in user_input:
        return True

    if any(token in user_input for token in ("```", "=", "->", "Traceback", "$ ")):
        return True

    return any(pattern in query for pattern in _GENERATION_FIRST_PATTERNS)


def _normalize(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"[^a-z0-9\s]", "", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered
