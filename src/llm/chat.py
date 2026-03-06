"""Interactive chat interface for the LLM.

After training a model with ``llm-train``, start a conversation with::

    llm-chat --checkpoint model.npz

The chat loop maintains a rolling conversation history that is fed as context
to the model at every turn, enabling multi-turn dialogue.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from llm.autograd import Tensor
from llm.config import ModelConfig, TrainingConfig
from llm.model import LanguageModel
from llm.retrieval import best_match_reply, load_pairs, should_skip_retrieval
from llm.tokenizer import Tokenizer
from llm.trainer import load_checkpoint, train


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

# Delimiters used to separate turns in the conversation context.
HUMAN_PROMPT = "<human>: "
ASSISTANT_PROMPT = "<assistant>: "
TURN_SEPARATOR = "\n"

# Optional steering instruction used when code-focused chat mode is enabled.
CODING_ASSISTANT_INSTRUCTION = (
    "You are a Python coding assistant. "
    "When asked to write code, provide runnable Python code with short comments. "
    "When asked to edit code, return the updated code and then a brief list of what changed."
)


def _build_prompt(
    history: list[tuple[str, str]],
    user_input: str,
    assistant_instruction: str | None = None,
) -> str:
    """Assemble the full conversation context string for the model.

    Each past turn is rendered as::

        <human>: {user message}
        <assistant>: {assistant reply}

    The current user message is appended, and the assistant prefix is added
    so the model continues from that point.

    Args:
        history:    List of (user_message, assistant_reply) pairs.
        user_input: The new user message.

    Returns:
        The full prompt string to feed into the model.
    """
    parts: list[str] = []
    if assistant_instruction:
        parts.append(f"{HUMAN_PROMPT}System instruction: {assistant_instruction}")
        parts.append(f"{ASSISTANT_PROMPT}Understood.")

    for user_msg, asst_msg in history:
        parts.append(f"{HUMAN_PROMPT}{user_msg}")
        parts.append(f"{ASSISTANT_PROMPT}{asst_msg}")
    # Append the new user message and the assistant prefix (model continues here).
    parts.append(f"{HUMAN_PROMPT}{user_input}")
    parts.append(ASSISTANT_PROMPT)
    return TURN_SEPARATOR.join(parts)


def _extract_response(generated_text: str, prompt: str) -> str:
    """Extract the assistant's reply from the full generated text.

    The model generates a continuation of *prompt*.  We strip the prompt,
    then take everything up to the first occurrence of the next human turn
    marker so only the current assistant reply is returned.

    Args:
        generated_text: Full text returned by the model (prompt + continuation).
        prompt:         The original prompt that was fed to the model.

    Returns:
        The assistant's reply as a clean string.
    """
    # Remove the prompt prefix.
    response = generated_text[len(prompt):]
    # Stop at the next human turn (avoid model hallucinating the next turn).
    if HUMAN_PROMPT in response:
        response = response[: response.index(HUMAN_PROMPT)]
    return response.strip()


# ---------------------------------------------------------------------------
# Chat session
# ---------------------------------------------------------------------------

class ChatSession:
    """Manages a multi-turn conversation with the LLM.

    Keeps a rolling history of recent turns and passes the full context to
    the model at every step, giving it memory of the conversation so far.

    Parameters
    ----------
    model:
        A trained :class:`~llm.model.LanguageModel`.
    tokenizer:
        The tokeniser that was used during training.
    max_history_turns:
        Maximum number of past turns to include in the context (older turns
        are dropped to stay within the model's context window).
    max_new_tokens:
        Maximum tokens to generate per assistant reply.
    temperature:
        Sampling temperature.  Higher values → more creative, lower → more
        conservative.
    top_k:
        Top-k sampling parameter.  Only the *k* most likely tokens are
        considered at each step.
    """

    def __init__(
        self,
        model: LanguageModel,
        tokenizer: Tokenizer,
        max_history_turns: int = 4,
        max_new_tokens: int = 120,
        temperature: float = 0.35,
        top_k: int = 20,
        retrieval_enabled: bool = True,
        retrieval_threshold: float = 0.4,
        coding_assistant: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_history_turns = max_history_turns
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.retrieval_enabled = retrieval_enabled
        self.retrieval_threshold = retrieval_threshold
        self.assistant_instruction = (
            CODING_ASSISTANT_INSTRUCTION if coding_assistant else None
        )
        self._retrieval_pairs = load_pairs()
        # History: list of (user_message, assistant_reply) pairs.
        self._history: list[tuple[str, str]] = []

    def chat(self, user_input: str) -> str:
        """Process one user turn and return the assistant reply.

        Args:
            user_input: The user's message.

        Returns:
            The assistant's response string.
        """
        if self.retrieval_enabled and not should_skip_retrieval(
            user_input,
            coding_assistant=self.assistant_instruction is not None,
        ):
            retrieved = best_match_reply(
                user_input,
                self._retrieval_pairs,
                threshold=self.retrieval_threshold,
            )
            if retrieved is not None:
                self._history.append((user_input, retrieved))
                return retrieved

        # Keep only the most recent turns to avoid overflowing context.
        recent = self._history[-self.max_history_turns :]

        # Build the prompt string.
        prompt = _build_prompt(
            recent,
            user_input,
            assistant_instruction=self.assistant_instruction,
        )

        # Trim to fit within the model's context window.
        ctx = self.model.config.context_length
        if len(prompt) > ctx:
            prompt = prompt[-ctx:]

        # Encode the prompt.
        prompt_ids = self.tokenizer.encode(prompt)
        # Trim again in token space (some characters may map to multiple tokens,
        # though in our char-level scheme it's 1-to-1).
        prompt_ids = prompt_ids[-ctx:]

        # Generate continuation.
        output_ids = self.model.generate(
            prompt_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
        )

        # Decode the full generated sequence and extract the reply.
        full_text = self.tokenizer.decode(output_ids)
        decoded_prompt = self.tokenizer.decode(prompt_ids)
        response = _extract_response(full_text, decoded_prompt)

        # Fallback if extraction produced nothing.
        if not response:
            response = "(…)"

        # Record this turn in history.
        self._history.append((user_input, response))
        return response

    def reset(self) -> None:
        """Clear the conversation history."""
        self._history.clear()


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

_WELCOME = """
╔═══════════════════════════════════════════════════════╗
║          LLM From Scratch — Interactive Chat          ║
╚═══════════════════════════════════════════════════════╝
Type your message and press Enter.  Special commands:
  /reset   — clear conversation history
  /quit    — exit
  /help    — show this message
"""


def _repl(session: ChatSession) -> None:
    """Run the interactive read-eval-print loop."""
    print(_WELCOME)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle special commands.
        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("Goodbye!")
            break
        elif user_input.lower() in ("/reset", "/clear"):
            session.reset()
            print("[Conversation history cleared]")
            continue
        elif user_input.lower() in ("/help", "/?"):
            print(_WELCOME)
            continue

        # Generate and display the assistant's response.
        print("Assistant: ", end="", flush=True)
        response = session.chat(user_input)
        print(response)
        print()  # blank line between turns for readability


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line entry point for ``llm-chat``."""
    parser = argparse.ArgumentParser(
        description="Interactive chat with a trained LLM."
    )
    parser.add_argument(
        "--checkpoint",
        default="model.npz",
        help="Path to the trained model checkpoint (default: model.npz).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.35,
        help="Sampling temperature (default: 0.35). Higher -> more creative.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling (default: 20). Use 0 to disable.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=120,
        help="Maximum tokens to generate per reply (default: 120).",
    )
    parser.add_argument(
        "--history-turns",
        type=int,
        default=4,
        help="Number of past turns to keep in context (default: 4).",
    )
    parser.add_argument(
        "--no-retrieval",
        action="store_true",
        help="Disable retrieval fallback from training conversation pairs.",
    )
    parser.add_argument(
        "--retrieval-threshold",
        type=float,
        default=0.4,
        help="Similarity threshold for retrieval fallback (default: 0.4).",
    )
    parser.add_argument(
        "--coding-assistant",
        action="store_true",
        help="Bias responses toward writing/editing Python code.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help=(
            "Train the model from scratch before chatting "
            "(useful when no checkpoint exists)."
        ),
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Training data path (used with --train).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs (used with --train, default: 100).",
    )
    args = parser.parse_args()

    # ---- Load or train the model ------------------------------------------
    if args.train or not os.path.isfile(args.checkpoint):
        if not args.train and not os.path.isfile(args.checkpoint):
            print(
                f"No checkpoint found at '{args.checkpoint}'.\n"
                "Training a new model from scratch…\n"
                "(Tip: use --train to train explicitly, or provide --checkpoint.)\n"
            )
        tc = TrainingConfig(
            checkpoint_path=args.checkpoint,
            max_epochs=args.epochs,
        )
        try:
            model, tokenizer = train(train_config=tc, data_path=args.data)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Loading model from '{args.checkpoint}'…")
        try:
            model, tokenizer = load_checkpoint(args.checkpoint)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        print("Model loaded.\n")

    # ---- Start the chat session -------------------------------------------
    session = ChatSession(
        model=model,
        tokenizer=tokenizer,
        max_history_turns=args.history_turns,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        retrieval_enabled=not args.no_retrieval,
        retrieval_threshold=args.retrieval_threshold,
        coding_assistant=args.coding_assistant,
    )
    _repl(session)


if __name__ == "__main__":
    main()
