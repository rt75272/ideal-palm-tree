"""Interactive chat interface for PyTorch checkpoints (.pt)."""

from __future__ import annotations

import argparse
import os
import sys

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "PyTorch is required for llm-chat-torch. Install with 'uv sync --extra torch'."
    ) from exc

from llm.chat import (
    ASSISTANT_PROMPT,
    CODING_ASSISTANT_INSTRUCTION,
    HUMAN_PROMPT,
    TURN_SEPARATOR,
    _WELCOME,
    _build_prompt,
    _extract_response,
)
from llm.config import ModelConfig
from llm.retrieval import best_match_reply, load_pairs, should_skip_retrieval
from llm.tokenizer import Tokenizer
from llm.torch_trainer import TorchLanguageModel


class TorchChatSession:
    """Maintains multi-turn chat state for a PyTorch model checkpoint."""

    def __init__(
        self,
        model: TorchLanguageModel,
        tokenizer: Tokenizer,
        max_history_turns: int = 4,
        max_new_tokens: int = 120,
        temperature: float = 0.35,
        top_k: int = 12,
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
        self._history: list[tuple[str, str]] = []

    @property
    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _generate_ids(self, prompt_ids: list[int]) -> list[int]:
        ctx = self.model.config.context_length
        ids = list(prompt_ids)

        for _ in range(self.max_new_tokens):
            window = ids[-ctx:]
            x = torch.tensor([window], dtype=torch.long, device=self._device)

            with torch.no_grad():
                logits = self.model(x)[0, -1, :] / max(self.temperature, 1e-8)
                if self.top_k > 0:
                    k = min(self.top_k, logits.numel())
                    vals, idx = torch.topk(logits, k)
                    filtered = torch.full_like(logits, -1e9)
                    filtered[idx] = vals
                    logits = filtered
                probs = torch.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probs, 1).item())

            ids.append(next_id)

        return ids

    def chat(self, user_input: str) -> str:
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

        recent = self._history[-self.max_history_turns :]
        prompt = _build_prompt(
            recent,
            user_input,
            assistant_instruction=self.assistant_instruction,
        )

        ctx = self.model.config.context_length
        if len(prompt) > ctx:
            prompt = prompt[-ctx:]

        prompt_ids = self.tokenizer.encode(prompt)[-ctx:]
        output_ids = self._generate_ids(prompt_ids)

        full_text = self.tokenizer.decode(output_ids)
        decoded_prompt = self.tokenizer.decode(prompt_ids)
        response = _extract_response(full_text, decoded_prompt).strip()
        if not response:
            response = "(...)"

        self._history.append((user_input, response))
        return response

    def reset(self) -> None:
        self._history.clear()


def _load_torch_checkpoint(path: str) -> tuple[TorchLanguageModel, Tokenizer]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: '{path}'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(path, map_location=device)

    config = ModelConfig(**data["model_config"])
    model = TorchLanguageModel(config).to(device)
    model.load_state_dict(data["model_state"])
    model.eval()

    tokenizer = Tokenizer()
    tokenizer_state = data.get("tokenizer_state", data.get("tokenizer_vocab"))
    tokenizer.load_state(tokenizer_state)
    return model, tokenizer


def _repl(session: TorchChatSession) -> None:
    print(_WELCOME)
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("Goodbye!")
            break
        if user_input.lower() in ("/reset", "/clear"):
            session.reset()
            print("[Conversation history cleared]")
            continue
        if user_input.lower() in ("/help", "/?"):
            print(_WELCOME)
            continue

        print("Assistant: ", end="", flush=True)
        print(session.chat(user_input))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive chat with a PyTorch LLM checkpoint.")
    parser.add_argument(
        "--checkpoint",
        default="model_torch.pt",
        help="Path to the trained PyTorch checkpoint (default: model_torch.pt).",
    )
    parser.add_argument("--temperature", type=float, default=0.35, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=12, help="Top-k sampling (0 disables).")
    parser.add_argument("--max-tokens", type=int, default=120, help="Max tokens per reply.")
    parser.add_argument("--history-turns", type=int, default=4, help="Conversation turns to keep.")
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
    args = parser.parse_args()

    try:
        model, tokenizer = _load_torch_checkpoint(args.checkpoint)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    session = TorchChatSession(
        model=model,
        tokenizer=tokenizer,
        temperature=args.temperature,
        top_k=args.top_k,
        max_new_tokens=args.max_tokens,
        max_history_turns=args.history_turns,
        retrieval_enabled=not args.no_retrieval,
        retrieval_threshold=args.retrieval_threshold,
        coding_assistant=args.coding_assistant,
    )
    _repl(session)


if __name__ == "__main__":
    main()
