#!/usr/bin/env python3
"""Main entry point for the LLM from scratch project.

Usage
-----
Train then chat (all-in-one)::

    python main.py

Train only::

    python main.py --mode train

Chat only (requires an existing checkpoint)::

    python main.py --mode chat

Run with custom settings::

    python main.py --mode train --epochs 200 --checkpoint my_model.npz
    python main.py --mode chat  --checkpoint my_model.npz --temperature 0.9
"""

from __future__ import annotations

import argparse
import sys

from llm.chat import ChatSession, _repl
from llm.config import ModelConfig, TrainingConfig
from llm.trainer import load_checkpoint, train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM from Scratch — train and chat with a language model."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "chat", "train-chat"],
        default="train-chat",
        help=(
            "Operation mode: 'train' trains the model, 'chat' starts an "
            "interactive session (requires a checkpoint), 'train-chat' does "
            "both (default)."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default="model.npz",
        help="Path to save/load model weights (default: model.npz).",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to the plain-text training corpus (default: data/conversations.txt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs (default: 100).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini-batch size (default: 16).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for generation (default: 0.8).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k sampling (default: 40). Use 0 to disable.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum tokens generated per reply (default: 200).",
    )
    args = parser.parse_args()

    tc = TrainingConfig(
        checkpoint_path=args.checkpoint,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )

    model, tokenizer = None, None

    # ---- Training phase ----------------------------------------------------
    if args.mode in ("train", "train-chat"):
        try:
            model, tokenizer = train(train_config=tc, data_path=args.data)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    # ---- Chat phase --------------------------------------------------------
    if args.mode in ("chat", "train-chat"):
        if model is None:
            # Load from checkpoint when chat-only mode.
            import os
            if not os.path.isfile(args.checkpoint):
                print(
                    f"No checkpoint found at '{args.checkpoint}'.\n"
                    "Run with --mode train first to train the model.",
                    file=sys.stderr,
                )
                sys.exit(1)
            print(f"Loading model from '{args.checkpoint}'…")
            try:
                model, tokenizer = load_checkpoint(args.checkpoint)
            except FileNotFoundError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(1)

        session = ChatSession(
            model=model,
            tokenizer=tokenizer,
            temperature=args.temperature,
            top_k=args.top_k,
            max_new_tokens=args.max_tokens,
        )
        _repl(session)


if __name__ == "__main__":
    main()
