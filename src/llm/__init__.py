"""LLM From Scratch — public package interface.

Importing this package exposes the main classes so users can run the model
from their own scripts without digging into sub-modules::

    from llm import LanguageModel, Tokenizer, ChatSession, ModelConfig

See ``llm.chat`` for the interactive CLI and ``llm.trainer`` for training.
"""

from llm.autograd import Tensor
from llm.chat import ChatSession
from llm.config import ModelConfig, TrainingConfig
from llm.model import LanguageModel
from llm.tokenizer import Tokenizer

__all__ = [
    "Tensor",
    "ChatSession",
    "ModelConfig",
    "TrainingConfig",
    "LanguageModel",
    "Tokenizer",
]
