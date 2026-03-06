# LLM From Scratch

A fully-featured **Large Language Model built from scratch** in Python — no
ML frameworks (PyTorch / TensorFlow / JAX / Keras) used anywhere.

By default it runs on **GPU (via CuPy)** when available, and falls back to
**CPU (NumPy)** otherwise. Everything else — automatic differentiation, the
transformer architecture, the Adam optimiser, the tokeniser, training loop,
and interactive chat interface — is written in plain Python.

Project dependencies and the virtual environment are managed entirely with
[**uv**](https://docs.astral.sh/uv/) and `pyproject.toml` (no `pip` or
`requirements.txt`).

---

## Architecture

| Component | Description |
|-----------|-------------|
| **Autograd engine** (`autograd.py`) | Reverse-mode automatic differentiation over NumPy/CuPy arrays |
| **Neural-network modules** (`nn.py`) | `Linear`, `LayerNorm`, `Embedding`, `Dropout`, `MultiHeadAttention`, `FeedForward`, `TransformerBlock` |
| **Language model** (`model.py`) | Decoder-only transformer (GPT-style): token + positional embeddings → N transformer blocks → LM head |
| **Tokeniser** (`tokenizer.py`) | Byte-level subword tokenizer with learned merges; saved/loaded as JSON |
| **Dataset** (`data.py`) | Loads text, builds next-token prediction pairs, serves random mini-batches |
| **Trainer** (`trainer.py`) | Adam optimiser (from scratch), gradient clipping, train/val split, `.npz` checkpointing |
| **Chat interface** (`chat.py`) | Multi-turn REPL that maintains conversation history as rolling context |

---

## Quick Start

### 1 — Install `uv`

```bash
pip install uv          # or: curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2 — Create the virtual environment and install dependencies

```bash
uv sync

# Optional: enable GPU backend (CuPy)
uv sync --extra gpu

# Optional: install PyTorch trainer dependencies
uv sync --extra torch

# Optional: install both extras
uv sync --extra gpu --extra torch
```

### 3 — Run the program

There are four supported entry-point styles:

```bash
# Unified entry point
uv run python main.py ...

# Installed console scripts from pyproject.toml
uv run llm-train ...
uv run llm-chat ...
uv run llm-train-torch ...
uv run llm-chat-torch ...

# Module execution
uv run python -m llm.trainer ...
uv run python -m llm.chat ...
uv run python -m llm.torch_trainer ...
uv run python -m llm.torch_chat ...
```

### 4 — Unified entry point via `main.py`

```bash
uv run python main.py
```

This trains the model on the bundled `data/conversations.txt` and immediately
opens the interactive chat.

Other supported `main.py` modes:

```bash
# Train only
uv run python main.py --mode train

# Chat only with an existing NumPy/CuPy checkpoint
uv run python main.py --mode chat --checkpoint model.npz

# Explicit all-in-one mode
uv run python main.py --mode train-chat

# Train/chat with custom settings
uv run python main.py --mode train --data data/conversations.txt --epochs 200 --lr 3e-4 --batch-size 32 --checkpoint my_model.npz
uv run python main.py --mode chat --checkpoint my_model.npz --temperature 0.35 --top-k 20 --max-tokens 120
```

### 5 — Train with the from-scratch backend

```bash
uv run llm-train

# Equivalent module form
uv run python -m llm.trainer

# Equivalent unified entry point
uv run python main.py --mode train
```

Examples:

```bash
# Custom dataset and checkpoint
uv run llm-train --data data/conversations.txt --checkpoint model.npz

# More epochs
uv run llm-train --epochs 300

# Custom learning rate and batch size
uv run llm-train --lr 0.0003 --batch-size 32

# Force backend selection
LLM_DEVICE=gpu uv run llm-train
LLM_DEVICE=cpu uv run llm-train

# PyTorch CUDA trainer (requires --extra torch)
uv run llm-train-torch
```

### 6 — Chat with the from-scratch backend

```bash
uv run llm-chat

# Equivalent module form
uv run python -m llm.chat

# Equivalent unified entry point
uv run python main.py --mode chat --checkpoint model.npz
```

Examples:

```bash
# Load a specific checkpoint
uv run llm-chat --checkpoint model.npz

# Train first, then drop into chat
uv run llm-chat --train --epochs 100 --data data/conversations.txt

# Lower-entropy sampling for more stable answers
uv run llm-chat --temperature 0.25 --top-k 12 --max-tokens 120

# Control retained history
uv run llm-chat --history-turns 6

# Retrieval options
uv run llm-chat --no-retrieval
uv run llm-chat --retrieval-threshold 0.4

# Bias toward code-writing and code-editing responses
uv run llm-chat --coding-assistant

# Force fully generative replies for every prompt
uv run llm-chat --no-retrieval --coding-assistant

# Pipe prompts non-interactively
printf 'hello\nwhat is python\n/quit\n' | uv run llm-chat --checkpoint model.npz
```

If `--checkpoint` does not exist, `llm-chat` will auto-train unless you point it at an existing model.
Code-like prompts now automatically prefer generation over retrieval.

### 7 — Train with the PyTorch CUDA backend

```bash
# Console script
uv run llm-train-torch

# Equivalent module form
uv run python -m llm.torch_trainer
```

Examples:

```bash
# Train and save to a custom PyTorch checkpoint
uv run llm-train-torch --checkpoint model_torch.pt

# Longer training run
uv run llm-train-torch --epochs 1000 --checkpoint model_torch.pt

# Custom dataset and batch size
uv run llm-train-torch --data data/conversations.txt --batch-size 128

# Disable CUDA throughput tuning
uv run llm-train-torch --no-gpu-boost
```

### 8 — Chat with the PyTorch checkpoint

```bash
# Console script
uv run llm-chat-torch --checkpoint model_torch.pt

# Equivalent module form
uv run python -m llm.torch_chat --checkpoint model_torch.pt
```

Examples:

```bash
# Recommended quality path
uv run llm-chat-torch --checkpoint model_torch.pt

# Coding assistant mode
uv run llm-chat-torch --checkpoint model_torch.pt --coding-assistant

# Lower or higher sampling entropy
uv run llm-chat-torch --checkpoint model_torch.pt --temperature 0.35 --top-k 12 --max-tokens 120

# Keep more conversation context
uv run llm-chat-torch --checkpoint model_torch.pt --history-turns 6

# Retrieval controls
uv run llm-chat-torch --checkpoint model_torch.pt --no-retrieval
uv run llm-chat-torch --checkpoint model_torch.pt --retrieval-threshold 0.4

# Force fully generative replies for every prompt
uv run llm-chat-torch --checkpoint model_torch.pt --no-retrieval --coding-assistant

# Pipe prompts non-interactively
printf 'write a python function to add two numbers\n/quit\n' | uv run llm-chat-torch --checkpoint model_torch.pt --coding-assistant
```

Code-like prompts now automatically prefer generation over retrieval here as well.

### 9 — Recommended combinations

```bash
# Fastest reliable GPU training path
uv sync --extra torch
uv run llm-train-torch --epochs 1000 --checkpoint model_torch.pt
uv run llm-chat-torch --checkpoint model_torch.pt --coding-assistant

# From-scratch backend on GPU via CuPy
uv sync --extra gpu
LLM_DEVICE=gpu uv run llm-train
uv run llm-chat --checkpoint model.npz

# CPU-only workflow
uv sync
LLM_DEVICE=cpu uv run llm-train
uv run llm-chat --checkpoint model.npz
```

### 10 — Chat only (after training)

```bash
# From-scratch checkpoint
uv run llm-chat --checkpoint model.npz

# PyTorch checkpoint
uv run llm-chat-torch --checkpoint model_torch.pt

# Chat with the PyTorch-trained checkpoint (recommended for quality)
uv run llm-chat-torch --checkpoint model_torch.pt

# Bias responses toward writing/editing code
uv run llm-chat-torch --checkpoint model_torch.pt --coding-assistant
```

If responses look noisy with `llm-chat`, use lower-entropy sampling:

```bash
uv run llm-chat --temperature 0.25 --top-k 12 --max-tokens 120
```

`llm-chat` now also uses a retrieval fallback over training pairs by default,
which greatly improves short/common prompts. Disable with:

```bash
uv run llm-chat --no-retrieval
```

Tune retrieval strictness with `--retrieval-threshold` (default `0.4`).

For coding-focused use, enable:

```bash
uv run llm-chat --coding-assistant
# or
uv run llm-chat-torch --checkpoint model_torch.pt --coding-assistant
```

### Improve coding answer quality

If you want smoother Python help, keep adding high-quality Q&A pairs to
`data/conversations.txt` using the same format:

```text
<human>: How do I read JSON in Python?
<assistant>: Use json.load(file_obj) for files and json.loads(text) for strings.
```

Then retrain (PyTorch path recommended):

```bash
uv run llm-train-torch --epochs 1000 --checkpoint model_torch.pt
uv run llm-chat-torch --checkpoint model_torch.pt --temperature 0.35 --top-k 12
```

Tips for better outputs:

- Keep answer style consistent and concise.
- Add many paraphrases of common Python questions.
- Include bug-fix prompts (e.g., `TypeError`, `NameError`, `ImportError`).
- Prefer lower sampling entropy for factual coding help.

### Chat commands

| Command | Action |
|---------|--------|
| `/reset` | Clear conversation history |
| `/quit` or `/exit` | Exit the chat |
| `/help` | Show the help message |

---

## Configuration

All hyper-parameters live in `src/llm/config.py`.  Key defaults:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 256 | Embedding / hidden dimension |
| `n_heads` | 8 | Number of attention heads |
| `n_layers` | 6 | Number of transformer blocks |
| `d_ff` | 1024 | Feed-forward hidden dimension |
| `context_length` | 256 | Maximum tokens in context |
| `learning_rate` | 3e-4 | Adam learning rate |
| `max_epochs` | 100 | Training epochs |
| `batch_size` | 32 | Mini-batch size (safer default for larger model) |

### GPU Selection

The backend is selected at import time:

- If CuPy is installed, GPU is used automatically.
- If CuPy is not installed, CPU/NumPy is used.
- Override explicitly with `LLM_DEVICE`:

```bash
LLM_DEVICE=gpu uv run llm-train
LLM_DEVICE=cpu uv run llm-train
```

When running on GPU, the trainer applies throughput-focused tuning by default
(`gpu_boost=True`): larger batches, less frequent validation, and lighter
dropout for faster training loops.

---

## CLI Options

### `main.py`

```text
uv run python main.py [--mode {train,chat,train-chat}]
                      [--checkpoint model.npz]
                      [--data data/conversations.txt]
                      [--epochs 100]
                      [--lr 3e-4]
                      [--batch-size 32]
                      [--temperature 0.8]
                      [--top-k 40]
                      [--max-tokens 200]
```

### `llm-train`

```text
uv run llm-train [--data PATH] [--checkpoint model.npz] [--epochs 100]
                 [--lr 3e-4] [--batch-size 32] [--no-gpu-boost]
```

### `llm-chat`

```text
uv run llm-chat [--checkpoint model.npz] [--temperature 0.35] [--top-k 20]
                [--max-tokens 120] [--history-turns 4] [--no-retrieval]
                [--retrieval-threshold 0.4] [--coding-assistant]
                [--train] [--data PATH] [--epochs 100]
```

### `llm-train-torch`

```text
uv run llm-train-torch [--data PATH] [--epochs 100] [--lr 3e-4]
                       [--batch-size 32] [--checkpoint model_torch.pt]
                       [--no-gpu-boost]
```

### `llm-chat-torch`

```text
uv run llm-chat-torch [--checkpoint model_torch.pt] [--temperature 0.35]
                      [--top-k 12] [--max-tokens 120] [--history-turns 4]
                      [--no-retrieval] [--retrieval-threshold 0.4]
                      [--coding-assistant]
```

---

## Running Tests

```bash
uv run pytest
```

---

## Project Structure

```
ideal-palm-tree/
├── pyproject.toml          # uv project: dependencies, scripts, build
├── .python-version         # Python version pin (3.11)
├── main.py                 # Unified entry point
├── data/
│   └── conversations.txt   # Bundled training corpus
└── src/
    └── llm/
        ├── __init__.py     # Public API
        ├── autograd.py     # Reverse-mode autodiff engine
        ├── nn.py           # Neural-network modules
        ├── model.py        # Transformer language model
        ├── tokenizer.py    # Byte-level subword tokenizer
        ├── data.py         # Dataset loader and sampler
        ├── trainer.py      # Training loop + Adam optimiser
        ├── chat.py         # Interactive chat REPL
        └── config.py       # Configuration dataclasses
```
