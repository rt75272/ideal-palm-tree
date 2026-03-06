# LLM From Scratch

A fully-featured **Large Language Model built from scratch** in Python — no
ML frameworks (PyTorch / TensorFlow / JAX / Keras) used anywhere.

The only external dependency is **NumPy** for efficient array arithmetic.
Everything else — automatic differentiation, the transformer architecture,
the Adam optimiser, the tokeniser, training loop, and interactive chat
interface — is written in plain Python.

Project dependencies and the virtual environment are managed entirely with
[**uv**](https://docs.astral.sh/uv/) and `pyproject.toml` (no `pip` or
`requirements.txt`).

---

## Architecture

| Component | Description |
|-----------|-------------|
| **Autograd engine** (`autograd.py`) | Reverse-mode automatic differentiation over NumPy arrays |
| **Neural-network modules** (`nn.py`) | `Linear`, `LayerNorm`, `Embedding`, `Dropout`, `MultiHeadAttention`, `FeedForward`, `TransformerBlock` |
| **Language model** (`model.py`) | Decoder-only transformer (GPT-style): token + positional embeddings → N transformer blocks → LM head |
| **Tokeniser** (`tokenizer.py`) | Character-level vocabulary; saved/loaded as JSON |
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
```

### 3 — Train then chat (all-in-one)

```bash
uv run python main.py
```

This trains the model on the bundled `data/conversations.txt` and immediately
opens the interactive chat.

### 4 — Train only

```bash
uv run llm-train
# or
uv run python main.py --mode train
```

### 5 — Chat only (after training)

```bash
uv run llm-chat
# or
uv run python main.py --mode chat
```

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
| `d_model` | 128 | Embedding / hidden dimension |
| `n_heads` | 4 | Number of attention heads |
| `n_layers` | 4 | Number of transformer blocks |
| `d_ff` | 512 | Feed-forward hidden dimension |
| `context_length` | 128 | Maximum tokens in context |
| `learning_rate` | 3e-4 | Adam learning rate |
| `max_epochs` | 100 | Training epochs |

---

## CLI Options

```
python main.py [--mode {train,chat,train-chat}]
               [--checkpoint model.npz]
               [--data data/conversations.txt]
               [--epochs 100]
               [--lr 3e-4]
               [--batch-size 16]
               [--temperature 0.8]
               [--top-k 40]
               [--max-tokens 200]
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
        ├── tokenizer.py    # Character-level tokeniser
        ├── data.py         # Dataset loader and sampler
        ├── trainer.py      # Training loop + Adam optimiser
        ├── chat.py         # Interactive chat REPL
        └── config.py       # Configuration dataclasses
```
