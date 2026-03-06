"""PyTorch-based GPU trainer for the byte-level subword language model."""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

from llm.config import ModelConfig, TrainingConfig
from llm.data import load_text, train_val_split
from llm.tokenizer import Tokenizer

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - depends on runtime environment
    raise ImportError(
        "PyTorch is required for llm-train-torch. Install with 'uv add torch' "
        "or 'pip install torch'."
    ) from exc


class TorchLanguageModel(nn.Module):
    """Small GPT-style decoder using PyTorch Transformer blocks."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.context_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = idx.shape
        if seq_len > self.config.context_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds context_length {self.config.context_length}."
            )

        positions = torch.arange(seq_len, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(positions)
        x = self.dropout(x)

        # True entries are masked by nn.TransformerEncoder.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=idx.device, dtype=torch.bool),
            diagonal=1,
        )

        x = self.blocks(x, mask=causal_mask)
        x = self.norm(x)
        return self.lm_head(x)

    def loss(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self.forward(idx)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )


class TorchBatchSampler:
    """Vectorized next-token batch sampler directly on a torch device."""

    def __init__(self, token_ids: list[int], context_length: int, device: torch.device) -> None:
        self.context_length = context_length
        self.device = device
        self.tokens = torch.tensor(token_ids, dtype=torch.long, device=device)
        self.max_start = len(token_ids) - context_length - 1
        self.offsets = torch.arange(context_length, device=device)
        if self.max_start <= 0:
            raise ValueError(
                f"Corpus too short ({len(token_ids)} tokens) for context_length={context_length}."
            )

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(0, self.max_start + 1, (batch_size, 1), device=self.device)
        idx = starts + self.offsets.view(1, -1)
        x = self.tokens[idx]
        y = self.tokens[idx + 1]
        return x, y


def _compute_loss(
    model: TorchLanguageModel,
    sampler: TorchBatchSampler,
    batch_size: int,
    n_batches: int,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = sampler.get_batch(batch_size)
            losses.append(float(model.loss(x, y).item()))
    model.train()
    return float(sum(losses) / len(losses))


def train_torch(
    model_config: ModelConfig | None = None,
    train_config: TrainingConfig | None = None,
    data_path: str | None = None,
    checkpoint_path: str = "model_torch.pt",
) -> tuple[TorchLanguageModel, Tokenizer]:
    """Train the model with PyTorch and CUDA."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available to PyTorch in this environment. "
            "Install a CUDA-enabled PyTorch build and NVIDIA driver, then retry."
        )

    device = torch.device("cuda")
    mc = model_config or ModelConfig()
    tc = train_config or TrainingConfig()

    # Throughput-oriented defaults for CUDA.
    if tc.gpu_boost:
        tc.batch_size = max(tc.batch_size, 64)
        tc.eval_interval = max(tc.eval_interval, 50)
        tc.eval_batches = min(tc.eval_batches, 2)
        mc.dropout = min(mc.dropout, 0.05)

    print("Using GPU backend (PyTorch CUDA).")
    print("Loading training data...")
    text = load_text(data_path)
    print(f"  Corpus size: {len(text):,} characters")

    tokenizer = Tokenizer()
    tokenizer.build_from_text(text)
    mc.vocab_size = tokenizer.vocab_size
    print(f"  Vocabulary size: {tokenizer.vocab_size} characters")

    token_ids = tokenizer.encode(text)
    train_ids, val_ids = train_val_split(token_ids, val_fraction=0.1)
    train_sampler = TorchBatchSampler(train_ids, mc.context_length, device)
    val_sampler = TorchBatchSampler(val_ids, mc.context_length, device)
    print(f"  Train tokens: {len(train_ids):,} | Val tokens: {len(val_ids):,}")

    torch.manual_seed(42)
    random.seed(42)
    model = TorchLanguageModel(mc).to(device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc.learning_rate,
        betas=(tc.beta1, tc.beta2),
        eps=tc.epsilon,
        weight_decay=tc.weight_decay,
    )

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    scaler = torch.amp.GradScaler("cuda")

    print(
        f"\nTraining for {tc.max_epochs} epochs "
        f"(batch_size={tc.batch_size}, eval_interval={tc.eval_interval})..."
    )
    start_time = time.time()

    model.train()
    for epoch in range(1, tc.max_epochs + 1):
        x, y = train_sampler.get_batch(tc.batch_size)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = model.loss(x, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if epoch % tc.eval_interval == 0 or epoch == tc.max_epochs:
            elapsed = time.time() - start_time
            val_loss = _compute_loss(model, val_sampler, tc.batch_size, tc.eval_batches)
            train_loss = float(loss.item())
            print(
                f"  Epoch {epoch:>5}/{tc.max_epochs} | "
                f"train loss: {train_loss:.4f} | "
                f"val loss: {val_loss:.4f} | "
                f"elapsed: {elapsed:.1f}s"
            )

    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": vars(mc),
        "tokenizer_state": tokenizer.to_state(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"\nSaved PyTorch checkpoint to '{checkpoint_path}'.")

    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the LLM with PyTorch on CUDA."
    )
    parser.add_argument("--data", default=None, help="Path to the text training file.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--checkpoint", default="model_torch.pt", help="Output .pt checkpoint.")
    parser.add_argument(
        "--no-gpu-boost",
        action="store_true",
        help="Disable throughput-oriented CUDA tuning.",
    )
    args = parser.parse_args()

    tc = TrainingConfig(
        max_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint,
        gpu_boost=not args.no_gpu_boost,
    )

    try:
        train_torch(train_config=tc, data_path=args.data, checkpoint_path=args.checkpoint)
    except (FileNotFoundError, RuntimeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
