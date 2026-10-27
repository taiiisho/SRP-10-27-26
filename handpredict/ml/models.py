"""The neural networks (PyTorch).

* ``RecurrentPredictor`` (primary ML model, "gru"): a small GRU (or LSTM)
  reads the window one sample at a time, oldest first, keeping a memory
  ("hidden state") of the motion so far. Its final memory, together with the
  static inputs (current position, target), goes through a small dense
  "head" that outputs the displacement at every horizon at once.
* ``MLPPredictor`` ("mlp"): a plain feed-forward network on the flattened
  window. Simpler, and made only of layers that AI accelerators such as the
  Hailo-10H support well, which is why it exists.

Both take the same two inputs (seq, static) and return (N, H, 2) outputs in
normalised units (see ``normalize.py``).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .windows import WindowSpec


class RecurrentPredictor(nn.Module):
    def __init__(
        self,
        spec: WindowSpec,
        hidden_size: int = 32,
        num_layers: int = 1,
        cell: str = "gru",
        dropout: float = 0.0,
        head_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.n_horizons = spec.n_horizons
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.cell = cell
        self.rnn = rnn_cls(
            input_size=spec.seq_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + spec.n_static, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, spec.n_horizons * 2),
        )

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        _, state = self.rnn(seq)
        hidden = state[0] if self.cell == "lstm" else state  # LSTM returns (h, c)
        features = torch.cat([hidden[-1], static], dim=-1)
        return self.head(features).reshape(-1, self.n_horizons, 2)


class MLPPredictor(nn.Module):
    def __init__(self, spec: WindowSpec, hidden_sizes: Sequence[int] = (64, 64), dropout: float = 0.0) -> None:
        super().__init__()
        self.n_horizons = spec.n_horizons
        self.n_inputs = spec.n_samples * spec.seq_features + spec.n_static
        layers: list[nn.Module] = []
        width = self.n_inputs
        for size in hidden_sizes:
            layers += [nn.Linear(width, int(size)), nn.ReLU(), nn.Dropout(dropout)]
            width = int(size)
        layers.append(nn.Linear(width, spec.n_horizons * 2))
        self.net = nn.Sequential(*layers)

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        flat = torch.cat([seq.flatten(1), static], dim=-1)
        return self.net(flat).reshape(-1, self.n_horizons, 2)


class FlatMLP(nn.Module):
    """The MLP with ONE flat input vector (for accelerator compilers that want a single input)."""

    def __init__(self, mlp: MLPPredictor) -> None:
        super().__init__()
        self.net = mlp.net

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        return self.net(flat)


class EnsembleModule(nn.Module):
    """Average of several trained copies (different random seeds) of the same network."""

    def __init__(self, members: Sequence[nn.Module]) -> None:
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        outputs = torch.stack([m(seq, static) for m in self.members], dim=0)
        return outputs.mean(dim=0)


def build_network(kind: str, spec: WindowSpec, params: Mapping[str, Any]) -> nn.Module:
    """Create an untrained network from its kind and hyperparameters."""
    if kind == "recurrent":
        return RecurrentPredictor(
            spec,
            hidden_size=int(params["hidden_size"]),
            num_layers=int(params.get("num_layers", 1)),
            cell=str(params.get("cell", "gru")),
            dropout=float(params.get("dropout", 0.0)),
            head_hidden=int(params.get("head_hidden", 64)),
        )
    if kind == "mlp":
        return MLPPredictor(spec, hidden_sizes=list(params["hidden_sizes"]), dropout=float(params.get("dropout", 0.0)))
    raise ValueError(f"unknown network kind {kind!r}")


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
