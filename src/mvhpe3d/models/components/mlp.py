"""Small MLP building block used by Stage 1 models."""

from __future__ import annotations

import torch.nn as nn


class MLP(nn.Module):
    """A simple feed-forward network with ReLU activations."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        if num_layers == 1:
            self.network = nn.Linear(input_dim, output_dim)
            return

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.network(inputs)
