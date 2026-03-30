"""Minimal permutation-invariant set pooling blocks."""

from __future__ import annotations

import torch
import torch.nn as nn


class MeanSetPooling(nn.Module):
    """Pool view features with mean aggregation."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                f"Expected inputs with shape [batch, num_views, dim], got {tuple(inputs.shape)}"
            )
        return inputs.mean(dim=1)
