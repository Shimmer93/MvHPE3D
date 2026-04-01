"""Losses for Stage 1 MHR parameter fusion."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(slots=True)
class Stage1LossConfig:
    """Weights for the Stage 1 objective."""

    mhr_weight: float = 1.0
    shape_weight: float = 1.0


class Stage1Loss(nn.Module):
    """Weighted MSE loss over MHR model params and shape params."""

    def __init__(self, config: Stage1LossConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        *,
        pred_mhr_params: torch.Tensor,
        pred_shape_params: torch.Tensor,
        target_mhr_params: torch.Tensor,
        target_shape_params: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mhr_loss = F.mse_loss(pred_mhr_params, target_mhr_params)
        shape_loss = F.mse_loss(pred_shape_params, target_shape_params)
        total_loss = (
            self.config.mhr_weight * mhr_loss
            + self.config.shape_weight * shape_loss
        )
        return {
            "loss": total_loss,
            "loss_mhr_params": mhr_loss,
            "loss_shape_params": shape_loss,
        }
