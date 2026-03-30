"""Losses for Stage 1 canonical SMPL fusion."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(slots=True)
class Stage1LossConfig:
    """Weights for the Stage 1 objective."""

    betas_weight: float = 1.0
    body_pose_weight: float = 1.0


class Stage1SMPLLoss(nn.Module):
    """Weighted MSE loss over canonical betas and body pose."""

    def __init__(self, config: Stage1LossConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        *,
        pred_betas: torch.Tensor,
        pred_body_pose: torch.Tensor,
        target_betas: torch.Tensor,
        target_body_pose: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        betas_loss = F.mse_loss(pred_betas, target_betas)
        body_pose_loss = F.mse_loss(pred_body_pose, target_body_pose)
        total_loss = (
            self.config.betas_weight * betas_loss
            + self.config.body_pose_weight * body_pose_loss
        )
        return {
            "loss": total_loss,
            "loss_betas": betas_loss,
            "loss_body_pose": body_pose_loss,
        }
