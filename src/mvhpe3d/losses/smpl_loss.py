"""Losses for Stage 1 SMPL parameter fusion."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(slots=True)
class Stage1LossConfig:
    """Weights for the Stage 1 objective."""

    body_pose_weight: float = 1.0
    betas_weight: float = 1.0
    joint_weight: float = 1.0
    supervise_betas: bool = True


class Stage1Loss(nn.Module):
    """Weighted MSE loss over canonical SMPL body pose and betas."""

    def __init__(self, config: Stage1LossConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        *,
        pred_body_pose: torch.Tensor,
        pred_betas: torch.Tensor,
        target_body_pose: torch.Tensor,
        target_betas: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        body_pose_loss = F.mse_loss(pred_body_pose, target_body_pose)
        if self.config.supervise_betas:
            betas_loss = F.mse_loss(pred_betas, target_betas)
        else:
            betas_loss = body_pose_loss.new_zeros(())
        total_loss = (
            self.config.body_pose_weight * body_pose_loss
            + self.config.betas_weight * betas_loss
        )
        return {
            "loss": total_loss,
            "loss_body_pose": body_pose_loss,
            "loss_betas": betas_loss,
        }
