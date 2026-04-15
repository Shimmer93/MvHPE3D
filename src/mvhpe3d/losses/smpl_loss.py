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


@dataclass(slots=True)
class Stage2LossConfig:
    """Weights for the Stage 2 objective."""

    pose_6d_weight: float = 1.0
    betas_weight: float = 0.1
    joint_weight: float = 1.0
    init_pose_6d_weight: float = 0.25
    init_betas_weight: float = 0.025
    supervise_betas: bool = True


class Stage2Loss(nn.Module):
    """Weighted losses over canonical SMPL parameters and initialization quality."""

    def __init__(self, config: Stage2LossConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        *,
        pred_pose_6d: torch.Tensor,
        pred_betas: torch.Tensor,
        target_pose_6d: torch.Tensor,
        target_betas: torch.Tensor,
        init_pose_6d: torch.Tensor,
        init_betas: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pose_6d_loss = F.mse_loss(pred_pose_6d, target_pose_6d)
        if self.config.supervise_betas:
            betas_loss = F.mse_loss(pred_betas, target_betas)
            init_betas_loss = F.mse_loss(init_betas, target_betas)
        else:
            betas_loss = pose_6d_loss.new_zeros(())
            init_betas_loss = pose_6d_loss.new_zeros(())
        init_pose_6d_loss = F.mse_loss(init_pose_6d, target_pose_6d)

        total_loss = (
            self.config.pose_6d_weight * pose_6d_loss
            + self.config.betas_weight * betas_loss
            + self.config.init_pose_6d_weight * init_pose_6d_loss
            + self.config.init_betas_weight * init_betas_loss
        )
        return {
            "loss": total_loss,
            "loss_pose_6d": pose_6d_loss,
            "loss_betas": betas_loss,
            "loss_init_pose_6d": init_pose_6d_loss,
            "loss_init_betas": init_betas_loss,
        }
