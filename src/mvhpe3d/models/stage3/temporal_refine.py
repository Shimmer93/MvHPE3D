"""Temporal Stage 3 refinement over per-frame Stage 2 fused predictions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from mvhpe3d.utils import rotation_6d_to_axis_angle

from ..components import MLP


class TemporalResidualBlock(nn.Module):
    """Simple residual 1D temporal block with same-length output."""

    def __init__(self, hidden_dim: int, *, kernel_size: int, dropout: float) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
        )
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.block(inputs))


@dataclass(slots=True)
class Stage3TemporalRefineConfig:
    """Configuration for the Stage 3 temporal refinement model."""

    name: str = "stage3_temporal_refine"
    backbone_name: str = "stage2_joint_residual"
    learn_betas: bool = False
    num_joints: int = 23
    pose_6d_dim: int = 138
    betas_dim: int = 10
    hidden_dim: int = 512
    temporal_dim: int = 256
    temporal_layers: int = 3
    temporal_kernel_size: int = 3
    head_layers: int = 3
    dropout: float = 0.1
    pose_residual_scale: float = 0.1
    betas_residual_scale: float = 0.05
    freeze_backbone: bool = True

    @property
    def temporal_feature_dim(self) -> int:
        return self.pose_6d_dim * 2 + self.betas_dim


class Stage3TemporalRefineModel(nn.Module):
    """Refine the center-frame Stage 2 prediction using short-range temporal context."""

    def __init__(self, config: Stage3TemporalRefineConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.temporal_feature_dim, config.temporal_dim)
        self.temporal_blocks = nn.ModuleList(
            [
                TemporalResidualBlock(
                    config.temporal_dim,
                    kernel_size=config.temporal_kernel_size,
                    dropout=config.dropout,
                )
                for _ in range(config.temporal_layers)
            ]
        )
        self.pose_head = MLP(
            input_dim=config.temporal_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.pose_6d_dim,
            num_layers=config.head_layers,
            dropout=config.dropout,
        )
        if config.learn_betas:
            self.betas_head: MLP | None = MLP(
                input_dim=config.temporal_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.betas_dim,
                num_layers=config.head_layers,
                dropout=config.dropout,
            )
        else:
            self.betas_head = None

    def forward(
        self,
        temporal_features: torch.Tensor,
        *,
        base_pose_6d: torch.Tensor,
        base_betas: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if temporal_features.ndim != 3:
            raise ValueError(
                "Stage3TemporalRefineModel expects temporal_features with shape "
                f"[batch, time, dim], got {tuple(temporal_features.shape)}"
            )
        if temporal_features.shape[-1] != self.config.temporal_feature_dim:
            raise ValueError(
                f"Expected temporal_features trailing dimension {self.config.temporal_feature_dim}, "
                f"got {temporal_features.shape[-1]}"
            )

        batch_size, num_frames, _ = temporal_features.shape
        center_index = num_frames // 2

        hidden = self.input_projection(temporal_features)
        hidden = hidden.transpose(1, 2)
        for block in self.temporal_blocks:
            hidden = block(hidden)
        center_feature = hidden[:, :, center_index]

        pose_delta = self.pose_head(center_feature)
        pred_pose_6d = base_pose_6d + self.config.pose_residual_scale * torch.tanh(pose_delta)

        if self.config.learn_betas:
            assert self.betas_head is not None
            betas_delta = self.betas_head(center_feature)
            pred_betas = base_betas + self.config.betas_residual_scale * torch.tanh(betas_delta)
        else:
            pred_betas = base_betas

        pred_pose_6d_reshaped = pred_pose_6d.reshape(batch_size, self.config.num_joints, 6)
        base_pose_6d_reshaped = base_pose_6d.reshape(batch_size, self.config.num_joints, 6)
        return {
            "base_pose_6d": base_pose_6d_reshaped,
            "base_body_pose": rotation_6d_to_axis_angle(base_pose_6d_reshaped).reshape(batch_size, -1),
            "base_betas": base_betas,
            "pred_pose_6d": pred_pose_6d_reshaped,
            "pred_body_pose": rotation_6d_to_axis_angle(pred_pose_6d_reshaped).reshape(batch_size, -1),
            "pred_betas": pred_betas,
        }
