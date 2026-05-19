"""Checkpoint-based RGB residual adapter for Stage 2 predictions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from mvhpe3d.utils import rotation_6d_to_axis_angle

from ..components import MLP


@dataclass(slots=True)
class Stage2RRGBGuidedResidualRefinerConfig:
    """Configuration for a small RGB residual head on top of a Stage 2 checkpoint."""

    name: str = "stage2r_rgb_guided_residual_refiner"
    backbone_name: str = "stage2_joint_graph_refiner"
    learn_betas: bool = True
    num_joints: int = 23
    pose_6d_dim: int = 138
    betas_dim: int = 10
    hidden_dim: int = 256
    latent_dim: int = 256
    encoder_layers: int = 2
    residual_layers: int = 3
    dropout: float = 0.1
    rgb_feature_dim: int = 384
    rgb_projection_dim: int = 128
    rgb_hidden_dim: int = 128
    rgb_layers: int = 2
    pose_residual_scale: float = 0.03
    freeze_backbone: bool = True
    detach_backbone_outputs: bool = True

    @property
    def input_dim(self) -> int:
        return self.pose_6d_dim + self.betas_dim


class Stage2RRGBGuidedResidualRefinerModel(nn.Module):
    """Refine a frozen Stage 2 prediction with RGB features pooled by Stage 2 weights."""

    def __init__(self, config: Stage2RRGBGuidedResidualRefinerConfig) -> None:
        super().__init__()
        self.config = config
        if config.rgb_feature_dim <= 0:
            raise ValueError("rgb_feature_dim must be positive")
        self.view_encoder = MLP(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.latent_dim,
            num_layers=config.encoder_layers,
            dropout=config.dropout,
        )
        self.rgb_projection = MLP(
            input_dim=config.rgb_feature_dim,
            hidden_dim=config.rgb_hidden_dim,
            output_dim=config.rgb_projection_dim,
            num_layers=config.rgb_layers,
            dropout=config.dropout,
        )
        residual_input_dim = config.latent_dim + config.rgb_projection_dim + 18
        self.pose_residual_head = MLP(
            input_dim=residual_input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=6,
            num_layers=config.residual_layers,
            dropout=config.dropout,
        )
        _zero_last_linear(self.pose_residual_head)

    def forward(
        self,
        views_input: torch.Tensor,
        *,
        view_rgb_feature: torch.Tensor,
        stage2_outputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if views_input.ndim != 3:
            raise ValueError(
                "Stage2RRGBGuidedResidualRefinerModel expects views_input with "
                f"shape [batch, num_views, dim], got {tuple(views_input.shape)}"
            )
        if views_input.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"Expected views_input trailing dimension {self.config.input_dim}, "
                f"got {views_input.shape[-1]}"
            )
        if view_rgb_feature.ndim != 3:
            raise ValueError(
                "view_rgb_feature must have shape [batch, num_views, rgb_feature_dim], "
                f"got {tuple(view_rgb_feature.shape)}"
            )

        batch_size, num_views, _ = views_input.shape
        expected_rgb_shape = (batch_size, num_views, self.config.rgb_feature_dim)
        if tuple(view_rgb_feature.shape) != expected_rgb_shape:
            raise ValueError(
                f"Expected view_rgb_feature shape {expected_rgb_shape}, "
                f"got {tuple(view_rgb_feature.shape)}"
            )
        view_rgb_feature = view_rgb_feature.to(
            device=views_input.device,
            dtype=views_input.dtype,
        )

        stage2_pred_pose_6d = _reshape_pose(
            stage2_outputs["pred_pose_6d"],
            batch_size=batch_size,
            num_joints=self.config.num_joints,
        )
        stage2_init_pose_6d = _reshape_pose(
            stage2_outputs["init_pose_6d"],
            batch_size=batch_size,
            num_joints=self.config.num_joints,
        )
        stage2_pred_betas = stage2_outputs["pred_betas"].reshape(
            batch_size,
            self.config.betas_dim,
        )
        stage2_init_betas = stage2_outputs.get("init_betas", stage2_pred_betas).reshape(
            batch_size,
            self.config.betas_dim,
        )
        pose_weights = self._get_pose_weights(
            stage2_outputs=stage2_outputs,
            batch_size=batch_size,
            num_views=num_views,
            device=views_input.device,
            dtype=views_input.dtype,
        )
        if self.config.detach_backbone_outputs:
            stage2_pred_pose_6d = stage2_pred_pose_6d.detach()
            stage2_init_pose_6d = stage2_init_pose_6d.detach()
            stage2_pred_betas = stage2_pred_betas.detach()
            stage2_init_betas = stage2_init_betas.detach()
            pose_weights = pose_weights.detach()

        encoded_views = self.view_encoder(views_input.reshape(batch_size * num_views, -1))
        encoded_views = encoded_views.reshape(batch_size, num_views, self.config.latent_dim)
        rgb_tokens = self.rgb_projection(
            view_rgb_feature.reshape(batch_size * num_views, self.config.rgb_feature_dim)
        )
        rgb_tokens = rgb_tokens.reshape(
            batch_size,
            num_views,
            self.config.rgb_projection_dim,
        )
        per_view_context = torch.cat((encoded_views, rgb_tokens), dim=-1)
        weighted_context = torch.sum(
            per_view_context.unsqueeze(2) * pose_weights.unsqueeze(-1),
            dim=1,
        )

        input_pose_6d = views_input[..., : self.config.pose_6d_dim].reshape(
            batch_size,
            num_views,
            self.config.num_joints,
            6,
        )
        input_init_pose = torch.sum(input_pose_6d * pose_weights.unsqueeze(-1), dim=1)
        input_pose_dispersion = torch.sum(
            pose_weights.unsqueeze(-1)
            * (input_pose_6d - input_init_pose.unsqueeze(1)).abs(),
            dim=1,
        )

        residual_input = torch.cat(
            (
                weighted_context,
                stage2_init_pose_6d,
                input_pose_dispersion,
                stage2_pred_pose_6d,
            ),
            dim=-1,
        )
        pose_residual = self.pose_residual_head(
            residual_input.reshape(batch_size * self.config.num_joints, -1)
        ).reshape(batch_size, self.config.num_joints, 6)
        pose_residual = self.config.pose_residual_scale * torch.tanh(pose_residual)
        pred_pose_6d = stage2_pred_pose_6d + pose_residual

        return {
            "init_pose_6d": stage2_init_pose_6d,
            "init_body_pose": rotation_6d_to_axis_angle(stage2_init_pose_6d).reshape(
                batch_size,
                -1,
            ),
            "init_betas": stage2_init_betas,
            "stage2_pred_pose_6d": stage2_pred_pose_6d,
            "stage2_pred_betas": stage2_pred_betas,
            "pred_pose_6d": pred_pose_6d,
            "pred_body_pose": rotation_6d_to_axis_angle(pred_pose_6d).reshape(
                batch_size,
                -1,
            ),
            "pred_betas": stage2_pred_betas,
            "view_weights": pose_weights.mean(dim=-1),
            "pose_view_weights": pose_weights,
            "stage2r_pose_residual_6d": pose_residual,
        }

    def _get_pose_weights(
        self,
        *,
        stage2_outputs: dict[str, torch.Tensor],
        batch_size: int,
        num_views: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        pose_weights = stage2_outputs.get("pose_view_weights")
        if pose_weights is not None:
            expected_shape = (batch_size, num_views, self.config.num_joints)
            if tuple(pose_weights.shape) != expected_shape:
                raise ValueError(
                    f"Expected pose_view_weights shape {expected_shape}, "
                    f"got {tuple(pose_weights.shape)}"
                )
            return pose_weights.to(device=device, dtype=dtype)
        view_weights = stage2_outputs.get("view_weights")
        if view_weights is None:
            raise ValueError(
                "Stage 2 backbone must return pose_view_weights or view_weights"
            )
        expected_view_shape = (batch_size, num_views)
        if tuple(view_weights.shape) != expected_view_shape:
            raise ValueError(
                f"Expected view_weights shape {expected_view_shape}, "
                f"got {tuple(view_weights.shape)}"
            )
        return view_weights.to(device=device, dtype=dtype).unsqueeze(-1).expand(
            batch_size,
            num_views,
            self.config.num_joints,
        )


def _reshape_pose(
    pose: torch.Tensor,
    *,
    batch_size: int,
    num_joints: int,
) -> torch.Tensor:
    return pose.reshape(batch_size, num_joints, 6)


def _zero_last_linear(module: nn.Module) -> None:
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Linear):
            nn.init.zeros_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
            return
