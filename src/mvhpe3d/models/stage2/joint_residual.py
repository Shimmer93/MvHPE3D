"""Joint-aware Stage 2 multiview fusion in canonical SMPL parameter space."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from mvhpe3d.utils import rotation_6d_to_axis_angle

from ..components import MLP


@dataclass(slots=True)
class Stage2JointResidualConfig:
    """Configuration for the joint-aware Stage 2 fusion model."""

    name: str = "stage2_joint_residual"
    learn_betas: bool = True
    num_joints: int = 23
    pose_6d_dim: int = 138
    betas_dim: int = 10
    hidden_dim: int = 512
    latent_dim: int = 512
    joint_latent_dim: int = 256
    encoder_layers: int = 3
    proposal_layers: int = 3
    weight_head_layers: int = 2
    refinement_layers: int = 3
    dropout: float = 0.1

    @property
    def input_dim(self) -> int:
        return self.pose_6d_dim + self.betas_dim


class Stage2JointResidualModel(nn.Module):
    """Fuse per-view pose proposals with joint-wise weights and local residuals."""

    def __init__(self, config: Stage2JointResidualConfig) -> None:
        super().__init__()
        self.config = config

        self.view_encoder = MLP(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.latent_dim,
            num_layers=config.encoder_layers,
            dropout=config.dropout,
        )
        self.pose_proposal_head = MLP(
            input_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.pose_6d_dim,
            num_layers=config.proposal_layers,
            dropout=config.dropout,
        )
        self.pose_weight_head = MLP(
            input_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.num_joints,
            num_layers=config.weight_head_layers,
            dropout=config.dropout,
        )
        self.beta_weight_head = MLP(
            input_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            output_dim=1,
            num_layers=config.weight_head_layers,
            dropout=config.dropout,
        )
        if config.learn_betas:
            self.betas_proposal_head: MLP | None = MLP(
                input_dim=config.latent_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.betas_dim,
                num_layers=config.proposal_layers,
                dropout=config.dropout,
            )
            self.betas_refinement_head: MLP | None = MLP(
                input_dim=config.latent_dim + 2 * config.betas_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.betas_dim,
                num_layers=config.refinement_layers,
                dropout=config.dropout,
            )
        else:
            self.betas_proposal_head = None
            self.betas_refinement_head = None
        self.pose_refinement_encoder = MLP(
            input_dim=config.latent_dim + 12,
            hidden_dim=config.hidden_dim,
            output_dim=config.joint_latent_dim,
            num_layers=config.refinement_layers,
            dropout=config.dropout,
        )
        self.pose_refinement_head = MLP(
            input_dim=config.joint_latent_dim,
            hidden_dim=config.hidden_dim,
            output_dim=6,
            num_layers=config.refinement_layers,
            dropout=config.dropout,
        )

    def forward(self, views_input: torch.Tensor) -> dict[str, torch.Tensor]:
        if views_input.ndim != 3:
            raise ValueError(
                "Stage2JointResidualModel expects views_input with shape "
                f"[batch, num_views, dim], got {tuple(views_input.shape)}"
            )
        if views_input.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"Expected views_input trailing dimension {self.config.input_dim}, "
                f"got {views_input.shape[-1]}"
            )

        batch_size, num_views, _ = views_input.shape
        encoded_views = self.view_encoder(views_input.reshape(batch_size * num_views, -1))
        encoded_views = encoded_views.reshape(batch_size, num_views, -1)

        per_view_pose = self.pose_proposal_head(encoded_views.reshape(batch_size * num_views, -1))
        per_view_pose = per_view_pose.reshape(batch_size, num_views, self.config.num_joints, 6)

        pose_logits = self.pose_weight_head(encoded_views.reshape(batch_size * num_views, -1))
        pose_logits = pose_logits.reshape(batch_size, num_views, self.config.num_joints)
        pose_weights = torch.softmax(pose_logits, dim=1)
        pose_weight_expanded = pose_weights.unsqueeze(-1)

        init_pose_6d = torch.sum(per_view_pose * pose_weight_expanded, dim=1)
        pose_dispersion = torch.sum(
            pose_weight_expanded * (per_view_pose - init_pose_6d.unsqueeze(1)).abs(),
            dim=1,
        )
        joint_features = torch.sum(
            encoded_views.unsqueeze(2) * pose_weight_expanded,
            dim=1,
        )

        refinement_input = torch.cat((joint_features, init_pose_6d, pose_dispersion), dim=-1)
        refinement_features = self.pose_refinement_encoder(
            refinement_input.reshape(batch_size * self.config.num_joints, -1)
        )
        refinement_features = refinement_features.reshape(
            batch_size,
            self.config.num_joints,
            self.config.joint_latent_dim,
        )
        pose_delta = self.pose_refinement_head(
            refinement_features.reshape(batch_size * self.config.num_joints, -1)
        ).reshape(batch_size, self.config.num_joints, 6)
        pred_pose_6d = init_pose_6d + pose_delta

        if self.config.learn_betas:
            assert self.betas_proposal_head is not None
            assert self.betas_refinement_head is not None
            per_view_betas = self.betas_proposal_head(
                encoded_views.reshape(batch_size * num_views, -1)
            ).reshape(batch_size, num_views, self.config.betas_dim)
            beta_logits = self.beta_weight_head(encoded_views.reshape(batch_size * num_views, -1))
            beta_logits = beta_logits.reshape(batch_size, num_views)
            beta_weights = torch.softmax(beta_logits, dim=1)
            init_betas = torch.sum(per_view_betas * beta_weights.unsqueeze(-1), dim=1)
            beta_dispersion = torch.sum(
                beta_weights.unsqueeze(-1) * (per_view_betas - init_betas.unsqueeze(1)).abs(),
                dim=1,
            )
            pooled_feature = torch.sum(encoded_views * beta_weights.unsqueeze(-1), dim=1)
            betas_delta = self.betas_refinement_head(
                torch.cat((pooled_feature, init_betas, beta_dispersion), dim=-1)
            )
            pred_betas = init_betas + betas_delta
            view_weights = beta_weights
        else:
            init_betas = views_input[:, :, self.config.pose_6d_dim :].mean(dim=1)
            pred_betas = init_betas
            view_weights = pose_weights.mean(dim=-1)

        return {
            "init_pose_6d": init_pose_6d,
            "init_body_pose": rotation_6d_to_axis_angle(init_pose_6d).reshape(batch_size, -1),
            "init_betas": init_betas,
            "pred_pose_6d": pred_pose_6d,
            "pred_body_pose": rotation_6d_to_axis_angle(pred_pose_6d).reshape(batch_size, -1),
            "pred_betas": pred_betas,
            "view_weights": view_weights,
            "pose_view_weights": pose_weights,
        }
