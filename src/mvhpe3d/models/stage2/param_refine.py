"""Stage 2 multiview fusion in canonical SMPL parameter space."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from mvhpe3d.utils import rotation_6d_to_axis_angle

from ..components import MLP


@dataclass(slots=True)
class Stage2ParamRefineConfig:
    """Configuration for the Stage 2 parameter refinement model."""

    name: str = "stage2_param_refine"
    learn_betas: bool = True
    num_joints: int = 23
    pose_6d_dim: int = 138
    betas_dim: int = 10
    hidden_dim: int = 512
    latent_dim: int = 512
    encoder_layers: int = 3
    weight_head_layers: int = 2
    refinement_layers: int = 3
    num_iterations: int = 1
    dropout: float = 0.1

    @property
    def input_dim(self) -> int:
        return self.pose_6d_dim + self.betas_dim


class Stage2ParamRefineModel(nn.Module):
    """Fuse per-view canonical SMPL parameters, then iteratively refine them."""

    def __init__(self, config: Stage2ParamRefineConfig) -> None:
        super().__init__()
        self.config = config

        self.view_encoder = MLP(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.latent_dim,
            num_layers=config.encoder_layers,
            dropout=config.dropout,
        )
        self.view_weight_head = MLP(
            input_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            output_dim=1,
            num_layers=config.weight_head_layers,
            dropout=config.dropout,
        )
        self.refinement_encoder = MLP(
            input_dim=config.latent_dim + 3 * config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.latent_dim,
            num_layers=config.refinement_layers,
            dropout=config.dropout,
        )
        self.refinement_head = MLP(
            input_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.pose_6d_dim + (config.betas_dim if config.learn_betas else 0),
            num_layers=config.refinement_layers,
            dropout=config.dropout,
        )

    def forward(self, views_input: torch.Tensor) -> dict[str, torch.Tensor]:
        if views_input.ndim != 3:
            raise ValueError(
                "Stage2ParamRefineModel expects views_input with shape "
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

        view_logits = self.view_weight_head(encoded_views.reshape(batch_size * num_views, -1))
        view_logits = view_logits.reshape(batch_size, num_views)
        view_weights = torch.softmax(view_logits, dim=1)

        initial_state = torch.sum(views_input * view_weights.unsqueeze(-1), dim=1)
        current_state = initial_state

        for _ in range(self.config.num_iterations):
            repeated_state = current_state.unsqueeze(1).expand(-1, num_views, -1)
            refinement_input = torch.cat(
                (
                    encoded_views,
                    views_input,
                    repeated_state,
                    views_input - repeated_state,
                ),
                dim=-1,
            )
            refinement_features = self.refinement_encoder(
                refinement_input.reshape(batch_size * num_views, -1)
            )
            refinement_features = refinement_features.reshape(batch_size, num_views, -1)
            pooled_refinement = torch.sum(
                refinement_features * view_weights.unsqueeze(-1),
                dim=1,
            )
            refinement_delta = self.refinement_head(pooled_refinement)
            if self.config.learn_betas:
                current_state = current_state + refinement_delta
            else:
                current_pose = current_state[:, : self.config.pose_6d_dim]
                current_betas = current_state[:, self.config.pose_6d_dim :]
                pose_delta = refinement_delta[:, : self.config.pose_6d_dim]
                current_state = torch.cat((current_pose + pose_delta, current_betas), dim=-1)

        init_pose_6d = initial_state[:, : self.config.pose_6d_dim].reshape(
            batch_size,
            self.config.num_joints,
            6,
        )
        pred_pose_6d = current_state[:, : self.config.pose_6d_dim].reshape(
            batch_size,
            self.config.num_joints,
            6,
        )
        init_betas = initial_state[:, self.config.pose_6d_dim :]
        pred_betas = current_state[:, self.config.pose_6d_dim :]

        return {
            "init_pose_6d": init_pose_6d,
            "init_body_pose": rotation_6d_to_axis_angle(init_pose_6d).reshape(batch_size, -1),
            "init_betas": init_betas,
            "pred_pose_6d": pred_pose_6d,
            "pred_body_pose": rotation_6d_to_axis_angle(pred_pose_6d).reshape(batch_size, -1),
            "pred_betas": pred_betas,
            "view_weights": view_weights,
        }
