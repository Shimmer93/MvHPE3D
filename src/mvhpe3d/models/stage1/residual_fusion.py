"""Residual multiview fusion model for Stage 1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..components import MLP


@dataclass(slots=True)
class Stage1ResidualFusionConfig:
    """Configuration for the residual Stage 1 fusion model."""

    input_dim: int = 249
    zero_mhr_root_input: bool = False
    learn_betas: bool = True
    body_pose_dim: int = 69
    betas_dim: int = 10
    hidden_dim: int = 256
    latent_dim: int = 256
    encoder_layers: int = 2
    proposal_layers: int = 2
    decoder_layers: int = 2
    weight_head_layers: int = 2
    dropout: float = 0.0


class Stage1ResidualFusionModel(nn.Module):
    """Fuse per-view pose proposals and predict a residual correction."""

    def __init__(self, config: Stage1ResidualFusionConfig) -> None:
        super().__init__()
        self.config = config

        residual_dim = config.body_pose_dim + (config.betas_dim if config.learn_betas else 0)
        decoder_input_dim = config.latent_dim + config.body_pose_dim
        if config.learn_betas:
            decoder_input_dim += config.betas_dim

        self.view_encoder = MLP(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.latent_dim,
            num_layers=config.encoder_layers,
            dropout=config.dropout,
        )
        self.body_pose_head = MLP(
            input_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.body_pose_dim,
            num_layers=config.proposal_layers,
            dropout=config.dropout,
        )
        self.view_weight_head = MLP(
            input_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            output_dim=1,
            num_layers=config.weight_head_layers,
            dropout=config.dropout,
        )
        if config.learn_betas:
            self.betas_head: MLP | None = MLP(
                input_dim=config.latent_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.betas_dim,
                num_layers=config.proposal_layers,
                dropout=config.dropout,
            )
        else:
            self.betas_head = None
        self.residual_decoder = MLP(
            input_dim=decoder_input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=residual_dim,
            num_layers=config.decoder_layers,
            dropout=config.dropout,
        )

    def forward(self, views_input: torch.Tensor) -> dict[str, torch.Tensor]:
        if views_input.ndim != 3:
            raise ValueError(
                "Stage1ResidualFusionModel expects views_input with shape "
                f"[batch, num_views, dim], got {tuple(views_input.shape)}"
            )

        batch_size, num_views, _ = views_input.shape
        encoded_views = self.view_encoder(views_input.reshape(batch_size * num_views, -1))
        encoded_views = encoded_views.reshape(batch_size, num_views, -1)

        view_logits = self.view_weight_head(encoded_views.reshape(batch_size * num_views, -1))
        view_logits = view_logits.reshape(batch_size, num_views)
        view_weights = torch.softmax(view_logits, dim=1)

        per_view_body_pose = self.body_pose_head(encoded_views.reshape(batch_size * num_views, -1))
        per_view_body_pose = per_view_body_pose.reshape(batch_size, num_views, self.config.body_pose_dim)
        reference_body_pose = torch.sum(
            per_view_body_pose * view_weights.unsqueeze(-1),
            dim=1,
        )

        pooled_feature = torch.sum(encoded_views * view_weights.unsqueeze(-1), dim=1)

        decoder_inputs = [pooled_feature, reference_body_pose]
        if self.config.learn_betas:
            assert self.betas_head is not None
            per_view_betas = self.betas_head(encoded_views.reshape(batch_size * num_views, -1))
            per_view_betas = per_view_betas.reshape(batch_size, num_views, self.config.betas_dim)
            reference_betas = torch.sum(
                per_view_betas * view_weights.unsqueeze(-1),
                dim=1,
            )
            decoder_inputs.append(reference_betas)
        else:
            reference_betas = torch.zeros(
                (batch_size, self.config.betas_dim),
                dtype=pooled_feature.dtype,
                device=pooled_feature.device,
            )

        residual_output = self.residual_decoder(torch.cat(decoder_inputs, dim=-1))
        split_index = self.config.body_pose_dim
        pred_body_pose = reference_body_pose + residual_output[:, :split_index]
        if self.config.learn_betas:
            pred_betas = reference_betas + residual_output[:, split_index:]
        else:
            pred_betas = reference_betas

        return {
            "pred_body_pose": pred_body_pose,
            "pred_betas": pred_betas,
            "fused_feature": pooled_feature,
            "view_weights": view_weights,
            "reference_body_pose": reference_body_pose,
            "reference_betas": reference_betas,
        }
