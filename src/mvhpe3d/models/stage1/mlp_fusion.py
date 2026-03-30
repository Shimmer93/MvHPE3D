"""Stage 1 MLP-based multiview fusion model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..components import MLP, MeanSetPooling


@dataclass(slots=True)
class Stage1MLPFusionConfig:
    """Configuration for the Stage 1 baseline fusion model."""

    input_dim: int = 79
    betas_dim: int = 10
    body_pose_dim: int = 69
    hidden_dim: int = 256
    latent_dim: int = 256
    encoder_layers: int = 2
    decoder_layers: int = 2
    dropout: float = 0.0


class Stage1MLPFusionModel(nn.Module):
    """DeepSets-style multiview fusion baseline for Stage 1."""

    def __init__(self, config: Stage1MLPFusionConfig) -> None:
        super().__init__()
        self.config = config

        output_dim = config.betas_dim + config.body_pose_dim

        self.view_encoder = MLP(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.latent_dim,
            num_layers=config.encoder_layers,
            dropout=config.dropout,
        )
        self.pool = MeanSetPooling()
        self.decoder = MLP(
            input_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            output_dim=output_dim,
            num_layers=config.decoder_layers,
            dropout=config.dropout,
        )

    def forward(self, views_input: torch.Tensor) -> dict[str, torch.Tensor]:
        if views_input.ndim != 3:
            raise ValueError(
                "Stage1MLPFusionModel expects views_input with shape "
                f"[batch, num_views, dim], got {tuple(views_input.shape)}"
            )

        batch_size, num_views, _ = views_input.shape
        encoded_views = self.view_encoder(views_input.reshape(batch_size * num_views, -1))
        encoded_views = encoded_views.reshape(batch_size, num_views, -1)
        fused_feature = self.pool(encoded_views)
        fused_output = self.decoder(fused_feature)

        split_index = self.config.betas_dim
        pred_betas = fused_output[:, :split_index]
        pred_body_pose = fused_output[:, split_index:]

        return {
            "pred_betas": pred_betas,
            "pred_body_pose": pred_body_pose,
            "fused_feature": fused_feature,
        }
