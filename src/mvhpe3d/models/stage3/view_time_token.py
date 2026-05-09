"""View-time token Stage 3 refinement model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from mvhpe3d.utils import rotation_6d_to_axis_angle

from ..components import MLP
from .temporal_refine import _zero_last_linear


@dataclass(slots=True)
class Stage3ViewTimeTokenConfig:
    """Configuration for token-based Stage 3 temporal fusion."""

    name: str = "stage3_view_time_token"
    backbone_name: str = "stage2_joint_graph_refiner"
    learn_betas: bool = False
    num_joints: int = 23
    pose_6d_dim: int = 138
    betas_dim: int = 10
    hidden_dim: int = 512
    token_dim: int = 256
    token_encoder_layers: int = 2
    transformer_layers: int = 3
    transformer_heads: int = 4
    transformer_ff_dim: int = 1024
    head_layers: int = 3
    dropout: float = 0.1
    pose_residual_scale: float = 0.05
    betas_residual_scale: float = 0.05
    max_camera_index: int = 256
    max_time_offset: int = 16
    freeze_backbone: bool = True

    @property
    def token_input_dim(self) -> int:
        return self.pose_6d_dim + self.betas_dim


class Stage3ViewTimeTokenModel(nn.Module):
    """Refine a target-frame Stage 2 prediction from sparse view-time tokens."""

    def __init__(self, config: Stage3ViewTimeTokenConfig) -> None:
        super().__init__()
        self.config = config
        if config.token_dim % config.transformer_heads != 0:
            raise ValueError(
                "token_dim must be divisible by transformer_heads, "
                f"got token_dim={config.token_dim}, transformer_heads={config.transformer_heads}"
            )
        self.token_encoder = MLP(
            input_dim=config.token_input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.token_dim,
            num_layers=config.token_encoder_layers,
            dropout=config.dropout,
        )
        self.time_encoder = MLP(
            input_dim=1,
            hidden_dim=config.token_dim,
            output_dim=config.token_dim,
            num_layers=2,
            dropout=0.0,
        )
        self.camera_embedding = nn.Embedding(config.max_camera_index, config.token_dim)
        self.target_query = nn.Parameter(torch.zeros(1, 1, config.token_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.token_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_layers,
        )
        self.pose_head = MLP(
            input_dim=config.token_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.pose_6d_dim,
            num_layers=config.head_layers,
            dropout=config.dropout,
        )
        _zero_last_linear(self.pose_head)
        if config.learn_betas:
            self.betas_head: MLP | None = MLP(
                input_dim=config.token_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.betas_dim,
                num_layers=config.head_layers,
                dropout=config.dropout,
            )
            _zero_last_linear(self.betas_head)
        else:
            self.betas_head = None

    def forward(
        self,
        *,
        view_time_tokens: torch.Tensor,
        token_time_offsets: torch.Tensor,
        token_camera_indices: torch.Tensor,
        token_valid_mask: torch.Tensor,
        base_pose_6d: torch.Tensor,
        base_betas: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if view_time_tokens.ndim != 3:
            raise ValueError(
                "Stage3ViewTimeTokenModel expects view_time_tokens with shape "
                f"[batch, tokens, dim], got {tuple(view_time_tokens.shape)}"
            )
        if view_time_tokens.shape[-1] != self.config.token_input_dim:
            raise ValueError(
                f"Expected token dim {self.config.token_input_dim}, "
                f"got {view_time_tokens.shape[-1]}"
            )
        batch_size, num_tokens, _ = view_time_tokens.shape
        base_pose_6d_flat = base_pose_6d.reshape(batch_size, -1)
        if base_pose_6d_flat.shape[-1] != self.config.pose_6d_dim:
            raise ValueError(
                f"Expected base_pose_6d to flatten to {self.config.pose_6d_dim}, "
                f"got {tuple(base_pose_6d.shape)}"
            )
        base_betas_flat = base_betas.reshape(batch_size, -1)
        if base_betas_flat.shape[-1] != self.config.betas_dim:
            raise ValueError(
                f"Expected base_betas to flatten to {self.config.betas_dim}, "
                f"got {tuple(base_betas.shape)}"
            )

        valid_mask = token_valid_mask.to(device=view_time_tokens.device, dtype=torch.bool)
        if valid_mask.shape != (batch_size, num_tokens):
            raise ValueError(
                "Expected token_valid_mask shape "
                f"{(batch_size, num_tokens)}, got {tuple(valid_mask.shape)}"
            )
        time_offsets = token_time_offsets.to(device=view_time_tokens.device, dtype=view_time_tokens.dtype)
        camera_indices = token_camera_indices.to(device=view_time_tokens.device, dtype=torch.long)
        camera_indices = torch.remainder(camera_indices, self.config.max_camera_index)

        token_features = self.token_encoder(view_time_tokens)
        normalized_offsets = time_offsets.unsqueeze(-1) / max(float(self.config.max_time_offset), 1.0)
        token_features = token_features + self.time_encoder(normalized_offsets)
        token_features = token_features + self.camera_embedding(camera_indices)
        token_features = token_features * valid_mask.unsqueeze(-1).to(token_features.dtype)

        target_query = self.target_query.expand(batch_size, -1, -1)
        transformer_input = torch.cat((target_query, token_features), dim=1)
        padding_mask = torch.cat(
            (
                torch.zeros((batch_size, 1), dtype=torch.bool, device=view_time_tokens.device),
                ~valid_mask,
            ),
            dim=1,
        )
        transformer_output = self.transformer(
            transformer_input,
            src_key_padding_mask=padding_mask,
        )
        target_feature = transformer_output[:, 0]

        pose_delta = self.pose_head(target_feature)
        pose_residual_6d = self.config.pose_residual_scale * torch.tanh(pose_delta)
        pred_pose_6d = base_pose_6d_flat + pose_residual_6d

        if self.config.learn_betas:
            assert self.betas_head is not None
            betas_delta = self.betas_head(target_feature)
            betas_residual = self.config.betas_residual_scale * torch.tanh(betas_delta)
            pred_betas = base_betas_flat + betas_residual
        else:
            betas_residual = torch.zeros_like(base_betas_flat)
            pred_betas = base_betas_flat

        pred_pose_6d_reshaped = pred_pose_6d.reshape(batch_size, self.config.num_joints, 6)
        base_pose_6d_reshaped = base_pose_6d_flat.reshape(batch_size, self.config.num_joints, 6)
        return {
            "base_pose_6d": base_pose_6d_reshaped,
            "base_body_pose": rotation_6d_to_axis_angle(base_pose_6d_reshaped).reshape(batch_size, -1),
            "base_betas": base_betas_flat,
            "pose_residual_6d": pose_residual_6d.reshape(batch_size, self.config.num_joints, 6),
            "betas_residual": betas_residual,
            "pred_pose_6d": pred_pose_6d_reshaped,
            "pred_body_pose": rotation_6d_to_axis_angle(pred_pose_6d_reshaped).reshape(batch_size, -1),
            "pred_betas": pred_betas,
            "target_frame_index": torch.zeros(
                (batch_size,),
                dtype=torch.long,
                device=view_time_tokens.device,
            ),
            "valid_token_count": valid_mask.sum(dim=1),
        }
