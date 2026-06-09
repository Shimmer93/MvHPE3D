"""Joint-aware Stage 2 multiview fusion in canonical SMPL parameter space."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
    proposal_delta_scale: float = 0.05
    refinement_delta_scale: float = 0.05
    betas_delta_scale: float = 0.05
    predict_external_joint_residual: bool = False
    external_joint_count: int = 17
    external_joint_delta_scale: float = 0.10
    external_joint_head_layers: int = 3
    pose_pca_basis_path: str | None = None
    pose_pca_components: int = 0
    pose_pca_coeff_scale: float = 3.0
    pose_pca_mean_scale: float = 0.0
    pose_pca_head_layers: int = 3
    pose_pca_freeze_base: bool = False
    pose_init_mode: str = "learned"
    zero_init_residual_heads: bool = False
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
        if config.predict_external_joint_residual:
            self.external_joint_residual_head: MLP | None = MLP(
                input_dim=(
                    config.latent_dim
                    + config.pose_6d_dim
                    + config.pose_6d_dim
                    + config.betas_dim
                ),
                hidden_dim=config.hidden_dim,
                output_dim=config.external_joint_count * 3,
                num_layers=config.external_joint_head_layers,
                dropout=config.dropout,
            )
        else:
            self.external_joint_residual_head = None
        if config.pose_pca_basis_path:
            (
                pose_pca_mean,
                pose_pca_components,
                pose_pca_coeff_std,
            ) = _load_pose_pca_basis(
                config.pose_pca_basis_path,
                num_components=config.pose_pca_components,
                pose_dim=config.pose_6d_dim,
            )
            self.register_buffer("pose_pca_mean", pose_pca_mean, persistent=True)
            self.register_buffer("pose_pca_components", pose_pca_components, persistent=True)
            self.register_buffer("pose_pca_coeff_std", pose_pca_coeff_std, persistent=True)
            self.pose_pca_head: MLP | None = MLP(
                input_dim=(
                    config.latent_dim
                    + config.pose_6d_dim
                    + config.pose_6d_dim
                    + config.betas_dim
                ),
                hidden_dim=config.hidden_dim,
                output_dim=pose_pca_components.shape[0],
                num_layers=config.pose_pca_head_layers,
                dropout=config.dropout,
            )
        else:
            self.pose_pca_head = None
        if config.zero_init_residual_heads:
            _zero_last_linear(self.pose_proposal_head)
            _zero_last_linear(self.pose_refinement_head)
            if self.betas_proposal_head is not None:
                _zero_last_linear(self.betas_proposal_head)
            if self.betas_refinement_head is not None:
                _zero_last_linear(self.betas_refinement_head)
            if self.external_joint_residual_head is not None:
                _zero_last_linear(self.external_joint_residual_head)
            if self.pose_pca_head is not None:
                _zero_last_linear(self.pose_pca_head)
        if config.pose_pca_freeze_base:
            if self.pose_pca_head is None:
                raise ValueError("pose_pca_freeze_base requires pose_pca_basis_path")
            _freeze_non_pca_parameters(self)

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
        input_pose = views_input[:, :, : self.config.pose_6d_dim].reshape(
            batch_size,
            num_views,
            self.config.num_joints,
            6,
        )
        input_betas = views_input[:, :, self.config.pose_6d_dim :]
        encoded_views = self.view_encoder(views_input.reshape(batch_size * num_views, -1))
        encoded_views = encoded_views.reshape(batch_size, num_views, -1)

        pose_proposal_delta = self.pose_proposal_head(
            encoded_views.reshape(batch_size * num_views, -1)
        )
        pose_proposal_delta = pose_proposal_delta.reshape(
            batch_size,
            num_views,
            self.config.num_joints,
            6,
        )
        if self.config.pose_init_mode == "learned":
            per_view_pose = pose_proposal_delta
        elif self.config.pose_init_mode == "input_residual":
            per_view_pose = (
                input_pose
                + self.config.proposal_delta_scale * torch.tanh(pose_proposal_delta)
            )
        elif self.config.pose_init_mode == "identity_residual":
            per_view_pose = (
                _identity_pose_6d(
                    batch_size=batch_size,
                    num_views=num_views,
                    num_joints=self.config.num_joints,
                    device=views_input.device,
                    dtype=views_input.dtype,
                )
                + self.config.proposal_delta_scale * torch.tanh(pose_proposal_delta)
            )
        else:
            raise ValueError(
                "pose_init_mode must be one of 'learned', 'input_residual', "
                f"or 'identity_residual', got {self.config.pose_init_mode!r}"
            )

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
        if self.config.pose_init_mode != "learned":
            pose_delta = self.config.refinement_delta_scale * torch.tanh(pose_delta)
        pred_pose_6d = init_pose_6d + pose_delta

        if self.config.learn_betas:
            assert self.betas_proposal_head is not None
            assert self.betas_refinement_head is not None
            betas_proposal_delta = self.betas_proposal_head(
                encoded_views.reshape(batch_size * num_views, -1)
            ).reshape(batch_size, num_views, self.config.betas_dim)
            if self.config.pose_init_mode == "learned":
                per_view_betas = betas_proposal_delta
            else:
                per_view_betas = (
                    input_betas
                    + self.config.betas_delta_scale * torch.tanh(betas_proposal_delta)
                )
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
            if self.config.pose_init_mode != "learned":
                betas_delta = self.config.betas_delta_scale * torch.tanh(betas_delta)
            pred_betas = init_betas + betas_delta
            view_weights = beta_weights
        else:
            init_betas = input_betas.mean(dim=1)
            pred_betas = init_betas
            view_weights = pose_weights.mean(dim=-1)

        stage2_pred_pose_6d = pred_pose_6d
        if self.pose_pca_head is not None:
            pooled_feature = torch.sum(encoded_views * view_weights.unsqueeze(-1), dim=1)
            pose_pca_input = torch.cat(
                (
                    pooled_feature,
                    init_pose_6d.reshape(batch_size, -1),
                    pose_dispersion.reshape(batch_size, -1),
                    init_betas,
                ),
                dim=-1,
            )
            pose_pca_coeff = self.config.pose_pca_coeff_scale * torch.tanh(
                self.pose_pca_head(pose_pca_input)
            )
            pose_pca_scaled_coeff = pose_pca_coeff * self.pose_pca_coeff_std.to(
                device=pose_pca_coeff.device,
                dtype=pose_pca_coeff.dtype,
            )
            pose_pca_delta = torch.matmul(
                pose_pca_scaled_coeff,
                self.pose_pca_components.to(
                    device=pose_pca_coeff.device,
                    dtype=pose_pca_coeff.dtype,
                ),
            )
            pose_pca_delta = pose_pca_delta + (
                self.config.pose_pca_mean_scale
                * self.pose_pca_mean.to(
                    device=pose_pca_coeff.device,
                    dtype=pose_pca_coeff.dtype,
                )
            )
            pose_pca_delta = pose_pca_delta.reshape(
                batch_size,
                self.config.num_joints,
                6,
            )
            pred_pose_6d = pred_pose_6d + pose_pca_delta
        else:
            pose_pca_coeff = None
            pose_pca_delta = None

        outputs = {
            "init_pose_6d": init_pose_6d,
            "init_body_pose": rotation_6d_to_axis_angle(init_pose_6d).reshape(batch_size, -1),
            "init_betas": init_betas,
            "stage2_pred_pose_6d": stage2_pred_pose_6d,
            "pred_pose_6d": pred_pose_6d,
            "pred_body_pose": rotation_6d_to_axis_angle(pred_pose_6d).reshape(batch_size, -1),
            "pred_betas": pred_betas,
            "view_weights": view_weights,
            "pose_view_weights": pose_weights,
        }
        if pose_pca_coeff is not None and pose_pca_delta is not None:
            outputs["pose_pca_coeff"] = pose_pca_coeff
            outputs["pose_pca_delta_6d"] = pose_pca_delta
        if self.external_joint_residual_head is not None:
            pooled_feature = torch.sum(encoded_views * view_weights.unsqueeze(-1), dim=1)
            external_joint_head_input = torch.cat(
                (
                    pooled_feature,
                    init_pose_6d.reshape(batch_size, -1),
                    pose_dispersion.reshape(batch_size, -1),
                    init_betas,
                ),
                dim=-1,
            )
            external_joint_delta = self.external_joint_residual_head(
                external_joint_head_input
            ).reshape(batch_size, self.config.external_joint_count, 3)
            external_joint_delta = (
                self.config.external_joint_delta_scale
                * torch.tanh(external_joint_delta)
            )
            outputs["pred_external_joint_delta"] = external_joint_delta
        return outputs


def _load_pose_pca_basis(
    path: str,
    *,
    num_components: int,
    pose_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Pose PCA basis does not exist: {resolved}")
    data = np.load(resolved)
    required = {"mean", "components", "coeff_std"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise KeyError(f"Pose PCA basis {resolved} is missing keys: {missing}")
    mean = np.asarray(data["mean"], dtype=np.float32).reshape(-1)
    components = np.asarray(data["components"], dtype=np.float32)
    coeff_std = np.asarray(data["coeff_std"], dtype=np.float32).reshape(-1)
    if mean.shape != (pose_dim,):
        raise ValueError(f"Expected PCA mean shape {(pose_dim,)}, got {mean.shape}")
    if components.ndim != 2 or components.shape[1] != pose_dim:
        raise ValueError(
            f"Expected PCA components shape [K, {pose_dim}], got {components.shape}"
        )
    if coeff_std.shape[0] != components.shape[0]:
        raise ValueError(
            "Expected coeff_std length to match number of components, got "
            f"{coeff_std.shape[0]} and {components.shape[0]}"
        )
    if num_components <= 0:
        num_components = int(components.shape[0])
    if num_components > components.shape[0]:
        raise ValueError(
            f"Requested {num_components} PCA components, but basis has "
            f"{components.shape[0]}"
        )
    components = components[:num_components]
    coeff_std = np.maximum(coeff_std[:num_components], 1.0e-6)
    return (
        torch.from_numpy(mean),
        torch.from_numpy(components),
        torch.from_numpy(coeff_std),
    )


def _freeze_non_pca_parameters(module: nn.Module) -> None:
    for name, parameter in module.named_parameters():
        parameter.requires_grad_(name.startswith("pose_pca_head."))


def _zero_last_linear(module: nn.Module) -> None:
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Linear):
            nn.init.zeros_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
            return


def _identity_pose_6d(
    *,
    batch_size: int,
    num_views: int,
    num_joints: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device, dtype=dtype)
    return identity.view(1, 1, 1, 6).expand(batch_size, num_views, num_joints, 6)
