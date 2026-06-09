"""Stage 2.2 gated body-pose correction adapter."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from mvhpe3d.utils import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    rotation_6d_to_axis_angle,
)
from mvhpe3d.utils.rotation import matrix_to_rotation_6d, rotation_6d_to_matrix

from ..components import MLP


@dataclass(slots=True)
class Stage22GatedBodyAdapterConfig:
    """Configuration for a conservative frozen-Stage2 body adapter."""

    name: str = "stage2_2_gated_body_adapter"
    view_input_dim: int = 148
    num_body_joints: int = 23
    betas_dim: int = 10
    image_joint_count: int = 17
    image_feature_dim: int = 250
    mask_feature_dim: int = 0
    evidence_token_dim: int = 32
    evidence_hidden_dim: int = 128
    evidence_layers: int = 2
    use_local_joint_update_head: bool = False
    local_joint_hidden_dim: int = 256
    local_joint_layers: int = 3
    local_joint_embedding_dim: int = 16
    hidden_dim: int = 512
    num_layers: int = 3
    dropout: float = 0.1
    body_delta_scale: float = 0.05
    betas_delta_scale: float = 0.05
    gate_bias: float = -4.0
    use_stage2_body_pose: bool = True
    use_input_pose_mean: bool = True
    use_input_pose_dispersion: bool = True
    use_input_betas: bool = True
    use_image_residual: bool = True
    use_image_confidence: bool = True
    use_image_valid: bool = True
    use_image_size: bool = True
    use_image_joint_feature: bool = False
    use_image_mask_feature: bool = False
    use_evidence_gate: bool = False
    use_evidence_gate_only: bool = False
    use_evidence_weighted_pose_fusion: bool = False
    evidence_weighted_pose_project_so3: bool = False
    use_evidence_weighted_betas_fusion: bool = False
    weighted_pose_joint_policy: str = "all"
    evidence_gate_bias: float = 2.0
    predict_betas_update: bool = False
    extra_candidate_count: int = 0
    extra_hidden_dim: int = 512
    extra_num_layers: int = 3
    extra_selector_bias: float = 2.0
    extra_use_local_evidence: bool = True
    extra_update_joint_policy: str = "all"
    normalize_image_joint_feature: bool = True
    normalize_image_mask_feature: bool = True
    zero_init: bool = True

    @property
    def body_pose_dim(self) -> int:
        return self.num_body_joints * 3

    @property
    def pose_6d_dim(self) -> int:
        return self.num_body_joints * 6

    @property
    def input_dim(self) -> int:
        dim = 0
        if self.use_stage2_body_pose:
            dim += self.body_pose_dim  # frozen Stage2 pose
        if self.use_input_pose_mean:
            dim += self.body_pose_dim  # mean input pose
        if self.use_input_pose_dispersion:
            dim += self.body_pose_dim  # input pose dispersion
        if self.use_input_betas:
            dim += self.betas_dim  # mean input betas
        if self.use_evidence_weighted_pose_fusion:
            dim += self.body_pose_dim  # evidence-weighted per-view pose hypothesis
        if self.use_evidence_weighted_betas_fusion:
            dim += self.betas_dim  # evidence-weighted per-view betas hypothesis
        if self.use_image_residual:
            dim += self.image_joint_count * 2
        if self.use_image_confidence:
            dim += self.image_joint_count
        if (
            (self.use_image_joint_feature or self.use_image_mask_feature)
            and not self.use_evidence_gate_only
        ):
            dim += self.num_body_joints * self.evidence_token_dim
        return dim

    @property
    def local_joint_input_dim(self) -> int:
        dim = 0
        if self.use_stage2_body_pose:
            dim += 3
        if self.use_input_pose_mean:
            dim += 3
        if self.use_input_pose_dispersion:
            dim += 3
        if self.use_input_betas:
            dim += self.betas_dim
        if self.use_evidence_weighted_pose_fusion:
            dim += 3
        if self.use_evidence_weighted_betas_fusion:
            dim += self.betas_dim
        if (
            (self.use_image_joint_feature or self.use_image_mask_feature)
            and not self.use_evidence_gate_only
        ):
            dim += self.evidence_token_dim
        dim += self.local_joint_embedding_dim
        return dim

    @property
    def extra_input_dim(self) -> int:
        dim = self.input_dim
        if (
            self.extra_use_local_evidence
            and (self.use_image_joint_feature or self.use_image_mask_feature)
            and self.use_evidence_gate_only
        ):
            dim += self.num_body_joints * self.evidence_token_dim
        return dim


class Stage22GatedBodyAdapterModel(nn.Module):
    """Predict small gated per-joint body-pose updates."""

    def __init__(self, config: Stage22GatedBodyAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.append_local_evidence_to_features = (
            (config.use_image_joint_feature or config.use_image_mask_feature)
            and not config.use_evidence_gate_only
        )
        self.use_view_weighted_fusion = (
            config.use_evidence_weighted_pose_fusion
            or config.use_evidence_weighted_betas_fusion
        )
        self.use_local_evidence = (
            config.use_image_joint_feature
            or config.use_image_mask_feature
            or config.use_evidence_gate
            or self.use_view_weighted_fusion
        )
        if self.use_local_evidence and config.evidence_token_dim <= 0:
            raise ValueError("evidence_token_dim must be positive when local evidence is enabled")
        evidence_input_dim = 4
        if config.use_image_joint_feature:
            evidence_input_dim += config.image_feature_dim
        if config.use_image_mask_feature:
            evidence_input_dim += config.mask_feature_dim
        self.image_feature_norm = (
            nn.LayerNorm(config.image_feature_dim)
            if config.use_image_joint_feature and config.normalize_image_joint_feature
            else nn.Identity()
        )
        self.mask_feature_norm = (
            nn.LayerNorm(config.mask_feature_dim)
            if config.use_image_mask_feature
            and config.mask_feature_dim > 0
            and config.normalize_image_mask_feature
            else nn.Identity()
        )
        self.evidence_encoder = (
            MLP(
                input_dim=evidence_input_dim,
                hidden_dim=config.evidence_hidden_dim,
                output_dim=config.evidence_token_dim,
                num_layers=config.evidence_layers,
                dropout=config.dropout,
            )
            if self.use_local_evidence
            else None
        )
        self.evidence_gate = (
            nn.Linear(config.evidence_token_dim, 1)
            if config.use_evidence_gate
            else None
        )
        self.view_reliability_head = (
            nn.Linear(config.evidence_token_dim, 1)
            if self.use_view_weighted_fusion
            else None
        )
        self.register_buffer(
            "body_to_image_joint",
            torch.tensor(
                _build_body_to_image_joint_index(
                    config.num_body_joints,
                    config.image_joint_count,
                ),
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.body_joint_embedding = (
            nn.Parameter(torch.empty(config.num_body_joints, config.local_joint_embedding_dim))
            if config.use_local_joint_update_head and config.local_joint_embedding_dim > 0
            else None
        )
        self.local_joint_network = (
            MLP(
                input_dim=config.local_joint_input_dim,
                hidden_dim=config.local_joint_hidden_dim,
                output_dim=4,
                num_layers=config.local_joint_layers,
                dropout=config.dropout,
            )
            if config.use_local_joint_update_head
            else None
        )
        self.network = (
            MLP(
                input_dim=config.input_dim,
                hidden_dim=config.hidden_dim,
                output_dim=(
                    config.num_body_joints * 4
                    + (config.betas_dim if config.predict_betas_update else 0)
                ),
                num_layers=config.num_layers,
                dropout=config.dropout,
            )
            if not config.use_local_joint_update_head
            else None
        )
        if config.extra_candidate_count > 0 and config.use_local_joint_update_head:
            raise ValueError("extra_candidate_count is only supported for the global body head")
        self.extra_network = (
            MLP(
                input_dim=config.extra_input_dim,
                hidden_dim=config.extra_hidden_dim,
                output_dim=config.extra_candidate_count * (config.num_body_joints * 4 + 1),
                num_layers=config.extra_num_layers,
                dropout=config.dropout,
            )
            if config.extra_candidate_count > 0
            else None
        )
        if self.body_joint_embedding is not None:
            nn.init.normal_(self.body_joint_embedding, mean=0.0, std=0.02)
        if self.evidence_gate is not None:
            nn.init.zeros_(self.evidence_gate.weight)
            nn.init.constant_(self.evidence_gate.bias, float(config.evidence_gate_bias))
        if self.view_reliability_head is not None:
            nn.init.zeros_(self.view_reliability_head.weight)
            nn.init.zeros_(self.view_reliability_head.bias)
        if config.zero_init:
            if self.network is not None:
                _zero_last_linear(self.network)
            if self.local_joint_network is not None:
                _zero_last_linear(self.local_joint_network)
            if self.extra_network is not None:
                _zero_last_linear(self.extra_network)

    def forward(
        self,
        *,
        views_input: torch.Tensor,
        stage2_body_pose: torch.Tensor,
        stage2_uv: torch.Tensor | None = None,
        measured_uv: torch.Tensor | None = None,
        measured_valid: torch.Tensor | None = None,
        measured_confidence: torch.Tensor | None = None,
        image_size: torch.Tensor | None = None,
        image_joint_feature: torch.Tensor | None = None,
        image_mask_feature: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if views_input.ndim != 3:
            raise ValueError(
                "views_input must have shape [B, V, D], "
                f"got {tuple(views_input.shape)}"
            )
        if stage2_body_pose.shape[-1] != self.config.body_pose_dim:
            raise ValueError(
                f"Expected stage2 body pose dim {self.config.body_pose_dim}, "
                f"got {stage2_body_pose.shape[-1]}"
            )
        if self.config.use_local_joint_update_head:
            raw = self._forward_local_joint_update(
                views_input=views_input,
                stage2_body_pose=stage2_body_pose,
                stage2_uv=stage2_uv,
                measured_uv=measured_uv,
                measured_valid=measured_valid,
                measured_confidence=measured_confidence,
                image_size=image_size,
                image_joint_feature=image_joint_feature,
                image_mask_feature=image_mask_feature,
            )
        else:
            if self.network is None:
                raise RuntimeError("Global body adapter network was not initialized")
            features = self._build_features(
                views_input=views_input,
                stage2_body_pose=stage2_body_pose,
                stage2_uv=stage2_uv,
                measured_uv=measured_uv,
                measured_valid=measured_valid,
                measured_confidence=measured_confidence,
                image_size=image_size,
                image_joint_feature=image_joint_feature,
                image_mask_feature=image_mask_feature,
            )
            raw_flat = self.network(features)
            pose_dim = self.config.num_body_joints * 4
            raw = raw_flat[..., :pose_dim].reshape(raw_flat.shape[0], self.config.num_body_joints, 4)
            raw_betas = raw_flat[..., pose_dim:]
        batch_size = raw.shape[0]
        delta = float(self.config.body_delta_scale) * torch.tanh(raw[..., :3])
        gate = torch.sigmoid(raw[..., 3] + float(self.config.gate_bias))
        evidence_gate = torch.ones_like(gate)
        if self.evidence_gate is not None:
            evidence_token = self._build_local_evidence_tokens(
                views_input=views_input,
                stage2_uv=stage2_uv,
                measured_uv=measured_uv,
                measured_valid=measured_valid,
                measured_confidence=measured_confidence,
                image_size=image_size,
                image_joint_feature=image_joint_feature,
                image_mask_feature=image_mask_feature,
            )
            evidence_gate = torch.sigmoid(self.evidence_gate(evidence_token).squeeze(-1))
            gate = gate * evidence_gate
        base_update = delta * gate[..., None]
        extra_selector = torch.zeros(
            (batch_size, self.config.extra_candidate_count),
            device=views_input.device,
            dtype=views_input.dtype,
        )
        extra_update_total = torch.zeros_like(base_update)
        if self.extra_network is not None:
            extra_features = self._build_extra_features(
                views_input=views_input,
                stage2_body_pose=stage2_body_pose,
                stage2_uv=stage2_uv,
                measured_uv=measured_uv,
                measured_valid=measured_valid,
                measured_confidence=measured_confidence,
                image_size=image_size,
                image_joint_feature=image_joint_feature,
                image_mask_feature=image_mask_feature,
            )
            raw_extra = self.extra_network(extra_features)
            candidate_count = self.config.extra_candidate_count
            pose_width = candidate_count * self.config.num_body_joints * 4
            raw_extra_pose = raw_extra[..., :pose_width].reshape(
                batch_size,
                candidate_count,
                self.config.num_body_joints,
                4,
            )
            raw_extra_selector = raw_extra[..., pose_width:].reshape(batch_size, candidate_count)
            extra_delta = float(self.config.body_delta_scale) * torch.tanh(raw_extra_pose[..., :3])
            extra_gate = torch.sigmoid(raw_extra_pose[..., 3] + float(self.config.gate_bias))
            extra_update = extra_delta * extra_gate[..., None]
            extra_update = self._apply_extra_update_joint_policy(extra_update)
            extra_selector = torch.sigmoid(
                raw_extra_selector + float(self.config.extra_selector_bias)
            )
            extra_update_total = (extra_update * extra_selector[..., None, None]).sum(dim=1)
        update = base_update + extra_update_total
        if self.config.predict_betas_update and not self.config.use_local_joint_update_head:
            betas_update = float(self.config.betas_delta_scale) * torch.tanh(raw_betas)
        else:
            betas_update = torch.zeros(
                (batch_size, self.config.betas_dim),
                device=views_input.device,
                dtype=views_input.dtype,
            )
        return {
            "pred_body_pose_delta": delta.reshape(batch_size, -1),
            "pred_body_pose_gate": gate,
            "pred_body_pose_evidence_gate": evidence_gate,
            "pred_body_pose_base_update": base_update.reshape(batch_size, -1),
            "pred_body_pose_extra_selector": extra_selector,
            "pred_body_pose_extra_update": extra_update_total.reshape(batch_size, -1),
            "pred_body_pose_update": update.reshape(batch_size, -1),
            "pred_betas_update": betas_update,
        }

    def _build_features(
        self,
        *,
        views_input: torch.Tensor,
        stage2_body_pose: torch.Tensor,
        stage2_uv: torch.Tensor | None,
        measured_uv: torch.Tensor | None,
        measured_valid: torch.Tensor | None,
        measured_confidence: torch.Tensor | None,
        image_size: torch.Tensor | None,
        image_joint_feature: torch.Tensor | None,
        image_mask_feature: torch.Tensor | None,
    ) -> torch.Tensor:
        input_body_pose, input_betas = self._decode_views_input(views_input)
        input_pose_mean = input_body_pose.mean(dim=1)
        input_pose_dispersion = (input_body_pose - input_pose_mean[:, None]).abs().mean(dim=1)
        input_betas_mean = input_betas.mean(dim=1)
        weighted_pose = weighted_betas = None
        if (
            self.config.use_evidence_weighted_pose_fusion
            or self.config.use_evidence_weighted_betas_fusion
        ):
            weighted_pose, weighted_betas, _, _ = self._build_evidence_weighted_input_features(
                views_input=views_input,
                stage2_uv=stage2_uv,
                measured_uv=measured_uv,
                measured_valid=measured_valid,
                measured_confidence=measured_confidence,
                image_size=image_size,
                image_joint_feature=image_joint_feature,
                image_mask_feature=image_mask_feature,
            )
        features = []
        if self.config.use_stage2_body_pose:
            features.append(stage2_body_pose)
        if self.config.use_input_pose_mean:
            features.append(input_pose_mean)
        if self.config.use_input_pose_dispersion:
            features.append(input_pose_dispersion)
        if self.config.use_input_betas:
            features.append(input_betas_mean)
        if self.config.use_evidence_weighted_pose_fusion:
            if weighted_pose is None:
                raise RuntimeError("Evidence-weighted pose features were not built")
            features.append(weighted_pose)
        if self.config.use_evidence_weighted_betas_fusion:
            if weighted_betas is None:
                raise RuntimeError("Evidence-weighted betas features were not built")
            features.append(weighted_betas)
        if self.config.use_image_residual or self.config.use_image_confidence:
            residual, confidence = self._build_image_features(
                views_input=views_input,
                stage2_uv=stage2_uv,
                measured_uv=measured_uv,
                measured_valid=measured_valid,
                measured_confidence=measured_confidence,
                image_size=image_size,
            )
            if self.config.use_image_residual:
                features.append(residual)
            if self.config.use_image_confidence:
                features.append(confidence)
        if self.append_local_evidence_to_features:
            features.append(
                self._build_local_evidence_features(
                    views_input=views_input,
                    stage2_uv=stage2_uv,
                    measured_uv=measured_uv,
                    measured_valid=measured_valid,
                    measured_confidence=measured_confidence,
                    image_size=image_size,
                    image_joint_feature=image_joint_feature,
                    image_mask_feature=image_mask_feature,
                )
            )
        if not features:
            raise ValueError("At least one Stage 2.2 adapter input feature must be enabled")
        result = torch.cat(features, dim=-1)
        if result.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"Expected adapter feature dim {self.config.input_dim}, "
                f"got {result.shape[-1]}"
            )
        return result

    def _build_extra_features(
        self,
        *,
        views_input: torch.Tensor,
        stage2_body_pose: torch.Tensor,
        stage2_uv: torch.Tensor | None,
        measured_uv: torch.Tensor | None,
        measured_valid: torch.Tensor | None,
        measured_confidence: torch.Tensor | None,
        image_size: torch.Tensor | None,
        image_joint_feature: torch.Tensor | None,
        image_mask_feature: torch.Tensor | None,
    ) -> torch.Tensor:
        base = self._build_features(
            views_input=views_input,
            stage2_body_pose=stage2_body_pose,
            stage2_uv=stage2_uv,
            measured_uv=measured_uv,
            measured_valid=measured_valid,
            measured_confidence=measured_confidence,
            image_size=image_size,
            image_joint_feature=image_joint_feature,
            image_mask_feature=image_mask_feature,
        )
        parts = [base]
        if (
            self.config.extra_use_local_evidence
            and (self.config.use_image_joint_feature or self.config.use_image_mask_feature)
            and self.config.use_evidence_gate_only
        ):
            parts.append(
                self._build_local_evidence_features(
                    views_input=views_input,
                    stage2_uv=stage2_uv,
                    measured_uv=measured_uv,
                    measured_valid=measured_valid,
                    measured_confidence=measured_confidence,
                    image_size=image_size,
                    image_joint_feature=image_joint_feature,
                    image_mask_feature=image_mask_feature,
                )
            )
        result = torch.cat(parts, dim=-1)
        if result.shape[-1] != self.config.extra_input_dim:
            raise ValueError(
                f"Expected extra body feature dim {self.config.extra_input_dim}, "
                f"got {result.shape[-1]}"
            )
        return result

    def _forward_local_joint_update(
        self,
        *,
        views_input: torch.Tensor,
        stage2_body_pose: torch.Tensor,
        stage2_uv: torch.Tensor | None,
        measured_uv: torch.Tensor | None,
        measured_valid: torch.Tensor | None,
        measured_confidence: torch.Tensor | None,
        image_size: torch.Tensor | None,
        image_joint_feature: torch.Tensor | None,
        image_mask_feature: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.local_joint_network is None:
            raise RuntimeError("Local joint update network was not initialized")
        input_body_pose, input_betas = self._decode_views_input(views_input)
        batch_size = views_input.shape[0]
        joint_count = self.config.num_body_joints
        stage2_joints = stage2_body_pose.reshape(batch_size, joint_count, 3)
        input_pose_mean = input_body_pose.mean(dim=1).reshape(batch_size, joint_count, 3)
        input_pose_dispersion = (input_body_pose - input_body_pose.mean(dim=1)[:, None]).abs()
        input_pose_dispersion = input_pose_dispersion.mean(dim=1).reshape(batch_size, joint_count, 3)
        input_betas_mean = input_betas.mean(dim=1)
        weighted_pose_joints = weighted_joint_betas = None
        if (
            self.config.use_evidence_weighted_pose_fusion
            or self.config.use_evidence_weighted_betas_fusion
        ):
            _, _, weighted_pose_joints, weighted_joint_betas = (
                self._build_evidence_weighted_input_features(
                    views_input=views_input,
                    stage2_uv=stage2_uv,
                    measured_uv=measured_uv,
                    measured_valid=measured_valid,
                    measured_confidence=measured_confidence,
                    image_size=image_size,
                    image_joint_feature=image_joint_feature,
                    image_mask_feature=image_mask_feature,
                )
            )
        parts = []
        if self.config.use_stage2_body_pose:
            parts.append(stage2_joints)
        if self.config.use_input_pose_mean:
            parts.append(input_pose_mean)
        if self.config.use_input_pose_dispersion:
            parts.append(input_pose_dispersion)
        if self.config.use_input_betas:
            parts.append(input_betas_mean[:, None, :].expand(-1, joint_count, -1))
        if self.config.use_evidence_weighted_pose_fusion:
            if weighted_pose_joints is None:
                raise RuntimeError("Evidence-weighted pose features were not built")
            parts.append(weighted_pose_joints)
        if self.config.use_evidence_weighted_betas_fusion:
            if weighted_joint_betas is None:
                raise RuntimeError("Evidence-weighted betas features were not built")
            parts.append(weighted_joint_betas)
        if self.append_local_evidence_to_features:
            parts.append(
                self._build_local_evidence_tokens(
                    views_input=views_input,
                    stage2_uv=stage2_uv,
                    measured_uv=measured_uv,
                    measured_valid=measured_valid,
                    measured_confidence=measured_confidence,
                    image_size=image_size,
                    image_joint_feature=image_joint_feature,
                    image_mask_feature=image_mask_feature,
                )
            )
        if self.body_joint_embedding is not None:
            parts.append(self.body_joint_embedding[None].expand(batch_size, -1, -1))
        local_input = torch.cat(parts, dim=-1)
        if local_input.shape[-1] != self.config.local_joint_input_dim:
            raise ValueError(
                f"Expected local joint input dim {self.config.local_joint_input_dim}, "
                f"got {local_input.shape[-1]}"
            )
        raw = self.local_joint_network(local_input.reshape(batch_size * joint_count, -1))
        return raw.reshape(batch_size, joint_count, 4)

    def _decode_views_input(
        self,
        views_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if views_input.shape[-1] < self.config.pose_6d_dim + self.config.betas_dim:
            raise ValueError(
                f"views_input dim {views_input.shape[-1]} is too small for "
                f"pose_6d_dim={self.config.pose_6d_dim} and betas_dim={self.config.betas_dim}"
            )
        batch_size, num_views = views_input.shape[:2]
        input_pose_6d = views_input[..., : self.config.pose_6d_dim].reshape(
            batch_size,
            num_views,
            self.config.num_body_joints,
            6,
        )
        input_body_pose = rotation_6d_to_axis_angle(input_pose_6d).reshape(
            batch_size,
            num_views,
            self.config.body_pose_dim,
        )
        input_betas = views_input[
            ...,
            self.config.pose_6d_dim : self.config.pose_6d_dim + self.config.betas_dim,
        ]
        return input_body_pose, input_betas

    def _build_image_features(
        self,
        *,
        views_input: torch.Tensor,
        stage2_uv: torch.Tensor | None,
        measured_uv: torch.Tensor | None,
        measured_valid: torch.Tensor | None,
        measured_confidence: torch.Tensor | None,
        image_size: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_views = views_input.shape[:2]
        residual, confidence, _ = self._build_per_view_measurement_features(
            views_input=views_input,
            stage2_uv=stage2_uv,
            measured_uv=measured_uv,
            measured_valid=measured_valid,
            measured_confidence=measured_confidence,
            image_size=image_size,
        )
        weight = confidence.clamp_min(0.0)
        denom = weight.sum(dim=1).clamp_min(1.0)
        residual_mean = (residual * weight[..., None]).sum(dim=1) / denom[..., None]
        confidence_mean = confidence.mean(dim=1)
        return residual_mean.reshape(batch_size, -1), confidence_mean

    def _build_per_view_measurement_features(
        self,
        *,
        views_input: torch.Tensor,
        stage2_uv: torch.Tensor | None,
        measured_uv: torch.Tensor | None,
        measured_valid: torch.Tensor | None,
        measured_confidence: torch.Tensor | None,
        image_size: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_views = views_input.shape[:2]
        shape = (batch_size, num_views, self.config.image_joint_count)
        if stage2_uv is None or measured_uv is None:
            residual = torch.zeros(
                (*shape, 2),
                device=views_input.device,
                dtype=views_input.dtype,
            )
            confidence = torch.zeros(shape, device=views_input.device, dtype=views_input.dtype)
            valid = torch.zeros(shape, device=views_input.device, dtype=torch.bool)
            return residual, confidence, valid
        stage2_uv = stage2_uv.to(device=views_input.device, dtype=views_input.dtype)
        measured_uv = measured_uv.to(device=views_input.device, dtype=views_input.dtype)
        if self.config.use_image_size and image_size is not None:
            image_size = image_size.to(device=views_input.device, dtype=views_input.dtype)
        else:
            image_size = torch.full(
                (batch_size, num_views, 2),
                1000.0,
                device=views_input.device,
                dtype=views_input.dtype,
            )
        if stage2_uv.shape[:3] != shape or measured_uv.shape[:3] != shape:
            raise ValueError(
                "stage2_uv and measured_uv must have shape [B, V, J, 2] with "
                f"J={self.config.image_joint_count}; got {tuple(stage2_uv.shape)} and "
                f"{tuple(measured_uv.shape)}"
            )
        if measured_valid is None or not self.config.use_image_valid:
            valid = torch.isfinite(measured_uv).all(dim=-1)
        else:
            valid = measured_valid.to(device=views_input.device, dtype=torch.bool)
        if measured_confidence is None:
            confidence = valid.to(dtype=views_input.dtype)
        else:
            confidence = measured_confidence.to(device=views_input.device, dtype=views_input.dtype)
            confidence = confidence * valid.to(dtype=confidence.dtype)
        normalizer = image_size.clamp_min(1.0)[..., None, :]
        residual = (measured_uv - stage2_uv) / normalizer
        valid = valid & torch.isfinite(residual).all(dim=-1)
        residual = torch.where(valid[..., None], residual, torch.zeros_like(residual))
        confidence = torch.where(valid, confidence, torch.zeros_like(confidence))
        return residual, confidence, valid

    def _build_evidence_weighted_input_features(
        self,
        *,
        views_input: torch.Tensor,
        stage2_uv: torch.Tensor | None,
        measured_uv: torch.Tensor | None,
        measured_valid: torch.Tensor | None,
        measured_confidence: torch.Tensor | None,
        image_size: torch.Tensor | None,
        image_joint_feature: torch.Tensor | None,
        image_mask_feature: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_body_pose, input_betas = self._decode_views_input(views_input)
        batch_size, num_views = views_input.shape[:2]
        joint_count = self.config.num_body_joints
        input_pose_joints = input_body_pose.reshape(batch_size, num_views, joint_count, 3)
        input_pose_mean_joints = input_pose_joints.mean(dim=1)
        view_joint_weight = self._build_view_reliability_weights(
            views_input=views_input,
            stage2_uv=stage2_uv,
            measured_uv=measured_uv,
            measured_valid=measured_valid,
            measured_confidence=measured_confidence,
            image_size=image_size,
            image_joint_feature=image_joint_feature,
            image_mask_feature=image_mask_feature,
        )
        if self.config.evidence_weighted_pose_project_so3:
            weighted_pose_joints = self._build_so3_weighted_pose_joints(
                input_pose_joints=input_pose_joints,
                view_joint_weight=view_joint_weight,
            )
        else:
            weighted_pose_joints = (input_pose_joints * view_joint_weight[..., None]).sum(dim=1)
        weighted_pose_joints = self._apply_weighted_pose_joint_policy(
            weighted_pose_joints=weighted_pose_joints,
            fallback_pose_joints=input_pose_mean_joints,
        )
        view_weight = view_joint_weight.mean(dim=-1)
        weighted_betas = (input_betas * view_weight[..., None]).sum(dim=1)
        weighted_joint_betas = (input_betas[:, :, None, :] * view_joint_weight[..., None]).sum(
            dim=1
        )
        return (
            weighted_pose_joints.reshape(batch_size, -1),
            weighted_betas,
            weighted_pose_joints,
            weighted_joint_betas,
        )

    def _build_so3_weighted_pose_joints(
        self,
        *,
        input_pose_joints: torch.Tensor,
        view_joint_weight: torch.Tensor,
    ) -> torch.Tensor:
        rotations = axis_angle_to_matrix(input_pose_joints)
        weighted_matrix = (rotations * view_joint_weight[..., None, None]).sum(dim=1)
        # SVD projection gives the closest rotation but has unstable gradients when
        # the averaged matrix has repeated singular values. The 6D projection uses
        # the same Gram-Schmidt map as the model's rotation heads and is stable in training.
        rotation_6d = matrix_to_rotation_6d(weighted_matrix)
        rotation = rotation_6d_to_matrix(rotation_6d)
        return matrix_to_axis_angle(rotation)

    def _apply_weighted_pose_joint_policy(
        self,
        *,
        weighted_pose_joints: torch.Tensor,
        fallback_pose_joints: torch.Tensor,
    ) -> torch.Tensor:
        policy = str(self.config.weighted_pose_joint_policy)
        if policy == "all":
            return weighted_pose_joints
        if policy not in {"lower_body", "distal_legs", "ankle_feet"}:
            raise ValueError(f"Unknown weighted_pose_joint_policy: {policy}")
        joint_count = int(weighted_pose_joints.shape[1])
        mask = _body_joint_policy_mask(
            policy=policy,
            joint_count=joint_count,
            device=weighted_pose_joints.device,
        )
        return torch.where(mask[None, :, None], weighted_pose_joints, fallback_pose_joints)

    def _apply_extra_update_joint_policy(self, extra_update: torch.Tensor) -> torch.Tensor:
        policy = str(self.config.extra_update_joint_policy)
        if policy == "all":
            return extra_update
        if policy not in {"lower_body", "distal_legs", "ankle_feet"}:
            raise ValueError(f"Unknown extra_update_joint_policy: {policy}")
        joint_count = int(extra_update.shape[-2])
        mask = _body_joint_policy_mask(
            policy=policy,
            joint_count=joint_count,
            device=extra_update.device,
        )
        return extra_update * mask[None, None, :, None].to(dtype=extra_update.dtype)

    def _build_local_evidence_features(
        self,
        *,
        views_input: torch.Tensor,
        stage2_uv: torch.Tensor | None,
        measured_uv: torch.Tensor | None,
        measured_valid: torch.Tensor | None,
        measured_confidence: torch.Tensor | None,
        image_size: torch.Tensor | None,
        image_joint_feature: torch.Tensor | None,
        image_mask_feature: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.evidence_encoder is None:
            raise RuntimeError("Local evidence encoder was not initialized")
        body_joint_token = self._build_local_evidence_tokens(
            views_input=views_input,
            stage2_uv=stage2_uv,
            measured_uv=measured_uv,
            measured_valid=measured_valid,
            measured_confidence=measured_confidence,
            image_size=image_size,
            image_joint_feature=image_joint_feature,
            image_mask_feature=image_mask_feature,
        )
        return body_joint_token.reshape(views_input.shape[0], -1)

    def _build_local_evidence_tokens(
        self,
        *,
        views_input: torch.Tensor,
        stage2_uv: torch.Tensor | None,
        measured_uv: torch.Tensor | None,
        measured_valid: torch.Tensor | None,
        measured_confidence: torch.Tensor | None,
        image_size: torch.Tensor | None,
        image_joint_feature: torch.Tensor | None,
        image_mask_feature: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.evidence_encoder is None:
            raise RuntimeError("Local evidence encoder was not initialized")
        batch_size, num_views = views_input.shape[:2]
        joint_count = self.config.image_joint_count
        residual, confidence, valid = self._build_per_view_measurement_features(
            views_input=views_input,
            stage2_uv=stage2_uv,
            measured_uv=measured_uv,
            measured_valid=measured_valid,
            measured_confidence=measured_confidence,
            image_size=image_size,
        )
        parts = []
        if self.config.use_image_joint_feature:
            parts.append(
                self._prepare_joint_feature(
                    feature=image_joint_feature,
                    expected_dim=self.config.image_feature_dim,
                    batch_size=batch_size,
                    num_views=num_views,
                    joint_count=joint_count,
                    device=views_input.device,
                    dtype=views_input.dtype,
                    norm=self.image_feature_norm,
                    name="image_joint_feature",
                )
            )
        if self.config.use_image_mask_feature:
            parts.append(
                self._prepare_joint_feature(
                    feature=image_mask_feature,
                    expected_dim=self.config.mask_feature_dim,
                    batch_size=batch_size,
                    num_views=num_views,
                    joint_count=joint_count,
                    device=views_input.device,
                    dtype=views_input.dtype,
                    norm=self.mask_feature_norm,
                    name="image_mask_feature",
                )
            )
        parts.extend(
            [
                residual,
                confidence[..., None],
                valid.to(dtype=views_input.dtype)[..., None],
            ]
        )
        evidence_input = torch.cat(parts, dim=-1)
        token = self.evidence_encoder(evidence_input.reshape(batch_size * num_views * joint_count, -1))
        token = token.reshape(batch_size, num_views, joint_count, self.config.evidence_token_dim)
        weight = confidence.clamp_min(0.0)
        denom = weight.sum(dim=1).clamp_min(1.0)
        image_joint_token = (token * weight[..., None]).sum(dim=1) / denom[..., None]
        image_joint_token = torch.where(
            torch.isfinite(image_joint_token),
            image_joint_token,
            torch.zeros_like(image_joint_token),
        )
        body_joint_token = image_joint_token.index_select(1, self.body_to_image_joint)
        return body_joint_token

    def _build_view_reliability_weights(
        self,
        *,
        views_input: torch.Tensor,
        stage2_uv: torch.Tensor | None,
        measured_uv: torch.Tensor | None,
        measured_valid: torch.Tensor | None,
        measured_confidence: torch.Tensor | None,
        image_size: torch.Tensor | None,
        image_joint_feature: torch.Tensor | None,
        image_mask_feature: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.view_reliability_head is None:
            raise RuntimeError("View reliability head was not initialized")
        body_joint_token, body_confidence, body_valid = self._build_per_view_local_evidence_tokens(
            views_input=views_input,
            stage2_uv=stage2_uv,
            measured_uv=measured_uv,
            measured_valid=measured_valid,
            measured_confidence=measured_confidence,
            image_size=image_size,
            image_joint_feature=image_joint_feature,
            image_mask_feature=image_mask_feature,
        )
        score = self.view_reliability_head(body_joint_token).squeeze(-1)
        confidence = torch.where(
            body_valid,
            body_confidence.clamp_min(0.0),
            torch.zeros_like(body_confidence),
        )
        score = score + confidence.clamp_min(1.0e-4).log()
        score = torch.where(confidence > 0.0, score, torch.full_like(score, -1.0e4))
        has_valid = (confidence > 0.0).any(dim=1, keepdim=True)
        score = torch.where(has_valid, score, torch.zeros_like(score))
        return torch.softmax(score, dim=1)

    def _build_per_view_local_evidence_tokens(
        self,
        *,
        views_input: torch.Tensor,
        stage2_uv: torch.Tensor | None,
        measured_uv: torch.Tensor | None,
        measured_valid: torch.Tensor | None,
        measured_confidence: torch.Tensor | None,
        image_size: torch.Tensor | None,
        image_joint_feature: torch.Tensor | None,
        image_mask_feature: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.evidence_encoder is None:
            raise RuntimeError("Local evidence encoder was not initialized")
        batch_size, num_views = views_input.shape[:2]
        joint_count = self.config.image_joint_count
        residual, confidence, valid = self._build_per_view_measurement_features(
            views_input=views_input,
            stage2_uv=stage2_uv,
            measured_uv=measured_uv,
            measured_valid=measured_valid,
            measured_confidence=measured_confidence,
            image_size=image_size,
        )
        parts = []
        if self.config.use_image_joint_feature:
            parts.append(
                self._prepare_joint_feature(
                    feature=image_joint_feature,
                    expected_dim=self.config.image_feature_dim,
                    batch_size=batch_size,
                    num_views=num_views,
                    joint_count=joint_count,
                    device=views_input.device,
                    dtype=views_input.dtype,
                    norm=self.image_feature_norm,
                    name="image_joint_feature",
                )
            )
        if self.config.use_image_mask_feature:
            parts.append(
                self._prepare_joint_feature(
                    feature=image_mask_feature,
                    expected_dim=self.config.mask_feature_dim,
                    batch_size=batch_size,
                    num_views=num_views,
                    joint_count=joint_count,
                    device=views_input.device,
                    dtype=views_input.dtype,
                    norm=self.mask_feature_norm,
                    name="image_mask_feature",
                )
            )
        parts.extend([residual, confidence[..., None], valid.to(dtype=views_input.dtype)[..., None]])
        evidence_input = torch.cat(parts, dim=-1)
        token = self.evidence_encoder(
            evidence_input.reshape(batch_size * num_views * joint_count, -1)
        )
        token = token.reshape(batch_size, num_views, joint_count, self.config.evidence_token_dim)
        token = torch.where(torch.isfinite(token), token, torch.zeros_like(token))
        body_joint_token = token.index_select(2, self.body_to_image_joint)
        body_confidence = confidence.index_select(2, self.body_to_image_joint)
        body_valid = valid.index_select(2, self.body_to_image_joint)
        return body_joint_token, body_confidence, body_valid

    def _prepare_joint_feature(
        self,
        *,
        feature: torch.Tensor | None,
        expected_dim: int,
        batch_size: int,
        num_views: int,
        joint_count: int,
        device: torch.device,
        dtype: torch.dtype,
        norm: nn.Module,
        name: str,
    ) -> torch.Tensor:
        shape = (batch_size, num_views, joint_count, expected_dim)
        if expected_dim <= 0:
            return torch.zeros(shape, device=device, dtype=dtype)
        if feature is None:
            return torch.zeros(shape, device=device, dtype=dtype)
        feature = feature.to(device=device, dtype=dtype)
        if feature.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(feature.shape)}")
        feature = torch.where(torch.isfinite(feature), feature, torch.zeros_like(feature))
        return norm(feature)


def _zero_last_linear(module: nn.Module) -> None:
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Linear):
            nn.init.zeros_(child.weight)
            nn.init.zeros_(child.bias)
            return


def _body_joint_policy_mask(*, policy: str, joint_count: int, device: torch.device) -> torch.Tensor:
    policy_indices = {
        "lower_body": [0, 1, 3, 4, 6, 7, 9, 10],
        "distal_legs": [3, 4, 6, 7, 9, 10],
        "ankle_feet": [6, 7, 9, 10],
    }
    if policy not in policy_indices:
        raise ValueError(f"Unsupported body joint policy mask: {policy}")
    mask = torch.zeros(joint_count, device=device, dtype=torch.bool)
    valid_indices = [index for index in policy_indices[policy] if index < joint_count]
    if valid_indices:
        mask[torch.tensor(valid_indices, device=device)] = True
    return mask

def _build_body_to_image_joint_index(num_body_joints: int, image_joint_count: int) -> list[int]:
    if image_joint_count <= 0:
        raise ValueError(f"image_joint_count must be positive, got {image_joint_count}")
    if num_body_joints == 23 and image_joint_count >= 17:
        return [
            4,  # left hip
            1,  # right hip
            7,  # spine
            5,  # left knee
            2,  # right knee
            8,  # upper spine / thorax
            6,  # left ankle
            3,  # right ankle
            8,  # chest
            6,  # left foot
            3,  # right foot
            9,  # neck
            11,  # left collar
            14,  # right collar
            10,  # head
            11,  # left shoulder
            14,  # right shoulder
            12,  # left elbow
            15,  # right elbow
            13,  # left wrist
            16,  # right wrist
            13,  # left hand
            16,  # right hand
        ]
    if num_body_joints == 1:
        return [0]
    return [
        min(image_joint_count - 1, round(index * (image_joint_count - 1) / (num_body_joints - 1)))
        for index in range(num_body_joints)
    ]
