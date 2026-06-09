"""Stage 2.3 combined root and gated body adapter."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .gated_body_adapter import Stage22GatedBodyAdapterConfig, Stage22GatedBodyAdapterModel
from .root_correction_adapter import (
    Stage21RootCorrectionAdapterConfig,
    Stage21RootCorrectionAdapterModel,
)


@dataclass(slots=True)
class Stage23RootBodyAdapterConfig:
    """Configuration for the frozen-Stage2 root+body adapter."""

    name: str = "stage2_3_root_body_adapter"
    view_input_dim: int = 148
    image_joint_count: int = 17
    image_feature_dim: int = 250
    mask_feature_dim: int = 0
    evidence_token_dim: int = 32
    evidence_hidden_dim: int = 128
    evidence_layers: int = 2
    use_body_local_joint_update_head: bool = False
    body_local_joint_hidden_dim: int = 256
    body_local_joint_layers: int = 3
    body_local_joint_embedding_dim: int = 16
    num_body_joints: int = 23
    betas_dim: int = 10
    root_hidden_dim: int = 256
    root_image_feature_dim: int = 250
    root_mask_feature_dim: int = 0
    root_evidence_token_dim: int = 32
    root_evidence_hidden_dim: int = 128
    root_evidence_layers: int = 2
    body_hidden_dim: int = 512
    body_extra_candidate_count: int = 0
    body_extra_hidden_dim: int = 512
    body_extra_num_layers: int = 3
    body_extra_selector_bias: float = 2.0
    body_extra_use_local_evidence: bool = True
    body_extra_update_joint_policy: str = "all"
    root_num_layers: int = 3
    body_num_layers: int = 3
    dropout: float = 0.1
    global_orient_delta_scale: float = 0.05
    transl_delta_scale: float = 0.05
    body_delta_scale: float = 0.15
    betas_delta_scale: float = 0.05
    gate_bias: float = -2.0
    use_root_input_global_orient: bool = True
    use_root_input_transl: bool = True
    use_root_measurement_residual: bool = True
    use_root_measurement_confidence: bool = True
    use_root_measurement_valid: bool = True
    use_root_image_size: bool = True
    use_root_image_joint_feature: bool = False
    use_root_image_mask_feature: bool = False
    use_root_image_evidence_delta: bool = False
    normalize_root_image_joint_feature: bool = True
    normalize_root_image_mask_feature: bool = True
    use_body_stage2_pose: bool = True
    use_body_input_pose_mean: bool = True
    use_body_input_pose_dispersion: bool = True
    use_body_input_betas: bool = True
    use_body_image_residual: bool = True
    use_body_image_confidence: bool = True
    use_body_image_valid: bool = True
    use_body_image_size: bool = True
    use_body_image_joint_feature: bool = False
    use_body_image_mask_feature: bool = False
    use_body_evidence_gate: bool = False
    use_body_evidence_gate_only: bool = False
    use_body_evidence_weighted_pose_fusion: bool = False
    evidence_weighted_pose_project_so3: bool = False
    use_body_evidence_weighted_betas_fusion: bool = False
    body_weighted_pose_joint_policy: str = "all"
    body_evidence_gate_bias: float = 2.0
    use_body_betas_update: bool = False
    normalize_body_image_joint_feature: bool = True
    normalize_body_image_mask_feature: bool = True
    zero_init: bool = True

    @property
    def body_pose_dim(self) -> int:
        return self.num_body_joints * 3

    def build_root_config(self) -> Stage21RootCorrectionAdapterConfig:
        return Stage21RootCorrectionAdapterConfig(
            view_input_dim=self.view_input_dim,
            joint_count=self.image_joint_count,
            image_feature_dim=self.root_image_feature_dim,
            mask_feature_dim=self.root_mask_feature_dim,
            evidence_token_dim=self.root_evidence_token_dim,
            evidence_hidden_dim=self.root_evidence_hidden_dim,
            evidence_layers=self.root_evidence_layers,
            hidden_dim=self.root_hidden_dim,
            num_layers=self.root_num_layers,
            dropout=self.dropout,
            global_orient_delta_scale=self.global_orient_delta_scale,
            transl_delta_scale=self.transl_delta_scale,
            use_input_global_orient=self.use_root_input_global_orient,
            use_input_transl=self.use_root_input_transl,
            use_measurement_residual=self.use_root_measurement_residual,
            use_measurement_confidence=self.use_root_measurement_confidence,
            use_measurement_valid=self.use_root_measurement_valid,
            use_image_size=self.use_root_image_size,
            use_image_joint_feature=self.use_root_image_joint_feature,
            use_image_mask_feature=self.use_root_image_mask_feature,
            use_image_evidence_delta=self.use_root_image_evidence_delta,
            normalize_image_joint_feature=self.normalize_root_image_joint_feature,
            normalize_image_mask_feature=self.normalize_root_image_mask_feature,
            zero_init=self.zero_init,
        )

    def build_body_config(self) -> Stage22GatedBodyAdapterConfig:
        return Stage22GatedBodyAdapterConfig(
            view_input_dim=self.view_input_dim,
            num_body_joints=self.num_body_joints,
            betas_dim=self.betas_dim,
            image_joint_count=self.image_joint_count,
            image_feature_dim=self.image_feature_dim,
            mask_feature_dim=self.mask_feature_dim,
            evidence_token_dim=self.evidence_token_dim,
            evidence_hidden_dim=self.evidence_hidden_dim,
            evidence_layers=self.evidence_layers,
            use_local_joint_update_head=self.use_body_local_joint_update_head,
            local_joint_hidden_dim=self.body_local_joint_hidden_dim,
            local_joint_layers=self.body_local_joint_layers,
            local_joint_embedding_dim=self.body_local_joint_embedding_dim,
            hidden_dim=self.body_hidden_dim,
            num_layers=self.body_num_layers,
            dropout=self.dropout,
            body_delta_scale=self.body_delta_scale,
            betas_delta_scale=self.betas_delta_scale,
            gate_bias=self.gate_bias,
            use_stage2_body_pose=self.use_body_stage2_pose,
            use_input_pose_mean=self.use_body_input_pose_mean,
            use_input_pose_dispersion=self.use_body_input_pose_dispersion,
            use_input_betas=self.use_body_input_betas,
            use_image_residual=self.use_body_image_residual,
            use_image_confidence=self.use_body_image_confidence,
            use_image_valid=self.use_body_image_valid,
            use_image_size=self.use_body_image_size,
            use_image_joint_feature=self.use_body_image_joint_feature,
            use_image_mask_feature=self.use_body_image_mask_feature,
            use_evidence_gate=self.use_body_evidence_gate,
            use_evidence_gate_only=self.use_body_evidence_gate_only,
            use_evidence_weighted_pose_fusion=self.use_body_evidence_weighted_pose_fusion,
            evidence_weighted_pose_project_so3=self.evidence_weighted_pose_project_so3,
            use_evidence_weighted_betas_fusion=self.use_body_evidence_weighted_betas_fusion,
            weighted_pose_joint_policy=self.body_weighted_pose_joint_policy,
            evidence_gate_bias=self.body_evidence_gate_bias,
            predict_betas_update=self.use_body_betas_update,
            extra_candidate_count=self.body_extra_candidate_count,
            extra_hidden_dim=self.body_extra_hidden_dim,
            extra_num_layers=self.body_extra_num_layers,
            extra_selector_bias=self.body_extra_selector_bias,
            extra_use_local_evidence=self.body_extra_use_local_evidence,
            extra_update_joint_policy=self.body_extra_update_joint_policy,
            normalize_image_joint_feature=self.normalize_body_image_joint_feature,
            normalize_image_mask_feature=self.normalize_body_image_mask_feature,
            zero_init=self.zero_init,
        )


class Stage23RootBodyAdapterModel(nn.Module):
    """Sequential Stage 2.1 root correction plus Stage 2.2 body correction."""

    def __init__(self, config: Stage23RootBodyAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.root_adapter = Stage21RootCorrectionAdapterModel(config.build_root_config())
        self.body_adapter = Stage22GatedBodyAdapterModel(config.build_body_config())

    def forward_root(
        self,
        *,
        views_input: torch.Tensor,
        view_aux: dict[str, torch.Tensor],
        stage2_uv: torch.Tensor | None = None,
        measured_uv: torch.Tensor | None = None,
        measured_valid: torch.Tensor | None = None,
        measured_confidence: torch.Tensor | None = None,
        image_joint_feature: torch.Tensor | None = None,
        image_mask_feature: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.root_adapter(
            views_input=views_input,
            view_aux=view_aux,
            stage2_uv=stage2_uv,
            measured_uv=measured_uv,
            measured_valid=measured_valid,
            measured_confidence=measured_confidence,
            image_joint_feature=image_joint_feature,
            image_mask_feature=image_mask_feature,
        )

    def forward_body(
        self,
        *,
        views_input: torch.Tensor,
        stage2_body_pose: torch.Tensor,
        corrected_stage2_uv: torch.Tensor | None = None,
        measured_uv: torch.Tensor | None = None,
        measured_valid: torch.Tensor | None = None,
        measured_confidence: torch.Tensor | None = None,
        image_size: torch.Tensor | None = None,
        image_joint_feature: torch.Tensor | None = None,
        image_mask_feature: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.body_adapter(
            views_input=views_input,
            stage2_body_pose=stage2_body_pose,
            stage2_uv=corrected_stage2_uv,
            measured_uv=measured_uv,
            measured_valid=measured_valid,
            measured_confidence=measured_confidence,
            image_size=image_size,
            image_joint_feature=image_joint_feature,
            image_mask_feature=image_mask_feature,
        )

    def forward(
        self,
        *,
        views_input: torch.Tensor,
        view_aux: dict[str, torch.Tensor],
        stage2_body_pose: torch.Tensor,
        stage2_uv: torch.Tensor | None = None,
        corrected_stage2_uv: torch.Tensor | None = None,
        measured_uv: torch.Tensor | None = None,
        measured_valid: torch.Tensor | None = None,
        measured_confidence: torch.Tensor | None = None,
        image_size: torch.Tensor | None = None,
        image_joint_feature: torch.Tensor | None = None,
        image_mask_feature: torch.Tensor | None = None,
        enable_body: bool = True,
    ) -> dict[str, torch.Tensor]:
        outputs = self.forward_root(
            views_input=views_input,
            view_aux=view_aux,
            stage2_uv=stage2_uv,
            measured_uv=measured_uv,
            measured_valid=measured_valid,
            measured_confidence=measured_confidence,
            image_joint_feature=image_joint_feature,
            image_mask_feature=image_mask_feature,
        )
        if enable_body:
            outputs.update(
                self.forward_body(
                    views_input=views_input,
                    stage2_body_pose=stage2_body_pose,
                    corrected_stage2_uv=corrected_stage2_uv,
                    measured_uv=measured_uv,
                    measured_valid=measured_valid,
                    measured_confidence=measured_confidence,
                    image_size=image_size,
                    image_joint_feature=image_joint_feature,
                    image_mask_feature=image_mask_feature,
                )
            )
        else:
            batch_size = stage2_body_pose.shape[0]
            device = stage2_body_pose.device
            dtype = stage2_body_pose.dtype
            outputs.update(
                {
                    "pred_body_pose_delta": torch.zeros(
                        (batch_size, self.config.body_pose_dim),
                        device=device,
                        dtype=dtype,
                    ),
                    "pred_body_pose_gate": torch.zeros(
                        (batch_size, self.config.num_body_joints),
                        device=device,
                        dtype=dtype,
                    ),
                    "pred_body_pose_update": torch.zeros(
                        (batch_size, self.config.body_pose_dim),
                        device=device,
                        dtype=dtype,
                    ),
                }
            )
        return outputs
