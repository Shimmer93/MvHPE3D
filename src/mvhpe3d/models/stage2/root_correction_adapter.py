"""Stage 2.1 per-view root correction adapter."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..components import MLP


@dataclass(slots=True)
class Stage21RootCorrectionAdapterConfig:
    """Configuration for the frozen-body Stage 2.1 root adapter."""

    name: str = "stage2_1_root_correction_adapter"
    view_input_dim: int = 148
    joint_count: int = 17
    image_feature_dim: int = 250
    mask_feature_dim: int = 0
    evidence_token_dim: int = 32
    evidence_hidden_dim: int = 128
    evidence_layers: int = 2
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    global_orient_delta_scale: float = 0.05
    transl_delta_scale: float = 0.05
    use_view_input: bool = True
    use_input_camera: bool = True
    use_input_global_orient: bool = True
    use_input_transl: bool = True
    use_measurement_residual: bool = True
    use_measurement_confidence: bool = True
    use_measurement_valid: bool = True
    use_image_size: bool = True
    use_image_joint_feature: bool = False
    use_image_mask_feature: bool = False
    use_image_evidence_delta: bool = False
    normalize_image_joint_feature: bool = True
    normalize_image_mask_feature: bool = True
    zero_init: bool = True

    @property
    def input_dim(self) -> int:
        dim = 0
        if self.use_view_input:
            dim += self.view_input_dim
        if self.use_input_camera:
            if self.use_input_global_orient:
                dim += 3
            if self.use_input_transl:
                dim += 3
        if self.use_measurement_residual:
            dim += self.joint_count * 2
        if self.use_measurement_confidence:
            dim += self.joint_count
        return dim


class Stage21RootCorrectionAdapterModel(nn.Module):
    """Predict small per-view root orientation/translation corrections."""

    def __init__(self, config: Stage21RootCorrectionAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.adapter = MLP(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=6,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        self.use_image_evidence = bool(config.use_image_evidence_delta)
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
            if self.use_image_evidence
            else None
        )
        self.evidence_adapter = (
            MLP(
                input_dim=config.joint_count * config.evidence_token_dim,
                hidden_dim=config.hidden_dim,
                output_dim=6,
                num_layers=config.num_layers,
                dropout=config.dropout,
            )
            if self.use_image_evidence
            else None
        )
        if config.zero_init:
            _zero_last_linear(self.adapter)
            if self.evidence_adapter is not None:
                _zero_last_linear(self.evidence_adapter)

    def forward(
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
        if views_input.ndim != 3:
            raise ValueError(
                "views_input must have shape [batch, num_views, dim], "
                f"got {tuple(views_input.shape)}"
            )
        batch_size, num_views, _ = views_input.shape
        features = []
        if self.config.use_view_input:
            if views_input.shape[-1] != self.config.view_input_dim:
                raise ValueError(
                    f"Expected views_input dim {self.config.view_input_dim}, "
                    f"got {views_input.shape[-1]}"
                )
            features.append(views_input)
        if self.config.use_input_camera:
            camera_features = []
            if self.config.use_input_global_orient:
                camera_features.append(
                    view_aux["input_global_orient"].to(
                        device=views_input.device,
                        dtype=views_input.dtype,
                    )
                )
            if self.config.use_input_transl:
                camera_features.append(
                    view_aux["input_transl"].to(
                        device=views_input.device,
                        dtype=views_input.dtype,
                    )
                )
            if camera_features:
                features.append(torch.cat(camera_features, dim=-1))
        if self.config.use_measurement_residual or self.config.use_measurement_confidence:
            residual, confidence = self._build_measurement_features(
                views_input=views_input,
                view_aux=view_aux,
                stage2_uv=stage2_uv,
                measured_uv=measured_uv,
                measured_valid=measured_valid,
                measured_confidence=measured_confidence,
            )
            if self.config.use_measurement_residual:
                features.append(residual.reshape(batch_size, num_views, -1))
            if self.config.use_measurement_confidence:
                features.append(confidence.reshape(batch_size, num_views, -1))
        if not features:
            raise ValueError("At least one Stage 2.1 adapter input feature must be enabled")
        adapter_input = torch.cat(features, dim=-1)
        if adapter_input.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"Expected adapter input dim {self.config.input_dim}, "
                f"got {adapter_input.shape[-1]}"
            )
        raw = self.adapter(adapter_input.reshape(batch_size * num_views, -1))
        raw = raw.reshape(batch_size, num_views, 6)
        global_orient_delta = float(self.config.global_orient_delta_scale) * torch.tanh(
            raw[..., :3]
        )
        transl_delta = float(self.config.transl_delta_scale) * torch.tanh(raw[..., 3:])
        evidence_global_delta = torch.zeros_like(global_orient_delta)
        evidence_transl_delta = torch.zeros_like(transl_delta)
        if self.evidence_adapter is not None:
            evidence_raw = self._forward_image_evidence_delta(
                views_input=views_input,
                view_aux=view_aux,
                stage2_uv=stage2_uv,
                measured_uv=measured_uv,
                measured_valid=measured_valid,
                measured_confidence=measured_confidence,
                image_joint_feature=image_joint_feature,
                image_mask_feature=image_mask_feature,
            )
            evidence_global_delta = float(self.config.global_orient_delta_scale) * torch.tanh(
                evidence_raw[..., :3]
            )
            evidence_transl_delta = float(self.config.transl_delta_scale) * torch.tanh(
                evidence_raw[..., 3:]
            )
            global_orient_delta = global_orient_delta + evidence_global_delta
            transl_delta = transl_delta + evidence_transl_delta
        return {
            "pred_view_global_orient_delta": global_orient_delta,
            "pred_view_transl_delta": transl_delta,
            "pred_view_image_global_orient_delta": evidence_global_delta,
            "pred_view_image_transl_delta": evidence_transl_delta,
        }

    def _build_measurement_features(
        self,
        *,
        views_input: torch.Tensor,
        view_aux: dict[str, torch.Tensor],
        stage2_uv: torch.Tensor | None,
        measured_uv: torch.Tensor | None,
        measured_valid: torch.Tensor | None,
        measured_confidence: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_views = views_input.shape[:2]
        shape = (batch_size, num_views, self.config.joint_count)
        if stage2_uv is None or measured_uv is None:
            residual = torch.zeros(
                (*shape, 2),
                device=views_input.device,
                dtype=views_input.dtype,
            )
            confidence = torch.zeros(shape, device=views_input.device, dtype=views_input.dtype)
            return residual, confidence
        stage2_uv = stage2_uv.to(device=views_input.device, dtype=views_input.dtype)
        measured_uv = measured_uv.to(device=views_input.device, dtype=views_input.dtype)
        if stage2_uv.shape[:3] != shape or measured_uv.shape[:3] != shape:
            raise ValueError(
                "stage2_uv and measured_uv must have shape [B, V, J, 2] with "
                f"J={self.config.joint_count}; got {tuple(stage2_uv.shape)} and "
                f"{tuple(measured_uv.shape)}"
            )
        if measured_valid is None or not self.config.use_measurement_valid:
            valid = torch.isfinite(measured_uv).all(dim=-1)
        else:
            valid = measured_valid.to(device=views_input.device, dtype=torch.bool)
        if measured_confidence is None:
            confidence = valid.to(dtype=views_input.dtype)
        else:
            confidence = measured_confidence.to(device=views_input.device, dtype=views_input.dtype)
            confidence = confidence * valid.to(dtype=confidence.dtype)
        if self.config.use_image_size:
            image_size = view_aux["image_size"].to(
                device=views_input.device,
                dtype=views_input.dtype,
            )
        else:
            image_size = torch.full(
                (batch_size, num_views, 2),
                1000.0,
                device=views_input.device,
                dtype=views_input.dtype,
            )
        normalizer = image_size.clamp_min(1.0)[..., None, :]
        residual = (measured_uv - stage2_uv) / normalizer
        valid = valid & torch.isfinite(residual).all(dim=-1)
        residual = torch.where(valid[..., None], residual, torch.zeros_like(residual))
        confidence = torch.where(valid, confidence, torch.zeros_like(confidence))
        return residual, confidence

    def _forward_image_evidence_delta(
        self,
        *,
        views_input: torch.Tensor,
        view_aux: dict[str, torch.Tensor],
        stage2_uv: torch.Tensor | None,
        measured_uv: torch.Tensor | None,
        measured_valid: torch.Tensor | None,
        measured_confidence: torch.Tensor | None,
        image_joint_feature: torch.Tensor | None,
        image_mask_feature: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.evidence_adapter is None:
            raise RuntimeError("Image evidence root adapter was not initialized")
        token = self._build_image_evidence_tokens(
            views_input=views_input,
            view_aux=view_aux,
            stage2_uv=stage2_uv,
            measured_uv=measured_uv,
            measured_valid=measured_valid,
            measured_confidence=measured_confidence,
            image_joint_feature=image_joint_feature,
            image_mask_feature=image_mask_feature,
        )
        batch_size, num_views = views_input.shape[:2]
        flat = token.reshape(batch_size * num_views, -1)
        return self.evidence_adapter(flat).reshape(batch_size, num_views, 6)

    def _build_image_evidence_tokens(
        self,
        *,
        views_input: torch.Tensor,
        view_aux: dict[str, torch.Tensor],
        stage2_uv: torch.Tensor | None,
        measured_uv: torch.Tensor | None,
        measured_valid: torch.Tensor | None,
        measured_confidence: torch.Tensor | None,
        image_joint_feature: torch.Tensor | None,
        image_mask_feature: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.evidence_encoder is None:
            raise RuntimeError("Image evidence encoder was not initialized")
        batch_size, num_views = views_input.shape[:2]
        joint_count = self.config.joint_count
        residual, confidence = self._build_measurement_features(
            views_input=views_input,
            view_aux=view_aux,
            stage2_uv=stage2_uv,
            measured_uv=measured_uv,
            measured_valid=measured_valid,
            measured_confidence=measured_confidence,
        )
        valid = confidence > 0
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
        return torch.where(torch.isfinite(token), token, torch.zeros_like(token))

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
