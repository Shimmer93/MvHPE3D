"""LightningModule for the Stage 2 parameter refinement model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F

from mvhpe3d.losses import Stage2Loss, Stage2LossConfig
from mvhpe3d.metrics import (
    SMPL_EVAL_NUM_JOINTS,
    batch_mpjpe,
    batch_pa_mpjpe,
    batch_similarity_align,
    compute_input_corrected_camera_joint_metrics,
    root_center_joints,
)
from mvhpe3d.models import (
    Stage2JointGraphRefinerConfig,
    Stage2JointGraphRefinerModel,
    Stage2JointResidualConfig,
    Stage2JointResidualModel,
    Stage2ParamRefineConfig,
    Stage2ParamRefineModel,
    Stage2RRGBGuidedResidualRefinerConfig,
    Stage2RRGBGuidedResidualRefinerModel,
)
from mvhpe3d.utils import axis_angle_to_matrix, build_smpl_model, rotation_6d_to_axis_angle


@dataclass(slots=True)
class Stage2OptimizationConfig:
    """Optimizer settings for Stage 2 training."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    stage2_backbone_lr_scale: float = 0.1


class Stage2FusionLightningModule(L.LightningModule):
    """Lightning wrapper around the Stage 2 fusion and refinement model."""

    strict_loading = False

    def __init__(
        self,
        *,
        model_config: (
                        Stage2JointGraphRefinerConfig
            | Stage2JointResidualConfig
            | Stage2ParamRefineConfig
            | dict
            | None
        ) = None,
        loss_config: Stage2LossConfig | dict | None = None,
        optimization_config: Stage2OptimizationConfig | dict | None = None,
        smpl_model_path: str | None = None,
        external_joint_source: str = "smpl24",
        external_joint_regressor_path: str | None = None,
        external_headtop_topk: int = 32,
        external_headtop_axis: str = "y",
    ) -> None:
        super().__init__()
        self.model_config = _coerce_stage2_model_config(model_config)
        self.loss_config = _coerce_dataclass_config(loss_config, Stage2LossConfig)
        if not self.model_config.learn_betas:
            self.loss_config.supervise_betas = False
            self.loss_config.betas_weight = 0.0
            self.loss_config.init_betas_weight = 0.0
        self.optimization_config = _coerce_dataclass_config(
            optimization_config,
            Stage2OptimizationConfig,
        )
        self.smpl_model_path = smpl_model_path
        self.external_joint_source = external_joint_source
        self.external_joint_regressor_path = external_joint_regressor_path
        self.external_headtop_topk = external_headtop_topk
        self.external_headtop_axis = external_headtop_axis

        self.model = _build_stage2_model(self.model_config)
        self.criterion = Stage2Loss(self.loss_config)
        self._runtime_cache: dict[str, object | None] = {
            "smpl_eval_model": None,
            "smpl_eval_models": {},
            "external_joint_regressor": None,
            "external_headtop_vertex_ids": None,
        }

        self.save_hyperparameters(
            {
                "model_config": asdict(self.model_config),
                "loss_config": asdict(self.loss_config),
                "optimization_config": asdict(self.optimization_config),
                "smpl_model_path": self.smpl_model_path,
                "external_joint_source": self.external_joint_source,
                "external_joint_regressor_path": self.external_joint_regressor_path,
                "external_headtop_topk": self.external_headtop_topk,
                "external_headtop_axis": self.external_headtop_axis,
            }
        )

    def forward(
        self,
        views_input: torch.Tensor,
        *,
        view_rgb_feature: torch.Tensor | None = None,
        view_image_joint_feature: torch.Tensor | None = None,
        view_image_joint_valid: torch.Tensor | None = None,
        view_image_joint_confidence: torch.Tensor | None = None,
        view_image_joint_uv: torch.Tensor | None = None,
        view_image_joint_projected_uv: torch.Tensor | None = None,
        view_image_mask_feature: torch.Tensor | None = None,
        view_segmentation_mask: torch.Tensor | None = None,
        view_segmentation_distance: torch.Tensor | None = None,
        view_segmentation_valid: torch.Tensor | None = None,
        view_image_size: torch.Tensor | None = None,
        view_aux: dict[str, torch.Tensor] | None = None,
        target_joint_smpl_indices: torch.Tensor | None = None,
        target_joint_root_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del (
            target_joint_smpl_indices,
            target_joint_root_index,
        )
        if view_rgb_feature is not None:
            self._assert_finite_tensor("batch/view_rgb_feature", view_rgb_feature)
        if getattr(self.model, "uses_image_measurements", False):
            return self.model(
                views_input,
                view_image_joint_feature=view_image_joint_feature,
                view_image_joint_valid=view_image_joint_valid,
                view_image_joint_confidence=view_image_joint_confidence,
                view_image_joint_uv=view_image_joint_uv,
                view_image_joint_projected_uv=view_image_joint_projected_uv,
                view_image_mask_feature=view_image_mask_feature,
                view_segmentation_mask=view_segmentation_mask,
                view_segmentation_distance=view_segmentation_distance,
                view_segmentation_valid=view_segmentation_valid,
                view_image_size=view_image_size,
                view_aux=view_aux,
            )
        return self.model(views_input)

    def training_step(self, batch, batch_idx):
        metrics = self._shared_step(batch, stage="train")
        self._log_metrics_if_possible(
            metrics,
            batch_size=batch["views_input"].shape[0],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
        )
        return metrics["train/loss"]

    def validation_step(self, batch, batch_idx):
        metrics = self._shared_step(batch, stage="val")
        self._log_metrics_if_possible(
            metrics,
            batch_size=batch["views_input"].shape[0],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return metrics["val/loss"]

    def test_step(self, batch, batch_idx):
        metrics = self._shared_step(batch, stage="test")
        self._log_metrics_if_possible(
            metrics,
            batch_size=batch["views_input"].shape[0],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        return metrics["test/loss"]

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        predictions = self(
            batch["views_input"],
            view_rgb_feature=batch.get("view_rgb_feature"),
            view_image_joint_feature=batch.get("view_image_joint_feature"),
            view_image_joint_valid=batch.get("view_image_joint_valid"),
            view_image_joint_confidence=batch.get("view_image_joint_confidence"),
            view_image_joint_uv=batch.get("view_image_joint_uv"),
            view_image_joint_projected_uv=batch.get("view_image_joint_projected_uv"),
            view_image_mask_feature=batch.get("view_image_mask_feature"),
            view_segmentation_mask=batch.get("view_segmentation_mask"),
            view_segmentation_distance=batch.get("view_segmentation_distance"),
            view_segmentation_valid=batch.get("view_segmentation_valid"),
            view_image_size=batch.get("view_aux", {}).get("image_size"),
            view_aux=batch.get("view_aux"),
            target_joint_smpl_indices=batch.get("target_joint_smpl_indices"),
            target_joint_root_index=batch.get("target_joint_root_index"),
        )
        return {
            "pred_body_pose": predictions["pred_body_pose"],
            "pred_betas": predictions["pred_betas"],
            "meta": batch["meta"],
        }

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.optimization_config.learning_rate,
            weight_decay=self.optimization_config.weight_decay,
        )

    def on_train_batch_start(self, batch, batch_idx) -> None:
        self._sanitize_parameters("train_batch_start")

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0) -> None:
        self._sanitize_parameters("validation_batch_start")

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
    ) -> None:
        self._sanitize_gradients()
        if gradient_clip_val is not None and gradient_clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=gradient_clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm,
            )

    def on_before_optimizer_step(self, optimizer) -> None:
        self._sanitize_parameters("before_optimizer_step")
        self._sanitize_gradients()

    def _shared_step(self, batch, *, stage: str) -> dict[str, torch.Tensor]:
        self._assert_finite_tensor("batch/views_input", batch["views_input"])
        use_joint_targets = "target_joints" in batch
        use_smpl_targets = self._batch_has_smpl_targets(
            batch,
            use_joint_targets=use_joint_targets,
        )
        if use_smpl_targets:
            self._assert_finite_tensor(
                "batch/target_body_pose_6d", batch["target_body_pose_6d"]
            )
            self._assert_finite_tensor("batch/target_betas", batch["target_betas"])
            self._assert_finite_tensor("batch/target_body_pose", batch["target_body_pose"])
        predictions = self(
            batch["views_input"],
            view_rgb_feature=batch.get("view_rgb_feature"),
            view_image_joint_feature=batch.get("view_image_joint_feature"),
            view_image_joint_valid=batch.get("view_image_joint_valid"),
            view_image_joint_confidence=batch.get("view_image_joint_confidence"),
            view_image_joint_uv=batch.get("view_image_joint_uv"),
            view_image_joint_projected_uv=batch.get("view_image_joint_projected_uv"),
            view_image_mask_feature=batch.get("view_image_mask_feature"),
            view_segmentation_mask=batch.get("view_segmentation_mask"),
            view_segmentation_distance=batch.get("view_segmentation_distance"),
            view_segmentation_valid=batch.get("view_segmentation_valid"),
            view_image_size=batch.get("view_aux", {}).get("image_size"),
            view_aux=batch.get("view_aux"),
            target_joint_smpl_indices=batch.get("target_joint_smpl_indices"),
            target_joint_root_index=batch.get("target_joint_root_index"),
        )
        for name, value in predictions.items():
            if torch.is_tensor(value):
                self._assert_finite_tensor(f"predictions/{name}", value)
        if use_joint_targets:
            self._assert_finite_tensor("batch/target_joints", batch["target_joints"])
            self._assert_finite_tensor(
                "batch/target_joint_confidence",
                batch["target_joint_confidence"],
            )
            if "target_camera_joints" in batch:
                self._assert_finite_tensor(
                    "batch/target_camera_joints",
                    batch["target_camera_joints"],
                )
            if "target_camera_joint_confidence" in batch:
                self._assert_finite_tensor(
                    "batch/target_camera_joint_confidence",
                    batch["target_camera_joint_confidence"],
                )
        if use_smpl_targets:
            pose_residual_6d, betas_residual = self._build_stage2_residuals(
                predictions
            )
            losses = self.criterion(
                pred_pose_6d=predictions["pred_pose_6d"],
                pred_betas=predictions["pred_betas"],
                target_pose_6d=batch["target_body_pose_6d"],
                target_betas=batch["target_betas"],
                init_pose_6d=predictions["init_pose_6d"],
                init_betas=predictions["init_betas"],
                pose_residual_6d=pose_residual_6d,
                betas_residual=betas_residual,
            )
        else:
            pose_residual_6d, betas_residual = self._build_stage2_residuals(
                predictions
            )
            losses = self._zero_stage2_parameter_losses(
                predictions["pred_pose_6d"],
                pose_residual_6d=pose_residual_6d,
                betas_residual=betas_residual,
            )
        for name, value in losses.items():
            self._assert_finite_tensor(f"losses/{name}", value)
        if use_joint_targets:
            if "pred_external_joints" in predictions:
                pred_joints = predictions["pred_external_joints"]
                target_joints, joint_weight = self._build_external_target_joints(
                    target_joints=batch["target_joints"],
                    target_joint_confidence=batch["target_joint_confidence"],
                    target_joint_root_index=batch["target_joint_root_index"],
                    view_aux=batch["view_aux"],
                    device=pred_joints.device,
                    dtype=pred_joints.dtype,
                )
            elif "pred_external_joint_delta" in predictions:
                pred_joints, target_joints, joint_weight = self._build_external_joint_pair(
                    pred_body_pose=predictions["pred_body_pose"],
                    pred_betas=predictions["pred_betas"],
                    target_joints=batch["target_joints"],
                    target_joint_confidence=batch["target_joint_confidence"],
                    target_joint_smpl_indices=batch["target_joint_smpl_indices"],
                    target_joint_root_index=batch["target_joint_root_index"],
                    view_aux=batch["view_aux"],
                )
                external_joint_delta = predictions["pred_external_joint_delta"].to(
                    device=pred_joints.device,
                    dtype=pred_joints.dtype,
                )
                if external_joint_delta.shape != pred_joints.shape:
                    raise ValueError(
                        "pred_external_joint_delta must match pred_joints shape "
                        f"{tuple(pred_joints.shape)}, got {tuple(external_joint_delta.shape)}"
                    )
                pred_joints = pred_joints + external_joint_delta
            else:
                pred_joints, target_joints, joint_weight = self._build_external_joint_pair(
                    pred_body_pose=predictions["pred_body_pose"],
                    pred_betas=predictions["pred_betas"],
                    target_joints=batch["target_joints"],
                    target_joint_confidence=batch["target_joint_confidence"],
                    target_joint_smpl_indices=batch["target_joint_smpl_indices"],
                    target_joint_root_index=batch["target_joint_root_index"],
                    view_aux=batch["view_aux"],
                )
            joint_loss = self._weighted_joint_mse(pred_joints, target_joints, joint_weight)
        else:
            pred_joints, target_joints = self._build_canonical_joint_pair(
                pred_body_pose=predictions["pred_body_pose"],
                pred_betas=predictions["pred_betas"],
                target_body_pose=batch["target_body_pose"],
                target_betas=batch["target_betas"],
            )
            joint_weight = None
            joint_loss = F.mse_loss(pred_joints, target_joints)
        self._assert_finite_tensor("losses/joint_loss", joint_loss)
        pa_joint_loss = self._compute_pa_joint_loss(
            pred_joints=pred_joints,
            target_joints=target_joints,
            joint_weight=joint_weight,
            target_joint_root_index=batch.get("target_joint_root_index"),
        )
        self._assert_finite_tensor("losses/pa_joint_loss", pa_joint_loss)
        canonical_pseudo_losses = self._compute_canonical_pseudo_joint_losses(
            pred_body_pose=predictions["pred_body_pose"],
            pred_betas=predictions["pred_betas"],
            target_body_pose=batch["target_body_pose"],
            target_betas=batch["target_betas"],
            use_smpl_targets=use_smpl_targets,
            target_joint_smpl_indices=batch.get("target_joint_smpl_indices"),
            target_joint_root_index=batch.get("target_joint_root_index"),
        )
        for name, value in canonical_pseudo_losses.items():
            self._assert_finite_tensor(f"losses/{name}", value)
        camera_joint_losses = self._compute_camera_joint_losses(
            pred_joints=pred_joints,
            batch=batch,
            use_joint_targets=use_joint_targets,
        )
        for name, value in camera_joint_losses.items():
            self._assert_finite_tensor(f"losses/{name}", value)
        camera_pa_joint_losses = self._compute_camera_pa_joint_losses(
            pred_joints=pred_joints,
            batch=batch,
            use_joint_targets=use_joint_targets,
        )
        for name, value in camera_pa_joint_losses.items():
            self._assert_finite_tensor(f"losses/{name}", value)
        if use_joint_targets:
            articulation_loss = joint_loss.new_zeros(())
        else:
            articulation_loss = self._compute_articulation_loss(
                pred_joints=pred_joints, target_joints=target_joints
            )
        self._assert_finite_tensor("losses/articulation_loss", articulation_loss)
        stage2_aux_losses = self._compute_stage2_aux_losses(
            predictions=predictions,
            batch=batch,
        )
        for name, value in stage2_aux_losses.items():
            self._assert_finite_tensor(f"losses/{name}", value)
        input_projection_losses = self._compute_input_projection_losses(
            predictions=predictions,
            batch=batch,
        )
        for name, value in input_projection_losses.items():
            self._assert_finite_tensor(f"losses/{name}", value)
        gt_projection_losses = self._compute_gt_projection_losses(
            predictions=predictions,
            batch=batch,
        )
        for name, value in gt_projection_losses.items():
            self._assert_finite_tensor(f"losses/{name}", value)
        image_projection_losses = self._compute_image_projection_losses(
            predictions=predictions,
            batch=batch,
        )
        for name, value in image_projection_losses.items():
            self._assert_finite_tensor(f"losses/{name}", value)
        projected_measurement_losses = self._compute_projected_measurement_losses(
            predictions=predictions,
            batch=batch,
        )
        for name, value in projected_measurement_losses.items():
            self._assert_finite_tensor(f"losses/{name}", value)
        mask_supervision_losses = self._compute_mask_supervision_losses(
            predictions=predictions,
            batch=batch,
        )
        for name, value in mask_supervision_losses.items():
            self._assert_finite_tensor(f"losses/{name}", value)
        pose_pca_coeff_loss = self._compute_pose_pca_coeff_loss(
            predictions=predictions,
            batch=batch,
        )
        self._assert_finite_tensor("losses/pose_pca_coeff_loss", pose_pca_coeff_loss)
        adapter_aux_losses = self._compute_adapter_aux_joint_losses(
            predictions=predictions,
            target_joints=target_joints,
            joint_weight=joint_weight,
            target_joint_root_index=batch.get("target_joint_root_index"),
        )
        for name, value in adapter_aux_losses.items():
            self._assert_finite_tensor(f"losses/{name}", value)
        triangulated_joint_losses = self._compute_triangulated_joint_losses(
            pred_joints=pred_joints,
            predictions=predictions,
            target_joint_root_index=batch.get("target_joint_root_index"),
        )
        for name, value in triangulated_joint_losses.items():
            self._assert_finite_tensor(f"losses/{name}", value)
        total_loss = (
            losses["loss"]
            + self.loss_config.joint_weight * joint_loss
            + self.loss_config.pa_joint_weight * pa_joint_loss
            + self.loss_config.canonical_pseudo_joint_weight
            * canonical_pseudo_losses["canonical_pseudo_joint_loss"]
            + self.loss_config.canonical_pseudo_pa_joint_weight
            * canonical_pseudo_losses["canonical_pseudo_pa_joint_loss"]
            + self.loss_config.camera_joint_weight
            * camera_joint_losses["camera_joint_loss"]
            + self.loss_config.camera_pa_joint_weight
            * camera_pa_joint_losses["camera_pa_joint_loss"]
            + self.loss_config.articulation_weight * articulation_loss
            + self.loss_config.stage2_aux_weight * stage2_aux_losses["stage2_aux_loss"]
            + self.loss_config.input_projection_weight
            * input_projection_losses["input_projection_loss"]
            + self.loss_config.gt_projection_weight
            * gt_projection_losses["gt_projection_loss"]
            + self.loss_config.image_projection_weight
            * image_projection_losses["image_projection_loss"]
            + self.loss_config.projected_measurement_weight
            * projected_measurement_losses["projected_measurement_loss"]
            + self.loss_config.projected_measurement_consistency_weight
            * projected_measurement_losses[
                "projected_measurement_consistency_loss"
            ]
            + self.loss_config.mask_supervision_weight
            * mask_supervision_losses["mask_supervision_loss"]
            + self.loss_config.pose_pca_coeff_weight * pose_pca_coeff_loss
            + self.loss_config.adapter_aux_joint_weight
            * adapter_aux_losses["adapter_aux_joint_loss"]
            + self.loss_config.adapter_aux_pa_joint_weight
            * adapter_aux_losses["adapter_aux_pa_joint_loss"]
            + self.loss_config.triangulated_joint_weight
            * triangulated_joint_losses["triangulated_joint_loss"]
            + self.loss_config.triangulated_pa_joint_weight
            * triangulated_joint_losses["triangulated_pa_joint_loss"]
        )
        self._assert_finite_tensor("losses/total_loss", total_loss)
        metrics = {
            f"{stage}/loss": total_loss,
            f"{stage}/loss_pose_6d": losses["loss_pose_6d"],
            f"{stage}/loss_betas": losses["loss_betas"],
            f"{stage}/loss_init_pose_6d": losses["loss_init_pose_6d"],
            f"{stage}/loss_init_betas": losses["loss_init_betas"],
            f"{stage}/loss_pose_residual": losses["loss_pose_residual"],
            f"{stage}/loss_betas_residual": losses["loss_betas_residual"],
            f"{stage}/loss_pose_pca_coeff": pose_pca_coeff_loss,
            f"{stage}/loss_adapter_aux_joints": adapter_aux_losses[
                "adapter_aux_joint_loss"
            ],
            f"{stage}/loss_adapter_aux_pa_joints": adapter_aux_losses[
                "adapter_aux_pa_joint_loss"
            ],
            f"{stage}/adapter_aux_mpjpe": adapter_aux_losses["adapter_aux_mpjpe"],
            f"{stage}/adapter_aux_pa_mpjpe": adapter_aux_losses[
                "adapter_aux_pa_mpjpe"
            ],
            f"{stage}/loss_triangulated_joints": triangulated_joint_losses[
                "triangulated_joint_loss"
            ],
            f"{stage}/loss_triangulated_pa_joints": triangulated_joint_losses[
                "triangulated_pa_joint_loss"
            ],
            f"{stage}/triangulated_mpjpe": triangulated_joint_losses[
                "triangulated_mpjpe"
            ],
            f"{stage}/triangulated_pa_mpjpe": triangulated_joint_losses[
                "triangulated_pa_mpjpe"
            ],
            f"{stage}/loss_joints": joint_loss,
            f"{stage}/loss_pa_joints": pa_joint_loss,
            f"{stage}/loss_canonical_pseudo_joints": canonical_pseudo_losses[
                "canonical_pseudo_joint_loss"
            ],
            f"{stage}/loss_canonical_pseudo_pa_joints": canonical_pseudo_losses[
                "canonical_pseudo_pa_joint_loss"
            ],
            f"{stage}/loss_camera_joints": camera_joint_losses["camera_joint_loss"],
            f"{stage}/loss_camera_pa_joints": camera_pa_joint_losses[
                "camera_pa_joint_loss"
            ],
            f"{stage}/loss_articulation": articulation_loss,
            f"{stage}/loss_stage2_aux": stage2_aux_losses["stage2_aux_loss"],
            f"{stage}/loss_stage2_aux_pose_6d": stage2_aux_losses[
                "stage2_aux_pose_6d"
            ],
            f"{stage}/loss_stage2_aux_betas": stage2_aux_losses[
                "stage2_aux_betas"
            ],
            f"{stage}/loss_input_projection": input_projection_losses[
                "input_projection_loss"
            ],
            f"{stage}/input_projection_error_px": input_projection_losses[
                "input_projection_error_px"
            ],
            f"{stage}/input_projection_valid_ratio": input_projection_losses[
                "input_projection_valid_ratio"
            ],
            f"{stage}/loss_gt_projection": gt_projection_losses[
                "gt_projection_loss"
            ],
            f"{stage}/gt_projection_error_px": gt_projection_losses[
                "gt_projection_error_px"
            ],
            f"{stage}/gt_projection_valid_ratio": gt_projection_losses[
                "gt_projection_valid_ratio"
            ],
            f"{stage}/loss_image_projection": image_projection_losses[
                "image_projection_loss"
            ],
            f"{stage}/image_projection_error_px": image_projection_losses[
                "image_projection_error_px"
            ],
            f"{stage}/image_projection_valid_ratio": image_projection_losses[
                "image_projection_valid_ratio"
            ],
            f"{stage}/loss_projected_measurement": projected_measurement_losses[
                "projected_measurement_loss"
            ],
            f"{stage}/projected_measurement_error_px": projected_measurement_losses[
                "projected_measurement_error_px"
            ],
            f"{stage}/projected_measurement_raw_error_px": (
                projected_measurement_losses["projected_measurement_raw_error_px"]
            ),
            f"{stage}/projected_measurement_valid_ratio": (
                projected_measurement_losses["projected_measurement_valid_ratio"]
            ),
            f"{stage}/loss_projected_measurement_consistency": (
                projected_measurement_losses[
                    "projected_measurement_consistency_loss"
                ]
            ),
            f"{stage}/projected_measurement_consistency_error_px": (
                projected_measurement_losses[
                    "projected_measurement_consistency_error_px"
                ]
            ),
            f"{stage}/projected_measurement_consistency_valid_ratio": (
                projected_measurement_losses[
                    "projected_measurement_consistency_valid_ratio"
                ]
            ),
            f"{stage}/loss_mask_supervision": mask_supervision_losses[
                "mask_supervision_loss"
            ],
            f"{stage}/mask_supervision_distance_loss": mask_supervision_losses[
                "mask_supervision_distance_loss"
            ],
            f"{stage}/mask_supervision_bbox_loss": mask_supervision_losses[
                "mask_supervision_bbox_loss"
            ],
            f"{stage}/mask_supervision_inside_ratio": mask_supervision_losses[
                "mask_supervision_inside_ratio"
            ],
            f"{stage}/mask_supervision_valid_ratio": mask_supervision_losses[
                "mask_supervision_valid_ratio"
            ],
        }
        metrics.update(
            self._compute_rgb_residual_diagnostics(
                predictions=predictions,
                batch=batch,
                stage=stage,
            )
        )

        if stage == "val":
            if use_joint_targets:
                joint_metrics = self._compute_external_joint_metrics(
                    pred_joints=pred_joints,
                    target_joints=target_joints,
                    joint_weight=joint_weight,
                    target_joint_root_index=batch.get("target_joint_root_index"),
                    target_camera_joints=batch.get("target_camera_joints"),
                    target_camera_joint_confidence=batch.get(
                        "target_camera_joint_confidence"
                    ),
                    view_aux=batch.get("view_aux"),
                )
            else:
                joint_metrics = self._compute_canonical_joint_metrics(
                    pred_body_pose=predictions["pred_body_pose"],
                    pred_betas=predictions["pred_betas"],
                    target_body_pose=batch["target_body_pose"],
                    target_betas=batch["target_betas"],
                )
            metrics.update(
                {
                    f"{stage}/mpjpe": joint_metrics["mpjpe"],
                    f"{stage}/pa_mpjpe": joint_metrics["pa_mpjpe"],
                }
            )
            if use_joint_targets:
                metrics.update(
                    {
                        f"{stage}/pck_150": joint_metrics["pck_150"],
                        f"{stage}/auc": joint_metrics["auc"],
                        f"{stage}/root_pck_150": joint_metrics["root_pck_150"],
                        f"{stage}/root_auc": joint_metrics["root_auc"],
                    }
                )
                if "target_camera_joints" in batch:
                    input_joint_metrics = self._compute_external_input_joint_metrics(
                        views_input=batch["views_input"],
                        view_aux=batch["view_aux"],
                        target_camera_joints=batch["target_camera_joints"],
                        target_camera_joint_confidence=batch.get(
                            "target_camera_joint_confidence"
                        ),
                        target_joint_smpl_indices=batch["target_joint_smpl_indices"],
                        target_joint_root_index=batch["target_joint_root_index"],
                    )
                    metrics.update(
                        {
                            f"{stage}/input_mpjpe": input_joint_metrics["mpjpe"],
                            f"{stage}/input_pa_mpjpe": input_joint_metrics["pa_mpjpe"],
                        }
                    )
                if use_smpl_targets:
                    canonical_pseudo_metrics = self._compute_canonical_joint_metrics(
                        pred_body_pose=predictions["pred_body_pose"],
                        pred_betas=predictions["pred_betas"],
                        target_body_pose=batch["target_body_pose"],
                        target_betas=batch["target_betas"],
                    )
                    metrics.update(
                        {
                            f"{stage}/canonical_pseudo_mpjpe": canonical_pseudo_metrics[
                                "mpjpe"
                            ],
                            f"{stage}/canonical_pseudo_pa_mpjpe": canonical_pseudo_metrics[
                                "pa_mpjpe"
                            ],
                        }
                    )
        if stage == "test":
            if use_joint_targets:
                joint_metrics = self._compute_external_joint_metrics(
                    pred_joints=pred_joints,
                    target_joints=target_joints,
                    joint_weight=joint_weight,
                    target_joint_root_index=batch.get("target_joint_root_index"),
                    target_camera_joints=batch.get("target_camera_joints"),
                    target_camera_joint_confidence=batch.get(
                        "target_camera_joint_confidence"
                    ),
                    view_aux=batch.get("view_aux"),
                )
            else:
                joint_metrics = self._compute_test_joint_metrics(
                    pred_body_pose=predictions["pred_body_pose"],
                    pred_betas=predictions["pred_betas"],
                    target_body_pose=batch["target_body_pose"],
                    target_betas=batch["target_betas"],
                    views_input=batch["views_input"],
                    view_aux=batch["view_aux"],
                )
            metrics.update(
                {
                    "test/mpjpe": joint_metrics["mpjpe"],
                    "test/pa_mpjpe": joint_metrics["pa_mpjpe"],
                }
            )
            if use_joint_targets:
                metrics.update(
                    {
                        "test/pck_150": joint_metrics["pck_150"],
                        "test/auc": joint_metrics["auc"],
                        "test/root_pck_150": joint_metrics["root_pck_150"],
                        "test/root_auc": joint_metrics["root_auc"],
                    }
                )
                if "target_camera_joints" in batch:
                    input_joint_metrics = self._compute_external_input_joint_metrics(
                        views_input=batch["views_input"],
                        view_aux=batch["view_aux"],
                        target_camera_joints=batch["target_camera_joints"],
                        target_camera_joint_confidence=batch.get(
                            "target_camera_joint_confidence"
                        ),
                        target_joint_smpl_indices=batch["target_joint_smpl_indices"],
                        target_joint_root_index=batch["target_joint_root_index"],
                    )
                    metrics.update(
                        {
                            "test/input_mpjpe": input_joint_metrics["mpjpe"],
                            "test/input_pa_mpjpe": input_joint_metrics["pa_mpjpe"],
                        }
                    )
                if use_smpl_targets:
                    canonical_pseudo_metrics = self._compute_canonical_joint_metrics(
                        pred_body_pose=predictions["pred_body_pose"],
                        pred_betas=predictions["pred_betas"],
                        target_body_pose=batch["target_body_pose"],
                        target_betas=batch["target_betas"],
                    )
                    metrics.update(
                        {
                            "test/canonical_pseudo_mpjpe": canonical_pseudo_metrics[
                                "mpjpe"
                            ],
                            "test/canonical_pseudo_pa_mpjpe": canonical_pseudo_metrics[
                                "pa_mpjpe"
                            ],
                        }
                    )
        return metrics

    @staticmethod
    def _batch_has_smpl_targets(
        batch: dict[str, torch.Tensor],
        *,
        use_joint_targets: bool,
    ) -> bool:
        if not use_joint_targets:
            return True
        target_smpl_valid = batch.get("target_smpl_valid")
        if target_smpl_valid is None:
            return False
        valid = target_smpl_valid.to(dtype=torch.bool)
        if bool(valid.any().item()) and not bool(valid.all().item()):
            raise ValueError(
                "Mixed batches with and without pseudo-SMPL targets are not supported"
            )
        return bool(valid.all().item())

    @staticmethod
    def _build_stage2_residuals(
        predictions: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        pose_residual_6d = predictions.get("stage2r_pose_residual_6d")
        if pose_residual_6d is None:
            pose_residual_6d = predictions["pred_pose_6d"] - predictions["init_pose_6d"]

        betas_residual = predictions.get("stage2r_betas_residual")
        if betas_residual is None:
            pred_betas = predictions.get("pred_betas")
            init_betas = predictions.get("init_betas")
            if pred_betas is not None and init_betas is not None:
                betas_residual = pred_betas - init_betas
        return pose_residual_6d, betas_residual

    def _zero_stage2_parameter_losses(
        self,
        reference: torch.Tensor,
        *,
        pose_residual_6d: torch.Tensor | None = None,
        betas_residual: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        zero = reference.sum() * 0.0
        if pose_residual_6d is not None and self.loss_config.pose_residual_weight > 0.0:
            pose_residual_loss = pose_residual_6d.square().mean()
        else:
            pose_residual_loss = zero
        if betas_residual is not None and self.loss_config.betas_residual_weight > 0.0:
            betas_residual_loss = betas_residual.square().mean()
        else:
            betas_residual_loss = zero
        total_loss = (
            self.loss_config.pose_residual_weight * pose_residual_loss
            + self.loss_config.betas_residual_weight * betas_residual_loss
        )
        return {
            "loss": total_loss,
            "loss_pose_6d": zero,
            "loss_betas": zero,
            "loss_init_pose_6d": zero,
            "loss_init_betas": zero,
            "loss_pose_residual": pose_residual_loss,
            "loss_betas_residual": betas_residual_loss,
        }

    def _compute_stage2_aux_losses(
        self,
        *,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        zero = predictions["pred_pose_6d"].new_zeros(())
        return {
            "stage2_aux_loss": zero,
            "stage2_aux_pose_6d": zero,
            "stage2_aux_betas": zero,
        }

    def _compute_triangulated_joint_losses(
        self,
        *,
        pred_joints: torch.Tensor,
        predictions: dict[str, torch.Tensor],
        target_joint_root_index: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        zero = pred_joints.new_zeros(())
        empty = {
            "triangulated_joint_loss": zero,
            "triangulated_pa_joint_loss": zero,
            "triangulated_mpjpe": zero,
            "triangulated_pa_mpjpe": zero,
        }
        triangulated_joints = predictions.get("triangulated_joints")
        if triangulated_joints is None:
            return empty
        triangulated_joints = triangulated_joints.to(
            device=pred_joints.device,
            dtype=pred_joints.dtype,
        )
        if tuple(triangulated_joints.shape) != tuple(pred_joints.shape):
            if (
                self.loss_config.triangulated_joint_weight > 0.0
                or self.loss_config.triangulated_pa_joint_weight > 0.0
            ):
                raise ValueError(
                    "triangulated_joints must match pred_joints shape when "
                    "triangulated consistency loss is enabled: "
                    f"{tuple(triangulated_joints.shape)} != {tuple(pred_joints.shape)}"
                )
            return empty
        triangulated_weight = predictions.get("triangulated_joint_weight")
        if triangulated_weight is None:
            weight = torch.ones(
                pred_joints.shape[:2],
                device=pred_joints.device,
                dtype=pred_joints.dtype,
            )
        else:
            weight = triangulated_weight.to(
                device=pred_joints.device,
                dtype=pred_joints.dtype,
            )
        weight = weight * (weight > 0.05).to(weight.dtype)
        valid_weight = weight.sum()
        if float(valid_weight.detach().item()) <= 0.0:
            return empty
        denom = valid_weight.clamp_min(1.0)
        delta = pred_joints - triangulated_joints
        per_joint_sq = delta.square().sum(dim=-1)
        joint_loss = (per_joint_sq * weight).sum() / denom
        mpjpe = (
            torch.linalg.norm(delta.detach(), dim=-1) * weight.detach()
        ).sum() / denom.detach()
        mpjpe = mpjpe * 1000.0
        root_index = (
            int(target_joint_root_index.reshape(-1)[0].item())
            if target_joint_root_index is not None
            else 0
        )
        root_pred = pred_joints - pred_joints[:, root_index : root_index + 1, :]
        root_target = (
            triangulated_joints
            - triangulated_joints[:, root_index : root_index + 1, :]
        )
        aligned_pred = self._weighted_similarity_align(
            root_pred.float(),
            root_target.float(),
            weight.float(),
        )
        pa_per_joint = torch.linalg.norm(aligned_pred - root_target.float(), dim=-1)
        pa_joint_loss = (pa_per_joint * weight.float()).sum() / denom.float()
        return {
            "triangulated_joint_loss": joint_loss,
            "triangulated_pa_joint_loss": pa_joint_loss,
            "triangulated_mpjpe": mpjpe,
            "triangulated_pa_mpjpe": pa_joint_loss.detach() * 1000.0,
        }

    def _compute_adapter_aux_joint_losses(
        self,
        *,
        predictions: dict[str, torch.Tensor],
        target_joints: torch.Tensor,
        joint_weight: torch.Tensor | None,
        target_joint_root_index: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        zero = predictions["pred_pose_6d"].new_zeros(())
        empty = {
            "adapter_aux_joint_loss": zero,
            "adapter_aux_pa_joint_loss": zero,
            "adapter_aux_mpjpe": zero,
            "adapter_aux_pa_mpjpe": zero,
        }
        aux_joints = predictions.get("adapter_aux_joints")
        if aux_joints is None:
            return empty
        aux_joints = aux_joints.to(
            device=target_joints.device,
            dtype=target_joints.dtype,
        )
        if tuple(aux_joints.shape) != tuple(target_joints.shape):
            if (
                self.loss_config.adapter_aux_joint_weight > 0.0
                or self.loss_config.adapter_aux_pa_joint_weight > 0.0
            ):
                raise ValueError(
                    "adapter_aux_joints must match target_joints shape when "
                    "adapter auxiliary loss is enabled: "
                    f"{tuple(aux_joints.shape)} != {tuple(target_joints.shape)}"
                )
            return empty
        if joint_weight is None:
            weight = torch.ones(
                aux_joints.shape[:2],
                device=aux_joints.device,
                dtype=aux_joints.dtype,
            )
        else:
            weight = joint_weight.to(device=aux_joints.device, dtype=aux_joints.dtype)
        aux_weight = predictions.get("adapter_aux_joint_weight")
        if aux_weight is not None:
            weight = weight * aux_weight.to(device=aux_joints.device, dtype=aux_joints.dtype)
        weight = weight * (weight > 0.05).to(weight.dtype)
        valid_weight = weight.sum()
        if float(valid_weight.detach().item()) <= 0.0:
            return empty
        denom = valid_weight.clamp_min(1.0)
        target = target_joints.to(device=aux_joints.device, dtype=aux_joints.dtype)
        delta = aux_joints - target
        per_joint_sq = delta.square().sum(dim=-1)
        aux_joint_loss = (per_joint_sq * weight).sum() / denom
        aux_mpjpe = (
            torch.linalg.norm(delta.detach(), dim=-1) * weight.detach()
        ).sum() / denom.detach()
        aux_mpjpe = aux_mpjpe * 1000.0

        root_index = (
            int(target_joint_root_index.reshape(-1)[0].item())
            if target_joint_root_index is not None
            else 0
        )
        root_aux = aux_joints - aux_joints[:, root_index : root_index + 1, :]
        root_target = target - target[:, root_index : root_index + 1, :]
        aligned_aux = self._weighted_similarity_align(
            root_aux.float(),
            root_target.float(),
            weight.float(),
        )
        aux_pa_per_joint = torch.linalg.norm(aligned_aux - root_target.float(), dim=-1)
        aux_pa_joint_loss = (aux_pa_per_joint * weight.float()).sum() / denom.float()
        return {
            "adapter_aux_joint_loss": aux_joint_loss,
            "adapter_aux_pa_joint_loss": aux_pa_joint_loss,
            "adapter_aux_mpjpe": aux_mpjpe,
            "adapter_aux_pa_mpjpe": aux_pa_joint_loss.detach() * 1000.0,
        }

    def _compute_pose_pca_coeff_loss(
        self,
        *,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        zero = predictions["pred_pose_6d"].new_zeros(())
        if self.loss_config.pose_pca_coeff_weight <= 0.0:
            return zero
        if "pose_pca_coeff" not in predictions or "target_pose_pca_coeff" not in batch:
            return zero
        pred_coeff = predictions["pose_pca_coeff"]
        target_coeff = batch["target_pose_pca_coeff"].to(
            device=pred_coeff.device,
            dtype=pred_coeff.dtype,
        )
        if target_coeff.shape[-1] < pred_coeff.shape[-1]:
            raise ValueError(
                "target_pose_pca_coeff has fewer components than pose_pca_coeff: "
                f"{target_coeff.shape[-1]} < {pred_coeff.shape[-1]}"
            )
        target_coeff = target_coeff[..., : pred_coeff.shape[-1]]
        valid = batch.get("target_pose_pca_coeff_valid")
        if valid is None:
            return F.mse_loss(pred_coeff, target_coeff)
        valid = valid.to(device=pred_coeff.device, dtype=torch.bool).reshape(-1)
        if not bool(valid.any().item()):
            return zero
        return F.mse_loss(pred_coeff[valid], target_coeff[valid])

    def _compute_pa_joint_loss(
        self,
        *,
        pred_joints: torch.Tensor,
        target_joints: torch.Tensor,
        joint_weight: torch.Tensor | None,
        target_joint_root_index: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.loss_config.pa_joint_weight <= 0.0:
            return pred_joints.new_zeros(())
        root_index = (
            int(target_joint_root_index.reshape(-1)[0].item())
            if target_joint_root_index is not None
            else 0
        )
        root_pred_joints = pred_joints - pred_joints[:, root_index : root_index + 1, :]
        root_target_joints = target_joints - target_joints[:, root_index : root_index + 1, :]
        if joint_weight is None:
            weight = torch.ones(
                root_pred_joints.shape[:2],
                device=root_pred_joints.device,
                dtype=root_pred_joints.dtype,
            )
        else:
            weight = joint_weight.to(
                device=root_pred_joints.device,
                dtype=root_pred_joints.dtype,
            )
            weight = weight * (weight > 0.05).to(weight.dtype)
        aligned_pred = self._weighted_similarity_align(
            root_pred_joints.float(),
            root_target_joints.float(),
            weight.float(),
        )
        per_joint = torch.linalg.norm(aligned_pred - root_target_joints.float(), dim=-1)
        return (per_joint * weight.float()).sum() / weight.float().sum().clamp_min(1.0)

    def _compute_rgb_residual_diagnostics(
        self,
        *,
        predictions: dict[str, torch.Tensor],
        batch: dict,
        stage: str,
    ) -> dict[str, torch.Tensor]:
        metrics: dict[str, torch.Tensor] = {}
        view_rgb_feature = batch.get("view_rgb_feature")
        if view_rgb_feature is not None:
            rgb_feature = view_rgb_feature.to(
                device=batch["views_input"].device,
                dtype=batch["views_input"].dtype,
            )
            rgb_norm = torch.linalg.vector_norm(rgb_feature, dim=-1)
            metrics.update(
                {
                    f"{stage}/rgb_feature_norm_mean": rgb_norm.mean(),
                    f"{stage}/rgb_feature_norm_std": rgb_norm.std(unbiased=False),
                }
            )
        view_image_joint_feature = batch.get("view_image_joint_feature")
        if view_image_joint_feature is not None:
            image_feature = view_image_joint_feature.to(
                device=batch["views_input"].device,
                dtype=batch["views_input"].dtype,
            )
            image_norm = torch.linalg.vector_norm(image_feature, dim=-1)
            metrics.update(
                {
                    f"{stage}/image_joint_feature_norm_mean": image_norm.mean(),
                    f"{stage}/image_joint_feature_norm_std": image_norm.std(
                        unbiased=False
                    ),
                }
            )
        image_gate = predictions.get("image_measurement_gate")
        if image_gate is not None:
            gate = image_gate.detach()
            metrics[f"{stage}/image_measurement_gate_mean"] = gate.mean()
            metrics[f"{stage}/image_measurement_gate_std"] = gate.std(unbiased=False)
        effective_confidence = predictions.get("image_measurement_effective_confidence")
        if effective_confidence is not None:
            confidence = effective_confidence.detach()
            metrics[f"{stage}/image_measurement_effective_confidence_mean"] = (
                confidence.mean()
            )
            metrics[f"{stage}/image_measurement_effective_confidence_std"] = (
                confidence.std(unbiased=False)
            )
        stage2r_pose_residual = predictions.get("stage2r_pose_residual_6d")
        if stage2r_pose_residual is not None:
            metrics[f"{stage}/stage2r_pose_residual_abs_mean"] = (
                stage2r_pose_residual.detach().abs().mean()
            )
            metrics[f"{stage}/stage2r_pose_residual_abs_max"] = (
                stage2r_pose_residual.detach().abs().amax()
            )
        for key in (
            "triangulated_joint_valid_ratio",
            "triangulated_joint_reprojection_error_px",
            "triangulated_joint_confidence_mean",
            "triangulated_joint_abs_mean",
            "triangulated_joint_root_valid_ratio",
            "triangulated_global_context_norm",
            "adapter_aux_joint_delta_abs_mean",
            "adapter_aux_joint_context_norm",
            "measurement_view_reweight_delta_abs_mean",
            "measurement_view_reweight_entropy",
            "measurement_view_reweight_residual_abs_mean",
            "projected_measurement_delta_px_abs_mean",
            "projected_measurement_sigma_px_mean",
            "projected_measurement_inlier_mean",
            "projected_measurement_confidence_mean",
            "projected_measurement_head_valid_ratio",
            "image_mask_context_norm",
            "segmentation_map_context_norm",
            "segmentation_map_sample_mask_mean",
            "segmentation_map_sample_distance_mean",
            "segmentation_map_valid_ratio",
        ):
            value = predictions.get(key)
            if value is not None:
                metrics[f"{stage}/{key}"] = value.detach()
        return metrics

    @staticmethod
    def _weighted_mean(
        values: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> torch.Tensor:
        if weights is None:
            return values.mean()
        weights = weights.to(device=values.device, dtype=values.dtype)
        return (values * weights).sum() / weights.sum().clamp_min(1.0)

    def _compute_camera_joint_losses(
        self,
        *,
        pred_joints: torch.Tensor,
        batch: dict[str, torch.Tensor],
        use_joint_targets: bool,
    ) -> dict[str, torch.Tensor]:
        zero = pred_joints.new_zeros(())
        if self.loss_config.camera_joint_weight <= 0.0:
            return {"camera_joint_loss": zero}
        if (
            not use_joint_targets
            or "target_camera_joints" not in batch
            or "view_aux" not in batch
        ):
            return {"camera_joint_loss": zero}
        view_aux = batch["view_aux"]
        if "input_global_orient" not in view_aux:
            raise KeyError(
                "camera_joint_weight requires view_aux['input_global_orient']"
            )
        target_camera_joints = batch["target_camera_joints"].to(
            device=pred_joints.device,
            dtype=pred_joints.dtype,
        )
        if target_camera_joints.ndim != 4 or target_camera_joints.shape[-1] != 3:
            raise ValueError(
                "target_camera_joints must have shape [B, V, J, 3], "
                f"got {tuple(target_camera_joints.shape)}"
            )
        batch_size, num_views, joint_count, _ = target_camera_joints.shape
        if pred_joints.shape[:2] != (batch_size, joint_count):
            raise ValueError(
                f"Expected pred_joints shape {(batch_size, joint_count, 3)}, "
                f"got {tuple(pred_joints.shape)}"
            )
        target_joint_root_index = batch.get("target_joint_root_index")
        root_index = (
            int(target_joint_root_index.reshape(-1)[0].item())
            if target_joint_root_index is not None
            else 0
        )
        root_pred_joints = pred_joints - pred_joints[:, root_index : root_index + 1, :]
        input_root_rotation = axis_angle_to_matrix(
            view_aux["input_global_orient"].to(
                device=pred_joints.device,
                dtype=pred_joints.dtype,
            )
        )
        pred_camera_joints = torch.matmul(
            root_pred_joints[:, None].unsqueeze(-2),
            input_root_rotation.transpose(-1, -2)[:, :, None],
        ).squeeze(-2)
        root_target_joints = (
            target_camera_joints
            - target_camera_joints[:, :, root_index : root_index + 1, :]
        )
        target_camera_joint_confidence = batch.get("target_camera_joint_confidence")
        if target_camera_joint_confidence is None:
            joint_weight = torch.ones(
                (batch_size, num_views, joint_count),
                device=pred_joints.device,
                dtype=pred_joints.dtype,
            )
        else:
            joint_weight = target_camera_joint_confidence.to(
                device=pred_joints.device,
                dtype=pred_joints.dtype,
            )
            joint_weight = joint_weight * (joint_weight > 0.05).to(joint_weight.dtype)
        per_joint = (
            pred_camera_joints.float() - root_target_joints.float()
        ).square().sum(dim=-1)
        camera_joint_loss = (
            per_joint * joint_weight.float()
        ).sum() / joint_weight.float().sum().clamp_min(1.0)
        return {"camera_joint_loss": camera_joint_loss}

    def _compute_camera_pa_joint_losses(
        self,
        *,
        pred_joints: torch.Tensor,
        batch: dict[str, torch.Tensor],
        use_joint_targets: bool,
    ) -> dict[str, torch.Tensor]:
        zero = pred_joints.new_zeros(())
        if self.loss_config.camera_pa_joint_weight <= 0.0:
            return {"camera_pa_joint_loss": zero}
        if (
            not use_joint_targets
            or "target_camera_joints" not in batch
            or "view_aux" not in batch
        ):
            return {"camera_pa_joint_loss": zero}
        view_aux = batch["view_aux"]
        if "input_global_orient" not in view_aux:
            raise KeyError(
                "camera_pa_joint_weight requires view_aux['input_global_orient']"
            )
        target_camera_joints = batch["target_camera_joints"].to(
            device=pred_joints.device,
            dtype=pred_joints.dtype,
        )
        if target_camera_joints.ndim != 4 or target_camera_joints.shape[-1] != 3:
            raise ValueError(
                "target_camera_joints must have shape [B, V, J, 3], "
                f"got {tuple(target_camera_joints.shape)}"
            )
        batch_size, num_views, joint_count, _ = target_camera_joints.shape
        if pred_joints.shape[:2] != (batch_size, joint_count):
            raise ValueError(
                f"Expected pred_joints shape {(batch_size, joint_count, 3)}, "
                f"got {tuple(pred_joints.shape)}"
            )
        target_joint_root_index = batch.get("target_joint_root_index")
        root_index = (
            int(target_joint_root_index.reshape(-1)[0].item())
            if target_joint_root_index is not None
            else 0
        )
        root_pred_joints = pred_joints - pred_joints[:, root_index : root_index + 1, :]
        input_root_rotation = axis_angle_to_matrix(
            view_aux["input_global_orient"].to(
                device=pred_joints.device,
                dtype=pred_joints.dtype,
            )
        )
        pred_camera_joints = torch.matmul(
            root_pred_joints[:, None].unsqueeze(-2),
            input_root_rotation.transpose(-1, -2)[:, :, None],
        ).squeeze(-2)
        root_target_joints = (
            target_camera_joints
            - target_camera_joints[:, :, root_index : root_index + 1, :]
        )
        flat_pred = pred_camera_joints.reshape(batch_size * num_views, joint_count, 3)
        flat_target = root_target_joints.reshape(batch_size * num_views, joint_count, 3)
        target_camera_joint_confidence = batch.get("target_camera_joint_confidence")
        if target_camera_joint_confidence is None:
            flat_weight = torch.ones(
                (batch_size * num_views, joint_count),
                device=pred_joints.device,
                dtype=pred_joints.dtype,
            )
        else:
            flat_weight = target_camera_joint_confidence.to(
                device=pred_joints.device,
                dtype=pred_joints.dtype,
            ).reshape(batch_size * num_views, joint_count)
            flat_weight = flat_weight * (flat_weight > 0.05).to(flat_weight.dtype)
        aligned_pred = self._weighted_similarity_align(
            flat_pred.float(),
            flat_target.float(),
            flat_weight.float(),
        )
        return {
            "camera_pa_joint_loss": self._weighted_mpjpe(
                aligned_pred,
                flat_target.float(),
                flat_weight.float(),
            )
        }

    def _compute_input_projection_losses(
        self,
        *,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        zero = predictions["pred_body_pose"].new_zeros(())
        if self.loss_config.input_projection_weight <= 0.0:
            return {
                "input_projection_loss": zero,
                "input_projection_error_px": zero,
                "input_projection_valid_ratio": zero,
            }
        view_aux = batch.get("view_aux")
        if view_aux is None:
            return {
                "input_projection_loss": zero,
                "input_projection_error_px": zero,
                "input_projection_valid_ratio": zero,
            }
        required_keys = {"cam_int", "image_size", "input_global_orient", "input_transl"}
        if not required_keys.issubset(view_aux):
            missing = sorted(required_keys.difference(view_aux))
            raise KeyError(
                "input_projection_weight requires view_aux fields: "
                f"{', '.join(missing)}"
            )

        pred_camera_joints, input_camera_joints = self._build_input_projection_joints(
            pred_body_pose=predictions["pred_body_pose"],
            pred_betas=predictions["pred_betas"],
            views_input=batch["views_input"],
            view_aux=view_aux,
            target_joint_smpl_indices=batch.get("target_joint_smpl_indices"),
        )
        cam_int = view_aux["cam_int"].to(
            device=pred_camera_joints.device,
            dtype=pred_camera_joints.dtype,
        )
        image_size = view_aux["image_size"].to(
            device=pred_camera_joints.device,
            dtype=pred_camera_joints.dtype,
        )
        pred_uv, pred_depth = self._project_camera_joints(
            pred_camera_joints,
            intrinsics=cam_int,
        )
        input_uv, input_depth = self._project_camera_joints(
            input_camera_joints.detach(),
            intrinsics=cam_int,
        )
        valid = self._input_projection_valid_mask(
            pred_uv=pred_uv,
            pred_depth=pred_depth,
            input_uv=input_uv,
            input_depth=input_depth,
            image_size=image_size,
        )
        valid_float = valid.to(dtype=pred_uv.dtype)
        focal_scale = (
            0.5 * (cam_int[..., 0, 0].abs() + cam_int[..., 1, 1].abs())
        ).clamp_min(1.0)
        normalized_delta = (pred_uv - input_uv) / focal_scale[..., None, None]
        normalized_delta = torch.where(
            valid[..., None],
            normalized_delta,
            torch.zeros_like(normalized_delta),
        )
        eps = float(self.loss_config.input_projection_charbonnier_eps)
        per_joint_loss = torch.sqrt(
            normalized_delta.square().sum(dim=-1) + eps * eps
        ) - eps
        denom = valid_float.sum().clamp_min(1.0)
        projection_loss = (per_joint_loss * valid_float).sum() / denom
        pixel_delta = pred_uv.detach() - input_uv.detach()
        pixel_delta = torch.where(
            valid[..., None],
            pixel_delta,
            torch.zeros_like(pixel_delta),
        )
        pixel_error = torch.linalg.norm(pixel_delta, dim=-1)
        pixel_error = (pixel_error * valid_float).sum() / denom
        valid_ratio = valid_float.mean()
        return {
            "input_projection_loss": projection_loss,
            "input_projection_error_px": pixel_error,
            "input_projection_valid_ratio": valid_ratio,
        }

    def _compute_gt_projection_losses(
        self,
        *,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        zero = predictions["pred_body_pose"].new_zeros(())
        if self.loss_config.gt_projection_weight <= 0.0:
            return {
                "gt_projection_loss": zero,
                "gt_projection_error_px": zero,
                "gt_projection_valid_ratio": zero,
            }
        if "target_joints_2d" not in batch:
            return {
                "gt_projection_loss": zero,
                "gt_projection_error_px": zero,
                "gt_projection_valid_ratio": zero,
            }
        view_aux = batch.get("view_aux")
        if view_aux is None:
            return {
                "gt_projection_loss": zero,
                "gt_projection_error_px": zero,
                "gt_projection_valid_ratio": zero,
            }
        required_keys = {"cam_int", "image_size", "input_global_orient", "input_transl"}
        if not required_keys.issubset(view_aux):
            missing = sorted(required_keys.difference(view_aux))
            raise KeyError(
                "gt_projection_weight requires view_aux fields: "
                f"{', '.join(missing)}"
            )

        pred_camera_joints = self._build_pred_projection_joints(
            pred_body_pose=predictions["pred_body_pose"],
            pred_betas=predictions["pred_betas"],
            views_input=batch["views_input"],
            view_aux=view_aux,
            target_joint_smpl_indices=batch.get("target_joint_smpl_indices"),
        )
        cam_int = view_aux["cam_int"].to(
            device=pred_camera_joints.device,
            dtype=pred_camera_joints.dtype,
        )
        image_size = view_aux["image_size"].to(
            device=pred_camera_joints.device,
            dtype=pred_camera_joints.dtype,
        )
        target_uv = batch["target_joints_2d"].to(
            device=pred_camera_joints.device,
            dtype=pred_camera_joints.dtype,
        )
        target_confidence = batch.get("target_joints_2d_confidence")
        if target_confidence is None:
            target_confidence = torch.ones(
                target_uv.shape[:-1],
                device=target_uv.device,
                dtype=target_uv.dtype,
            )
        else:
            target_confidence = target_confidence.to(
                device=target_uv.device,
                dtype=target_uv.dtype,
            )
        pred_uv, pred_depth = self._project_camera_joints(
            pred_camera_joints,
            intrinsics=cam_int,
        )
        valid = self._gt_projection_valid_mask(
            pred_uv=pred_uv,
            pred_depth=pred_depth,
            target_uv=target_uv,
            target_confidence=target_confidence,
            image_size=image_size,
        )
        valid_float = valid.to(dtype=pred_uv.dtype)
        focal_scale = (
            0.5 * (cam_int[..., 0, 0].abs() + cam_int[..., 1, 1].abs())
        ).clamp_min(1.0)
        normalized_delta = (pred_uv - target_uv) / focal_scale[..., None, None]
        normalized_delta = torch.where(
            valid[..., None],
            normalized_delta,
            torch.zeros_like(normalized_delta),
        )
        eps = float(self.loss_config.gt_projection_charbonnier_eps)
        per_joint_loss = torch.sqrt(
            normalized_delta.square().sum(dim=-1) + eps * eps
        ) - eps
        denom = valid_float.sum().clamp_min(1.0)
        projection_loss = (per_joint_loss * valid_float).sum() / denom
        pixel_delta = pred_uv.detach() - target_uv.detach()
        pixel_delta = torch.where(
            valid[..., None],
            pixel_delta,
            torch.zeros_like(pixel_delta),
        )
        pixel_error = torch.linalg.norm(pixel_delta, dim=-1)
        pixel_error = (pixel_error * valid_float).sum() / denom
        valid_ratio = valid_float.mean()
        return {
            "gt_projection_loss": projection_loss,
            "gt_projection_error_px": pixel_error,
            "gt_projection_valid_ratio": valid_ratio,
        }

    def _compute_image_projection_losses(
        self,
        *,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        zero = predictions["pred_body_pose"].new_zeros(())
        if self.loss_config.image_projection_weight <= 0.0:
            return {
                "image_projection_loss": zero,
                "image_projection_error_px": zero,
                "image_projection_valid_ratio": zero,
            }
        if "view_image_joint_uv" not in batch:
            return {
                "image_projection_loss": zero,
                "image_projection_error_px": zero,
                "image_projection_valid_ratio": zero,
            }
        view_aux = batch.get("view_aux")
        if view_aux is None:
            return {
                "image_projection_loss": zero,
                "image_projection_error_px": zero,
                "image_projection_valid_ratio": zero,
            }
        required_keys = {"cam_int", "image_size", "input_global_orient", "input_transl"}
        if not required_keys.issubset(view_aux):
            missing = sorted(required_keys.difference(view_aux))
            raise KeyError(
                "image_projection_weight requires view_aux fields: "
                f"{', '.join(missing)}"
            )

        pred_camera_joints = self._build_pred_projection_joints(
            pred_body_pose=predictions["pred_body_pose"],
            pred_betas=predictions["pred_betas"],
            views_input=batch["views_input"],
            view_aux=view_aux,
            target_joint_smpl_indices=batch.get("target_joint_smpl_indices"),
        )
        cam_int = view_aux["cam_int"].to(
            device=pred_camera_joints.device,
            dtype=pred_camera_joints.dtype,
        )
        image_size = view_aux["image_size"].to(
            device=pred_camera_joints.device,
            dtype=pred_camera_joints.dtype,
        )
        target_uv = batch["view_image_joint_uv"].to(
            device=pred_camera_joints.device,
            dtype=pred_camera_joints.dtype,
        )
        target_confidence = batch.get("view_image_joint_confidence")
        if target_confidence is None:
            target_confidence = torch.ones(
                target_uv.shape[:-1],
                device=target_uv.device,
                dtype=target_uv.dtype,
            )
        else:
            target_confidence = target_confidence.to(
                device=target_uv.device,
                dtype=target_uv.dtype,
            )
        target_valid = batch.get("view_image_joint_valid")
        if target_valid is None:
            target_valid = torch.ones(
                target_uv.shape[:-1],
                device=target_uv.device,
                dtype=torch.bool,
            )
        else:
            target_valid = target_valid.to(device=target_uv.device, dtype=torch.bool)

        pred_uv, pred_depth = self._project_camera_joints(
            pred_camera_joints,
            intrinsics=cam_int,
        )
        valid = self._image_projection_valid_mask(
            pred_uv=pred_uv,
            pred_depth=pred_depth,
            target_uv=target_uv,
            target_confidence=target_confidence,
            target_valid=target_valid,
            image_size=image_size,
        )
        valid_float = valid.to(dtype=pred_uv.dtype)
        focal_scale = (
            0.5 * (cam_int[..., 0, 0].abs() + cam_int[..., 1, 1].abs())
        ).clamp_min(1.0)
        normalized_delta = (pred_uv - target_uv) / focal_scale[..., None, None]
        normalized_delta = torch.where(
            valid[..., None],
            normalized_delta,
            torch.zeros_like(normalized_delta),
        )
        eps = float(self.loss_config.image_projection_charbonnier_eps)
        per_joint_loss = torch.sqrt(
            normalized_delta.square().sum(dim=-1) + eps * eps
        ) - eps
        denom = valid_float.sum().clamp_min(1.0)
        projection_loss = (per_joint_loss * valid_float).sum() / denom
        pixel_delta = pred_uv.detach() - target_uv.detach()
        pixel_delta = torch.where(
            valid[..., None],
            pixel_delta,
            torch.zeros_like(pixel_delta),
        )
        pixel_error = torch.linalg.norm(pixel_delta, dim=-1)
        pixel_error = (pixel_error * valid_float).sum() / denom
        valid_ratio = valid_float.mean()
        return {
            "image_projection_loss": projection_loss,
            "image_projection_error_px": pixel_error,
            "image_projection_valid_ratio": valid_ratio,
        }

    def _compute_projected_measurement_losses(
        self,
        *,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        zero = predictions["pred_body_pose"].new_zeros(())
        empty = {
            "projected_measurement_loss": zero,
            "projected_measurement_error_px": zero,
            "projected_measurement_raw_error_px": zero,
            "projected_measurement_valid_ratio": zero,
            "projected_measurement_consistency_loss": zero,
            "projected_measurement_consistency_error_px": zero,
            "projected_measurement_consistency_valid_ratio": zero,
        }
        corrected_uv = predictions.get("measurement_corrected_uv")
        if corrected_uv is None:
            return empty
        corrected_uv = corrected_uv.to(
            device=predictions["pred_body_pose"].device,
            dtype=predictions["pred_body_pose"].dtype,
        )
        log_sigma = predictions.get("measurement_log_sigma")
        if log_sigma is None:
            log_sigma = corrected_uv.new_zeros(corrected_uv.shape[:-1])
        else:
            log_sigma = log_sigma.to(
                device=corrected_uv.device,
                dtype=corrected_uv.dtype,
            )
        measurement_valid = predictions.get("measurement_valid")
        if measurement_valid is None:
            measurement_valid_mask = torch.ones(
                corrected_uv.shape[:-1],
                device=corrected_uv.device,
                dtype=torch.bool,
            )
        else:
            measurement_valid_mask = (
                measurement_valid.to(device=corrected_uv.device) > 0.5
            )

        measurement_loss = zero
        measurement_error_px = zero
        raw_error_px = zero
        measurement_valid_ratio = zero
        view_aux = batch.get("view_aux")
        if (
            self.loss_config.projected_measurement_weight > 0.0
            and "target_joints_2d" not in batch
        ):
            raise KeyError(
                "projected_measurement_weight requires batch['target_joints_2d']"
            )
        if (
            self.loss_config.projected_measurement_weight > 0.0
            and view_aux is None
        ):
            raise KeyError("projected_measurement_weight requires batch['view_aux']")
        if "target_joints_2d" in batch and view_aux is not None:
            if "image_size" not in view_aux:
                raise KeyError(
                    "projected measurement supervision requires view_aux['image_size']"
                )
            target_uv = batch["target_joints_2d"].to(
                device=corrected_uv.device,
                dtype=corrected_uv.dtype,
            )
            if tuple(target_uv.shape) != tuple(corrected_uv.shape):
                raise ValueError(
                    "target_joints_2d must match measurement_corrected_uv shape: "
                    f"{tuple(target_uv.shape)} != {tuple(corrected_uv.shape)}"
                )
            target_confidence = batch.get("target_joints_2d_confidence")
            if target_confidence is None:
                target_confidence = torch.ones(
                    target_uv.shape[:-1],
                    device=target_uv.device,
                    dtype=target_uv.dtype,
                )
            else:
                target_confidence = target_confidence.to(
                    device=target_uv.device,
                    dtype=target_uv.dtype,
                )
            image_size = view_aux["image_size"].to(
                device=target_uv.device,
                dtype=target_uv.dtype,
            )
            valid = self._projected_measurement_valid_mask(
                source_uv=corrected_uv,
                target_uv=target_uv,
                target_confidence=target_confidence,
                base_valid=measurement_valid_mask,
                image_size=image_size,
            )
            valid_float = valid.to(dtype=corrected_uv.dtype)
            denom = valid_float.sum().clamp_min(1.0)
            delta_px = corrected_uv - target_uv
            delta_px = torch.where(
                valid[..., None],
                delta_px,
                torch.zeros_like(delta_px),
            )
            eps = float(self.loss_config.projected_measurement_charbonnier_eps)
            per_joint_loss = torch.sqrt(
                delta_px.square().sum(dim=-1) + eps * eps
            ) - eps
            precision = torch.exp(-log_sigma).clamp(max=100.0)
            per_joint_loss = (
                precision * per_joint_loss
                + float(self.loss_config.projected_measurement_uncertainty_weight)
                * log_sigma
            )
            measurement_loss = (per_joint_loss * valid_float).sum() / denom
            measurement_error = torch.linalg.norm(delta_px.detach(), dim=-1)
            measurement_error_px = (
                measurement_error * valid_float.detach()
            ).sum() / denom.detach()
            raw_uv = predictions.get("measurement_measured_uv")
            if raw_uv is not None:
                raw_uv = raw_uv.to(device=target_uv.device, dtype=target_uv.dtype)
                raw_delta = raw_uv.detach() - target_uv.detach()
                raw_delta = torch.where(
                    valid[..., None],
                    raw_delta,
                    torch.zeros_like(raw_delta),
                )
                raw_error = torch.linalg.norm(raw_delta, dim=-1)
                raw_error_px = (raw_error * valid_float.detach()).sum() / denom.detach()
            measurement_valid_ratio = valid_float.mean()

        consistency_loss = zero
        consistency_error_px = zero
        consistency_valid_ratio = zero
        if self.loss_config.projected_measurement_consistency_weight > 0.0:
            if view_aux is None:
                raise KeyError(
                    "projected_measurement_consistency_weight requires batch['view_aux']"
                )
            required_keys = {"cam_int", "image_size", "input_global_orient", "input_transl"}
            if not required_keys.issubset(view_aux):
                missing = sorted(required_keys.difference(view_aux))
                raise KeyError(
                    "projected_measurement_consistency_weight requires view_aux fields: "
                    f"{', '.join(missing)}"
                )
            pred_camera_joints = self._build_pred_projection_joints(
                pred_body_pose=predictions["pred_body_pose"],
                pred_betas=predictions["pred_betas"],
                views_input=batch["views_input"],
                view_aux=view_aux,
                target_joint_smpl_indices=batch.get("target_joint_smpl_indices"),
            )
            cam_int = view_aux["cam_int"].to(
                device=pred_camera_joints.device,
                dtype=pred_camera_joints.dtype,
            )
            image_size = view_aux["image_size"].to(
                device=pred_camera_joints.device,
                dtype=pred_camera_joints.dtype,
            )
            pred_uv, pred_depth = self._project_camera_joints(
                pred_camera_joints,
                intrinsics=cam_int,
            )
            measurement_confidence = predictions.get("measurement_confidence")
            if measurement_confidence is None:
                measurement_confidence = torch.ones(
                    corrected_uv.shape[:-1],
                    device=corrected_uv.device,
                    dtype=corrected_uv.dtype,
                )
            else:
                measurement_confidence = measurement_confidence.to(
                    device=corrected_uv.device,
                    dtype=corrected_uv.dtype,
                )
            valid = self._projected_measurement_consistency_valid_mask(
                pred_uv=pred_uv,
                pred_depth=pred_depth,
                corrected_uv=corrected_uv,
                measurement_confidence=measurement_confidence,
                base_valid=measurement_valid_mask,
                image_size=image_size,
            )
            weight = measurement_confidence.clamp_min(0.0) * valid.to(
                dtype=corrected_uv.dtype
            )
            denom = weight.sum().clamp_min(1.0)
            delta_px = pred_uv - corrected_uv
            delta_px = torch.where(
                valid[..., None],
                delta_px,
                torch.zeros_like(delta_px),
            )
            eps = float(
                self.loss_config.projected_measurement_consistency_charbonnier_eps
            )
            per_joint_loss = torch.sqrt(
                delta_px.square().sum(dim=-1) + eps * eps
            ) - eps
            precision = torch.exp(-log_sigma).clamp(max=100.0)
            consistency_loss = (per_joint_loss * precision * weight).sum() / denom
            consistency_error = torch.linalg.norm(delta_px.detach(), dim=-1)
            consistency_error_px = (
                consistency_error * valid.to(dtype=corrected_uv.dtype)
            ).sum() / valid.to(dtype=corrected_uv.dtype).sum().clamp_min(1.0)
            consistency_valid_ratio = valid.to(dtype=corrected_uv.dtype).mean()

        return {
            "projected_measurement_loss": measurement_loss,
            "projected_measurement_error_px": measurement_error_px,
            "projected_measurement_raw_error_px": raw_error_px,
            "projected_measurement_valid_ratio": measurement_valid_ratio,
            "projected_measurement_consistency_loss": consistency_loss,
            "projected_measurement_consistency_error_px": consistency_error_px,
            "projected_measurement_consistency_valid_ratio": consistency_valid_ratio,
        }

    def _compute_mask_supervision_losses(
        self,
        *,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        zero = predictions["pred_body_pose"].new_zeros(())
        if self.loss_config.mask_supervision_weight <= 0.0:
            return {
                "mask_supervision_loss": zero,
                "mask_supervision_distance_loss": zero,
                "mask_supervision_bbox_loss": zero,
                "mask_supervision_inside_ratio": zero,
                "mask_supervision_valid_ratio": zero,
            }
        required_batch_keys = {
            "view_segmentation_mask",
            "view_segmentation_distance",
            "view_segmentation_bbox",
            "view_segmentation_valid",
        }
        missing_batch_keys = sorted(required_batch_keys.difference(batch))
        if missing_batch_keys:
            raise KeyError(
                "mask_supervision_weight requires batch fields: "
                f"{', '.join(missing_batch_keys)}"
            )
        view_aux = batch.get("view_aux")
        if view_aux is None:
            raise KeyError("mask_supervision_weight requires batch['view_aux']")
        required_aux_keys = {"cam_int", "image_size", "input_global_orient", "input_transl"}
        if not required_aux_keys.issubset(view_aux):
            missing = sorted(required_aux_keys.difference(view_aux))
            raise KeyError(
                "mask_supervision_weight requires view_aux fields: "
                f"{', '.join(missing)}"
            )

        vertices = self._build_pred_projection_vertices(
            pred_body_pose=predictions["pred_body_pose"],
            pred_betas=predictions["pred_betas"],
            views_input=batch["views_input"],
            view_aux=view_aux,
        )
        cam_int = view_aux["cam_int"].to(device=vertices.device, dtype=vertices.dtype)
        image_size = view_aux["image_size"].to(device=vertices.device, dtype=vertices.dtype)
        vertex_uv, vertex_depth = self._project_camera_joints(vertices, intrinsics=cam_int)

        mask_valid = batch["view_segmentation_valid"].to(device=vertices.device, dtype=torch.bool)
        if mask_valid.ndim == 3 and mask_valid.shape[-1] == 1:
            mask_valid = mask_valid.squeeze(-1)
        if mask_valid.ndim != 2 or mask_valid.shape != vertex_uv.shape[:2]:
            raise ValueError(
                "view_segmentation_valid must have shape [batch, views], got "
                f"{tuple(mask_valid.shape)}"
            )
        vertex_valid = self._mask_supervision_valid_mask(
            vertex_uv=vertex_uv,
            vertex_depth=vertex_depth,
            image_size=image_size,
            mask_valid=mask_valid,
        )

        vertex_indices = self._mask_supervision_vertex_indices(
            num_vertices=vertex_uv.shape[2],
            device=vertex_uv.device,
        )
        sampled_uv = vertex_uv.index_select(2, vertex_indices)
        sampled_valid = vertex_valid.index_select(2, vertex_indices)
        distance_map = batch["view_segmentation_distance"].to(
            device=vertices.device,
            dtype=vertices.dtype,
        )
        mask_map = batch["view_segmentation_mask"].to(
            device=vertices.device,
            dtype=vertices.dtype,
        )
        sampled_distance = self._sample_view_map(
            distance_map,
            uv=sampled_uv,
            image_size=image_size,
        )
        sampled_mask = self._sample_view_map(
            mask_map,
            uv=sampled_uv,
            image_size=image_size,
        ).clamp(0.0, 1.0)
        sampled_valid_float = sampled_valid.to(dtype=vertices.dtype)
        denom = sampled_valid_float.sum().clamp_min(1.0)
        distance_loss = (sampled_distance * sampled_valid_float).sum() / denom
        inside_ratio = (sampled_mask * sampled_valid_float).sum() / denom
        valid_ratio = sampled_valid_float.mean()

        bbox_loss = self._compute_mask_bbox_loss(
            vertex_uv=vertex_uv,
            vertex_valid=vertex_valid,
            target_bbox=batch["view_segmentation_bbox"].to(
                device=vertices.device,
                dtype=vertices.dtype,
            ),
            image_size=image_size,
            mask_valid=mask_valid,
        )
        total = (
            distance_loss
            + float(self.loss_config.mask_supervision_bbox_weight) * bbox_loss
        )
        return {
            "mask_supervision_loss": total,
            "mask_supervision_distance_loss": distance_loss,
            "mask_supervision_bbox_loss": bbox_loss,
            "mask_supervision_inside_ratio": inside_ratio.detach(),
            "mask_supervision_valid_ratio": valid_ratio.detach(),
        }

    def _build_pred_projection_vertices(
        self,
        *,
        pred_body_pose: torch.Tensor,
        pred_betas: torch.Tensor,
        views_input: torch.Tensor,
        view_aux: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        batch_size, num_views = views_input.shape[:2]
        camera_global_orient = view_aux["input_global_orient"].to(
            device=pred_body_pose.device,
            dtype=pred_body_pose.dtype,
        ).reshape(batch_size * num_views, 3)
        camera_transl = view_aux["input_transl"].to(
            device=pred_body_pose.device,
            dtype=pred_body_pose.dtype,
        ).reshape(batch_size * num_views, 3)
        pred_output = self._build_smpl_output(
            body_pose=pred_body_pose[:, None, :]
            .expand(batch_size, num_views, -1)
            .reshape(batch_size * num_views, -1),
            betas=pred_betas[:, None, :]
            .expand(batch_size, num_views, -1)
            .reshape(batch_size * num_views, -1),
            global_orient=camera_global_orient,
            transl=camera_transl,
        )
        vertices = pred_output.vertices
        return vertices.reshape(batch_size, num_views, vertices.shape[1], 3)

    def _mask_supervision_vertex_indices(
        self,
        *,
        num_vertices: int,
        device: torch.device,
    ) -> torch.Tensor:
        sample_count = int(self.loss_config.mask_supervision_sample_count)
        if sample_count <= 0 or sample_count >= num_vertices:
            return torch.arange(num_vertices, device=device, dtype=torch.long)
        cache_key = ("mask_supervision_vertex_indices", num_vertices, sample_count, device)
        cached = self._runtime_cache.get("mask_supervision_vertex_indices")
        if (
            isinstance(cached, tuple)
            and len(cached) == 4
            and cached[0] == num_vertices
            and cached[1] == sample_count
            and cached[2] == device
            and isinstance(cached[3], torch.Tensor)
        ):
            return cached[3]
        del cache_key
        indices = torch.linspace(
            0,
            num_vertices - 1,
            steps=sample_count,
            device=device,
            dtype=torch.float32,
        ).round().to(dtype=torch.long)
        indices = torch.unique(indices, sorted=True)
        self._runtime_cache["mask_supervision_vertex_indices"] = (
            num_vertices,
            sample_count,
            device,
            indices,
        )
        return indices

    def _mask_supervision_valid_mask(
        self,
        *,
        vertex_uv: torch.Tensor,
        vertex_depth: torch.Tensor,
        image_size: torch.Tensor,
        mask_valid: torch.Tensor,
    ) -> torch.Tensor:
        min_depth = float(self.loss_config.mask_supervision_min_depth)
        border = float(self.loss_config.mask_supervision_border_px)
        image_width = image_size[..., 0].clamp_min(1.0)[..., None]
        image_height = image_size[..., 1].clamp_min(1.0)[..., None]
        return (
            mask_valid[..., None]
            & torch.isfinite(vertex_uv).all(dim=-1)
            & (vertex_depth > min_depth)
            & (vertex_uv[..., 0] >= border)
            & (vertex_uv[..., 0] <= image_width - 1.0 - border)
            & (vertex_uv[..., 1] >= border)
            & (vertex_uv[..., 1] <= image_height - 1.0 - border)
        )

    @staticmethod
    def _sample_view_map(
        value_map: torch.Tensor,
        *,
        uv: torch.Tensor,
        image_size: torch.Tensor,
    ) -> torch.Tensor:
        if value_map.ndim != 4:
            raise ValueError(f"Expected value_map shape [B, V, H, W], got {tuple(value_map.shape)}")
        batch_size, num_views, height, width = value_map.shape
        if uv.shape[:2] != (batch_size, num_views) or uv.shape[-1] != 2:
            raise ValueError(
                "Expected uv shape [B, V, N, 2] matching value_map, got "
                f"{tuple(uv.shape)}"
            )
        image_width = image_size[..., 0].clamp_min(2.0)[..., None]
        image_height = image_size[..., 1].clamp_min(2.0)[..., None]
        grid_x = 2.0 * uv[..., 0] / (image_width - 1.0) - 1.0
        grid_y = 2.0 * uv[..., 1] / (image_height - 1.0) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1)
        flat_map = value_map.reshape(batch_size * num_views, 1, height, width)
        flat_grid = grid.reshape(batch_size * num_views, uv.shape[2], 1, 2)
        sampled = F.grid_sample(
            flat_map,
            flat_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.reshape(batch_size, num_views, uv.shape[2])

    @staticmethod
    def _compute_mask_bbox_loss(
        *,
        vertex_uv: torch.Tensor,
        vertex_valid: torch.Tensor,
        target_bbox: torch.Tensor,
        image_size: torch.Tensor,
        mask_valid: torch.Tensor,
    ) -> torch.Tensor:
        if target_bbox.shape != vertex_uv.shape[:2] + (4,):
            raise ValueError(
                "view_segmentation_bbox must have shape [B, V, 4], got "
                f"{tuple(target_bbox.shape)}"
            )
        image_width = image_size[..., 0].clamp_min(2.0)[..., None]
        image_height = image_size[..., 1].clamp_min(2.0)[..., None]
        x_norm = vertex_uv[..., 0] / (image_width - 1.0)
        y_norm = vertex_uv[..., 1] / (image_height - 1.0)
        valid = vertex_valid & mask_valid[..., None]
        large = torch.full_like(x_norm, 1.0e4)
        x0 = torch.where(valid, x_norm, large).amin(dim=-1)
        y0 = torch.where(valid, y_norm, large).amin(dim=-1)
        x1 = torch.where(valid, x_norm, -large).amax(dim=-1)
        y1 = torch.where(valid, y_norm, -large).amax(dim=-1)
        pred_bbox = torch.stack((x0, y0, x1, y1), dim=-1).clamp(0.0, 1.0)
        has_valid = valid.any(dim=-1) & mask_valid
        has_valid_float = has_valid.to(dtype=vertex_uv.dtype)
        per_view = (pred_bbox - target_bbox.clamp(0.0, 1.0)).abs().mean(dim=-1)
        return (per_view * has_valid_float).sum() / has_valid_float.sum().clamp_min(1.0)

    def _build_pred_projection_joints(
        self,
        *,
        pred_body_pose: torch.Tensor,
        pred_betas: torch.Tensor,
        views_input: torch.Tensor,
        view_aux: dict[str, torch.Tensor],
        target_joint_smpl_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, num_views = views_input.shape[:2]
        camera_global_orient = view_aux["input_global_orient"].to(
            device=pred_body_pose.device,
            dtype=pred_body_pose.dtype,
        ).reshape(batch_size * num_views, 3)
        camera_transl = view_aux["input_transl"].to(
            device=pred_body_pose.device,
            dtype=pred_body_pose.dtype,
        ).reshape(batch_size * num_views, 3)
        pred_output = self._build_smpl_output(
            body_pose=pred_body_pose[:, None, :]
            .expand(batch_size, num_views, -1)
            .reshape(batch_size * num_views, -1),
            betas=pred_betas[:, None, :]
            .expand(batch_size, num_views, -1)
            .reshape(batch_size * num_views, -1),
            global_orient=camera_global_orient,
            transl=camera_transl,
        )
        smpl_indices = self._resolve_projection_smpl_indices(
            target_joint_smpl_indices=target_joint_smpl_indices,
            device=pred_body_pose.device,
        )
        pred_joints = self._select_external_pred_joints(
            smpl_output=pred_output,
            smpl_indices=smpl_indices,
        )
        return pred_joints.reshape(batch_size, num_views, pred_joints.shape[1], 3)

    def _build_input_projection_joints(
        self,
        *,
        pred_body_pose: torch.Tensor,
        pred_betas: torch.Tensor,
        views_input: torch.Tensor,
        view_aux: dict[str, torch.Tensor],
        target_joint_smpl_indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_views = views_input.shape[:2]
        pose_6d_dim = self.model_config.pose_6d_dim
        input_pose_6d = views_input[..., :pose_6d_dim].reshape(
            batch_size,
            num_views,
            self.model_config.num_joints,
            6,
        )
        input_body_pose = rotation_6d_to_axis_angle(
            input_pose_6d.reshape(-1, self.model_config.num_joints, 6)
        ).reshape(batch_size * num_views, -1)
        input_betas = views_input[..., pose_6d_dim:].reshape(batch_size * num_views, -1)
        camera_global_orient = view_aux["input_global_orient"].to(
            device=pred_body_pose.device,
            dtype=pred_body_pose.dtype,
        ).reshape(batch_size * num_views, 3)
        camera_transl = view_aux["input_transl"].to(
            device=pred_body_pose.device,
            dtype=pred_body_pose.dtype,
        ).reshape(batch_size * num_views, 3)

        pred_output = self._build_smpl_output(
            body_pose=pred_body_pose[:, None, :]
            .expand(batch_size, num_views, -1)
            .reshape(batch_size * num_views, -1),
            betas=pred_betas[:, None, :]
            .expand(batch_size, num_views, -1)
            .reshape(batch_size * num_views, -1),
            global_orient=camera_global_orient,
            transl=camera_transl,
        )
        input_output = self._build_smpl_output(
            body_pose=input_body_pose.to(
                device=pred_body_pose.device,
                dtype=pred_body_pose.dtype,
            ),
            betas=input_betas.to(device=pred_body_pose.device, dtype=pred_body_pose.dtype),
            global_orient=camera_global_orient,
            transl=camera_transl,
        )
        smpl_indices = self._resolve_projection_smpl_indices(
            target_joint_smpl_indices=target_joint_smpl_indices,
            device=pred_body_pose.device,
        )
        pred_joints = self._select_external_pred_joints(
            smpl_output=pred_output,
            smpl_indices=smpl_indices,
        )
        input_joints = self._select_external_pred_joints(
            smpl_output=input_output,
            smpl_indices=smpl_indices,
        )
        joint_count = pred_joints.shape[1]
        return (
            pred_joints.reshape(batch_size, num_views, joint_count, 3),
            input_joints.reshape(batch_size, num_views, joint_count, 3),
        )

    @staticmethod
    def _project_camera_joints(
        joints: torch.Tensor,
        *,
        intrinsics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        depth = joints[..., 2].clamp_min(1e-6)
        fx = intrinsics[..., 0, 0][..., None]
        fy = intrinsics[..., 1, 1][..., None]
        cx = intrinsics[..., 0, 2][..., None]
        cy = intrinsics[..., 1, 2][..., None]
        u_coord = fx * joints[..., 0] / depth + cx
        v_coord = fy * joints[..., 1] / depth + cy
        return torch.stack((u_coord, v_coord), dim=-1), joints[..., 2]

    def _input_projection_valid_mask(
        self,
        *,
        pred_uv: torch.Tensor,
        pred_depth: torch.Tensor,
        input_uv: torch.Tensor,
        input_depth: torch.Tensor,
        image_size: torch.Tensor,
    ) -> torch.Tensor:
        min_depth = float(self.loss_config.input_projection_min_depth)
        border = float(self.loss_config.input_projection_border_px)
        image_width = image_size[..., 0].clamp_min(1.0)[..., None]
        image_height = image_size[..., 1].clamp_min(1.0)[..., None]
        return (
            torch.isfinite(pred_uv).all(dim=-1)
            & torch.isfinite(input_uv).all(dim=-1)
            & (pred_depth > min_depth)
            & (input_depth > min_depth)
            & (input_uv[..., 0] >= border)
            & (input_uv[..., 0] <= image_width - 1.0 - border)
            & (input_uv[..., 1] >= border)
            & (input_uv[..., 1] <= image_height - 1.0 - border)
        )

    def _gt_projection_valid_mask(
        self,
        *,
        pred_uv: torch.Tensor,
        pred_depth: torch.Tensor,
        target_uv: torch.Tensor,
        target_confidence: torch.Tensor,
        image_size: torch.Tensor,
    ) -> torch.Tensor:
        min_depth = float(self.loss_config.gt_projection_min_depth)
        border = float(self.loss_config.gt_projection_border_px)
        image_width = image_size[..., 0].clamp_min(1.0)[..., None]
        image_height = image_size[..., 1].clamp_min(1.0)[..., None]
        return (
            torch.isfinite(pred_uv).all(dim=-1)
            & torch.isfinite(target_uv).all(dim=-1)
            & (pred_depth > min_depth)
            & (target_confidence > 0.05)
            & (target_uv[..., 0] >= border)
            & (target_uv[..., 0] <= image_width - 1.0 - border)
            & (target_uv[..., 1] >= border)
            & (target_uv[..., 1] <= image_height - 1.0 - border)
        )

    def _image_projection_valid_mask(
        self,
        *,
        pred_uv: torch.Tensor,
        pred_depth: torch.Tensor,
        target_uv: torch.Tensor,
        target_confidence: torch.Tensor,
        target_valid: torch.Tensor,
        image_size: torch.Tensor,
    ) -> torch.Tensor:
        min_depth = float(self.loss_config.image_projection_min_depth)
        border = float(self.loss_config.image_projection_border_px)
        confidence_threshold = float(
            self.loss_config.image_projection_confidence_threshold
        )
        image_width = image_size[..., 0].clamp_min(1.0)[..., None]
        image_height = image_size[..., 1].clamp_min(1.0)[..., None]
        return (
            torch.isfinite(pred_uv).all(dim=-1)
            & torch.isfinite(target_uv).all(dim=-1)
            & (pred_depth > min_depth)
            & target_valid
            & (target_confidence > confidence_threshold)
            & (target_uv[..., 0] >= border)
            & (target_uv[..., 0] <= image_width - 1.0 - border)
            & (target_uv[..., 1] >= border)
            & (target_uv[..., 1] <= image_height - 1.0 - border)
        )

    def _projected_measurement_valid_mask(
        self,
        *,
        source_uv: torch.Tensor,
        target_uv: torch.Tensor,
        target_confidence: torch.Tensor,
        base_valid: torch.Tensor,
        image_size: torch.Tensor,
    ) -> torch.Tensor:
        border = float(self.loss_config.projected_measurement_border_px)
        confidence_threshold = float(
            self.loss_config.projected_measurement_confidence_threshold
        )
        image_width = image_size[..., 0].clamp_min(1.0)[..., None]
        image_height = image_size[..., 1].clamp_min(1.0)[..., None]
        return (
            torch.isfinite(source_uv).all(dim=-1)
            & torch.isfinite(target_uv).all(dim=-1)
            & base_valid
            & (target_confidence > confidence_threshold)
            & (target_uv[..., 0] >= border)
            & (target_uv[..., 0] <= image_width - 1.0 - border)
            & (target_uv[..., 1] >= border)
            & (target_uv[..., 1] <= image_height - 1.0 - border)
        )

    def _projected_measurement_consistency_valid_mask(
        self,
        *,
        pred_uv: torch.Tensor,
        pred_depth: torch.Tensor,
        corrected_uv: torch.Tensor,
        measurement_confidence: torch.Tensor,
        base_valid: torch.Tensor,
        image_size: torch.Tensor,
    ) -> torch.Tensor:
        min_depth = float(self.loss_config.projected_measurement_min_depth)
        border = float(self.loss_config.projected_measurement_border_px)
        confidence_threshold = float(
            self.loss_config.projected_measurement_confidence_threshold
        )
        image_width = image_size[..., 0].clamp_min(1.0)[..., None]
        image_height = image_size[..., 1].clamp_min(1.0)[..., None]
        return (
            torch.isfinite(pred_uv).all(dim=-1)
            & torch.isfinite(corrected_uv).all(dim=-1)
            & (pred_depth > min_depth)
            & base_valid
            & (measurement_confidence > confidence_threshold)
            & (corrected_uv[..., 0] >= border)
            & (corrected_uv[..., 0] <= image_width - 1.0 - border)
            & (corrected_uv[..., 1] >= border)
            & (corrected_uv[..., 1] <= image_height - 1.0 - border)
        )

    @staticmethod
    def _resolve_projection_smpl_indices(
        *,
        target_joint_smpl_indices: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        if target_joint_smpl_indices is not None:
            smpl_indices = target_joint_smpl_indices
            if smpl_indices.ndim > 1:
                smpl_indices = smpl_indices[0]
            return smpl_indices.reshape(-1).to(device=device, dtype=torch.long)
        return torch.arange(SMPL_EVAL_NUM_JOINTS, device=device, dtype=torch.long)

    def _build_canonical_joint_pair(
        self,
        *,
        pred_body_pose: torch.Tensor,
        pred_betas: torch.Tensor,
        target_body_pose: torch.Tensor,
        target_betas: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = pred_body_pose.shape[0]
        zero_root = torch.zeros(
            (batch_size, 3),
            dtype=pred_body_pose.dtype,
            device=pred_body_pose.device,
        )
        zero_transl = torch.zeros_like(zero_root)
        pred_joints = root_center_joints(
            self._build_smpl_joints(
                body_pose=pred_body_pose,
                betas=pred_betas,
                global_orient=zero_root,
                transl=zero_transl,
            )
        )
        target_joints = root_center_joints(
            self._build_smpl_joints(
                body_pose=target_body_pose,
                betas=target_betas,
                global_orient=zero_root,
                transl=zero_transl,
            )
        )
        return pred_joints, target_joints

    def _compute_canonical_joint_loss(
        self,
        *,
        pred_body_pose: torch.Tensor,
        pred_betas: torch.Tensor,
        target_body_pose: torch.Tensor,
        target_betas: torch.Tensor,
    ) -> torch.Tensor:
        pred_joints, target_joints = self._build_canonical_joint_pair(
            pred_body_pose=pred_body_pose,
            pred_betas=pred_betas,
            target_body_pose=target_body_pose,
            target_betas=target_betas,
        )
        return F.mse_loss(pred_joints, target_joints)

    def _compute_canonical_pseudo_joint_losses(
        self,
        *,
        pred_body_pose: torch.Tensor,
        pred_betas: torch.Tensor,
        target_body_pose: torch.Tensor,
        target_betas: torch.Tensor,
        use_smpl_targets: bool,
        target_joint_smpl_indices: torch.Tensor | None,
        target_joint_root_index: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        zero = pred_body_pose.new_zeros(())
        if (
            not use_smpl_targets
            or (
                self.loss_config.canonical_pseudo_joint_weight <= 0.0
                and self.loss_config.canonical_pseudo_pa_joint_weight <= 0.0
            )
        ):
            return {
                "canonical_pseudo_joint_loss": zero,
                "canonical_pseudo_pa_joint_loss": zero,
            }

        batch_size = pred_body_pose.shape[0]
        zero_root = torch.zeros(
            (batch_size, 3),
            dtype=pred_body_pose.dtype,
            device=pred_body_pose.device,
        )
        zero_transl = torch.zeros_like(zero_root)
        pred_smpl_output = self._build_smpl_output(
            body_pose=pred_body_pose,
            betas=pred_betas,
            global_orient=zero_root,
            transl=zero_transl,
        )
        target_smpl_output = self._build_smpl_output(
            body_pose=target_body_pose.to(
                device=pred_body_pose.device,
                dtype=pred_body_pose.dtype,
            ),
            betas=target_betas.to(
                device=pred_betas.device,
                dtype=pred_betas.dtype,
            ),
            global_orient=zero_root,
            transl=zero_transl,
        )
        if target_joint_smpl_indices is not None:
            smpl_indices = self._resolve_projection_smpl_indices(
                target_joint_smpl_indices=target_joint_smpl_indices,
                device=pred_body_pose.device,
            )
            pred_joints = self._select_external_pred_joints(
                smpl_output=pred_smpl_output,
                smpl_indices=smpl_indices,
            )
            target_joints = self._select_external_pred_joints(
                smpl_output=target_smpl_output,
                smpl_indices=smpl_indices,
            )
            root_index = (
                int(target_joint_root_index.reshape(-1)[0].item())
                if target_joint_root_index is not None
                else 0
            )
        else:
            pred_joints = pred_smpl_output.joints[:, :SMPL_EVAL_NUM_JOINTS, :]
            target_joints = target_smpl_output.joints[:, :SMPL_EVAL_NUM_JOINTS, :]
            root_index = 0

        pred_joints = pred_joints - pred_joints[:, root_index : root_index + 1, :]
        target_joints = target_joints - target_joints[:, root_index : root_index + 1, :]
        weight = torch.ones(
            pred_joints.shape[:2],
            dtype=pred_joints.dtype,
            device=pred_joints.device,
        )
        joint_loss = self._weighted_joint_mse(pred_joints, target_joints, weight)
        aligned_pred = self._weighted_similarity_align(
            pred_joints.float(),
            target_joints.float(),
            weight.float(),
        )
        per_joint = torch.linalg.norm(aligned_pred - target_joints.float(), dim=-1)
        pa_joint_loss = (per_joint * weight.float()).sum() / weight.float().sum().clamp_min(1.0)
        return {
            "canonical_pseudo_joint_loss": joint_loss,
            "canonical_pseudo_pa_joint_loss": pa_joint_loss,
        }

    def _compute_canonical_joint_metrics(
        self,
        *,
        pred_body_pose: torch.Tensor,
        pred_betas: torch.Tensor,
        target_body_pose: torch.Tensor,
        target_betas: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pred_joints, target_joints = self._build_canonical_joint_pair(
            pred_body_pose=pred_body_pose,
            pred_betas=pred_betas,
            target_body_pose=target_body_pose,
            target_betas=target_betas,
        )
        return {
            "mpjpe": batch_mpjpe(pred_joints, target_joints).mean(),
            "pa_mpjpe": batch_pa_mpjpe(pred_joints, target_joints).mean(),
        }

    def _build_external_joint_pair(
        self,
        *,
        pred_body_pose: torch.Tensor,
        pred_betas: torch.Tensor,
        target_joints: torch.Tensor,
        target_joint_confidence: torch.Tensor,
        target_joint_smpl_indices: torch.Tensor,
        target_joint_root_index: torch.Tensor,
        view_aux: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = pred_body_pose.shape[0]
        zero_root = torch.zeros(
            (batch_size, 3),
            dtype=pred_body_pose.dtype,
            device=pred_body_pose.device,
        )
        pred_smpl_output = self._build_smpl_output(
            body_pose=pred_body_pose,
            betas=pred_betas,
            global_orient=zero_root,
            transl=torch.zeros_like(zero_root),
        )
        smpl_indices = target_joint_smpl_indices[0].to(
            device=pred_body_pose.device,
            dtype=torch.long,
        )
        root_index = int(target_joint_root_index.reshape(-1)[0].item())
        pred_joints = self._select_external_pred_joints(
            smpl_output=pred_smpl_output,
            smpl_indices=smpl_indices,
        )
        pred_joints = pred_joints - pred_joints[:, root_index : root_index + 1, :]
        transformed_target_joints, joint_weight = self._build_external_target_joints(
            target_joints=target_joints,
            target_joint_confidence=target_joint_confidence,
            target_joint_root_index=target_joint_root_index,
            view_aux=view_aux,
            device=pred_body_pose.device,
            dtype=pred_joints.dtype,
        )
        return pred_joints, transformed_target_joints, joint_weight

    def _build_external_target_joints(
        self,
        *,
        target_joints: torch.Tensor,
        target_joint_confidence: torch.Tensor,
        target_joint_root_index: torch.Tensor,
        view_aux: dict[str, torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        root_index = int(target_joint_root_index.reshape(-1)[0].item())
        transformed_target_joints = self._transform_world_joints_to_input_root_frame(
            target_joints=target_joints.to(device=device, dtype=dtype),
            view_aux=view_aux,
            root_index=root_index,
        )
        joint_weight = target_joint_confidence.to(device=device, dtype=dtype).clamp_min(0.0)
        joint_weight = joint_weight * (joint_weight > 0.05).to(joint_weight.dtype)
        return transformed_target_joints, joint_weight

    def _select_external_pred_joints(
        self,
        *,
        smpl_output,
        smpl_indices: torch.Tensor,
    ) -> torch.Tensor:
        source = self.external_joint_source
        if source == "smpl24":
            pred_smpl_joints = smpl_output.joints[:, :SMPL_EVAL_NUM_JOINTS, :]
            return pred_smpl_joints.index_select(1, smpl_indices)
        if source not in {"regressor", "regressor_headtop_proxy", "body25_regressor15"}:
            raise ValueError(
                "Unsupported external_joint_source="
                f"{source!r}. Expected smpl24, regressor, regressor_headtop_proxy, "
                "or body25_regressor15."
            )

        regressor, regressor_input = self._get_external_joint_regressor(
            device=smpl_output.vertices.device,
            num_smpl_joints=int(smpl_output.joints.shape[1]),
        )
        if regressor_input == "vertices":
            pred_joints = torch.einsum("jv,bvc->bjc", regressor, smpl_output.vertices)
        elif regressor_input == "joints":
            pred_joints = torch.einsum("jk,bkc->bjc", regressor, smpl_output.joints)
        else:
            raise ValueError(f"Unsupported external joint regressor input {regressor_input!r}")

        if source == "body25_regressor15":
            if pred_joints.shape[1] < 15:
                raise ValueError(
                    f"body25_regressor15 requires at least 15 joints, got {pred_joints.shape[1]}"
                )
            return pred_joints[:, :15, :]

        if source == "regressor_headtop_proxy":
            headtop_vertex_ids = self._get_external_headtop_vertex_ids(
                device=smpl_output.vertices.device,
                smpl_output=smpl_output,
            )
            headtop = smpl_output.vertices.index_select(1, headtop_vertex_ids).mean(dim=1)
            pred_joints = pred_joints.clone()
            pred_joints[:, 10, :] = headtop
        return pred_joints

    def _get_external_joint_regressor(
        self,
        *,
        device: torch.device,
        num_smpl_joints: int,
    ) -> tuple[torch.Tensor, str]:
        if self.external_joint_regressor_path is None:
            raise ValueError(
                f"external_joint_source={self.external_joint_source!r} requires "
                "external_joint_regressor_path."
            )
        resolved_path = Path(self.external_joint_regressor_path).resolve()
        cached = self._runtime_cache.get("external_joint_regressor")
        if (
            isinstance(cached, tuple)
            and len(cached) == 3
            and cached[0] == resolved_path
            and isinstance(cached[1], torch.Tensor)
            and cached[1].device == device
        ):
            return cached[1], str(cached[2])

        if not resolved_path.exists():
            raise FileNotFoundError(f"External joint regressor does not exist: {resolved_path}")
        if resolved_path.suffix == ".npz":
            with np.load(resolved_path, allow_pickle=False) as payload:
                preferred_keys = ("J_regressor_h36m", "J_regressor", "regressor")
                key = next((item for item in preferred_keys if item in payload.files), payload.files[0])
                regressor = np.asarray(payload[key], dtype=np.float32)
        else:
            regressor = np.asarray(np.load(resolved_path, allow_pickle=False), dtype=np.float32)
        if regressor.ndim != 2:
            raise ValueError(f"External joint regressor must be 2D, got {regressor.shape}")
        if regressor.shape[0] not in {15, 17, 25} and regressor.shape[1] in {15, 17, 25}:
            regressor = regressor.T
        if regressor.shape[0] not in {15, 17, 25}:
            raise ValueError(
                "Expected external joint regressor to have 15, 17, or 25 output joints, "
                f"got {regressor.shape}"
            )
        if regressor.shape[1] == 6890:
            regressor_input = "vertices"
        elif regressor.shape[1] <= num_smpl_joints:
            regressor_input = "joints"
        else:
            raise ValueError(
                f"Unsupported external joint regressor shape {regressor.shape}. Expected "
                f"[17, 6890] or [17, <= {num_smpl_joints}]."
            )
        tensor = torch.from_numpy(np.ascontiguousarray(regressor)).to(
            device=device,
            dtype=torch.float32,
        )
        self._runtime_cache["external_joint_regressor"] = (
            resolved_path,
            tensor,
            regressor_input,
        )
        return tensor, regressor_input

    def _get_external_headtop_vertex_ids(
        self,
        *,
        device: torch.device,
        smpl_output,
    ) -> torch.Tensor:
        if self.external_headtop_topk < 1:
            raise ValueError(
                f"external_headtop_topk must be >= 1, got {self.external_headtop_topk}"
            )
        axis_index = {"x": 0, "y": 1, "z": 2}[self.external_headtop_axis]
        cached = self._runtime_cache.get("external_headtop_vertex_ids")
        if isinstance(cached, torch.Tensor) and cached.device == device:
            return cached

        smpl_model = self._runtime_cache.get("smpl_eval_model")
        template_vertices = getattr(smpl_model, "v_template", None)
        if template_vertices is None:
            template_vertices = smpl_output.vertices[0].detach()
        else:
            template_vertices = template_vertices.detach().to(device=device)
            if template_vertices.ndim == 3:
                template_vertices = template_vertices[0]
        values = template_vertices[:, axis_index]
        k = min(int(self.external_headtop_topk), int(values.numel()))
        vertex_ids = torch.topk(values, k=k, largest=True).indices.to(
            device=device,
            dtype=torch.long,
        )
        self._runtime_cache["external_headtop_vertex_ids"] = vertex_ids
        return vertex_ids

    @staticmethod
    def _transform_world_joints_to_input_root_frame(
        *,
        target_joints: torch.Tensor,
        view_aux: dict[str, torch.Tensor],
        root_index: int,
    ) -> torch.Tensor:
        camera_rotation = view_aux["camera_rotation"].to(
            device=target_joints.device,
            dtype=torch.float32,
        )
        camera_translation = view_aux["camera_translation"].to(
            device=target_joints.device,
            dtype=torch.float32,
        )
        input_global_orient = view_aux["input_global_orient"].to(
            device=target_joints.device,
            dtype=torch.float32,
        )
        target_camera = (
            torch.einsum("bvij,bkj->bvki", camera_rotation, target_joints.float())
            + camera_translation[:, :, None, :]
        )
        target_camera = target_camera - target_camera[:, :, root_index : root_index + 1, :]
        input_root_rotation = axis_angle_to_matrix(input_global_orient)
        target_input_root = torch.matmul(
            target_camera.unsqueeze(-2),
            input_root_rotation[:, :, None, :, :],
        ).squeeze(-2)
        return target_input_root.mean(dim=1).to(target_joints.dtype)

    def _compute_external_joint_metrics(
        self,
        *,
        pred_joints: torch.Tensor,
        target_joints: torch.Tensor,
        joint_weight: torch.Tensor,
        target_joint_root_index: torch.Tensor | None = None,
        target_camera_joints: torch.Tensor | None = None,
        target_camera_joint_confidence: torch.Tensor | None = None,
        view_aux: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        root_index = (
            int(target_joint_root_index.reshape(-1)[0].item())
            if target_joint_root_index is not None
            else 0
        )
        if target_camera_joints is not None:
            return self._compute_heatformer_external_joint_metrics(
                pred_joints=pred_joints,
                target_camera_joints=target_camera_joints,
                target_camera_joint_confidence=target_camera_joint_confidence,
                view_aux=view_aux,
                root_index=root_index,
            )

        root_pred_joints = pred_joints - pred_joints[:, root_index : root_index + 1, :]
        root_target_joints = target_joints - target_joints[:, root_index : root_index + 1, :]
        mpjpe = self._weighted_mpjpe(root_pred_joints, root_target_joints, joint_weight) * 1000.0
        with torch.no_grad(), torch.autocast(device_type=pred_joints.device.type, enabled=False):
            aligned_pred = self._weighted_similarity_align(
                root_pred_joints.float(),
                root_target_joints.float(),
                joint_weight.float(),
            )
        pa_mpjpe = self._weighted_mpjpe(
            aligned_pred,
            root_target_joints.float(),
            joint_weight.float(),
        ) * 1000.0
        pck_150, auc = self._weighted_pck_auc(
            aligned_pred,
            root_target_joints.float(),
            joint_weight.float(),
        )
        root_pck_150, root_auc = self._weighted_pck_auc(
            root_pred_joints,
            root_target_joints,
            joint_weight,
        )
        return {
            "mpjpe": mpjpe,
            "pa_mpjpe": pa_mpjpe,
            "pck_150": pck_150,
            "auc": auc,
            "root_pck_150": root_pck_150,
            "root_auc": root_auc,
        }

    def _compute_heatformer_external_joint_metrics(
        self,
        *,
        pred_joints: torch.Tensor,
        target_camera_joints: torch.Tensor,
        target_camera_joint_confidence: torch.Tensor | None,
        view_aux: dict[str, torch.Tensor] | None,
        root_index: int = 0,
    ) -> dict[str, torch.Tensor]:
        if target_camera_joints.ndim != 4 or target_camera_joints.shape[-1] != 3:
            raise ValueError(
                "target_camera_joints must have shape [B, V, J, 3], "
                f"got {tuple(target_camera_joints.shape)}"
            )
        if view_aux is None or "input_global_orient" not in view_aux:
            raise KeyError(
                "HeatFormer-style external metrics require view_aux['input_global_orient']"
            )

        target_camera_joints = target_camera_joints.to(
            device=pred_joints.device,
            dtype=pred_joints.dtype,
        )
        batch_size, num_views, joint_count, _ = target_camera_joints.shape
        if pred_joints.shape[:2] != (batch_size, joint_count):
            raise ValueError(
                f"Expected pred_joints shape {(batch_size, joint_count, 3)}, "
                f"got {tuple(pred_joints.shape)}"
            )

        root_pred_joints = pred_joints - pred_joints[:, root_index : root_index + 1, :]
        input_root_rotation = axis_angle_to_matrix(
            view_aux["input_global_orient"].to(
                device=pred_joints.device,
                dtype=pred_joints.dtype,
            )
        )
        pred_camera_joints = torch.matmul(
            root_pred_joints[:, None].unsqueeze(-2),
            input_root_rotation.transpose(-1, -2)[:, :, None],
        ).squeeze(-2)

        root_target_joints = (
            target_camera_joints
            - target_camera_joints[:, :, root_index : root_index + 1, :]
        )
        flat_pred = pred_camera_joints.reshape(batch_size * num_views, joint_count, 3)
        flat_target = root_target_joints.reshape(batch_size * num_views, joint_count, 3)
        if target_camera_joint_confidence is None:
            flat_weight = torch.ones(
                (batch_size * num_views, joint_count),
                device=pred_joints.device,
                dtype=pred_joints.dtype,
            )
        else:
            flat_weight = target_camera_joint_confidence.to(
                device=pred_joints.device,
                dtype=pred_joints.dtype,
            ).reshape(batch_size * num_views, joint_count)
            flat_weight = flat_weight * (flat_weight > 0.05).to(flat_weight.dtype)

        if joint_count == 15 and root_index == 8:
            visibility = flat_weight[..., None]
            mpjpe = (
                torch.linalg.norm(flat_pred * visibility - flat_target * visibility, dim=-1)
                .mean(dim=-1)
                .mean()
                * 1000.0
            )
            with torch.no_grad(), torch.autocast(device_type=pred_joints.device.type, enabled=False):
                aligned_pred = batch_similarity_align(flat_pred.float(), flat_target.float())
            pa_mpjpe = (
                torch.linalg.norm(
                    aligned_pred * visibility.float() - flat_target.float() * visibility.float(),
                    dim=-1,
                )
                .mean(dim=-1)
                .mean()
                * 1000.0
            )
            pck_150, auc = self._weighted_pck_auc(
                aligned_pred,
                flat_target.float(),
                flat_weight.float(),
            )
            root_pck_150, root_auc = self._weighted_pck_auc(
                flat_pred,
                flat_target,
                flat_weight,
            )
            return {
                "mpjpe": mpjpe,
                "pa_mpjpe": pa_mpjpe,
                "pck_150": pck_150,
                "auc": auc,
                "root_pck_150": root_pck_150,
                "root_auc": root_auc,
            }

        mpjpe = self._weighted_mpjpe(flat_pred, flat_target, flat_weight) * 1000.0
        with torch.no_grad(), torch.autocast(device_type=pred_joints.device.type, enabled=False):
            aligned_pred = self._weighted_similarity_align(
                flat_pred.float(),
                flat_target.float(),
                flat_weight.float(),
            )
        pa_mpjpe = self._weighted_mpjpe(
            aligned_pred,
            flat_target.float(),
            flat_weight.float(),
        ) * 1000.0
        pck_150, auc = self._weighted_pck_auc(
            aligned_pred,
            flat_target.float(),
            flat_weight.float(),
        )
        root_pck_150, root_auc = self._weighted_pck_auc(
            flat_pred,
            flat_target,
            flat_weight,
        )
        return {
            "mpjpe": mpjpe,
            "pa_mpjpe": pa_mpjpe,
            "pck_150": pck_150,
            "auc": auc,
            "root_pck_150": root_pck_150,
            "root_auc": root_auc,
        }

    def _compute_external_input_joint_metrics(
        self,
        *,
        views_input: torch.Tensor,
        view_aux: dict[str, torch.Tensor],
        target_camera_joints: torch.Tensor,
        target_camera_joint_confidence: torch.Tensor | None,
        target_joint_smpl_indices: torch.Tensor,
        target_joint_root_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, num_views = views_input.shape[:2]
        pose_6d_dim = self.model_config.pose_6d_dim
        input_pose_6d = views_input[..., :pose_6d_dim].reshape(
            batch_size,
            num_views,
            self.model_config.num_joints,
            6,
        )
        input_body_pose = rotation_6d_to_axis_angle(
            input_pose_6d.reshape(-1, self.model_config.num_joints, 6)
        ).reshape(batch_size * num_views, -1)
        input_betas = views_input[..., pose_6d_dim:].reshape(batch_size * num_views, -1)
        camera_global_orient = view_aux["input_global_orient"].to(
            device=views_input.device,
            dtype=views_input.dtype,
        ).reshape(batch_size * num_views, 3)
        camera_transl = view_aux["input_transl"].to(
            device=views_input.device,
            dtype=views_input.dtype,
        ).reshape(batch_size * num_views, 3)
        input_output = self._build_smpl_output(
            body_pose=input_body_pose.to(device=views_input.device, dtype=views_input.dtype),
            betas=input_betas.to(device=views_input.device, dtype=views_input.dtype),
            global_orient=camera_global_orient,
            transl=camera_transl,
        )
        smpl_indices = self._resolve_projection_smpl_indices(
            target_joint_smpl_indices=target_joint_smpl_indices,
            device=views_input.device,
        )
        input_joints = self._select_external_pred_joints(
            smpl_output=input_output,
            smpl_indices=smpl_indices,
        )
        joint_count = input_joints.shape[1]
        input_joints = input_joints.reshape(batch_size, num_views, joint_count, 3)
        target_camera_joints = target_camera_joints.to(
            device=views_input.device,
            dtype=views_input.dtype,
        )
        root_index = int(target_joint_root_index.reshape(-1)[0].item())
        flat_pred = (
            input_joints - input_joints[:, :, root_index : root_index + 1, :]
        ).reshape(batch_size * num_views, joint_count, 3)
        flat_target = (
            target_camera_joints
            - target_camera_joints[:, :, root_index : root_index + 1, :]
        ).reshape(batch_size * num_views, joint_count, 3)
        if target_camera_joint_confidence is None:
            flat_weight = torch.ones(
                (batch_size * num_views, joint_count),
                device=views_input.device,
                dtype=views_input.dtype,
            )
        else:
            flat_weight = target_camera_joint_confidence.to(
                device=views_input.device,
                dtype=views_input.dtype,
            ).reshape(batch_size * num_views, joint_count)
            flat_weight = flat_weight * (flat_weight > 0.05).to(flat_weight.dtype)

        if joint_count == 15 and root_index == 8:
            visibility = flat_weight[..., None]
            mpjpe = (
                torch.linalg.norm(flat_pred * visibility - flat_target * visibility, dim=-1)
                .mean(dim=-1)
                .mean()
                * 1000.0
            )
            with torch.no_grad(), torch.autocast(device_type=views_input.device.type, enabled=False):
                aligned_pred = batch_similarity_align(flat_pred.float(), flat_target.float())
            pa_mpjpe = (
                torch.linalg.norm(
                    aligned_pred * visibility.float() - flat_target.float() * visibility.float(),
                    dim=-1,
                )
                .mean(dim=-1)
                .mean()
                * 1000.0
            )
        else:
            mpjpe = self._weighted_mpjpe(flat_pred, flat_target, flat_weight) * 1000.0
            pa_mpjpe = (
                self._weighted_pa_mpjpe(flat_pred, flat_target, flat_weight) * 1000.0
            )
        return {"mpjpe": mpjpe, "pa_mpjpe": pa_mpjpe}

    @staticmethod
    def _weighted_joint_mse(
        pred_joints: torch.Tensor,
        target_joints: torch.Tensor,
        joint_weight: torch.Tensor,
    ) -> torch.Tensor:
        per_joint = (pred_joints.float() - target_joints.float()).square().sum(dim=-1)
        weight = joint_weight.to(device=per_joint.device, dtype=per_joint.dtype)
        return (per_joint * weight).sum() / weight.sum().clamp_min(1.0)

    @staticmethod
    def _weighted_mpjpe(
        pred_joints: torch.Tensor,
        target_joints: torch.Tensor,
        joint_weight: torch.Tensor,
    ) -> torch.Tensor:
        per_joint = torch.linalg.norm(pred_joints.float() - target_joints.float(), dim=-1)
        weight = joint_weight.to(device=per_joint.device, dtype=per_joint.dtype)
        return (per_joint * weight).sum() / weight.sum().clamp_min(1.0)

    @classmethod
    def _weighted_pa_mpjpe(
        cls,
        pred_joints: torch.Tensor,
        target_joints: torch.Tensor,
        joint_weight: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad(), torch.autocast(device_type=pred_joints.device.type, enabled=False):
            aligned_pred = cls._weighted_similarity_align(
                pred_joints.float(),
                target_joints.float(),
                joint_weight.float(),
            )
        return cls._weighted_mpjpe(aligned_pred, target_joints.float(), joint_weight.float())

    @staticmethod
    def _weighted_similarity_align(
        source: torch.Tensor,
        target: torch.Tensor,
        joint_weight: torch.Tensor,
    ) -> torch.Tensor:
        weight = joint_weight.to(device=source.device, dtype=source.dtype).clamp_min(0.0)
        weight_sum = weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
        weight = weight / weight_sum
        weight_3d = weight.unsqueeze(-1)
        mu_source = (source * weight_3d).sum(dim=1, keepdim=True)
        mu_target = (target * weight_3d).sum(dim=1, keepdim=True)
        source_centered = source - mu_source
        target_centered = target - mu_target
        covariance = source_centered.transpose(1, 2) @ (target_centered * weight_3d)
        u, singular_values, vh = torch.linalg.svd(covariance)
        v = vh.transpose(1, 2)
        sign_correction = torch.ones(
            (source.shape[0], 3),
            dtype=source.dtype,
            device=source.device,
        )
        det = torch.det(v @ u.transpose(1, 2))
        sign_correction[:, -1] = torch.where(det < 0, -1.0, 1.0).to(source.dtype)
        rotation = u @ torch.diag_embed(sign_correction) @ v.transpose(1, 2)
        source_variance = (weight_3d * source_centered.square()).sum(dim=(1, 2)).clamp_min(1e-8)
        trace = (singular_values * sign_correction).sum(dim=1)
        scale = (trace / source_variance).view(-1, 1, 1)
        translation = mu_target - scale * (mu_source @ rotation)
        return scale * (source @ rotation) + translation

    @staticmethod
    def _weighted_pck_auc(
        pred_joints: torch.Tensor,
        target_joints: torch.Tensor,
        joint_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        err_mm = torch.linalg.norm(pred_joints.float() - target_joints.float(), dim=-1) * 1000.0
        valid = joint_weight.to(device=err_mm.device, dtype=torch.bool)
        denom = valid.sum().clamp_min(1)
        pck_150 = ((err_mm < 150.0) & valid).sum().to(err_mm.dtype) / denom
        auc = err_mm.new_zeros(())
        for threshold in range(1, 151, 5):
            auc = auc + ((err_mm < float(threshold)) & valid).sum().to(err_mm.dtype) / denom
        return pck_150, auc / 30.0

    def _compute_articulation_loss(
        self,
        *,
        pred_joints: torch.Tensor,
        target_joints: torch.Tensor,
    ) -> torch.Tensor:
        with torch.autocast(device_type=pred_joints.device.type, enabled=False):
            aligned_pred = batch_similarity_align(
                pred_joints.float(),
                target_joints.float(),
            )
            articulation_loss = F.mse_loss(aligned_pred, target_joints.float())
        return articulation_loss.to(pred_joints.dtype)

    def _compute_test_joint_metrics(
        self,
        *,
        pred_body_pose: torch.Tensor,
        pred_betas: torch.Tensor,
        target_body_pose: torch.Tensor,
        target_betas: torch.Tensor,
        views_input: torch.Tensor,
        view_aux: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        pose_6d_dim = self.model_config.pose_6d_dim
        input_pose_6d = views_input[..., :pose_6d_dim].reshape(
            views_input.shape[0],
            views_input.shape[1],
            self.model_config.num_joints,
            6,
        )
        input_betas = views_input[..., pose_6d_dim:]
        return compute_input_corrected_camera_joint_metrics(
            build_smpl_joints=self._build_smpl_joints,
            pred_body_pose=pred_body_pose,
            pred_betas=pred_betas,
            target_body_pose=target_body_pose,
            target_betas=target_betas,
            input_views_pose_6d=input_pose_6d,
            input_views_betas=input_betas,
            input_camera_global_orient=view_aux["input_global_orient"].to(
                pred_body_pose.device
            ),
            input_transl=view_aux["input_transl"].to(pred_body_pose.device),
        )

    def _build_smpl_joints(
        self,
        *,
        body_pose: torch.Tensor,
        betas: torch.Tensor,
        global_orient: torch.Tensor,
        transl: torch.Tensor,
    ) -> torch.Tensor:
        output = self._build_smpl_output(
            body_pose=body_pose,
            betas=betas,
            global_orient=global_orient,
            transl=transl,
        )
        return output.joints[:, :SMPL_EVAL_NUM_JOINTS, :]

    def _build_smpl_output(
        self,
        *,
        body_pose: torch.Tensor,
        betas: torch.Tensor,
        global_orient: torch.Tensor,
        transl: torch.Tensor,
    ):
        smpl_model = self._get_smpl_eval_model(
            device=body_pose.device, batch_size=body_pose.shape[0]
        )
        with torch.autocast(device_type=body_pose.device.type, enabled=False):
            output = smpl_model(
                body_pose=body_pose.float(),
                betas=betas.float(),
                global_orient=global_orient.float(),
                transl=transl.float(),
            )
        return output

    def _get_smpl_eval_model(self, *, device: torch.device, batch_size: int):
        cache = self._runtime_cache.get("smpl_eval_models")
        if not isinstance(cache, dict):
            cache = {}
        key = (str(device), int(batch_size))
        smpl_eval_model = cache.get(key)
        if smpl_eval_model is None or next(smpl_eval_model.parameters()).device != device:
            smpl_eval_model = build_smpl_model(
                device=device,
                smpl_model_path=self.smpl_model_path,
                batch_size=batch_size,
            )
            smpl_eval_model.eval()
            cache[key] = smpl_eval_model
        self._runtime_cache["smpl_eval_models"] = cache
        self._runtime_cache["smpl_eval_model"] = smpl_eval_model
        return smpl_eval_model

    def _log_metrics_if_possible(
        self,
        metrics: dict[str, torch.Tensor],
        *,
        batch_size: int,
        on_step: bool,
        on_epoch: bool,
        prog_bar: bool,
        sync_dist: bool = False,
    ) -> None:
        if getattr(self, "_trainer", None) is None:
            return
        self.log_dict(
            metrics,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
        )

    def _sanitize_parameters(self, location: str) -> None:
        sanitized = 0
        with torch.no_grad():
            for parameter in self.parameters():
                if torch.isfinite(parameter).all():
                    continue
                parameter.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                sanitized += 1
        if sanitized and getattr(self, "global_rank", 0) == 0:
            print(
                f"Sanitized {sanitized} non-finite parameter tensors at {location}.",
                flush=True,
            )

    def _sanitize_gradients(self) -> None:
        sanitized = 0
        for parameter in self.parameters():
            grad = parameter.grad
            if grad is None:
                continue
            if not torch.isfinite(grad).all():
                grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                sanitized += 1
            grad.clamp_(min=-100.0, max=100.0)
        if sanitized and getattr(self, "global_rank", 0) == 0:
            print(
                f"Sanitized {sanitized} non-finite gradient tensors.",
                flush=True,
            )

    def _assert_finite_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if torch.isfinite(tensor).all():
            return
        detached = tensor.detach()
        nonfinite_mask = ~torch.isfinite(detached)
        bad_count = int(nonfinite_mask.sum().item())
        min_value = (
            float(detached[torch.isfinite(detached)].min().item())
            if torch.isfinite(detached).any()
            else float("nan")
        )
        max_value = (
            float(detached[torch.isfinite(detached)].max().item())
            if torch.isfinite(detached).any()
            else float("nan")
        )
        raise RuntimeError(
            f"Non-finite tensor detected at {name}: "
            f"shape={tuple(detached.shape)}, bad_count={bad_count}, "
            f"finite_min={min_value}, finite_max={max_value}"
        )


class Stage2RRGBGuidedResidualRefinerLightningModule(Stage2FusionLightningModule):
    """Train a small RGB residual adapter on top of a Stage 2 checkpoint."""

    def __init__(
        self,
        *,
        model_config: Stage2RRGBGuidedResidualRefinerConfig | dict | None = None,
        loss_config: Stage2LossConfig | dict | None = None,
        optimization_config: Stage2OptimizationConfig | dict | None = None,
        smpl_model_path: str | None = None,
        stage2_checkpoint_path: str | None = None,
        stage2_backbone_model: torch.nn.Module | None = None,
        external_joint_source: str = "smpl24",
        external_joint_regressor_path: str | None = None,
        external_headtop_topk: int = 32,
        external_headtop_axis: str = "y",
    ) -> None:
        L.LightningModule.__init__(self)
        self.model_config = _coerce_rgb_residual_model_config(model_config)
        self.loss_config = _coerce_dataclass_config(loss_config, Stage2LossConfig)
        if not self.model_config.learn_betas:
            self.loss_config.supervise_betas = False
            self.loss_config.betas_weight = 0.0
            self.loss_config.init_betas_weight = 0.0
        self.optimization_config = _coerce_dataclass_config(
            optimization_config,
            Stage2OptimizationConfig,
        )
        self.smpl_model_path = smpl_model_path
        self.stage2_checkpoint_path = stage2_checkpoint_path
        self.external_joint_source = external_joint_source
        self.external_joint_regressor_path = external_joint_regressor_path
        self.external_headtop_topk = external_headtop_topk
        self.external_headtop_axis = external_headtop_axis

        self.stage2_backbone = self._load_stage2_backbone(
            stage2_checkpoint_path=stage2_checkpoint_path,
            stage2_backbone_model=stage2_backbone_model,
        )
        self.model = Stage2RRGBGuidedResidualRefinerModel(self.model_config)
        self.criterion = Stage2Loss(self.loss_config)
        self._runtime_cache: dict[str, object | None] = {
            "smpl_eval_model": None,
            "external_joint_regressor": None,
            "external_headtop_vertex_ids": None,
        }

        self.save_hyperparameters(
            {
                "model_config": asdict(self.model_config),
                "loss_config": asdict(self.loss_config),
                "optimization_config": asdict(self.optimization_config),
                "smpl_model_path": self.smpl_model_path,
                "stage2_checkpoint_path": self.stage2_checkpoint_path,
                "external_joint_source": self.external_joint_source,
                "external_joint_regressor_path": self.external_joint_regressor_path,
                "external_headtop_topk": self.external_headtop_topk,
                "external_headtop_axis": self.external_headtop_axis,
            }
        )

    def forward(
        self,
        views_input: torch.Tensor,
        *,
        view_rgb_feature: torch.Tensor | None = None,
        view_image_joint_feature: torch.Tensor | None = None,
        view_image_joint_valid: torch.Tensor | None = None,
        view_image_joint_confidence: torch.Tensor | None = None,
        view_image_joint_uv: torch.Tensor | None = None,
        view_image_joint_projected_uv: torch.Tensor | None = None,
        view_image_mask_feature: torch.Tensor | None = None,
        view_segmentation_mask: torch.Tensor | None = None,
        view_segmentation_distance: torch.Tensor | None = None,
        view_segmentation_valid: torch.Tensor | None = None,
        view_image_size: torch.Tensor | None = None,
        view_aux: dict[str, torch.Tensor] | None = None,
        target_joint_smpl_indices: torch.Tensor | None = None,
        target_joint_root_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del (
            view_image_size,
            view_image_joint_feature,
            view_image_joint_valid,
            view_image_joint_confidence,
            view_image_joint_uv,
            view_image_joint_projected_uv,
            view_image_mask_feature,
            view_segmentation_mask,
            view_segmentation_distance,
            view_segmentation_valid,
            view_aux,
            target_joint_smpl_indices,
            target_joint_root_index,
        )
        if view_rgb_feature is None:
            raise ValueError(
                "Stage2RRGBGuidedResidualRefinerLightningModule requires view_rgb_feature."
            )
        self._assert_finite_tensor("batch/view_rgb_feature", view_rgb_feature)
        stage2_outputs = self._run_stage2_backbone(
            views_input,
        )
        return self.model(
            views_input,
            view_rgb_feature=view_rgb_feature,
            stage2_outputs=stage2_outputs,
        )

    def configure_optimizers(self):
        adapter_parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        parameter_groups: list[dict[str, object]] = [
            {
                "params": adapter_parameters,
                "lr": self.optimization_config.learning_rate,
            }
        ]
        stage2_parameters = [
            parameter
            for parameter in self.stage2_backbone.parameters()
            if parameter.requires_grad
        ]
        if stage2_parameters:
            parameter_groups.append(
                {
                    "params": stage2_parameters,
                    "lr": (
                        self.optimization_config.learning_rate
                        * self.optimization_config.stage2_backbone_lr_scale
                    ),
                }
            )
        return torch.optim.AdamW(
            parameter_groups,
            weight_decay=self.optimization_config.weight_decay,
        )

    def _compute_stage2_aux_losses(
        self,
        *,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if self.loss_config.stage2_aux_weight <= 0.0 or not any(
            parameter.requires_grad for parameter in self.stage2_backbone.parameters()
        ):
            zero = predictions["pred_pose_6d"].new_zeros(())
            return {
                "stage2_aux_loss": zero,
                "stage2_aux_pose_6d": zero,
                "stage2_aux_betas": zero,
            }

        stage2_pose = predictions.get("stage2_pred_pose_6d", predictions["pred_pose_6d"])
        stage2_betas = predictions.get("stage2_pred_betas", predictions["pred_betas"])
        target_pose = batch["target_body_pose_6d"].to(
            device=stage2_pose.device,
            dtype=stage2_pose.dtype,
        )
        target_betas = batch["target_betas"].to(
            device=stage2_betas.device,
            dtype=stage2_betas.dtype,
        )
        pose_loss = F.mse_loss(stage2_pose, target_pose)
        if self.loss_config.supervise_betas:
            betas_loss = F.mse_loss(stage2_betas, target_betas)
        else:
            betas_loss = pose_loss.new_zeros(())
        aux_loss = (
            self.loss_config.pose_6d_weight * pose_loss
            + self.loss_config.betas_weight * betas_loss
        )
        return {
            "stage2_aux_loss": aux_loss,
            "stage2_aux_pose_6d": pose_loss,
            "stage2_aux_betas": betas_loss,
        }

    def _load_stage2_backbone(
        self,
        *,
        stage2_checkpoint_path: str | None,
        stage2_backbone_model: torch.nn.Module | None,
    ) -> torch.nn.Module:
        if stage2_backbone_model is not None:
            backbone = stage2_backbone_model
        elif stage2_checkpoint_path is not None:
            loaded_module = Stage2FusionLightningModule.load_from_checkpoint(
                stage2_checkpoint_path,
                map_location="cpu",
                smpl_model_path=self.smpl_model_path,
                strict=False,
            )
            backbone = loaded_module.model
        else:
            raise ValueError(
                "stage2_checkpoint_path is required for "
                "stage2r_rgb_guided_residual_refiner."
            )
        if self.model_config.freeze_backbone:
            backbone.requires_grad_(False)
            backbone.eval()
        return backbone

    def _run_stage2_backbone(
        self,
        views_input: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.model_config.freeze_backbone:
            self.stage2_backbone.eval()
            with torch.no_grad():
                return self.stage2_backbone(views_input)
        return self.stage2_backbone(views_input)


def _coerce_dataclass_config(value, config_type):
    if value is None:
        return config_type()
    if isinstance(value, config_type):
        return value
    if isinstance(value, dict):
        return config_type(**value)
    raise TypeError(
        f"Expected {config_type.__name__}, dict, or None; got {type(value)!r}"
    )


def _coerce_rgb_residual_model_config(value):
    if value is None:
        return Stage2RRGBGuidedResidualRefinerConfig()
    if isinstance(value, Stage2RRGBGuidedResidualRefinerConfig):
        return value
    if isinstance(value, dict):
        return Stage2RRGBGuidedResidualRefinerConfig(**value)
    raise TypeError(
        "Expected Stage2RRGBGuidedResidualRefinerConfig, dict, or None; "
        f"got {type(value)!r}"
    )


def _coerce_stage2_model_config(value):
    if value is None:
        return Stage2ParamRefineConfig()
    if isinstance(
        value,
        (
            Stage2ParamRefineConfig,
            Stage2JointResidualConfig,
            Stage2JointGraphRefinerConfig,
        ),
    ):
        return value
    if isinstance(value, dict):
        model_name = str(value.get("name", "stage2_param_refine"))
        if model_name == "stage2_joint_graph_refiner":
            return Stage2JointGraphRefinerConfig(**value)
        if model_name == "stage2_joint_residual":
            return Stage2JointResidualConfig(**value)
        return Stage2ParamRefineConfig(**value)
    raise TypeError(
        "Expected Stage2JointGraphRefinerConfig, Stage2JointResidualConfig, "
        "Stage2ParamRefineConfig, dict, or None; "
        f"got {type(value)!r}"
    )


def _build_stage2_model(config):
    if isinstance(config, Stage2JointGraphRefinerConfig):
        return Stage2JointGraphRefinerModel(config)
    if isinstance(config, Stage2JointResidualConfig):
        return Stage2JointResidualModel(config)
    return Stage2ParamRefineModel(config)
