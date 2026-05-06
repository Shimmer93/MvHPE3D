"""LightningModule for the Stage 1 baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import lightning as L
import torch
import torch.nn.functional as F

from mvhpe3d.metrics import (
    SMPL_EVAL_NUM_JOINTS,
    batch_mpjpe,
    batch_pa_mpjpe,
    root_center_joints,
)
from mvhpe3d.losses import Stage1LossConfig, Stage1Loss
from mvhpe3d.models import (
    Stage1MLPFusionConfig,
    Stage1MLPFusionModel,
    Stage1ResidualFusionConfig,
    Stage1ResidualFusionModel,
)
from mvhpe3d.utils import (
    MHRToSMPLConverter,
    PANOPTIC_EVAL_JOINT_INDICES,
    PANOPTIC_EVAL_ROOT_COLUMN,
    PANOPTIC_EVAL_SMPL24_INDICES,
    build_smpl_model,
)


@dataclass(slots=True)
class Stage1OptimizationConfig:
    """Optimizer settings for Stage 1 training."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


Stage1ModelConfig = Stage1MLPFusionConfig | Stage1ResidualFusionConfig


class Stage1FusionLightningModule(L.LightningModule):
    """Lightning wrapper around the Stage 1 fusion baseline."""

    strict_loading = False

    def __init__(
        self,
        *,
        model_config: Stage1ModelConfig | dict | None = None,
        loss_config: Stage1LossConfig | None = None,
        optimization_config: Stage1OptimizationConfig | None = None,
        smpl_model_path: str | None = None,
        mhr_assets_dir: str | None = None,
        input_smpl_cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.model_config = _coerce_model_config(model_config)
        self.loss_config = _coerce_dataclass_config(
            loss_config,
            Stage1LossConfig,
        )
        if not self.model_config.learn_betas:
            self.loss_config.supervise_betas = False
            self.loss_config.betas_weight = 0.0
        self.optimization_config = _coerce_dataclass_config(
            optimization_config,
            Stage1OptimizationConfig,
        )
        self.smpl_model_path = smpl_model_path
        self.mhr_assets_dir = mhr_assets_dir
        self.input_smpl_cache_dir = input_smpl_cache_dir

        self.model = _build_stage1_model(self.model_config)
        self.criterion = Stage1Loss(self.loss_config)
        self._runtime_cache: dict[str, object | None] = {
            "smpl_eval_model": None,
            "mhr_to_smpl_converter": None,
        }

        self.save_hyperparameters(
            {
                "model_config": asdict(self.model_config),
                "loss_config": asdict(self.loss_config),
                "optimization_config": asdict(self.optimization_config),
                "smpl_model_path": self.smpl_model_path,
                "mhr_assets_dir": self.mhr_assets_dir,
                "input_smpl_cache_dir": self.input_smpl_cache_dir,
            }
        )

    def forward(self, views_input: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(self._preprocess_views_input(views_input))

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
        predictions = self(batch["views_input"])
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

    def _shared_step(self, batch, *, stage: str) -> dict[str, torch.Tensor]:
        predictions = self(batch["views_input"])
        losses = self.criterion(
            pred_body_pose=predictions["pred_body_pose"],
            pred_betas=predictions["pred_betas"],
            target_body_pose=batch["target_body_pose"],
            target_betas=batch["target_betas"],
        )
        joint_loss = self._compute_canonical_joint_loss(
            pred_body_pose=predictions["pred_body_pose"],
            pred_betas=predictions["pred_betas"],
            target_body_pose=batch["target_body_pose"],
            target_betas=batch["target_betas"],
        )
        total_loss = losses["loss"] + self.loss_config.joint_weight * joint_loss
        metrics = {
            f"{stage}/loss": total_loss,
            f"{stage}/loss_body_pose": losses["loss_body_pose"],
            f"{stage}/loss_betas": losses["loss_betas"],
            f"{stage}/loss_joints": joint_loss,
        }
        if stage == "val":
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
        if stage == "test":
            input_view_smpl_params = self.convert_input_views_to_smpl(
                views_input=batch["views_input"],
                pred_cam_t=batch["view_aux"]["pred_cam_t"],
                batch_meta=batch["meta"],
            )
            joint_metrics = self._compute_test_joint_metrics(
                pred_body_pose=predictions["pred_body_pose"],
                pred_betas=predictions["pred_betas"],
                target_body_pose=batch["target_body_pose"],
                target_betas=batch["target_betas"],
                target_aux=batch["target_aux"],
                pred_cam_t=batch["view_aux"]["pred_cam_t"],
                input_view_smpl_params=input_view_smpl_params,
            )
            metrics.update(
                {
                    "test/mpjpe": joint_metrics["mpjpe"],
                    "test/pa_mpjpe": joint_metrics["pa_mpjpe"],
                }
            )
            input_view_metrics = self._compute_input_view_joint_metrics(
                input_view_smpl_params=input_view_smpl_params,
                target_body_pose=batch["target_body_pose"],
                target_betas=batch["target_betas"],
                target_aux=batch["target_aux"],
                pred_cam_t=batch["view_aux"]["pred_cam_t"],
            )
            metrics.update(
                {
                    "test/input_avg_mpjpe": input_view_metrics["mpjpe"],
                    "test/input_avg_pa_mpjpe": input_view_metrics["pa_mpjpe"],
                }
            )
            if "panoptic_joints_world" in batch["target_aux"]:
                panoptic_pred_metrics = self._compute_panoptic_prediction_joint_metrics(
                    pred_body_pose=predictions["pred_body_pose"],
                    pred_betas=predictions["pred_betas"],
                    target_body_pose=batch["target_body_pose"],
                    target_betas=batch["target_betas"],
                    target_aux=batch["target_aux"],
                )
                panoptic_input_metrics = (
                    self._compute_panoptic_input_view_joint_metrics(
                        input_view_smpl_params=input_view_smpl_params,
                        target_aux=batch["target_aux"],
                    )
                )
                metrics.update(
                    {
                        "test/panoptic_mpjpe": panoptic_pred_metrics["mpjpe"],
                        "test/panoptic_pa_mpjpe": panoptic_pred_metrics["pa_mpjpe"],
                        "test/panoptic_target_fit_mpjpe": panoptic_pred_metrics[
                            "target_fit_mpjpe"
                        ],
                        "test/panoptic_input_avg_mpjpe": panoptic_input_metrics[
                            "mpjpe"
                        ],
                        "test/panoptic_input_avg_pa_mpjpe": panoptic_input_metrics[
                            "pa_mpjpe"
                        ],
                        "test/panoptic_input_avg_abs_mpjpe": panoptic_input_metrics[
                            "abs_mpjpe"
                        ],
                    }
                )
        return metrics

    def _compute_canonical_joint_loss(
        self,
        *,
        pred_body_pose: torch.Tensor,
        pred_betas: torch.Tensor,
        target_body_pose: torch.Tensor,
        target_betas: torch.Tensor,
    ) -> torch.Tensor:
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
        return F.mse_loss(pred_joints, target_joints)

    def _compute_canonical_joint_metrics(
        self,
        *,
        pred_body_pose: torch.Tensor,
        pred_betas: torch.Tensor,
        target_body_pose: torch.Tensor,
        target_betas: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
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
        return {
            "mpjpe": batch_mpjpe(pred_joints, target_joints),
            "pa_mpjpe": batch_pa_mpjpe(pred_joints, target_joints),
        }

    def _compute_test_joint_metrics(
        self,
        *,
        pred_body_pose: torch.Tensor,
        pred_betas: torch.Tensor,
        target_body_pose: torch.Tensor,
        target_betas: torch.Tensor,
        target_aux: dict[str, torch.Tensor],
        pred_cam_t: torch.Tensor,
        input_view_smpl_params: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        batch_size = pred_body_pose.shape[0]
        num_views = pred_cam_t.shape[1]
        repeated_pred_body_pose = self._expand_per_view_tensor(
            pred_body_pose, num_views=num_views
        )
        repeated_pred_betas = self._expand_per_view_tensor(
            pred_betas, num_views=num_views
        )
        repeated_target_body_pose = self._expand_per_view_tensor(
            target_body_pose,
            num_views=num_views,
        )
        repeated_target_betas = self._expand_per_view_tensor(
            target_betas, num_views=num_views
        )
        input_global_orient = self._require_converted_parameter(
            input_view_smpl_params,
            "global_orient",
            device=pred_body_pose.device,
        )
        input_transl = self._require_converted_parameter(
            input_view_smpl_params,
            "transl",
            device=pred_body_pose.device,
        )
        pred_joints = self._build_smpl_joints(
            body_pose=repeated_pred_body_pose,
            betas=repeated_pred_betas,
            global_orient=input_global_orient,
            transl=input_transl,
        )
        target_joints = self._build_smpl_joints(
            body_pose=repeated_target_body_pose,
            betas=repeated_target_betas,
            global_orient=input_global_orient,
            transl=input_transl,
        )
        return {
            "mpjpe": batch_mpjpe(pred_joints, target_joints),
            "pa_mpjpe": batch_pa_mpjpe(pred_joints, target_joints),
        }

    def _compute_input_view_joint_metrics(
        self,
        *,
        input_view_smpl_params: dict[str, torch.Tensor],
        target_body_pose: torch.Tensor,
        target_betas: torch.Tensor,
        target_aux: dict[str, torch.Tensor],
        pred_cam_t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = target_body_pose.shape[0]
        num_views = pred_cam_t.shape[1]
        repeated_target_body_pose = (
            target_body_pose[:, None, :]
            .expand(batch_size, num_views, target_body_pose.shape[-1])
            .reshape(batch_size * num_views, -1)
        )
        repeated_target_betas = (
            target_betas[:, None, :]
            .expand(batch_size, num_views, target_betas.shape[-1])
            .reshape(batch_size * num_views, -1)
        )
        input_global_orient = self._require_converted_parameter(
            input_view_smpl_params,
            "global_orient",
            device=target_body_pose.device,
        )
        input_transl = self._require_converted_parameter(
            input_view_smpl_params,
            "transl",
            device=target_body_pose.device,
        )
        pred_joints = self._build_smpl_joints(
            body_pose=self._require_converted_parameter(
                input_view_smpl_params,
                "body_pose",
                device=target_body_pose.device,
            ),
            betas=self._require_converted_parameter(
                input_view_smpl_params,
                "betas",
                device=target_betas.device,
            ),
            global_orient=input_global_orient,
            transl=input_transl,
        )
        target_joints = self._build_smpl_joints(
            body_pose=repeated_target_body_pose,
            betas=repeated_target_betas,
            global_orient=input_global_orient,
            transl=input_transl,
        )
        return {
            "mpjpe": batch_mpjpe(pred_joints, target_joints),
            "pa_mpjpe": batch_pa_mpjpe(pred_joints, target_joints),
        }

    def _compute_panoptic_prediction_joint_metrics(
        self,
        *,
        pred_body_pose: torch.Tensor,
        pred_betas: torch.Tensor,
        target_body_pose: torch.Tensor,
        target_betas: torch.Tensor,
        target_aux: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        batch_size = pred_body_pose.shape[0]
        num_views = target_aux["camera_global_orient"].shape[1]
        repeated_pred_body_pose = self._expand_per_view_tensor(
            pred_body_pose, num_views=num_views
        )
        repeated_pred_betas = self._expand_per_view_tensor(
            pred_betas, num_views=num_views
        )
        repeated_target_body_pose = self._expand_per_view_tensor(
            target_body_pose,
            num_views=num_views,
        )
        repeated_target_betas = self._expand_per_view_tensor(
            target_betas, num_views=num_views
        )
        target_global_orient = target_aux["camera_global_orient"].reshape(
            batch_size * num_views, -1
        )
        target_transl = target_aux["camera_transl"].reshape(batch_size * num_views, -1)

        pred_joints = self._build_smpl_joints(
            body_pose=repeated_pred_body_pose,
            betas=repeated_pred_betas,
            global_orient=target_global_orient.to(pred_body_pose.device),
            transl=target_transl.to(pred_body_pose.device),
        )
        target_joints = self._build_smpl_joints(
            body_pose=repeated_target_body_pose,
            betas=repeated_target_betas,
            global_orient=target_global_orient.to(pred_body_pose.device),
            transl=target_transl.to(pred_body_pose.device),
        )
        panoptic_joints, panoptic_mask = self._build_panoptic_camera_targets(
            target_aux,
            device=pred_joints.device,
            num_views=num_views,
        )
        pred_selected = self._select_panoptic_smpl_joints(pred_joints)
        target_selected = self._select_panoptic_smpl_joints(target_joints)
        panoptic_pred_mpjpe = self._masked_root_centered_mpjpe(
            pred_selected,
            panoptic_joints,
            panoptic_mask,
        )
        panoptic_pred_pa_mpjpe = self._masked_root_centered_pa_mpjpe(
            pred_selected,
            panoptic_joints,
            panoptic_mask,
        )
        target_fit_mpjpe = self._masked_mpjpe(
            target_selected,
            panoptic_joints,
            panoptic_mask,
        )
        return {
            "mpjpe": panoptic_pred_mpjpe,
            "pa_mpjpe": panoptic_pred_pa_mpjpe,
            "target_fit_mpjpe": target_fit_mpjpe,
        }

    def _compute_panoptic_input_view_joint_metrics(
        self,
        *,
        input_view_smpl_params: dict[str, torch.Tensor],
        target_aux: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        num_views = target_aux["camera_rotation"].shape[1]
        input_body_pose = self._require_converted_parameter(
            input_view_smpl_params,
            "body_pose",
            device=target_aux["camera_rotation"].device,
        )
        input_joints = self._build_smpl_joints(
            body_pose=input_body_pose,
            betas=self._require_converted_parameter(
                input_view_smpl_params,
                "betas",
                device=input_body_pose.device,
            ),
            global_orient=self._require_converted_parameter(
                input_view_smpl_params,
                "global_orient",
                device=input_body_pose.device,
            ),
            transl=self._require_converted_parameter(
                input_view_smpl_params,
                "transl",
                device=input_body_pose.device,
            ),
        )
        panoptic_joints, panoptic_mask = self._build_panoptic_camera_targets(
            target_aux,
            device=input_joints.device,
            num_views=num_views,
        )
        input_selected = self._select_panoptic_smpl_joints(input_joints)
        return {
            "mpjpe": self._masked_root_centered_mpjpe(
                input_selected,
                panoptic_joints,
                panoptic_mask,
            ),
            "pa_mpjpe": self._masked_root_centered_pa_mpjpe(
                input_selected,
                panoptic_joints,
                panoptic_mask,
            ),
            "abs_mpjpe": self._masked_mpjpe(
                input_selected,
                panoptic_joints,
                panoptic_mask,
            ),
        }

    def _build_panoptic_camera_targets(
        self,
        target_aux: dict[str, torch.Tensor],
        *,
        device: torch.device,
        num_views: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        panoptic_joints_world = target_aux["panoptic_joints_world"].to(device)
        panoptic_confidence = target_aux["panoptic_confidence"].to(device)
        panoptic_joint_indices = torch.as_tensor(
            PANOPTIC_EVAL_JOINT_INDICES,
            dtype=torch.long,
            device=device,
        )
        selected_world = panoptic_joints_world.index_select(1, panoptic_joint_indices)
        selected_mask = (
            panoptic_confidence.index_select(1, panoptic_joint_indices) > 0.05
        )

        batch_size = selected_world.shape[0]
        repeated_world = (
            selected_world[:, None, :, :]
            .expand(batch_size, num_views, selected_world.shape[1], 3)
            .reshape(batch_size * num_views, selected_world.shape[1], 3)
        )
        repeated_mask = (
            selected_mask[:, None, :]
            .expand(batch_size, num_views, selected_mask.shape[1])
            .reshape(batch_size * num_views, selected_mask.shape[1])
        )
        camera_rotation = (
            target_aux["camera_rotation"]
            .to(device)
            .reshape(
                batch_size * num_views,
                3,
                3,
            )
        )
        camera_translation = (
            target_aux["camera_translation"]
            .to(device)
            .reshape(
                batch_size * num_views,
                3,
            )
        )
        return (
            self._transform_points_world_to_camera(
                points_world=repeated_world,
                rotation=camera_rotation,
                translation=camera_translation,
            ),
            repeated_mask,
        )

    @staticmethod
    def _select_panoptic_smpl_joints(joints: torch.Tensor) -> torch.Tensor:
        smpl_joint_indices = torch.as_tensor(
            PANOPTIC_EVAL_SMPL24_INDICES,
            dtype=torch.long,
            device=joints.device,
        )
        return joints.index_select(1, smpl_joint_indices)

    @staticmethod
    def _masked_mpjpe(
        pred_joints: torch.Tensor,
        target_joints: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        errors = torch.linalg.norm(pred_joints - target_joints, dim=-1)
        weights = mask.to(dtype=errors.dtype)
        valid_counts = weights.sum(dim=1)
        valid_samples = valid_counts > 0
        if not bool(valid_samples.any()):
            return errors.new_tensor(float("nan"))
        per_sample = (errors * weights).sum(dim=1) / valid_counts.clamp_min(1.0)
        return per_sample[valid_samples].mean()

    @classmethod
    def _masked_root_centered_mpjpe(
        cls,
        pred_joints: torch.Tensor,
        target_joints: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return cls._masked_mpjpe(
            cls._root_center_panoptic_joints(pred_joints),
            cls._root_center_panoptic_joints(target_joints),
            mask,
        )

    @classmethod
    def _masked_root_centered_pa_mpjpe(
        cls,
        pred_joints: torch.Tensor,
        target_joints: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        pred_centered = cls._root_center_panoptic_joints(pred_joints)
        target_centered = cls._root_center_panoptic_joints(target_joints)
        valid_samples = mask.sum(dim=1) >= 3
        if not bool(valid_samples.any()):
            return pred_joints.new_tensor(float("nan"))
        if bool(mask[valid_samples].all()):
            return batch_pa_mpjpe(
                pred_centered[valid_samples],
                target_centered[valid_samples],
            )

        values = []
        for sample_index in valid_samples.nonzero(as_tuple=False).flatten().tolist():
            sample_mask = mask[sample_index]
            values.append(
                batch_pa_mpjpe(
                    pred_centered[sample_index : sample_index + 1, sample_mask],
                    target_centered[sample_index : sample_index + 1, sample_mask],
                )
            )
        return torch.stack(values).mean()

    @staticmethod
    def _root_center_panoptic_joints(joints: torch.Tensor) -> torch.Tensor:
        return (
            joints
            - joints[:, PANOPTIC_EVAL_ROOT_COLUMN : PANOPTIC_EVAL_ROOT_COLUMN + 1, :]
        )

    def convert_input_views_to_smpl(
        self,
        *,
        views_input: torch.Tensor,
        pred_cam_t: torch.Tensor,
        batch_meta: list[dict[str, object]] | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, num_views, _ = views_input.shape
        flat_inputs = views_input.reshape(batch_size * num_views, -1)
        flat_pred_cam_t = pred_cam_t.reshape(batch_size * num_views, -1)
        flat_source_npz_paths = None
        if batch_meta is not None:
            flat_source_npz_paths = []
            for sample_meta in batch_meta:
                sample_paths = sample_meta.get("view_npz_paths")
                if not isinstance(sample_paths, list):
                    raise KeyError("Expected batch meta to contain 'view_npz_paths'")
                if len(sample_paths) != num_views:
                    raise ValueError(
                        "Expected each meta['view_npz_paths'] entry to match num_views, "
                        f"got {len(sample_paths)} paths for num_views={num_views}"
                    )
                flat_source_npz_paths.extend(str(path) for path in sample_paths)
        return self._get_mhr_to_smpl_converter().convert(
            mhr_model_params=flat_inputs[:, :204],
            shape_params=flat_inputs[:, 204:249],
            pred_cam_t=flat_pred_cam_t,
            source_npz_paths=flat_source_npz_paths,
        )

    def _build_smpl_joints(
        self,
        *,
        body_pose: torch.Tensor,
        betas: torch.Tensor,
        global_orient: torch.Tensor,
        transl: torch.Tensor,
    ) -> torch.Tensor:
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
        return output.joints[:, :SMPL_EVAL_NUM_JOINTS, :]

    def _get_smpl_eval_model(self, *, device: torch.device, batch_size: int):
        smpl_eval_model = self._runtime_cache["smpl_eval_model"]
        if smpl_eval_model is None:
            smpl_eval_model = build_smpl_model(
                device=device,
                smpl_model_path=self.smpl_model_path,
                batch_size=batch_size,
            )
            smpl_eval_model.eval()
        else:
            needs_rebuild = False
            assert smpl_eval_model is not None
            if next(smpl_eval_model.parameters()).device != device:
                needs_rebuild = True
            if getattr(smpl_eval_model, "batch_size", None) != batch_size:
                needs_rebuild = True
            if needs_rebuild:
                smpl_eval_model = build_smpl_model(
                    device=device,
                    smpl_model_path=self.smpl_model_path,
                    batch_size=batch_size,
                )
                smpl_eval_model.eval()
        self._runtime_cache["smpl_eval_model"] = smpl_eval_model
        return smpl_eval_model

    @staticmethod
    def _expand_per_view_tensor(
        tensor: torch.Tensor, *, num_views: int
    ) -> torch.Tensor:
        return (
            tensor[:, None, :]
            .expand(tensor.shape[0], num_views, tensor.shape[-1])
            .reshape(tensor.shape[0] * num_views, tensor.shape[-1])
        )

    @staticmethod
    def _transform_points_world_to_camera(
        *,
        points_world: torch.Tensor,
        rotation: torch.Tensor,
        translation: torch.Tensor,
    ) -> torch.Tensor:
        rotated = torch.bmm(
            rotation.float(), points_world.float().transpose(1, 2)
        ).transpose(1, 2)
        return rotated + translation.float()[:, None, :]

    @staticmethod
    def _require_converted_parameter(
        converted_parameters: dict[str, torch.Tensor],
        key: str,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        try:
            return converted_parameters[key].to(device)
        except KeyError as exc:
            raise KeyError(
                f"Expected converted SMPL parameters to contain '{key}'"
            ) from exc

    def _get_mhr_to_smpl_converter(self) -> MHRToSMPLConverter:
        converter = self._runtime_cache["mhr_to_smpl_converter"]
        if converter is None:
            converter = MHRToSMPLConverter(
                smpl_model_path=self.smpl_model_path,
                mhr_assets_dir=self.mhr_assets_dir,
                cache_dir=self.input_smpl_cache_dir,
            )
            self._runtime_cache["mhr_to_smpl_converter"] = converter
        return converter

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

    def _preprocess_views_input(self, views_input: torch.Tensor) -> torch.Tensor:
        if not self.model_config.zero_mhr_root_input:
            return views_input
        if views_input.shape[-1] < 6:
            raise ValueError(
                "Expected views_input trailing dimension >= 6 when zero_mhr_root_input is enabled, "
                f"got {views_input.shape[-1]}"
            )
        processed = views_input.clone()
        processed[..., :6] = 0.0
        return processed


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


def _coerce_model_config(value) -> Stage1ModelConfig:
    if value is None:
        return Stage1MLPFusionConfig()
    if isinstance(
        value,
        (
            Stage1MLPFusionConfig,
            Stage1ResidualFusionConfig,
        ),
    ):
        return value
    if isinstance(value, dict):
        model_kwargs = dict(value)
        model_name = str(model_kwargs.pop("name", "stage1_mlp_fusion"))
        if model_name == "stage1_residual_fusion":
            return Stage1ResidualFusionConfig(**model_kwargs)
        return Stage1MLPFusionConfig(**model_kwargs)
    raise TypeError(
        f"Expected model config dataclass, dict, or None; got {type(value)!r}"
    )


def _build_stage1_model(config: Stage1ModelConfig):
    if isinstance(config, Stage1ResidualFusionConfig):
        return Stage1ResidualFusionModel(config)
    return Stage1MLPFusionModel(config)
