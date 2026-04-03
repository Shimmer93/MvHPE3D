"""LightningModule for the Stage 1 baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import lightning as L
import torch

from mvhpe3d.metrics import SMPL_EVAL_NUM_JOINTS, batch_mpjpe, batch_pa_mpjpe, root_center_joints
from mvhpe3d.losses import Stage1LossConfig, Stage1Loss
from mvhpe3d.models import Stage1MLPFusionConfig, Stage1MLPFusionModel
from mvhpe3d.utils import MHRToSMPLConverter, build_smpl_model


@dataclass(slots=True)
class Stage1OptimizationConfig:
    """Optimizer settings for Stage 1 training."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


class Stage1FusionLightningModule(L.LightningModule):
    """Lightning wrapper around the Stage 1 fusion baseline."""

    def __init__(
        self,
        *,
        model_config: Stage1MLPFusionConfig | None = None,
        loss_config: Stage1LossConfig | None = None,
        optimization_config: Stage1OptimizationConfig | None = None,
        smpl_model_path: str | None = None,
        mhr_assets_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.model_config = _coerce_dataclass_config(
            model_config,
            Stage1MLPFusionConfig,
        )
        self.loss_config = _coerce_dataclass_config(
            loss_config,
            Stage1LossConfig,
        )
        self.optimization_config = _coerce_dataclass_config(
            optimization_config,
            Stage1OptimizationConfig,
        )
        self.smpl_model_path = smpl_model_path
        self.mhr_assets_dir = mhr_assets_dir

        self.model = Stage1MLPFusionModel(self.model_config)
        self.criterion = Stage1Loss(self.loss_config)
        self._smpl_eval_model = None
        self._mhr_to_smpl_converter = None

        self.save_hyperparameters(
            {
                "model_config": asdict(self.model_config),
                "loss_config": asdict(self.loss_config),
                "optimization_config": asdict(self.optimization_config),
                "smpl_model_path": self.smpl_model_path,
                "mhr_assets_dir": self.mhr_assets_dir,
            }
        )

    def forward(self, views_input: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(views_input)

    def training_step(self, batch, batch_idx):
        metrics = self._shared_step(batch, stage="train")
        self._log_metrics_if_possible(
            metrics,
            batch_size=batch["views_input"].shape[0],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
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
        metrics = {
            f"{stage}/loss": losses["loss"],
            f"{stage}/loss_body_pose": losses["loss_body_pose"],
            f"{stage}/loss_betas": losses["loss_betas"],
        }
        if stage != "train":
            joint_metrics = self._compute_joint_metrics(
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
            input_view_metrics = self._compute_input_view_joint_metrics(
                views_input=batch["views_input"],
                target_body_pose=batch["target_body_pose"],
                target_betas=batch["target_betas"],
            )
            metrics.update(
                {
                    "test/input_avg_mpjpe": input_view_metrics["mpjpe"],
                    "test/input_avg_pa_mpjpe": input_view_metrics["pa_mpjpe"],
                }
            )
        return metrics

    def _compute_joint_metrics(
        self,
        *,
        pred_body_pose: torch.Tensor,
        pred_betas: torch.Tensor,
        target_body_pose: torch.Tensor,
        target_betas: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self._current_joint_metric_batch_size = pred_body_pose.shape[0]
        smpl_model = self._get_smpl_eval_model(device=pred_body_pose.device)
        batch_size = pred_body_pose.shape[0]
        zero_root = torch.zeros(
            (batch_size, 3),
            dtype=pred_body_pose.dtype,
            device=pred_body_pose.device,
        )
        zero_transl = torch.zeros_like(zero_root)

        pred_smpl = smpl_model(
            body_pose=pred_body_pose,
            betas=pred_betas,
            global_orient=zero_root,
            transl=zero_transl,
        )
        target_smpl = smpl_model(
            body_pose=target_body_pose,
            betas=target_betas,
            global_orient=zero_root,
            transl=zero_transl,
        )

        pred_joints = root_center_joints(pred_smpl.joints[:, :SMPL_EVAL_NUM_JOINTS, :])
        target_joints = root_center_joints(target_smpl.joints[:, :SMPL_EVAL_NUM_JOINTS, :])
        return {
            "mpjpe": batch_mpjpe(pred_joints, target_joints),
            "pa_mpjpe": batch_pa_mpjpe(pred_joints, target_joints),
        }

    def _compute_input_view_joint_metrics(
        self,
        *,
        views_input: torch.Tensor,
        target_body_pose: torch.Tensor,
        target_betas: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, num_views, _ = views_input.shape
        flat_inputs = views_input.reshape(batch_size * num_views, -1)
        converted_parameters = self._get_mhr_to_smpl_converter().convert(
            mhr_model_params=flat_inputs[:, :204],
            shape_params=flat_inputs[:, 204:249],
        )
        try:
            converted_body_pose = converted_parameters["body_pose"].to(target_body_pose.device)
            converted_betas = converted_parameters["betas"].to(target_betas.device)
        except KeyError as exc:
            raise KeyError(
                "Expected converted SMPL parameters to contain 'body_pose' and 'betas'"
            ) from exc

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
        return self._compute_joint_metrics(
            pred_body_pose=converted_body_pose,
            pred_betas=converted_betas,
            target_body_pose=repeated_target_body_pose,
            target_betas=repeated_target_betas,
        )

    def _get_smpl_eval_model(self, *, device: torch.device):
        batch_size = getattr(self, "_current_joint_metric_batch_size", 1)
        if self._smpl_eval_model is None:
            self._smpl_eval_model = build_smpl_model(
                device=device,
                smpl_model_path=self.smpl_model_path,
                batch_size=batch_size,
            )
            self._smpl_eval_model.eval()
        else:
            needs_rebuild = False
            if next(self._smpl_eval_model.parameters()).device != device:
                needs_rebuild = True
            if getattr(self._smpl_eval_model, "batch_size", None) != batch_size:
                needs_rebuild = True
            if needs_rebuild:
                self._smpl_eval_model = build_smpl_model(
                    device=device,
                    smpl_model_path=self.smpl_model_path,
                    batch_size=batch_size,
                )
                self._smpl_eval_model.eval()
        return self._smpl_eval_model

    def _get_mhr_to_smpl_converter(self) -> MHRToSMPLConverter:
        if self._mhr_to_smpl_converter is None:
            self._mhr_to_smpl_converter = MHRToSMPLConverter(
                smpl_model_path=self.smpl_model_path,
                mhr_assets_dir=self.mhr_assets_dir,
            )
        return self._mhr_to_smpl_converter

    def _log_metrics_if_possible(
        self,
        metrics: dict[str, torch.Tensor],
        *,
        batch_size: int,
        on_step: bool,
        on_epoch: bool,
        prog_bar: bool,
    ) -> None:
        if getattr(self, "_trainer", None) is None:
            return
        self.log_dict(
            metrics,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
        )


def _coerce_dataclass_config(value, config_type):
    if value is None:
        return config_type()
    if isinstance(value, config_type):
        return value
    if isinstance(value, dict):
        return config_type(**value)
    raise TypeError(f"Expected {config_type.__name__}, dict, or None; got {type(value)!r}")
