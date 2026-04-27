"""LightningModule for the Stage 3 temporal refinement model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import lightning as L
import torch
import torch.nn.functional as F

from mvhpe3d.losses import Stage3Loss, Stage3LossConfig
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
    Stage3TemporalRefineConfig,
    Stage3TemporalRefineModel,
)
from mvhpe3d.utils import build_smpl_model

from .stage2_module import Stage2FusionLightningModule


@dataclass(slots=True)
class Stage3OptimizationConfig:
    """Optimizer settings for Stage 3 training."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


class Stage3TemporalLightningModule(L.LightningModule):
    """Lightning wrapper around the Stage 3 temporal refinement model."""

    strict_loading = False

    def __init__(
        self,
        *,
        model_config: Stage3TemporalRefineConfig | dict | None = None,
        loss_config: Stage3LossConfig | dict | None = None,
        optimization_config: Stage3OptimizationConfig | dict | None = None,
        smpl_model_path: str | None = None,
        stage2_checkpoint_path: str | None = None,
        stage2_backbone_model: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.model_config = _coerce_dataclass_config(model_config, Stage3TemporalRefineConfig)
        self.loss_config = _coerce_dataclass_config(loss_config, Stage3LossConfig)
        if not self.model_config.learn_betas:
            self.loss_config.supervise_betas = False
            self.loss_config.betas_weight = 0.0
        self.optimization_config = _coerce_dataclass_config(
            optimization_config,
            Stage3OptimizationConfig,
        )
        self.smpl_model_path = smpl_model_path
        self.stage2_checkpoint_path = stage2_checkpoint_path

        self.model = Stage3TemporalRefineModel(self.model_config)
        self.criterion = Stage3Loss(self.loss_config)
        self.stage2_backbone = self._load_stage2_backbone(
            stage2_checkpoint_path=stage2_checkpoint_path,
            stage2_backbone_model=stage2_backbone_model,
        )
        self._runtime_cache: dict[str, object | None] = {
            "smpl_eval_model": None,
        }

        self.save_hyperparameters(
            {
                "model_config": asdict(self.model_config),
                "loss_config": asdict(self.loss_config),
                "optimization_config": asdict(self.optimization_config),
                "smpl_model_path": self.smpl_model_path,
                "stage2_checkpoint_path": self.stage2_checkpoint_path,
            }
        )

    def forward(self, views_input: torch.Tensor) -> dict[str, torch.Tensor]:
        if views_input.ndim != 4:
            raise ValueError(
                "Stage3TemporalLightningModule expects views_input with shape "
                f"[batch, time, num_views, dim], got {tuple(views_input.shape)}"
            )
        batch_size, num_frames, num_views, dim = views_input.shape
        flat_input = views_input.reshape(batch_size * num_frames, num_views, dim)
        stage2_outputs = self._run_stage2_backbone(flat_input)
        init_pose_6d = stage2_outputs["init_pose_6d"].reshape(batch_size, num_frames, -1)
        pred_pose_6d = stage2_outputs["pred_pose_6d"].reshape(batch_size, num_frames, -1)
        pred_betas = stage2_outputs["pred_betas"].reshape(batch_size, num_frames, -1)
        temporal_features = torch.cat((init_pose_6d, pred_pose_6d, pred_betas), dim=-1)
        target_index = self._target_frame_index(num_frames)
        outputs = self.model(
            temporal_features,
            base_pose_6d=pred_pose_6d[:, target_index],
            base_betas=pred_betas[:, target_index],
        )
        outputs["stage2_pred_pose_6d"] = pred_pose_6d[:, target_index].reshape(
            batch_size,
            self.model_config.num_joints,
            6,
        )
        outputs["stage2_pred_betas"] = pred_betas[:, target_index]
        outputs["stage2_init_pose_6d"] = init_pose_6d[:, target_index].reshape(
            batch_size,
            self.model_config.num_joints,
            6,
        )
        return outputs

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
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        return torch.optim.AdamW(
            parameters,
            lr=self.optimization_config.learning_rate,
            weight_decay=self.optimization_config.weight_decay,
        )

    def _shared_step(self, batch, *, stage: str) -> dict[str, torch.Tensor]:
        self._assert_finite_tensor("batch/views_input", batch["views_input"])
        self._assert_finite_tensor("batch/target_body_pose_6d", batch["target_body_pose_6d"])
        self._assert_finite_tensor("batch/target_betas", batch["target_betas"])
        self._assert_finite_tensor("batch/target_body_pose", batch["target_body_pose"])
        predictions = self(batch["views_input"])
        for name, value in predictions.items():
            if torch.is_tensor(value):
                self._assert_finite_tensor(f"predictions/{name}", value)
        losses = self.criterion(
            pred_pose_6d=predictions["pred_pose_6d"],
            pred_betas=predictions["pred_betas"],
            target_pose_6d=batch["target_body_pose_6d"],
            target_betas=batch["target_betas"],
            pose_residual_6d=predictions["pose_residual_6d"],
            betas_residual=predictions["betas_residual"],
        )
        for name, value in losses.items():
            self._assert_finite_tensor(f"losses/{name}", value)

        pred_joints, target_joints = self._build_canonical_joint_pair(
            pred_body_pose=predictions["pred_body_pose"],
            pred_betas=predictions["pred_betas"],
            target_body_pose=batch["target_body_pose"],
            target_betas=batch["target_betas"],
        )
        joint_loss = F.mse_loss(pred_joints, target_joints)
        self._assert_finite_tensor("losses/joint_loss", joint_loss)
        articulation_loss = self._compute_articulation_loss(
            pred_joints=pred_joints,
            target_joints=target_joints,
        )
        self._assert_finite_tensor("losses/articulation_loss", articulation_loss)
        total_loss = (
            losses["loss"]
            + self.loss_config.joint_weight * joint_loss
            + self.loss_config.articulation_weight * articulation_loss
        )
        self._assert_finite_tensor("losses/total_loss", total_loss)
        metrics = {
            f"{stage}/loss": total_loss,
            f"{stage}/loss_pose_6d": losses["loss_pose_6d"],
            f"{stage}/loss_betas": losses["loss_betas"],
            f"{stage}/loss_pose_residual": losses["loss_pose_residual"],
            f"{stage}/loss_betas_residual": losses["loss_betas_residual"],
            f"{stage}/loss_joints": joint_loss,
            f"{stage}/loss_articulation": articulation_loss,
            f"{stage}/pose_residual_abs_mean": (
                predictions["pose_residual_6d"].detach().abs().mean()
            ),
            f"{stage}/pose_residual_abs_max": (
                predictions["pose_residual_6d"].detach().abs().amax()
            ),
            f"{stage}/pred_pose_6d_abs_max": predictions["pred_pose_6d"].detach().abs().amax(),
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
        return metrics

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
            )
            backbone = loaded_module.model
        else:
            backbone = _build_stage2_backbone_from_name(
                self.model_config.backbone_name,
                learn_betas=True,
            )
        if self.model_config.freeze_backbone:
            backbone.requires_grad_(False)
            backbone.eval()
        return backbone

    def _run_stage2_backbone(self, views_input: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.model_config.freeze_backbone:
            self.stage2_backbone.eval()
            with torch.no_grad():
                return self.stage2_backbone(views_input)
        return self.stage2_backbone(views_input)

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

    def _compute_articulation_loss(
        self,
        *,
        pred_joints: torch.Tensor,
        target_joints: torch.Tensor,
    ) -> torch.Tensor:
        with torch.autocast(device_type=pred_joints.device.type, enabled=False):
            aligned_pred = batch_similarity_align(pred_joints.float(), target_joints.float())
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
        target_index = self._target_frame_index(views_input.shape[1])
        target_views_input = views_input[:, target_index]
        input_pose_6d = target_views_input[..., : self.model_config.pose_6d_dim].reshape(
            target_views_input.shape[0],
            target_views_input.shape[1],
            self.model_config.num_joints,
            6,
        )
        input_betas = target_views_input[..., self.model_config.pose_6d_dim :]
        return compute_input_corrected_camera_joint_metrics(
            build_smpl_joints=self._build_smpl_joints,
            pred_body_pose=pred_body_pose,
            pred_betas=pred_betas,
            target_body_pose=target_body_pose,
            target_betas=target_betas,
            input_views_pose_6d=input_pose_6d,
            input_views_betas=input_betas,
            input_camera_global_orient=view_aux["input_global_orient"].to(pred_body_pose.device),
            input_transl=view_aux["input_transl"].to(pred_body_pose.device),
        )

    def _target_frame_index(self, num_frames: int) -> int:
        return self.model.target_frame_index(num_frames)

    def _build_smpl_joints(
        self,
        *,
        body_pose: torch.Tensor,
        betas: torch.Tensor,
        global_orient: torch.Tensor,
        transl: torch.Tensor,
    ) -> torch.Tensor:
        smpl_model = self._get_smpl_eval_model(device=body_pose.device, batch_size=body_pose.shape[0])
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

    def _assert_finite_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if torch.isfinite(tensor).all():
            return
        detached = tensor.detach()
        nonfinite_mask = ~torch.isfinite(detached)
        bad_count = int(nonfinite_mask.sum().item())
        finite_mask = torch.isfinite(detached)
        min_value = float(detached[finite_mask].min().item()) if finite_mask.any() else float("nan")
        max_value = float(detached[finite_mask].max().item()) if finite_mask.any() else float("nan")
        raise RuntimeError(
            f"Non-finite tensor detected at {name}: "
            f"shape={tuple(detached.shape)}, bad_count={bad_count}, "
            f"finite_min={min_value}, finite_max={max_value}"
        )


def _coerce_dataclass_config(value, config_type):
    if value is None:
        return config_type()
    if isinstance(value, config_type):
        return value
    if isinstance(value, dict):
        return config_type(**value)
    raise TypeError(f"Expected {config_type.__name__} or mapping, got {type(value)!r}")


def _build_stage2_backbone_from_name(name: str, *, learn_betas: bool) -> torch.nn.Module:
    if name == "stage2_joint_graph_refiner":
        return Stage2JointGraphRefinerModel(Stage2JointGraphRefinerConfig(learn_betas=learn_betas))
    if name == "stage2_param_refine":
        return Stage2ParamRefineModel(Stage2ParamRefineConfig(learn_betas=learn_betas))
    return Stage2JointResidualModel(Stage2JointResidualConfig(learn_betas=learn_betas))
