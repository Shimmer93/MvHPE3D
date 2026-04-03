"""LightningModule for the Stage 1 baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import lightning as L
import torch

from mvhpe3d.losses import Stage1LossConfig, Stage1Loss
from mvhpe3d.models import Stage1MLPFusionConfig, Stage1MLPFusionModel


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

        self.model = Stage1MLPFusionModel(self.model_config)
        self.criterion = Stage1Loss(self.loss_config)

        self.save_hyperparameters(
            {
                "model_config": asdict(self.model_config),
                "loss_config": asdict(self.loss_config),
                "optimization_config": asdict(self.optimization_config),
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
        return {
            f"{stage}/loss": losses["loss"],
            f"{stage}/loss_body_pose": losses["loss_body_pose"],
            f"{stage}/loss_betas": losses["loss_betas"],
        }

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
