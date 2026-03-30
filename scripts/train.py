#!/usr/bin/env python
"""Train the Stage 1 baseline with PyTorch Lightning."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from mvhpe3d.data import Stage1DataConfig, Stage1HuMManDataModule
from mvhpe3d.lightning import Stage1FusionLightningModule, Stage1OptimizationConfig
from mvhpe3d.losses import Stage1LossConfig
from mvhpe3d.models import Stage1MLPFusionConfig
from mvhpe3d.utils import load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Stage 1 MvHPE3D baseline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment/stage1_cross_camera.yaml",
        help="Path to the experiment YAML file",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Optional override for the data manifest path",
    )
    parser.add_argument(
        "--default-root-dir",
        type=str,
        default="outputs/stage1",
        help="Trainer default root dir for checkpoints and logs",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Optional override for trainer.max_epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for the experiment seed",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run a single Lightning fast-dev-run iteration",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = load_experiment_config(args.config)

    data_config = build_data_config(experiment["data"], args)
    model_config = build_model_config(experiment["model"])
    loss_config = Stage1LossConfig(**experiment["loss"])
    optimization_config = Stage1OptimizationConfig(**experiment["optimizer"])

    L.seed_everything(data_config.seed, workers=True)

    datamodule = Stage1HuMManDataModule(data_config)
    module = Stage1FusionLightningModule(
        model_config=model_config,
        loss_config=loss_config,
        optimization_config=optimization_config,
    )

    trainer_config = build_trainer_config(experiment["trainer"], args, experiment["experiment_name"])
    trainer = L.Trainer(**trainer_config)
    trainer.fit(module, datamodule=datamodule)


def build_data_config(config: dict[str, Any], args: argparse.Namespace) -> Stage1DataConfig:
    data_kwargs = dict(config)
    data_kwargs.pop("name", None)
    data_kwargs.pop("_config_path", None)

    if args.manifest_path is not None:
        data_kwargs["manifest_path"] = args.manifest_path
    if args.seed is not None:
        data_kwargs["seed"] = args.seed
    if args.fast_dev_run:
        data_kwargs["drop_last_train"] = False

    return Stage1DataConfig(**data_kwargs)


def build_model_config(config: dict[str, Any]) -> Stage1MLPFusionConfig:
    model_kwargs = dict(config)
    model_kwargs.pop("name", None)
    model_kwargs.pop("_config_path", None)
    return Stage1MLPFusionConfig(**model_kwargs)


def build_trainer_config(
    config: dict[str, Any],
    args: argparse.Namespace,
    experiment_name: str,
) -> dict[str, Any]:
    trainer_kwargs = dict(config)
    trainer_kwargs.pop("_config_path", None)

    if args.max_epochs is not None:
        trainer_kwargs["max_epochs"] = args.max_epochs
    if args.fast_dev_run:
        trainer_kwargs["fast_dev_run"] = True

    root_dir = Path(args.default_root_dir).resolve()
    logger = CSVLogger(save_dir=str(root_dir / "logs"), name=experiment_name)
    checkpoint = ModelCheckpoint(
        dirpath=str(root_dir / "checkpoints" / experiment_name),
        filename="epoch{epoch:03d}-step{step:06d}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    trainer_kwargs["default_root_dir"] = str(root_dir)
    trainer_kwargs["logger"] = logger
    trainer_kwargs["callbacks"] = [checkpoint]
    return trainer_kwargs


if __name__ == "__main__":
    main()
