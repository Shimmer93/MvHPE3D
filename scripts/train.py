#!/usr/bin/env python
"""Train the Stage 1 baseline with PyTorch Lightning."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, Logger, WandbLogger

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mvhpe3d.data import Stage1DataConfig, Stage1HuMManDataModule
from mvhpe3d.lightning import Stage1FusionLightningModule, Stage1OptimizationConfig
from mvhpe3d.losses import Stage1LossConfig
from mvhpe3d.models import Stage1MLPFusionConfig
from mvhpe3d.utils import load_experiment_config, validate_mhr_asset_folder


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
        "--gt-smpl-dir",
        type=str,
        default=None,
        help="Optional override for the HuMMan GT SMPL directory",
    )
    parser.add_argument(
        "--cameras-dir",
        type=str,
        default=None,
        help="Optional override for the HuMMan camera JSON directory",
    )
    parser.add_argument(
        "--split-config-path",
        type=str,
        default=None,
        help="Optional override for the split policy YAML path",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default=None,
        help="Optional override for the named split policy in the split config YAML",
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
        "--accelerator",
        type=str,
        default=None,
        help="Optional override for trainer.accelerator",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Optional override for trainer.devices, e.g. 2, auto, or 0,1",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Optional override for trainer.strategy, e.g. ddp",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=None,
        help="Optional override for trainer.num_nodes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for the experiment seed",
    )
    parser.add_argument(
        "--smpl-model-path",
        type=str,
        default=None,
        help="Optional override for the neutral SMPL model used for validation metrics",
    )
    parser.add_argument(
        "--mhr-assets-dir",
        type=str,
        default=None,
        help="Optional override for the MHR asset directory used for test-time input conversion",
    )
    parser.add_argument(
        "--input-smpl-cache-dir",
        type=str,
        default=None,
        help="Optional directory for caching fitted SMPL parameters converted from input views",
    )
    parser.add_argument(
        "--test-after-train",
        action="store_true",
        help="Run a test pass immediately after training finishes",
    )
    parser.add_argument(
        "--test-ckpt",
        type=str,
        choices=("best", "last", "current"),
        default="best",
        help="Checkpoint source to use for --test-after-train",
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
    validate_required_mhr_assets(args)

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
        smpl_model_path=args.smpl_model_path,
        mhr_assets_dir=args.mhr_assets_dir,
        input_smpl_cache_dir=resolve_input_smpl_cache_dir(args, data_config),
    )

    trainer_config = build_trainer_config(experiment["trainer"], args, experiment["experiment_name"])
    trainer = L.Trainer(**trainer_config)
    trainer.fit(module, datamodule=datamodule)
    maybe_run_test_after_train(
        trainer,
        module=module,
        datamodule=datamodule,
        args=args,
    )


def build_data_config(config: dict[str, Any], args: argparse.Namespace) -> Stage1DataConfig:
    data_kwargs = dict(config)
    data_kwargs.pop("name", None)
    data_kwargs.pop("_config_path", None)

    if args.manifest_path is not None:
        data_kwargs["manifest_path"] = args.manifest_path
    if args.gt_smpl_dir is not None:
        data_kwargs["gt_smpl_dir"] = args.gt_smpl_dir
    if args.cameras_dir is not None:
        data_kwargs["cameras_dir"] = args.cameras_dir
    if args.split_config_path is not None:
        data_kwargs["split_config_path"] = args.split_config_path
    if args.split_name is not None:
        data_kwargs["split_name"] = args.split_name
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
    if args.accelerator is not None:
        trainer_kwargs["accelerator"] = args.accelerator
    if args.devices is not None:
        trainer_kwargs["devices"] = _parse_devices_arg(args.devices)
    if args.strategy is not None:
        trainer_kwargs["strategy"] = args.strategy
    if args.num_nodes is not None:
        trainer_kwargs["num_nodes"] = args.num_nodes
    if args.fast_dev_run:
        trainer_kwargs["fast_dev_run"] = True

    root_dir = Path(args.default_root_dir).resolve()
    logger = build_loggers(root_dir=root_dir, experiment_name=experiment_name)
    csv_logger = logger[0] if isinstance(logger, list) else logger
    checkpoint = ModelCheckpoint(
        dirpath=str(Path(csv_logger.log_dir) / "checkpoints"),
        filename="{epoch:03d}-{step:06d}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    trainer_kwargs["default_root_dir"] = str(root_dir)
    trainer_kwargs["logger"] = logger
    trainer_kwargs["callbacks"] = [checkpoint]
    trainer_kwargs["inference_mode"] = False
    return trainer_kwargs


def build_loggers(root_dir: Path, experiment_name: str) -> Logger | list[Logger]:
    csv_logger = CSVLogger(save_dir=str(root_dir), name=experiment_name)
    wandb_logger = build_optional_wandb_logger(root_dir=root_dir, experiment_name=experiment_name)
    if wandb_logger is None:
        return csv_logger
    return [csv_logger, wandb_logger]


def build_optional_wandb_logger(root_dir: Path, experiment_name: str) -> WandbLogger | None:
    try:
        import wandb
    except Exception:
        return None

    settings = wandb.Settings(start_method="thread", init_timeout=60)
    project = os.environ.get("WANDB_PROJECT", "MvHPE3D")
    mode = os.environ.get("WANDB_MODE")
    if mode is None and os.environ.get("WANDB_API_KEY") is None:
        mode = "offline"

    try:
        return WandbLogger(
            save_dir=str(root_dir),
            project=project,
            name=experiment_name,
            settings=settings,
            mode=mode,
            log_model=False,
        )
    except Exception as exc:
        print(f"W&B logging disabled: {exc}")
        return None


def _parse_devices_arg(value: str) -> int | str | list[int]:
    normalized = value.strip()
    if normalized.isdigit():
        return int(normalized)
    if "," in normalized:
        return [int(item.strip()) for item in normalized.split(",") if item.strip()]
    return normalized


def maybe_run_test_after_train(
    trainer: L.Trainer,
    *,
    module: Stage1FusionLightningModule,
    datamodule: Stage1HuMManDataModule,
    args: argparse.Namespace,
) -> None:
    if not args.test_after_train:
        return

    ckpt_path = resolve_test_after_train_ckpt_path(
        trainer,
        requested=args.test_ckpt,
    )
    if ckpt_path is None:
        print("Post-train test: using in-memory model weights")
        trainer.test(module, datamodule=datamodule)
        return

    print(f"Post-train test: using checkpoint {ckpt_path}")
    trainer.test(model=None, datamodule=datamodule, ckpt_path=ckpt_path)


def validate_required_mhr_assets(args: argparse.Namespace) -> None:
    if not args.test_after_train:
        return
    validate_mhr_asset_folder(args.mhr_assets_dir or "/opt/data/assets")


def resolve_test_after_train_ckpt_path(
    trainer: L.Trainer,
    *,
    requested: str,
) -> str | None:
    if requested == "current":
        return None

    checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
    if checkpoint_callback is None:
        if requested == "best":
            return None
        return "last"

    if requested == "best":
        best_model_path = getattr(checkpoint_callback, "best_model_path", "")
        return best_model_path or None
    if requested == "last":
        last_model_path = getattr(checkpoint_callback, "last_model_path", "")
        return last_model_path or None
    return None


def resolve_input_smpl_cache_dir(
    args: argparse.Namespace,
    data_config: Stage1DataConfig,
) -> str:
    if args.input_smpl_cache_dir is not None:
        return str(Path(args.input_smpl_cache_dir).resolve())
    manifest_parent = Path(data_config.manifest_path).resolve().parent
    return str((manifest_parent / "sam3dbody_fitted_smpl").resolve())


if __name__ == "__main__":
    main()
