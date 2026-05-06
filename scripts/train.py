#!/usr/bin/env python
"""Train Stage 1, Stage 2, or Stage 3 experiments with PyTorch Lightning."""

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
from mvhpe3d.data import Stage2DataConfig, Stage2HuMManDataModule
from mvhpe3d.data import Stage3DataConfig, Stage3HuMManDataModule
from mvhpe3d.lightning import (
    Stage1FusionLightningModule,
    Stage1OptimizationConfig,
    Stage2FusionLightningModule,
    Stage2OptimizationConfig,
    Stage3OptimizationConfig,
    Stage3TemporalLightningModule,
)
from mvhpe3d.losses import Stage1LossConfig, Stage2LossConfig, Stage3LossConfig
from mvhpe3d.models import (
    Stage1MLPFusionConfig,
    Stage1ResidualFusionConfig,
    Stage2JointGraphRefinerConfig,
    Stage2JointResidualConfig,
    Stage2ParamRefineConfig,
    Stage3TemporalRefineConfig,
)
from mvhpe3d.utils import load_experiment_config, validate_mhr_asset_folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MvHPE3D experiment")
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
        default=None,
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
        "--stage2-checkpoint-path",
        type=str,
        default=None,
        help="Optional Stage 2 checkpoint used to initialize the Stage 3 backbone",
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
    parser.add_argument(
        "--disable-learn-betas",
        action="store_true",
        help="Disable the learned beta head and beta supervision",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = load_experiment_config(args.config)
    validate_required_mhr_assets(args, experiment=experiment)

    data_config = build_data_config(experiment["data"], args)
    model_config = build_model_config(experiment["model"], args)
    loss_config = build_loss_config(experiment["loss"], args, model_config=model_config)
    optimization_config = build_optimization_config(experiment["optimizer"], model_config=model_config)

    L.seed_everything(data_config.seed, workers=True)

    datamodule = build_datamodule(data_config)
    module = build_lightning_module(
        model_config=model_config,
        loss_config=loss_config,
        optimization_config=optimization_config,
        data_config=data_config,
        args=args,
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


def build_data_config(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> Stage1DataConfig | Stage2DataConfig | Stage3DataConfig:
    data_kwargs = dict(config)
    data_name = str(data_kwargs.pop("name", "humman_stage1"))
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
    if data_name in {"humman_stage2", "humman_stage3"}:
        input_smpl_cache_dir = getattr(args, "input_smpl_cache_dir", None)
        if input_smpl_cache_dir is not None:
            data_kwargs["input_smpl_cache_dir"] = input_smpl_cache_dir
    if data_name == "humman_stage3":
        return Stage3DataConfig(**data_kwargs)
    if data_name == "humman_stage2":
        return Stage2DataConfig(**data_kwargs)

    data_kwargs["name"] = data_name
    return Stage1DataConfig(**data_kwargs)


def build_model_config(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> (
    Stage1MLPFusionConfig
    | Stage1ResidualFusionConfig
    | Stage2JointGraphRefinerConfig
    | Stage2JointResidualConfig
    | Stage2ParamRefineConfig
    | Stage3TemporalRefineConfig
):
    model_kwargs = dict(config)
    model_name = str(model_kwargs.pop("name", "stage1_mlp_fusion"))
    model_kwargs.pop("_config_path", None)
    if model_name == "stage2_joint_graph_refiner":
        if args.disable_learn_betas:
            model_kwargs["learn_betas"] = False
        return Stage2JointGraphRefinerConfig(**model_kwargs)
    if model_name == "stage3_temporal_refine":
        if args.disable_learn_betas:
            model_kwargs["learn_betas"] = False
        return Stage3TemporalRefineConfig(**model_kwargs)
    if model_name == "stage2_joint_residual":
        if args.disable_learn_betas:
            model_kwargs["learn_betas"] = False
        return Stage2JointResidualConfig(**model_kwargs)
    if model_name == "stage2_param_refine":
        if args.disable_learn_betas:
            model_kwargs["learn_betas"] = False
        return Stage2ParamRefineConfig(**model_kwargs)
    if args.disable_learn_betas:
        model_kwargs["learn_betas"] = False
    if model_name == "stage1_residual_fusion":
        return Stage1ResidualFusionConfig(**model_kwargs)
    return Stage1MLPFusionConfig(**model_kwargs)


def build_loss_config(
    config: dict[str, Any],
    args: argparse.Namespace,
    *,
    model_config: (
        Stage1MLPFusionConfig
        | Stage1ResidualFusionConfig
        | Stage2JointGraphRefinerConfig
        | Stage2JointResidualConfig
        | Stage2ParamRefineConfig
        | Stage3TemporalRefineConfig
    ),
) -> Stage1LossConfig | Stage2LossConfig | Stage3LossConfig:
    loss_kwargs = dict(config)
    if isinstance(
        model_config,
        (
            Stage2ParamRefineConfig,
            Stage2JointResidualConfig,
            Stage2JointGraphRefinerConfig,
        ),
    ):
        if args.disable_learn_betas:
            loss_kwargs["supervise_betas"] = False
            loss_kwargs["betas_weight"] = 0.0
            loss_kwargs["init_betas_weight"] = 0.0
        return Stage2LossConfig(**loss_kwargs)
    if isinstance(model_config, Stage3TemporalRefineConfig):
        if args.disable_learn_betas:
            loss_kwargs["supervise_betas"] = False
            loss_kwargs["betas_weight"] = 0.0
        return Stage3LossConfig(**loss_kwargs)
    if args.disable_learn_betas:
        loss_kwargs["supervise_betas"] = False
        loss_kwargs["betas_weight"] = 0.0
    return Stage1LossConfig(**loss_kwargs)


def build_optimization_config(
    config: dict[str, Any],
    *,
    model_config: (
        Stage1MLPFusionConfig
        | Stage1ResidualFusionConfig
        | Stage2JointGraphRefinerConfig
        | Stage2JointResidualConfig
        | Stage2ParamRefineConfig
        | Stage3TemporalRefineConfig
    ),
) -> Stage1OptimizationConfig | Stage2OptimizationConfig | Stage3OptimizationConfig:
    if isinstance(model_config, Stage3TemporalRefineConfig):
        return Stage3OptimizationConfig(**config)
    if isinstance(
        model_config,
        (
            Stage2ParamRefineConfig,
            Stage2JointResidualConfig,
            Stage2JointGraphRefinerConfig,
        ),
    ):
        return Stage2OptimizationConfig(**config)
    return Stage1OptimizationConfig(**config)


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

    root_dir = resolve_default_root_dir(args, experiment_name=experiment_name)
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


def resolve_default_root_dir(args: argparse.Namespace, *, experiment_name: str) -> Path:
    if args.default_root_dir is not None:
        return Path(args.default_root_dir).resolve()
    if "stage3" in experiment_name.lower():
        return Path("outputs/stage3").resolve()
    if "stage2" in experiment_name.lower():
        return Path("outputs/stage2").resolve()
    return Path("outputs/stage1").resolve()


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
    module,
    datamodule,
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


def validate_required_mhr_assets(
    args: argparse.Namespace,
    *,
    experiment: dict[str, Any],
) -> None:
    if not args.test_after_train:
        return
    if is_stage2_or_stage3_experiment(experiment):
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
    data_config: Stage1DataConfig | Stage2DataConfig | Stage3DataConfig,
) -> str:
    if args.input_smpl_cache_dir is not None:
        return str(Path(args.input_smpl_cache_dir).resolve())
    manifest_parent = Path(data_config.manifest_path).resolve().parent
    return str((manifest_parent / "sam3dbody_fitted_smpl").resolve())


def build_datamodule(data_config: Stage1DataConfig | Stage2DataConfig | Stage3DataConfig):
    if isinstance(data_config, Stage3DataConfig):
        return Stage3HuMManDataModule(data_config)
    if isinstance(data_config, Stage2DataConfig):
        return Stage2HuMManDataModule(data_config)
    return Stage1HuMManDataModule(data_config)


def build_lightning_module(
    *,
    model_config,
    loss_config,
    optimization_config,
    data_config: Stage1DataConfig | Stage2DataConfig | Stage3DataConfig,
    args: argparse.Namespace,
):
    if isinstance(model_config, Stage3TemporalRefineConfig):
        return Stage3TemporalLightningModule(
            model_config=model_config,
            loss_config=loss_config,
            optimization_config=optimization_config,
            smpl_model_path=args.smpl_model_path,
            stage2_checkpoint_path=args.stage2_checkpoint_path,
        )
    if isinstance(
        model_config,
        (
            Stage2ParamRefineConfig,
            Stage2JointResidualConfig,
            Stage2JointGraphRefinerConfig,
        ),
    ):
        return Stage2FusionLightningModule(
            model_config=model_config,
            loss_config=loss_config,
            optimization_config=optimization_config,
            smpl_model_path=args.smpl_model_path,
        )
    return Stage1FusionLightningModule(
        model_config=model_config,
        loss_config=loss_config,
        optimization_config=optimization_config,
        smpl_model_path=args.smpl_model_path,
        mhr_assets_dir=args.mhr_assets_dir,
        input_smpl_cache_dir=resolve_input_smpl_cache_dir(args, data_config),
    )


def is_stage2_or_stage3_experiment(experiment: dict[str, Any]) -> bool:
    model_name = str(experiment.get("model", {}).get("name", ""))
    data_name = str(experiment.get("data", {}).get("name", ""))
    return model_name in {
        "stage2_param_refine",
        "stage2_joint_residual",
        "stage2_joint_graph_refiner",
        "stage3_temporal_refine",
    } or data_name in {"humman_stage2", "humman_stage3"}


if __name__ == "__main__":
    main()
