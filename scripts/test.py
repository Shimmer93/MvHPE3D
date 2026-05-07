#!/usr/bin/env python
"""Evaluate a trained Stage 1, Stage 2, or Stage 3 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import lightning as L
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mvhpe3d.data import (
    Stage1DataConfig,
    Stage1HuMManDataModule,
    Stage2DataConfig,
    Stage2HuMManDataModule,
    Stage3DataConfig,
    Stage3HuMManDataModule,
)
from mvhpe3d.lightning import (
    Stage1FusionLightningModule,
    Stage2FusionLightningModule,
    Stage3TemporalLightningModule,
)
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
    parser = argparse.ArgumentParser(description="Evaluate a trained MvHPE3D checkpoint")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment/stage1_cross_camera.yaml",
        help="Path to the experiment YAML file",
    )
    parser.add_argument("--manifest-path", type=str, default=None, help="Optional manifest override")
    parser.add_argument("--gt-smpl-dir", type=str, default=None, help="Optional GT SMPL override")
    parser.add_argument("--cameras-dir", type=str, default=None, help="Optional camera JSON override")
    parser.add_argument(
        "--split-config-path",
        type=str,
        default=None,
        help="Optional split policy YAML override",
    )
    parser.add_argument("--split-name", type=str, default=None, help="Optional split name override")
    parser.add_argument(
        "--stage",
        type=str,
        choices=("test", "val"),
        default="test",
        help="Which evaluation split to run",
    )
    parser.add_argument(
        "--default-root-dir",
        type=str,
        default=None,
        help="Trainer default root dir",
    )
    parser.add_argument("--accelerator", type=str, default=None, help="Override trainer.accelerator")
    parser.add_argument(
        "--devices",
        type=str,
        default="1",
        help="Override trainer.devices, e.g. 1, 2, auto, or 0,1",
    )
    parser.add_argument("--strategy", type=str, default=None, help="Override trainer.strategy")
    parser.add_argument("--num-nodes", type=int, default=None, help="Override trainer.num_nodes")
    parser.add_argument("--seed", type=int, default=None, help="Optional override for experiment seed")
    parser.add_argument(
        "--smpl-model-path",
        type=str,
        default=None,
        help="Optional override for the neutral SMPL model used for keypoint metrics",
    )
    parser.add_argument(
        "--mhr-assets-dir",
        type=str,
        default=None,
        help="Optional override for the MHR asset directory used for Stage 1 input conversion",
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
        help="Optional Stage 2 checkpoint path used when loading a Stage 3 model manually",
    )
    parser.add_argument(
        "--pred-camera-mode",
        type=str,
        choices=("input", "input_corrected", "gt"),
        default="input_corrected",
        help=(
            "Compatibility flag for evaluation/visualization workflows. "
            "Stage 2/3 test metrics already use input_corrected camera placement internally."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional path to save aggregated metrics as JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = load_experiment_config(args.config)
    data_config = build_data_config(experiment["data"], args)
    model_config = build_model_config(
        experiment["model"],
        checkpoint_path=args.checkpoint_path,
    )
    trainer_config = build_eval_trainer_config(experiment["trainer"], args, experiment["experiment_name"])
    validate_optional_mhr_assets(args, experiment=experiment)

    if args.seed is not None:
        L.seed_everything(args.seed, workers=True)
    else:
        L.seed_everything(data_config.seed, workers=True)

    datamodule = build_datamodule(data_config)
    module = load_eval_module(
        checkpoint_path=args.checkpoint_path,
        model_config=model_config,
        data_config=data_config,
        args=args,
    )

    trainer = L.Trainer(**trainer_config)
    if args.stage == "test":
        results = trainer.test(module, datamodule=datamodule)
    else:
        results = trainer.validate(module, datamodule=datamodule)

    payload = {
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "stage": args.stage,
        "pred_camera_mode": args.pred_camera_mode,
        "results": results,
    }
    print(json.dumps(payload, indent=2))

    if args.output_path is not None:
        output_path = Path(args.output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved metrics to {output_path}")


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
    if data_name in {"humman_stage2", "humman_stage3"} and args.input_smpl_cache_dir is not None:
        data_kwargs["input_smpl_cache_dir"] = args.input_smpl_cache_dir

    if data_name == "humman_stage3":
        return Stage3DataConfig(**data_kwargs)
    if data_name == "humman_stage2":
        return Stage2DataConfig(**data_kwargs)
    return Stage1DataConfig(**data_kwargs)


def build_model_config(
    config: dict[str, Any],
    *,
    checkpoint_path: str,
):
    model_kwargs = dict(config)
    model_name = str(model_kwargs.pop("name", "stage1_mlp_fusion"))
    model_kwargs.pop("_config_path", None)
    model_kwargs.update(load_checkpoint_model_overrides(checkpoint_path))

    if model_name == "stage3_temporal_refine":
        return Stage3TemporalRefineConfig(**model_kwargs)
    if model_name == "stage2_joint_graph_refiner":
        return Stage2JointGraphRefinerConfig(**model_kwargs)
    if model_name == "stage2_joint_residual":
        return Stage2JointResidualConfig(**model_kwargs)
    if model_name == "stage2_param_refine":
        return Stage2ParamRefineConfig(**model_kwargs)
    if model_name == "stage1_residual_fusion":
        return Stage1ResidualFusionConfig(**model_kwargs)
    return Stage1MLPFusionConfig(**model_kwargs)


def load_checkpoint_model_overrides(checkpoint_path: str) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint.get("hyper_parameters", {}).get("model_config", {})
    if not isinstance(model_config, dict):
        return {}
    return dict(model_config)


def build_eval_trainer_config(
    config: dict[str, Any],
    args: argparse.Namespace,
    experiment_name: str,
) -> dict[str, Any]:
    trainer_kwargs = dict(config)
    trainer_kwargs.pop("_config_path", None)

    if args.accelerator is not None:
        trainer_kwargs["accelerator"] = args.accelerator
    if args.devices is not None:
        trainer_kwargs["devices"] = parse_devices_arg(args.devices)
    if args.strategy is not None:
        trainer_kwargs["strategy"] = args.strategy
    if args.num_nodes is not None:
        trainer_kwargs["num_nodes"] = args.num_nodes

    trainer_kwargs["default_root_dir"] = str(resolve_default_root_dir(args, experiment_name=experiment_name))
    trainer_kwargs["logger"] = False
    trainer_kwargs["enable_checkpointing"] = False
    trainer_kwargs["inference_mode"] = False
    return trainer_kwargs


def resolve_default_root_dir(args: argparse.Namespace, *, experiment_name: str) -> Path:
    if args.default_root_dir is not None:
        return Path(args.default_root_dir).resolve()
    if "stage3" in experiment_name.lower():
        return Path("outputs/stage3_eval").resolve()
    if "stage2" in experiment_name.lower():
        return Path("outputs/stage2_eval").resolve()
    return Path("outputs/stage1_eval").resolve()


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


def load_eval_module(
    *,
    checkpoint_path: str,
    model_config,
    data_config: Stage1DataConfig | Stage2DataConfig | Stage3DataConfig,
    args: argparse.Namespace,
):
    if isinstance(model_config, Stage3TemporalRefineConfig):
        return Stage3TemporalLightningModule.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
            model_config=model_config,
            smpl_model_path=args.smpl_model_path,
            stage2_checkpoint_path=args.stage2_checkpoint_path,
            strict=False,
        )
    if isinstance(
        model_config,
        (
            Stage2ParamRefineConfig,
            Stage2JointResidualConfig,
            Stage2JointGraphRefinerConfig,
        ),
    ):
        return Stage2FusionLightningModule.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
            model_config=model_config,
            smpl_model_path=args.smpl_model_path,
            strict=False,
        )
    return Stage1FusionLightningModule.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
        model_config=model_config,
        smpl_model_path=args.smpl_model_path,
        mhr_assets_dir=args.mhr_assets_dir,
        input_smpl_cache_dir=resolve_input_smpl_cache_dir(args, data_config),
        strict=False,
    )


def parse_devices_arg(value: str) -> int | str | list[int]:
    normalized = value.strip()
    if normalized.isdigit():
        return int(normalized)
    if "," in normalized:
        return [int(item.strip()) for item in normalized.split(",") if item.strip()]
    return normalized


def validate_optional_mhr_assets(
    args: argparse.Namespace,
    *,
    experiment: dict[str, Any],
) -> None:
    if args.stage != "test":
        return
    if is_stage2_or_stage3_experiment(experiment):
        return
    validate_mhr_asset_folder(args.mhr_assets_dir or "/opt/data/assets")


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
