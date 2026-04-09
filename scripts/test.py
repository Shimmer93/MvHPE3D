#!/usr/bin/env python
"""Evaluate a trained Stage 1 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import lightning as L

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mvhpe3d.data import Stage1DataConfig, Stage1HuMManDataModule
from mvhpe3d.lightning import Stage1FusionLightningModule
from mvhpe3d.utils import load_experiment_config, validate_mhr_asset_folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Stage 1 checkpoint")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to .ckpt file")
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
        "--stage",
        type=str,
        choices=("test", "val"),
        default="test",
        help="Which evaluation split to run",
    )
    parser.add_argument(
        "--default-root-dir",
        type=str,
        default="outputs/stage1_eval",
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
        help="Optional override for the MHR asset directory used for input-view conversion",
    )
    parser.add_argument(
        "--input-smpl-cache-dir",
        type=str,
        default=None,
        help="Optional directory for caching fitted SMPL parameters converted from input views",
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
    trainer_config = build_eval_trainer_config(experiment["trainer"], args)
    validate_optional_mhr_assets(args)

    if args.seed is not None:
        L.seed_everything(args.seed, workers=True)
    else:
        L.seed_everything(data_config.seed, workers=True)

    datamodule = Stage1HuMManDataModule(data_config)
    module = Stage1FusionLightningModule.load_from_checkpoint(
        args.checkpoint_path,
        map_location="cpu",
        smpl_model_path=args.smpl_model_path,
        mhr_assets_dir=args.mhr_assets_dir,
        input_smpl_cache_dir=resolve_input_smpl_cache_dir(args, data_config),
        strict=False,
    )

    trainer = L.Trainer(**trainer_config)
    if args.stage == "test":
        results = trainer.test(module, datamodule=datamodule)
    else:
        results = trainer.validate(module, datamodule=datamodule)

    payload = {
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "stage": args.stage,
        "results": results,
    }
    print(json.dumps(payload, indent=2))

    if args.output_path is not None:
        output_path = Path(args.output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved metrics to {output_path}")


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

    return Stage1DataConfig(**data_kwargs)


def build_eval_trainer_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
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

    trainer_kwargs["default_root_dir"] = str(Path(args.default_root_dir).resolve())
    trainer_kwargs["logger"] = False
    trainer_kwargs["enable_checkpointing"] = False
    trainer_kwargs["inference_mode"] = False
    return trainer_kwargs


def resolve_input_smpl_cache_dir(
    args: argparse.Namespace,
    data_config: Stage1DataConfig,
) -> str:
    if args.input_smpl_cache_dir is not None:
        return str(Path(args.input_smpl_cache_dir).resolve())
    manifest_parent = Path(data_config.manifest_path).resolve().parent
    return str((manifest_parent / "sam3dbody_fitted_smpl").resolve())


def parse_devices_arg(value: str) -> int | str | list[int]:
    normalized = value.strip()
    if normalized.isdigit():
        return int(normalized)
    if "," in normalized:
        return [int(item.strip()) for item in normalized.split(",") if item.strip()]
    return normalized


def validate_optional_mhr_assets(args: argparse.Namespace) -> None:
    if args.stage != "test":
        return
    validate_mhr_asset_folder(args.mhr_assets_dir or "/opt/data/assets")


if __name__ == "__main__":
    main()
