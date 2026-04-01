from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

from mvhpe3d.data import Stage1DataConfig, Stage1HuMManDataModule
from mvhpe3d.lightning import Stage1FusionLightningModule
from mvhpe3d.models import Stage1MLPFusionConfig, Stage1MLPFusionModel
from mvhpe3d.utils import load_experiment_config


def _load_train_script_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("mvhpe3d_train_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load training script from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage1_model_forward_shapes() -> None:
    model = Stage1MLPFusionModel(Stage1MLPFusionConfig())
    outputs = model.forward(__import__("torch").zeros(2, 3, 249))

    assert tuple(outputs["pred_mhr_params"].shape) == (2, 204)
    assert tuple(outputs["pred_shape_params"].shape) == (2, 45)
    assert tuple(outputs["fused_feature"].shape) == (2, 256)


def test_stage1_lightning_module_training_step(sample_manifest: Path) -> None:
    datamodule = Stage1HuMManDataModule(
        Stage1DataConfig(
            manifest_path=str(sample_manifest),
            num_views=2,
            batch_size=1,
            drop_last_train=False,
        )
    )
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))

    module = Stage1FusionLightningModule(model_config=Stage1MLPFusionConfig())
    loss = module.training_step(batch, 0)

    assert loss.ndim == 0


def test_load_experiment_config_resolves_sections() -> None:
    experiment = load_experiment_config("configs/experiment/stage1_cross_camera.yaml")

    assert experiment["experiment_name"] == "stage1_cross_camera"
    assert experiment["data"]["manifest_path"].endswith("humman_stage1_manifest.example.json")
    assert experiment["model"]["input_dim"] == 249
    assert experiment["trainer"]["max_epochs"] == 100


def test_train_script_build_data_config_disables_drop_last_in_fast_dev_run() -> None:
    train_module = _load_train_script_module()
    args = Namespace(
        manifest_path=None,
        seed=None,
        fast_dev_run=True,
        max_epochs=None,
        default_root_dir="outputs/stage1",
        config="configs/experiment/stage1_cross_camera.yaml",
    )
    data_config = train_module.build_data_config(
        {"manifest_path": "dummy.json", "drop_last_train": True},
        args,
    )

    assert data_config.drop_last_train is False
