from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

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
    Stage1MLPFusionModel,
    Stage2JointGraphRefinerConfig,
    Stage2JointGraphRefinerModel,
    Stage2JointResidualConfig,
    Stage2JointResidualModel,
    Stage2ParamRefineConfig,
    Stage2ParamRefineModel,
    Stage3TemporalRefineConfig,
    Stage3TemporalRefineModel,
)
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

    assert tuple(outputs["pred_body_pose"].shape) == (2, 69)
    assert tuple(outputs["pred_betas"].shape) == (2, 10)
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


def test_stage2_model_forward_shapes() -> None:
    model = Stage2ParamRefineModel(Stage2ParamRefineConfig())
    outputs = model.forward(__import__("torch").zeros(2, 3, 148))

    assert tuple(outputs["init_pose_6d"].shape) == (2, 23, 6)
    assert tuple(outputs["pred_body_pose"].shape) == (2, 69)
    assert tuple(outputs["pred_betas"].shape) == (2, 10)
    assert tuple(outputs["view_weights"].shape) == (2, 3)


def test_stage2_joint_residual_model_forward_shapes() -> None:
    model = Stage2JointResidualModel(Stage2JointResidualConfig())
    outputs = model.forward(__import__("torch").zeros(2, 3, 148))

    assert tuple(outputs["init_pose_6d"].shape) == (2, 23, 6)
    assert tuple(outputs["pred_body_pose"].shape) == (2, 69)
    assert tuple(outputs["pred_betas"].shape) == (2, 10)
    assert tuple(outputs["view_weights"].shape) == (2, 3)
    assert tuple(outputs["pose_view_weights"].shape) == (2, 3, 23)


def test_stage2_joint_graph_refiner_model_forward_shapes() -> None:
    model = Stage2JointGraphRefinerModel(Stage2JointGraphRefinerConfig())
    outputs = model.forward(__import__("torch").zeros(2, 3, 148))

    assert tuple(outputs["init_pose_6d"].shape) == (2, 23, 6)
    assert tuple(outputs["pred_body_pose"].shape) == (2, 69)
    assert tuple(outputs["pred_betas"].shape) == (2, 10)
    assert tuple(outputs["view_weights"].shape) == (2, 3)
    assert tuple(outputs["pose_view_weights"].shape) == (2, 3, 23)


def test_stage3_temporal_refine_model_forward_shapes() -> None:
    model = Stage3TemporalRefineModel(Stage3TemporalRefineConfig())
    outputs = model.forward(
        __import__("torch").zeros(2, 3, 286),
        base_pose_6d=__import__("torch").zeros(2, 138),
        base_betas=__import__("torch").zeros(2, 10),
    )

    assert tuple(outputs["base_pose_6d"].shape) == (2, 23, 6)
    assert tuple(outputs["pred_pose_6d"].shape) == (2, 23, 6)
    assert tuple(outputs["pred_body_pose"].shape) == (2, 69)
    assert tuple(outputs["pred_betas"].shape) == (2, 10)


def test_stage2_lightning_module_training_step(
    sample_manifest: Path,
    sample_input_smpl_cache: Path,
) -> None:
    datamodule = Stage2HuMManDataModule(
        Stage2DataConfig(
            manifest_path=str(sample_manifest),
            input_smpl_cache_dir=str(sample_input_smpl_cache),
            num_views=2,
            batch_size=1,
            drop_last_train=False,
        )
    )
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))

    module = Stage2FusionLightningModule(model_config=Stage2ParamRefineConfig())
    monkeypatch_torch = __import__("torch")
    dummy_joints = monkeypatch_torch.zeros(1, 24, 3)
    module._runtime_cache["smpl_eval_model"] = lambda **_: None
    module._build_smpl_joints = lambda **_: dummy_joints
    loss = module.training_step(batch, 0)

    assert loss.ndim == 0


def test_stage2_lightning_module_validation_step_adds_joint_metrics(
    sample_manifest: Path,
    sample_input_smpl_cache: Path,
    monkeypatch,
) -> None:
    datamodule = Stage2HuMManDataModule(
        Stage2DataConfig(
            manifest_path=str(sample_manifest),
            input_smpl_cache_dir=str(sample_input_smpl_cache),
            num_views=2,
            batch_size=1,
            drop_last_train=False,
        )
    )
    datamodule.setup("fit")
    batch = next(iter(datamodule.val_dataloader()))

    module = Stage2FusionLightningModule(model_config=Stage2ParamRefineConfig())
    monkeypatch.setattr(
        module,
        "_compute_canonical_joint_metrics",
        lambda **_: {
            "mpjpe": __import__("torch").tensor(10.0),
            "pa_mpjpe": __import__("torch").tensor(5.0),
        },
    )
    monkeypatch.setattr(module, "_compute_canonical_joint_loss", lambda **_: __import__("torch").tensor(0.0))

    metrics = module._shared_step(batch, stage="val")

    assert metrics["val/mpjpe"].item() == 10.0
    assert metrics["val/pa_mpjpe"].item() == 5.0


def test_stage2_lightning_module_disabling_betas_zeros_beta_losses(
    sample_manifest: Path,
    sample_input_smpl_cache: Path,
    monkeypatch,
) -> None:
    datamodule = Stage2HuMManDataModule(
        Stage2DataConfig(
            manifest_path=str(sample_manifest),
            input_smpl_cache_dir=str(sample_input_smpl_cache),
            num_views=2,
            batch_size=1,
            drop_last_train=False,
        )
    )
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))

    module = Stage2FusionLightningModule(
        model_config=Stage2ParamRefineConfig(learn_betas=False),
    )
    monkeypatch.setattr(module, "_compute_canonical_joint_loss", lambda **_: __import__("torch").tensor(0.0))

    metrics = module._shared_step(batch, stage="train")

    assert metrics["train/loss_betas"].item() == 0.0
    assert metrics["train/loss_init_betas"].item() == 0.0


def test_stage2_lightning_module_test_step_uses_camera_metrics(
    sample_manifest: Path,
    sample_input_smpl_cache: Path,
) -> None:
    datamodule = Stage2HuMManDataModule(
        Stage2DataConfig(
            manifest_path=str(sample_manifest),
            input_smpl_cache_dir=str(sample_input_smpl_cache),
            num_views=2,
            batch_size=1,
            drop_last_train=False,
        )
    )
    datamodule.setup("test")
    batch = next(iter(datamodule.test_dataloader()))

    module = Stage2FusionLightningModule(model_config=Stage2ParamRefineConfig())
    monkeypatch_torch = __import__("torch")
    module._runtime_cache["smpl_eval_model"] = lambda **_: None
    module._build_smpl_joints = lambda **kwargs: monkeypatch_torch.zeros(
        kwargs["body_pose"].shape[0], 24, 3
    )

    metrics = module._shared_step(batch, stage="test")

    assert "test/mpjpe" in metrics
    assert "test/pa_mpjpe" in metrics




def test_stage3_lightning_module_training_step(
    sample_manifest: Path,
    sample_input_smpl_cache: Path,
) -> None:
    datamodule = Stage3HuMManDataModule(
        Stage3DataConfig(
            manifest_path=str(sample_manifest),
            input_smpl_cache_dir=str(sample_input_smpl_cache),
            num_views=2,
            window_size=3,
            batch_size=1,
            drop_last_train=False,
        )
    )
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))

    module = Stage3TemporalLightningModule(
        model_config=Stage3TemporalRefineConfig(),
        stage2_backbone_model=Stage2JointResidualModel(Stage2JointResidualConfig()),
    )
    monkeypatch_torch = __import__("torch")
    dummy_joints = monkeypatch_torch.zeros(1, 24, 3)
    module._runtime_cache["smpl_eval_model"] = lambda **_: None
    module._build_smpl_joints = lambda **_: dummy_joints
    loss = module.training_step(batch, 0)

    assert loss.ndim == 0


def test_stage3_lightning_module_test_step_uses_camera_metrics(
    sample_manifest: Path,
    sample_input_smpl_cache: Path,
) -> None:
    datamodule = Stage3HuMManDataModule(
        Stage3DataConfig(
            manifest_path=str(sample_manifest),
            input_smpl_cache_dir=str(sample_input_smpl_cache),
            num_views=2,
            window_size=3,
            batch_size=1,
            drop_last_train=False,
        )
    )
    datamodule.setup("test")
    batch = next(iter(datamodule.test_dataloader()))

    module = Stage3TemporalLightningModule(
        model_config=Stage3TemporalRefineConfig(),
        stage2_backbone_model=Stage2JointResidualModel(Stage2JointResidualConfig()),
    )
    monkeypatch_torch = __import__("torch")
    module._runtime_cache["smpl_eval_model"] = lambda **_: None
    module._build_smpl_joints = lambda **kwargs: monkeypatch_torch.zeros(
        kwargs["body_pose"].shape[0], 24, 3
    )

    metrics = module._shared_step(batch, stage="test")

    assert "test/mpjpe" in metrics
    assert "test/pa_mpjpe" in metrics


def test_load_stage2_experiment_config_resolves_sections() -> None:
    experiment = load_experiment_config("configs/experiment/stage2_cross_camera.yaml")

    assert experiment["experiment_name"] == "stage2_cross_camera"
    assert experiment["data"]["name"] == "humman_stage2"
    assert experiment["model"]["name"] == "stage2_param_refine"


def test_load_stage2_joint_residual_experiment_config_resolves_sections() -> None:
    experiment = load_experiment_config("configs/experiment/stage2_cross_camera_joint_residual.yaml")

    assert experiment["experiment_name"] == "stage2_cross_camera_joint_residual"
    assert experiment["data"]["name"] == "humman_stage2"
    assert experiment["model"]["name"] == "stage2_joint_residual"


def test_load_stage2_joint_graph_refiner_experiment_config_resolves_sections() -> None:
    experiment = load_experiment_config(
        "configs/experiment/stage2_cross_camera_joint_graph_refiner.yaml"
    )

    assert experiment["experiment_name"] == "stage2_cross_camera_joint_graph_refiner"
    assert experiment["data"]["name"] == "humman_stage2"
    assert experiment["model"]["name"] == "stage2_joint_graph_refiner"


def test_load_stage3_experiment_config_resolves_sections() -> None:
    experiment = load_experiment_config("configs/experiment/stage3_temporal_refine.yaml")

    assert experiment["experiment_name"] == "stage3_temporal_refine"
    assert experiment["data"]["name"] == "humman_stage3"
    assert experiment["model"]["name"] == "stage3_temporal_refine"


def test_stage1_lightning_module_validation_step_adds_joint_metrics(
    sample_manifest: Path,
    monkeypatch,
) -> None:
    datamodule = Stage1HuMManDataModule(
        Stage1DataConfig(
            manifest_path=str(sample_manifest),
            num_views=2,
            batch_size=1,
            drop_last_train=False,
        )
    )
    datamodule.setup("fit")
    batch = next(iter(datamodule.val_dataloader()))

    module = Stage1FusionLightningModule(model_config=Stage1MLPFusionConfig())
    monkeypatch.setattr(
        module,
        "_compute_canonical_joint_metrics",
        lambda **_: {
            "mpjpe": __import__("torch").tensor(12.0),
            "pa_mpjpe": __import__("torch").tensor(7.0),
        },
    )

    metrics = module._shared_step(batch, stage="val")

    assert metrics["val/mpjpe"].item() == 12.0
    assert metrics["val/pa_mpjpe"].item() == 7.0


def test_stage1_lightning_module_test_step_adds_input_view_metrics(
    sample_manifest: Path,
    monkeypatch,
) -> None:
    datamodule = Stage1HuMManDataModule(
        Stage1DataConfig(
            manifest_path=str(sample_manifest),
            num_views=2,
            batch_size=1,
            drop_last_train=False,
        )
    )
    datamodule.setup("test")
    batch = next(iter(datamodule.test_dataloader()))

    module = Stage1FusionLightningModule(model_config=Stage1MLPFusionConfig())
    monkeypatch.setattr(
        module,
        "convert_input_views_to_smpl",
        lambda **_: {
            "body_pose": __import__("torch").zeros(2, 69),
            "betas": __import__("torch").zeros(2, 10),
            "global_orient": __import__("torch").zeros(2, 3),
            "transl": __import__("torch").zeros(2, 3),
        },
    )
    monkeypatch.setattr(
        module,
        "_compute_test_joint_metrics",
        lambda **_: {
            "mpjpe": __import__("torch").tensor(12.0),
            "pa_mpjpe": __import__("torch").tensor(7.0),
        },
    )
    monkeypatch.setattr(
        module,
        "_compute_input_view_joint_metrics",
        lambda **_: {
            "mpjpe": __import__("torch").tensor(21.0),
            "pa_mpjpe": __import__("torch").tensor(11.0),
        },
    )

    metrics = module._shared_step(batch, stage="test")

    assert metrics["test/mpjpe"].item() == 12.0
    assert metrics["test/pa_mpjpe"].item() == 7.0
    assert metrics["test/input_avg_mpjpe"].item() == 21.0
    assert metrics["test/input_avg_pa_mpjpe"].item() == 11.0


def test_load_experiment_config_resolves_sections() -> None:
    experiment = load_experiment_config("configs/experiment/stage1_cross_camera.yaml")

    assert experiment["experiment_name"] == "stage1_cross_camera"
    assert experiment["data"]["manifest_path"].endswith("humman_stage1_manifest.example.json")
    assert experiment["data"]["split_name"] == "cross_camera_split"
    assert experiment["model"]["input_dim"] == 249
    assert experiment["trainer"]["max_epochs"] == 100


def test_train_script_build_data_config_disables_drop_last_in_fast_dev_run() -> None:
    train_module = _load_train_script_module()
    args = Namespace(
        manifest_path=None,
        gt_smpl_dir=None,
        cameras_dir=None,
        split_config_path=None,
        split_name=None,
        seed=None,
        smpl_model_path=None,
        mhr_assets_dir=None,
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


def test_train_script_build_data_config_overrides_split_selection() -> None:
    train_module = _load_train_script_module()
    args = Namespace(
        manifest_path=None,
        gt_smpl_dir="dummy_smpl",
        cameras_dir="dummy_cameras",
        split_config_path="configs/data/humman_stage1_splits.yaml",
        split_name="random_split",
        seed=None,
        smpl_model_path=None,
        mhr_assets_dir=None,
        fast_dev_run=False,
        max_epochs=None,
        default_root_dir="outputs/stage1",
        config="configs/experiment/stage1_cross_camera.yaml",
    )
    data_config = train_module.build_data_config(
        {
            "manifest_path": "dummy.json",
            "split_name": "cross_camera_split",
            "gt_smpl_dir": None,
        },
        args,
    )

    assert data_config.gt_smpl_dir == "dummy_smpl"
    assert data_config.cameras_dir == "dummy_cameras"
    assert data_config.split_config_path == "configs/data/humman_stage1_splits.yaml"
    assert data_config.split_name == "random_split"


def test_train_script_build_trainer_config_overrides_multi_gpu_settings(monkeypatch) -> None:
    train_module = _load_train_script_module()
    monkeypatch.setattr(train_module, "build_optional_wandb_logger", lambda **_: None)
    args = Namespace(
        manifest_path=None,
        gt_smpl_dir=None,
        cameras_dir=None,
        split_config_path=None,
        split_name=None,
        seed=None,
        smpl_model_path=None,
        mhr_assets_dir=None,
        fast_dev_run=False,
        max_epochs=50,
        accelerator="gpu",
        devices="2",
        strategy="ddp",
        num_nodes=1,
        default_root_dir="outputs/stage1",
        config="configs/experiment/stage1_cross_camera.yaml",
    )

    trainer_config = train_module.build_trainer_config(
        {"devices": 1, "accelerator": "auto"},
        args,
        "stage1_cross_camera",
    )

    assert trainer_config["max_epochs"] == 50
    assert trainer_config["accelerator"] == "gpu"
    assert trainer_config["devices"] == 2
    assert trainer_config["strategy"] == "ddp"
    assert trainer_config["num_nodes"] == 1
    logger = trainer_config["logger"]
    checkpoint = trainer_config["callbacks"][0]
    assert Path(checkpoint.dirpath) == Path(logger.log_dir) / "checkpoints"


def test_build_loggers_returns_csv_only_when_wandb_unavailable(monkeypatch) -> None:
    train_module = _load_train_script_module()
    monkeypatch.setattr(train_module, "build_optional_wandb_logger", lambda **_: None)

    logger = train_module.build_loggers(
        root_dir=Path("/tmp/outputs"),
        experiment_name="stage1_cross_camera",
    )

    assert logger.__class__.__name__ == "CSVLogger"


def test_build_loggers_adds_wandb_when_available(monkeypatch) -> None:
    train_module = _load_train_script_module()

    class DummyWandbLogger:
        pass

    dummy_logger = DummyWandbLogger()
    monkeypatch.setattr(train_module, "build_optional_wandb_logger", lambda **_: dummy_logger)

    logger = train_module.build_loggers(
        root_dir=Path("/tmp/outputs"),
        experiment_name="stage1_cross_camera",
    )

    assert isinstance(logger, list)
    assert logger[0].__class__.__name__ == "CSVLogger"
    assert logger[1] is dummy_logger


def test_resolve_test_after_train_ckpt_path_prefers_best_checkpoint() -> None:
    train_module = _load_train_script_module()

    class DummyCheckpoint:
        best_model_path = "/tmp/best.ckpt"
        last_model_path = "/tmp/last.ckpt"

    class DummyTrainer:
        checkpoint_callback = DummyCheckpoint()

    ckpt_path = train_module.resolve_test_after_train_ckpt_path(
        DummyTrainer(),
        requested="best",
    )

    assert ckpt_path == "/tmp/best.ckpt"


def test_resolve_test_after_train_ckpt_path_can_fallback_to_current_model() -> None:
    train_module = _load_train_script_module()

    class DummyCheckpoint:
        best_model_path = ""
        last_model_path = ""

    class DummyTrainer:
        checkpoint_callback = DummyCheckpoint()

    ckpt_path = train_module.resolve_test_after_train_ckpt_path(
        DummyTrainer(),
        requested="best",
    )

    assert ckpt_path is None
