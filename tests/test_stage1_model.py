from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn

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
from mvhpe3d.losses import Stage3Loss, Stage3LossConfig
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
from mvhpe3d.utils import (
    PANOPTIC_EVAL_JOINT_INDICES,
    PANOPTIC_EVAL_SMPL24_INDICES,
    load_experiment_config,
)


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


def test_stage3_temporal_refine_model_zero_init_matches_stage2_baseline() -> None:
    config = Stage3TemporalRefineConfig(
        learn_betas=True,
        hidden_dim=32,
        temporal_dim=16,
        temporal_layers=1,
        head_layers=2,
    )
    model = Stage3TemporalRefineModel(config)
    temporal_features = torch.randn(4, 5, config.temporal_feature_dim)
    base_pose_6d = torch.randn(4, config.num_joints, 6)
    base_betas = torch.randn(4, config.betas_dim)

    outputs = model(
        temporal_features,
        base_pose_6d=base_pose_6d,
        base_betas=base_betas,
    )

    assert torch.equal(
        outputs["pose_residual_6d"],
        torch.zeros_like(outputs["pose_residual_6d"]),
    )
    assert torch.equal(
        outputs["betas_residual"], torch.zeros_like(outputs["betas_residual"])
    )
    assert torch.equal(outputs["base_pose_6d"], base_pose_6d)
    assert torch.equal(outputs["base_betas"], base_betas)
    assert torch.equal(outputs["pred_pose_6d"], base_pose_6d)
    assert torch.equal(outputs["pred_betas"], base_betas)


def test_stage3_loss_includes_residual_regularization_when_configured() -> None:
    criterion = Stage3Loss(
        Stage3LossConfig(
            pose_6d_weight=0.0,
            betas_weight=0.0,
            pose_residual_weight=2.0,
            betas_residual_weight=3.0,
            supervise_betas=True,
        )
    )
    pred_pose_6d = torch.zeros(1, 23, 6)
    pred_betas = torch.zeros(1, 10)
    pose_residual_6d = torch.full((1, 23, 6), 0.5)
    betas_residual = torch.full((1, 10), 0.25)

    losses = criterion(
        pred_pose_6d=pred_pose_6d,
        pred_betas=pred_betas,
        target_pose_6d=pred_pose_6d,
        target_betas=pred_betas,
        pose_residual_6d=pose_residual_6d,
        betas_residual=betas_residual,
    )

    assert torch.isclose(losses["loss_pose_residual"], torch.tensor(0.25))
    assert torch.isclose(losses["loss_betas_residual"], torch.tensor(0.0625))
    assert torch.isclose(losses["loss"], torch.tensor(0.6875))


class _EchoStage2Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pose_offset = nn.Parameter(torch.tensor(0.25))

    def forward(self, views_input: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = views_input.shape[0]
        pooled_input = views_input.mean(dim=1)
        init_pose_6d = pooled_input[:, :138].reshape(batch_size, 23, 6)
        pred_pose_6d = init_pose_6d + self.pose_offset
        pred_betas = pooled_input[:, 138:]
        return {
            "init_pose_6d": init_pose_6d,
            "pred_pose_6d": pred_pose_6d,
            "pred_betas": pred_betas,
        }


def test_stage3_lightning_module_zero_init_matches_frozen_stage2_center_prediction() -> (
    None
):
    config = Stage3TemporalRefineConfig(
        hidden_dim=32,
        temporal_dim=16,
        temporal_layers=1,
        head_layers=2,
    )
    module = Stage3TemporalLightningModule(
        model_config=config,
        stage2_backbone_model=_EchoStage2Backbone(),
    )
    views_input = torch.randn(2, 5, 3, config.pose_6d_dim + config.betas_dim)

    outputs = module(views_input)

    assert not any(
        parameter.requires_grad for parameter in module.stage2_backbone.parameters()
    )
    with torch.no_grad():
        flat_input = views_input.reshape(10, 3, config.pose_6d_dim + config.betas_dim)
        stage2_outputs = module.stage2_backbone(flat_input)
    expected_pose_6d = stage2_outputs["pred_pose_6d"].reshape(2, 5, 23, 6)[:, 2]
    expected_betas = stage2_outputs["pred_betas"].reshape(2, 5, 10)[:, 2]

    assert torch.equal(outputs["stage2_pred_pose_6d"], expected_pose_6d)
    assert torch.equal(outputs["stage2_pred_betas"], expected_betas)
    assert torch.equal(outputs["base_pose_6d"], expected_pose_6d)
    assert torch.equal(outputs["base_betas"], expected_betas)
    assert torch.equal(
        outputs["pose_residual_6d"],
        torch.zeros_like(outputs["pose_residual_6d"]),
    )
    assert torch.equal(
        outputs["betas_residual"], torch.zeros_like(outputs["betas_residual"])
    )
    assert torch.equal(outputs["pred_pose_6d"], expected_pose_6d)
    assert torch.equal(outputs["pred_betas"], expected_betas)


def test_stage3_lightning_module_causal_zero_init_matches_last_stage2_prediction() -> (
    None
):
    config = Stage3TemporalRefineConfig(
        hidden_dim=32,
        temporal_dim=16,
        temporal_layers=1,
        head_layers=2,
        causal=True,
    )
    module = Stage3TemporalLightningModule(
        model_config=config,
        stage2_backbone_model=_EchoStage2Backbone(),
    )
    views_input = torch.randn(2, 5, 3, config.pose_6d_dim + config.betas_dim)

    outputs = module(views_input)

    with torch.no_grad():
        flat_input = views_input.reshape(10, 3, config.pose_6d_dim + config.betas_dim)
        stage2_outputs = module.stage2_backbone(flat_input)
    expected_pose_6d = stage2_outputs["pred_pose_6d"].reshape(2, 5, 23, 6)[:, -1]
    expected_betas = stage2_outputs["pred_betas"].reshape(2, 5, 10)[:, -1]

    assert torch.equal(outputs["target_frame_index"], torch.full((2,), 4))
    assert torch.equal(outputs["stage2_pred_pose_6d"], expected_pose_6d)
    assert torch.equal(outputs["stage2_pred_betas"], expected_betas)
    assert torch.equal(outputs["pred_pose_6d"], expected_pose_6d)
    assert torch.equal(outputs["pred_betas"], expected_betas)


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
    monkeypatch.setattr(
        module,
        "_compute_canonical_joint_loss",
        lambda **_: __import__("torch").tensor(0.0),
    )

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
    monkeypatch.setattr(
        module,
        "_compute_canonical_joint_loss",
        lambda **_: __import__("torch").tensor(0.0),
    )

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
    experiment = load_experiment_config(
        "configs/experiment/stage2_cross_camera_joint_residual.yaml"
    )

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
    experiment = load_experiment_config(
        "configs/experiment/stage3_temporal_refine.yaml"
    )

    assert experiment["experiment_name"] == "stage3_temporal_refine"
    assert experiment["data"]["name"] == "humman_stage3"
    assert experiment["model"]["name"] == "stage3_temporal_refine"


def test_load_stage3_causal_experiment_config_resolves_sections() -> None:
    experiment = load_experiment_config(
        "configs/experiment/stage3_temporal_refine_causal.yaml"
    )

    assert experiment["experiment_name"] == "stage3_temporal_refine_causal"
    assert experiment["data"]["name"] == "humman_stage3"
    assert experiment["data"]["causal"] is True
    assert experiment["model"]["name"] == "stage3_temporal_refine"
    assert experiment["model"]["causal"] is True


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


def test_stage1_test_metrics_use_converted_input_smpl_translation() -> None:
    module = Stage1FusionLightningModule(model_config=Stage1MLPFusionConfig())
    converted_transl = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=torch.float32,
    )
    input_view_smpl_params = {
        "body_pose": torch.zeros(2, 69),
        "betas": torch.zeros(2, 10),
        "global_orient": torch.zeros(2, 3),
        "transl": converted_transl,
    }
    pred_cam_t = torch.full((1, 2, 3), 100.0)
    used_translations: list[torch.Tensor] = []

    def fake_build_smpl_joints(**kwargs):
        used_translations.append(kwargs["transl"].detach().clone())
        return kwargs["transl"][:, None, :].expand(kwargs["transl"].shape[0], 24, 3)

    module._build_smpl_joints = fake_build_smpl_joints

    module._compute_test_joint_metrics(
        pred_body_pose=torch.zeros(1, 69),
        pred_betas=torch.zeros(1, 10),
        target_body_pose=torch.zeros(1, 69),
        target_betas=torch.zeros(1, 10),
        target_aux={},
        pred_cam_t=pred_cam_t,
        input_view_smpl_params=input_view_smpl_params,
    )
    module._compute_input_view_joint_metrics(
        input_view_smpl_params=input_view_smpl_params,
        target_body_pose=torch.zeros(1, 69),
        target_betas=torch.zeros(1, 10),
        target_aux={},
        pred_cam_t=pred_cam_t,
    )

    assert used_translations
    for transl in used_translations:
        assert torch.equal(transl, converted_transl)


def test_stage1_panoptic_input_metrics_use_native_camera_gt_root_centering() -> None:
    module = Stage1FusionLightningModule(model_config=Stage1MLPFusionConfig())
    panoptic_template = torch.zeros(19, 3, dtype=torch.float32)
    selected_template = torch.stack(
        [
            torch.tensor(
                [
                    float(index % 4) * 0.1,
                    float(index // 4) * 0.05,
                    float(index % 3) * 0.02,
                ],
                dtype=torch.float32,
            )
            for index in range(len(PANOPTIC_EVAL_JOINT_INDICES))
        ],
        dim=0,
    )
    for column, panoptic_index in enumerate(PANOPTIC_EVAL_JOINT_INDICES):
        panoptic_template[panoptic_index] = selected_template[column]

    def fake_build_smpl_joints(**kwargs):
        transl = kwargs["transl"].detach().cpu()
        joints = torch.zeros(transl.shape[0], 24, 3, dtype=torch.float32)
        for column, smpl_index in enumerate(PANOPTIC_EVAL_SMPL24_INDICES):
            joints[:, smpl_index, :] = selected_template[column] + transl
        return joints

    module._build_smpl_joints = fake_build_smpl_joints
    target_aux = {
        "panoptic_joints_world": panoptic_template[None],
        "panoptic_confidence": torch.ones(1, 19, dtype=torch.float32),
        "camera_rotation": torch.eye(3, dtype=torch.float32)
        .reshape(1, 1, 3, 3)
        .expand(1, 2, 3, 3),
        "camera_translation": torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
    }
    input_view_smpl_params = {
        "body_pose": torch.zeros(2, 69),
        "betas": torch.zeros(2, 10),
        "global_orient": torch.zeros(2, 3),
        "transl": torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    }

    metrics = module._compute_panoptic_input_view_joint_metrics(
        input_view_smpl_params=input_view_smpl_params,
        target_aux=target_aux,
    )

    assert metrics["mpjpe"].item() < 1e-6
    assert metrics["pa_mpjpe"].item() < 1e-6
    assert torch.isclose(metrics["abs_mpjpe"], torch.tensor(0.5))


def test_load_experiment_config_resolves_sections() -> None:
    experiment = load_experiment_config("configs/experiment/stage1_cross_camera.yaml")

    assert experiment["experiment_name"] == "stage1_cross_camera"
    assert experiment["data"]["manifest_path"].endswith(
        "humman_stage1_manifest.example.json"
    )
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


def test_train_script_build_trainer_config_overrides_multi_gpu_settings(
    monkeypatch,
) -> None:
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
    monkeypatch.setattr(
        train_module, "build_optional_wandb_logger", lambda **_: dummy_logger
    )

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
