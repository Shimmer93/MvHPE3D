from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import numpy as np

from mvhpe3d.data import Stage1DataConfig, Stage1HuMManDataModule
from mvhpe3d.data.datasets import PanopticStage1Dataset
from mvhpe3d.data.splits import filter_records_by_split, load_sample_records
from mvhpe3d.utils import load_panoptic_camera_parameters


def _load_panoptic_smpl_precompute_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "precompute_panoptic_gt_smpl.py"
    )
    spec = importlib.util.spec_from_file_location(
        "mvhpe3d_precompute_panoptic_gt_smpl", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Failed to load Panoptic SMPL precompute script from {module_path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_prediction_npz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        mhr_model_params=np.zeros((1, 204), dtype=np.float32),
        shape_params=np.zeros((1, 45), dtype=np.float32),
        pred_cam_t=np.zeros((1, 3), dtype=np.float32),
        cam_int=np.eye(3, dtype=np.float32),
        image_size=np.array([224, 224], dtype=np.int32),
    )


def _write_target_npz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        global_orient=np.zeros(3, dtype=np.float32),
        body_pose=np.zeros(69, dtype=np.float32),
        betas=np.ones(10, dtype=np.float32),
        transl=np.zeros(3, dtype=np.float32),
    )


def _write_gt3d(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = np.zeros((19, 4), dtype=np.float32)
    payload[:, :3] = np.arange(57, dtype=np.float32).reshape(19, 3)
    payload[:, 3] = 1.0
    np.save(path, payload)


def _write_panoptic_camera_json(dataset_root: Path, *, sequence_id: str) -> None:
    camera_json = dataset_root / sequence_id / "meta" / "cameras_kinect_cropped.json"
    camera_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kinect_1": {
            "K_color": [[100.0, 0.0, 112.0], [0.0, 100.0, 112.0], [0.0, 0.0, 1.0]],
            "extrinsic_world_to_color": [
                [1.0, 0.0, 0.0, 100.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            "extrinsic_world_to_color_unit": "cm",
        },
        "kinect_7": {
            "K_color": [[120.0, 0.0, 112.0], [0.0, 120.0, 112.0], [0.0, 0.0, 1.0]],
            "extrinsic_world_to_color": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 200.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            "extrinsic_world_to_color_unit": "cm",
        },
    }
    camera_json.write_text(json.dumps(payload), encoding="utf-8")


def _write_panoptic_manifest(
    tmp_path: Path, *, sequence_id: str = "170307_dance5"
) -> Path:
    predictions_dir = tmp_path / "sam3dbody"
    target_path = tmp_path / "smpl" / sequence_id / "00001011.npz"
    for camera_id in ("kinect_001", "kinect_007"):
        _write_prediction_npz(
            predictions_dir / f"{sequence_id}_{camera_id}_00001011.npz"
        )
    _write_target_npz(target_path)

    manifest = {
        "samples": [
            {
                "sample_id": f"{sequence_id}_frame00001011",
                "sequence_id": sequence_id,
                "frame_id": "00001011",
                "split": "train",
                "target_path": str(target_path),
                "views": [
                    {
                        "camera_id": "kinect_001",
                        "npz_path": str(
                            predictions_dir / f"{sequence_id}_kinect_001_00001011.npz"
                        ),
                    },
                    {
                        "camera_id": "kinect_007",
                        "npz_path": str(
                            predictions_dir / f"{sequence_id}_kinect_007_00001011.npz"
                        ),
                    },
                ],
            }
        ]
    }
    manifest_path = tmp_path / "panoptic_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_load_panoptic_camera_parameters_maps_padded_camera_ids(tmp_path: Path) -> None:
    sequence_id = "170307_dance5"
    _write_panoptic_camera_json(tmp_path, sequence_id=sequence_id)

    camera = load_panoptic_camera_parameters(
        tmp_path,
        sequence_id=sequence_id,
        camera_id="kinect_001",
    )

    assert tuple(camera.intrinsics.shape) == (3, 3)
    assert np.allclose(camera.rotation, np.eye(3, dtype=np.float32))
    assert np.allclose(camera.translation, np.array([1.0, 0.0, 0.0], dtype=np.float32))


def test_panoptic_stage1_dataset_reads_target_path_and_native_cameras(
    tmp_path: Path,
) -> None:
    sequence_id = "170307_dance5"
    manifest_path = _write_panoptic_manifest(tmp_path, sequence_id=sequence_id)
    _write_panoptic_camera_json(tmp_path, sequence_id=sequence_id)
    _write_gt3d(tmp_path / sequence_id / "gt3d" / "00001011.npy")
    records = filter_records_by_split(load_sample_records(manifest_path), "train")

    dataset = PanopticStage1Dataset(
        records,
        num_views=2,
        train=False,
        gt_smpl_dir=tmp_path / "smpl",
        cameras_dir=tmp_path,
    )

    sample = dataset[0]

    assert tuple(sample["views_input"].shape) == (2, 249)
    assert tuple(sample["target_body_pose"].shape) == (69,)
    assert np.allclose(sample["target_betas"].numpy(), np.ones(10, dtype=np.float32))
    assert sample["meta"]["camera_ids"] == ["kinect_001", "kinect_007"]
    assert np.allclose(
        sample["target_aux"]["camera_translation"].numpy(),
        np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32),
    )
    assert tuple(sample["target_aux"]["panoptic_joints_world"].shape) == (19, 3)
    assert tuple(sample["target_aux"]["panoptic_confidence"].shape) == (19,)
    assert np.allclose(
        sample["target_aux"]["panoptic_joints_world"][0].numpy(), [0.0, 0.01, 0.02]
    )


def test_panoptic_stage1_dataset_sequence_cache_uses_frame_ids(tmp_path: Path) -> None:
    sequence_id = "170307_dance5"
    predictions_dir = tmp_path / "sam3dbody"
    for camera_id in ("kinect_001", "kinect_007"):
        _write_prediction_npz(
            predictions_dir / f"{sequence_id}_{camera_id}_00001011.npz"
        )
    _write_panoptic_camera_json(tmp_path, sequence_id=sequence_id)

    smpl_dir = tmp_path / "smpl"
    smpl_dir.mkdir()
    body_pose = np.zeros((2, 69), dtype=np.float32)
    body_pose[1, 0] = 3.0
    np.savez_compressed(
        smpl_dir / f"{sequence_id}_smpl_params.npz",
        frame_ids=np.array([161, 1011], dtype=np.int32),
        global_orient=np.zeros((2, 3), dtype=np.float32),
        body_pose=body_pose,
        betas=np.zeros((2, 10), dtype=np.float32),
        transl=np.zeros((2, 3), dtype=np.float32),
    )
    manifest = {
        "samples": [
            {
                "sample_id": f"{sequence_id}_frame00001011",
                "sequence_id": sequence_id,
                "frame_id": "00001011",
                "split": "train",
                "views": [
                    {
                        "camera_id": "kinect_001",
                        "npz_path": str(
                            predictions_dir / f"{sequence_id}_kinect_001_00001011.npz"
                        ),
                    },
                    {
                        "camera_id": "kinect_007",
                        "npz_path": str(
                            predictions_dir / f"{sequence_id}_kinect_007_00001011.npz"
                        ),
                    },
                ],
            }
        ]
    }
    manifest_path = tmp_path / "panoptic_sequence_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    records = filter_records_by_split(load_sample_records(manifest_path), "train")

    dataset = PanopticStage1Dataset(
        records,
        num_views=2,
        train=False,
        gt_smpl_dir=smpl_dir,
        cameras_dir=tmp_path,
    )

    sample = dataset[0]

    assert sample["target_body_pose"][0].item() == 3.0


def test_panoptic_stage1_datamodule_routes_by_config_name(tmp_path: Path) -> None:
    sequence_id = "170307_dance5"
    manifest_path = _write_panoptic_manifest(tmp_path, sequence_id=sequence_id)
    _write_panoptic_camera_json(tmp_path, sequence_id=sequence_id)

    datamodule = Stage1HuMManDataModule(
        Stage1DataConfig(
            name="panoptic_stage1",
            manifest_path=str(manifest_path),
            gt_smpl_dir=str(tmp_path / "smpl"),
            cameras_dir=str(tmp_path),
            train_split="train",
            val_split="train",
            test_split="train",
            num_views=2,
            batch_size=1,
            drop_last_train=False,
        )
    )

    datamodule.prepare_data()
    datamodule.setup(None)

    assert isinstance(datamodule.train_dataset, PanopticStage1Dataset)
    batch = next(iter(datamodule.train_dataloader()))
    assert tuple(batch["views_input"].shape) == (1, 2, 249)


def test_panoptic_datamodule_accepts_target_paths_without_gt_smpl_dir(
    tmp_path: Path,
) -> None:
    sequence_id = "170307_dance5"
    manifest_path = _write_panoptic_manifest(tmp_path, sequence_id=sequence_id)
    _write_panoptic_camera_json(tmp_path, sequence_id=sequence_id)

    datamodule = Stage1HuMManDataModule(
        Stage1DataConfig(
            name="panoptic_stage1",
            manifest_path=str(manifest_path),
            gt_smpl_dir=str(tmp_path / "missing_smpl"),
            cameras_dir=str(tmp_path),
            train_split="train",
            val_split="train",
            test_split="train",
            num_views=2,
            batch_size=1,
            drop_last_train=False,
        )
    )

    datamodule.prepare_data()
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))

    assert tuple(batch["views_input"].shape) == (1, 2, 249)


def test_panoptic_datamodule_explains_missing_smpl_targets(tmp_path: Path) -> None:
    sequence_id = "170307_dance5"
    predictions_dir = tmp_path / "sam3dbody"
    for camera_id in ("kinect_001", "kinect_007"):
        _write_prediction_npz(
            predictions_dir / f"{sequence_id}_{camera_id}_00001011.npz"
        )
    _write_panoptic_camera_json(tmp_path, sequence_id=sequence_id)
    manifest = {
        "samples": [
            {
                "sample_id": f"{sequence_id}_frame00001011",
                "sequence_id": sequence_id,
                "frame_id": "00001011",
                "split": "train",
                "views": [
                    {
                        "camera_id": "kinect_001",
                        "npz_path": str(
                            predictions_dir / f"{sequence_id}_kinect_001_00001011.npz"
                        ),
                    },
                    {
                        "camera_id": "kinect_007",
                        "npz_path": str(
                            predictions_dir / f"{sequence_id}_kinect_007_00001011.npz"
                        ),
                    },
                ],
            }
        ]
    }
    manifest_path = tmp_path / "panoptic_no_targets_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    datamodule = Stage1HuMManDataModule(
        Stage1DataConfig(
            name="panoptic_stage1",
            manifest_path=str(manifest_path),
            gt_smpl_dir=str(tmp_path / "missing_smpl"),
            cameras_dir=str(tmp_path),
        )
    )

    try:
        datamodule.prepare_data()
    except FileNotFoundError as exc:
        message = str(exc)
    else:
        raise AssertionError(
            "Expected missing Panoptic SMPL targets to raise FileNotFoundError"
        )

    assert "Panoptic Stage 1 requires SMPL supervision targets" in message
    assert "SAM3DBody" in message


def test_panoptic_gt_smpl_precompute_collects_manifest_requested_frames(
    tmp_path: Path,
) -> None:
    precompute = _load_panoptic_smpl_precompute_module()
    sequence_dir = tmp_path / "170307_dance5"
    gt3d_dir = sequence_dir / "gt3d"
    gt3d_dir.mkdir(parents=True)
    for frame_id in ("00000161", "00001011", "00002000"):
        np.save(gt3d_dir / f"{frame_id}.npy", np.zeros((19, 4), dtype=np.float32))

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "sample",
                        "sequence_id": "170307_dance5",
                        "frame_id": "00001011",
                        "views": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    requested = precompute.load_requested_frames(str(manifest_path))
    frames = precompute.collect_sequence_frames(
        sequence_dir,
        requested_frame_ids=requested["170307_dance5"],
        max_frames=None,
    )

    assert [frame.frame_id for frame in frames] == ["1011"]


def test_panoptic_gt_smpl_precompute_loads_gt3d_shape(tmp_path: Path) -> None:
    precompute = _load_panoptic_smpl_precompute_module()
    gt3d_path = tmp_path / "00001011.npy"
    payload = np.zeros((19, 4), dtype=np.float16)
    payload[2] = [1.0, 2.0, 3.0, 0.5]
    np.save(gt3d_path, payload)

    loaded = precompute.load_panoptic_gt3d_file(gt3d_path)

    assert loaded.shape == (19, 4)
    assert loaded.dtype == np.float32
    assert np.allclose(loaded[2], [1.0, 2.0, 3.0, 0.5])
