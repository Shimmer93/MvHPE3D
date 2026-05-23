from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

from mvhpe3d.data import Stage2DataConfig, Stage2HuMManDataModule
from mvhpe3d.data.datasets import HuMManStage2Dataset
from mvhpe3d.data.mpi_inf_3dhp import (
    MPII3D_HEATFORMER_CAMERA_IDS,
    MPII3D_HEATFORMER_ROOT_INDEX,
    annotation_frame_count,
    load_mpii3d_joint_target,
)
from mvhpe3d.data.splits import load_sample_records
from mvhpe3d.utils import cache_path_for_source_npz, load_camera_parameters


def _load_mpii3d_manifest_builder_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_mpi_inf_3dhp_stage2_manifest.py"
    )
    spec = importlib.util.spec_from_file_location("mvhpe3d_mpii3d_manifest_builder", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load manifest builder from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_mpii3d_sequence(root: Path, *, num_frames: int = 12) -> None:
    sequence_dir = root / "S1" / "Seq1"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    (sequence_dir / "camera.calibration").write_text(
        "\n".join(_camera_block(camera_index) for camera_index in range(14)),
        encoding="utf-8",
    )
    annot = np.empty((14, 1), dtype=object)
    univ_annot = np.empty((14, 1), dtype=object)
    annot2 = np.empty((14, 1), dtype=object)
    base = np.arange(num_frames * 28 * 3, dtype=np.float32).reshape(num_frames, 84)
    for camera_index in range(14):
        annot[camera_index, 0] = base + camera_index
        univ_annot[camera_index, 0] = base + camera_index
        annot2[camera_index, 0] = np.zeros((num_frames, 56), dtype=np.float32)
    sio.savemat(
        sequence_dir / "annot.mat",
        {
            "annot2": annot2,
            "annot3": annot,
            "univ_annot3": univ_annot,
        },
    )


def _camera_block(camera_index: int) -> str:
    intrinsic = " ".join(str(value) for value in np.eye(4, dtype=np.float32).reshape(-1))
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[0, 3] = float(camera_index)
    extrinsic_text = " ".join(str(value) for value in extrinsic.reshape(-1))
    return "\n".join(
        [
            f"name          {camera_index}",
            "  sensor      10 10",
            "  size        2048 2048",
            "  animated    0",
            f"  intrinsic   {intrinsic} ",
            f"  extrinsic   {extrinsic_text} ",
            "  radial      0",
        ]
    )


def _write_valid_prediction(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        mhr_model_params=np.zeros((1, 204), dtype=np.float32),
        shape_params=np.zeros((1, 45), dtype=np.float32),
        pred_cam_t=np.zeros((1, 3), dtype=np.float32),
        cam_int=np.eye(3, dtype=np.float32),
        image_size=np.array([224, 224], dtype=np.int32),
    )


def _write_cached_input_smpl(cache_dir: Path, source_npz_path: Path) -> None:
    cache_path = cache_path_for_source_npz(cache_dir, source_npz_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        body_pose=np.zeros(69, dtype=np.float32),
        betas=np.zeros(10, dtype=np.float32),
        global_orient=np.zeros(3, dtype=np.float32),
        transl=np.zeros(3, dtype=np.float32),
    )


def test_mpii3d_camera_parser_and_joint_target(tmp_path: Path) -> None:
    _write_mpii3d_sequence(tmp_path)

    camera = load_camera_parameters(
        tmp_path,
        sequence_id="S1_Seq1",
        camera_id="video_8",
    )
    assert tuple(camera.intrinsics.shape) == (3, 3)
    assert tuple(camera.rotation.shape) == (3, 3)
    assert np.allclose(camera.translation, np.array([0.008, 0.0, 0.0], dtype=np.float32))
    assert annotation_frame_count(tmp_path, sequence_id="S1_Seq1") == 12

    target = load_mpii3d_joint_target(
        tmp_path,
        sequence_id="S1_Seq1",
        frame_id="00000005",
        camera_ids=MPII3D_HEATFORMER_CAMERA_IDS,
    )
    assert tuple(target["joints"].shape) == (17, 3)
    assert tuple(target["confidence"].shape) == (17,)
    assert tuple(target["smpl_indices"].shape) == (17,)
    assert int(target["root_index"]) == MPII3D_HEATFORMER_ROOT_INDEX


def test_mpii3d_stage2_manifest_builder_and_dataset(tmp_path: Path) -> None:
    _write_mpii3d_sequence(tmp_path)
    predictions_root = tmp_path / "sam3dbody"
    for camera_id in MPII3D_HEATFORMER_CAMERA_IDS:
        _write_valid_prediction(
            predictions_root / "S1_Seq1" / camera_id / "frame_00000000.npz"
        )

    builder = _load_mpii3d_manifest_builder_module()
    samples, report = builder.build_sequence_samples(
        dataset_root=tmp_path,
        sam3dbody_root=predictions_root,
        manifest_dir=tmp_path,
        sequence_id="S1_Seq1",
        split_name="train",
        cameras=list(MPII3D_HEATFORMER_CAMERA_IDS),
        sampling=10,
        min_views=4,
        max_frames=1,
        use_absolute_paths=True,
    )
    assert len(samples) == 1
    assert report["samples"] == 1
    assert [view["camera_id"] for view in samples[0]["views"]] == list(
        MPII3D_HEATFORMER_CAMERA_IDS
    )

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    cache_dir = tmp_path / "input_smpl_cache"
    for view in samples[0]["views"]:
        _write_cached_input_smpl(cache_dir, Path(view["npz_path"]))

    records = load_sample_records(manifest_path)
    dataset = HuMManStage2Dataset(
        records,
        num_views=4,
        train=False,
        gt_smpl_dir=tmp_path / "unused_smpl",
        cameras_dir=tmp_path,
        input_smpl_cache_dir=cache_dir,
        joint_target_dataset="mpi_inf_3dhp",
        joint_target_root=tmp_path,
    )
    sample = dataset[0]

    assert tuple(sample["views_input"].shape) == (4, 148)
    assert tuple(sample["target_joints"].shape) == (17, 3)
    assert tuple(sample["target_joint_confidence"].shape) == (17,)
    assert tuple(sample["view_aux"]["camera_rotation"].shape) == (4, 3, 3)
    assert tuple(sample["view_aux"]["camera_translation"].shape) == (4, 3)


def test_mpii3d_manifest_preserves_partial_views_and_datamodule_filters_num_views(
    tmp_path: Path,
) -> None:
    _write_mpii3d_sequence(tmp_path, num_frames=21)
    predictions_root = tmp_path / "sam3dbody"
    for camera_id in MPII3D_HEATFORMER_CAMERA_IDS[:3]:
        _write_valid_prediction(
            predictions_root / "S1_Seq1" / camera_id / "frame_00000000.npz"
        )
    for camera_id in MPII3D_HEATFORMER_CAMERA_IDS:
        _write_valid_prediction(
            predictions_root / "S1_Seq1" / camera_id / "frame_00000010.npz"
        )

    builder = _load_mpii3d_manifest_builder_module()
    samples, report = builder.build_sequence_samples(
        dataset_root=tmp_path,
        sam3dbody_root=predictions_root,
        manifest_dir=tmp_path,
        sequence_id="S1_Seq1",
        split_name="train",
        cameras=list(MPII3D_HEATFORMER_CAMERA_IDS),
        sampling=10,
        min_views=0,
        max_frames=2,
        use_absolute_paths=True,
    )
    assert len(samples) == 2
    assert [len(sample["views"]) for sample in samples] == [3, 4]
    assert report["available_view_count_distribution"] == {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 1,
        "4": 1,
    }

    manifest_path = tmp_path / "partial_manifest.json"
    manifest_path.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    records = load_sample_records(manifest_path)
    config_kwargs = {
        "manifest_path": str(manifest_path),
        "gt_smpl_dir": str(tmp_path / "unused_smpl"),
        "cameras_dir": str(tmp_path),
        "input_smpl_cache_dir": str(tmp_path / "input_smpl_cache"),
        "joint_target_dataset": "mpi_inf_3dhp",
        "joint_target_root": str(tmp_path),
        "train_split": "train",
        "val_split": "train",
        "test_split": "train",
    }

    four_view_module = Stage2HuMManDataModule(
        Stage2DataConfig(num_views=4, **config_kwargs)
    )
    three_view_module = Stage2HuMManDataModule(
        Stage2DataConfig(num_views=3, **config_kwargs)
    )

    assert len(four_view_module._resolve_dataset_records(records)["train"]) == 1
    assert len(three_view_module._resolve_dataset_records(records)["train"]) == 2
