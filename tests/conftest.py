from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _write_prediction_npz(path: Path) -> None:
    payload = {
        "mhr_model_params": np.zeros(204, dtype=np.float32),
        "shape_params": np.zeros(45, dtype=np.float32),
        "pred_cam_t": np.array([0.1, 0.2, 2.0], dtype=np.float32),
        "cam_int": np.eye(3, dtype=np.float32),
        "image_size": np.array([224, 224], dtype=np.float32),
    }
    np.savez(path, **payload)


def _write_gt_smpl_sequence(path: Path, *, num_frames: int) -> None:
    payload = {
        "global_orient": np.zeros((num_frames, 3), dtype=np.float32),
        "body_pose": np.zeros((num_frames, 69), dtype=np.float32),
        "betas": np.zeros((num_frames, 10), dtype=np.float32),
        "transl": np.zeros((num_frames, 3), dtype=np.float32),
    }
    np.savez(path, **payload)


def _write_camera_json(path: Path, *, camera_ids: list[str]) -> None:
    payload = {}
    for index, camera_id in enumerate(camera_ids):
        suffix = camera_id.split("_", maxsplit=1)[1]
        payload[f"kinect_color_{suffix}"] = {
            "K": [[100.0, 0.0, 112.0], [0.0, 100.0, 112.0], [0.0, 0.0, 1.0]],
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "T": [0.05 * index, 0.0, 0.0],
        }
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def sample_manifest(tmp_path: Path) -> Path:
    cameras = ["kinect_000", "kinect_001", "kinect_002"]
    gt_smpl_dir = tmp_path / "smpl"
    gt_smpl_dir.mkdir()
    cameras_dir = tmp_path / "cameras"
    cameras_dir.mkdir()
    for camera_id in cameras:
        _write_prediction_npz(tmp_path / f"{camera_id}.npz")

    _write_gt_smpl_sequence(gt_smpl_dir / "seq_001_smpl_params.npz", num_frames=3)
    _write_camera_json(cameras_dir / "seq_001_cameras.json", camera_ids=cameras)

    samples = []
    for split, frame_id in [("train", "000001"), ("val", "000002"), ("test", "000003")]:
        samples.append(
            {
                "sample_id": f"sample_{split}",
                "sequence_id": "seq_001",
                "frame_id": frame_id,
                "split": split,
                "views": [
                    {
                        "camera_id": camera_id,
                        "npz_path": str(tmp_path / f"{camera_id}.npz"),
                    }
                    for camera_id in cameras
                ],
            }
        )

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    return manifest_path


@pytest.fixture()
def sample_inventory_manifest(tmp_path: Path) -> Path:
    cameras = ["kinect_000", "kinect_001", "kinect_002", "kinect_003"]
    gt_smpl_dir = tmp_path / "smpl"
    gt_smpl_dir.mkdir(exist_ok=True)
    cameras_dir = tmp_path / "cameras"
    cameras_dir.mkdir(exist_ok=True)
    for camera_id in cameras:
        _write_prediction_npz(tmp_path / f"{camera_id}.npz")

    samples = []
    sample_specs = [
        ("sample_0", "subject_a", "action_walk", "training", "000001"),
        ("sample_1", "subject_a", "action_jump", "training", "000002"),
        ("sample_2", "subject_b", "action_walk", "training", "000003"),
        ("sample_3", "subject_b", "action_jump", "validation", "000004"),
    ]
    for sample_id, subject_id, action_id, split, frame_id in sample_specs:
        sequence_id = f"{subject_id}_{action_id}"
        _write_gt_smpl_sequence(
            gt_smpl_dir / f"{sequence_id}_smpl_params.npz",
            num_frames=4,
        )
        _write_camera_json(cameras_dir / f"{sequence_id}_cameras.json", camera_ids=cameras)
        samples.append(
            {
                "sample_id": sample_id,
                "sequence_id": sequence_id,
                "frame_id": frame_id,
                "split": split,
                "subject_id": subject_id,
                "action_id": action_id,
                "views": [
                    {
                        "camera_id": camera_id,
                        "npz_path": str(tmp_path / f"{camera_id}.npz"),
                    }
                    for camera_id in cameras
                ],
            }
        )

    manifest_path = tmp_path / "inventory_manifest.json"
    manifest_path.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    return manifest_path


@pytest.fixture()
def sample_split_config(tmp_path: Path) -> Path:
    payload = """
cross_camera_split:
  train_dataset:
    split: null
    cameras: [kinect_000, kinect_001]
    subjects: null
    actions: null
  val_dataset:
    split: null
    cameras: [kinect_002, kinect_003]
    subjects: null
    actions: null
  test_dataset:
    split: null
    cameras: [kinect_002, kinect_003]
    subjects: null
    actions: null

random_training_only:
  ratio: 0.5
  random_seed: 3
  candidate_dataset:
    split: training
    cameras: null
    subjects: null
    actions: null
  train_dataset:
    split: null
    cameras: [kinect_000, kinect_001]
    subjects: null
    actions: null
  val_dataset:
    split: null
    cameras: [kinect_002, kinect_003]
    subjects: null
    actions: null
  test_dataset:
    split: null
    cameras: [kinect_002, kinect_003]
    subjects: null
    actions: null

cross_subject_split:
  train_dataset:
    split: null
    cameras: null
    subjects: [subject_a]
    actions: null
  val_dataset:
    split: null
    cameras: null
    subjects: [subject_b]
    actions: null
  test_dataset:
    split: null
    cameras: null
    subjects: [subject_b]
    actions: null
"""
    split_config_path = tmp_path / "splits.yaml"
    split_config_path.write_text(payload.strip() + "\n", encoding="utf-8")
    return split_config_path
