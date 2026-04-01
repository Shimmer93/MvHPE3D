from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _write_prediction_npz(path: Path) -> None:
    payload = {
        "mhr_model_params": np.zeros(204, dtype=np.float32),
        "shape_params": np.zeros(45, dtype=np.float32),
        "pred_cam_t": np.array([0.1, 0.2, 2.0], dtype=np.float32),
        "cam_int": np.eye(3, dtype=np.float32),
        "image_size": np.array([224, 224], dtype=np.float32),
    }
    np.savez(path, **payload)


def _write_target_npz(path: Path) -> None:
    payload = {
        "mhr_model_params": np.zeros(204, dtype=np.float32),
        "shape_params": np.zeros(45, dtype=np.float32),
    }
    np.savez(path, **payload)


@pytest.fixture()
def sample_manifest(tmp_path: Path) -> Path:
    cameras = ["kinect_000", "kinect_001", "kinect_002"]
    for camera_id in cameras:
        _write_prediction_npz(tmp_path / f"{camera_id}.npz")

    target_path = tmp_path / "target.npz"
    _write_target_npz(target_path)

    samples = []
    for split, frame_id in [("train", "000001"), ("val", "000002"), ("test", "000003")]:
        samples.append(
            {
                "sample_id": f"sample_{split}",
                "sequence_id": "seq_001",
                "frame_id": frame_id,
                "split": split,
                "target_path": str(target_path),
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
