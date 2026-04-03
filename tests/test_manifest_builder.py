from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_manifest_builder_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "build_humman_stage1_manifest.py"
    spec = importlib.util.spec_from_file_location("mvhpe3d_manifest_builder", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load manifest builder from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_valid_prediction(path: Path) -> None:
    np.savez_compressed(
        path,
        mhr_model_params=np.zeros((1, 204), dtype=np.float32),
        shape_params=np.zeros((1, 45), dtype=np.float32),
        pred_cam_t=np.zeros((1, 3), dtype=np.float32),
        cam_int=np.eye(3, dtype=np.float32),
        image_size=np.array([224, 224], dtype=np.int32),
    )


def _write_invalid_prediction(path: Path) -> None:
    np.savez_compressed(
        path,
        mhr_model_params=np.zeros((0, 204), dtype=np.float32),
        shape_params=np.zeros((0, 45), dtype=np.float32),
        pred_cam_t=np.zeros((0, 3), dtype=np.float32),
        cam_int=np.eye(3, dtype=np.float32),
        image_size=np.array([224, 224], dtype=np.int32),
    )


def test_manifest_builder_groups_valid_views_and_skips_invalid_exports(tmp_path: Path) -> None:
    manifest_builder = _load_manifest_builder_module()
    predictions_dir = tmp_path / "sam3dbody"
    predictions_dir.mkdir()

    _write_valid_prediction(predictions_dir / "p000001_a000123_kinect_000_000001.npz")
    _write_valid_prediction(predictions_dir / "p000001_a000123_kinect_001_000001.npz")
    _write_invalid_prediction(predictions_dir / "p000001_a000123_kinect_002_000001.npz")
    _write_valid_prediction(predictions_dir / "bad_name.npz")

    parsed_predictions, skipped_files = manifest_builder.collect_predictions(predictions_dir)
    samples = manifest_builder.build_manifest_samples(
        parsed_predictions,
        min_views=2,
        manifest_dir=tmp_path,
        use_absolute_paths=False,
    )

    assert len(parsed_predictions) == 2
    assert len(skipped_files) == 2
    assert len(samples) == 1
    assert samples[0]["sample_id"] == "p000001_a000123_frame000001"
    assert samples[0]["sequence_id"] == "p000001_a000123"
    assert samples[0]["frame_id"] == "000001"
    assert samples[0]["subject_id"] == "p000001"
    assert samples[0]["action_id"] == "a000123"
    assert [view["camera_id"] for view in samples[0]["views"]] == ["kinect_000", "kinect_001"]
    assert samples[0]["views"][0]["npz_path"].endswith("sam3dbody/p000001_a000123_kinect_000_000001.npz")
    assert "target_path" not in samples[0]
