from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _load_filter_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "filter_panoptic_stage1_manifest.py"
    spec = importlib.util.spec_from_file_location("mvhpe3d_filter_panoptic_manifest", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load Panoptic filter script from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_prediction(path: Path, *, image_size: tuple[int, int] = (224, 224)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        mhr_model_params=np.zeros((1, 204), dtype=np.float32),
        shape_params=np.zeros((1, 45), dtype=np.float32),
        pred_cam_t=np.zeros((1, 3), dtype=np.float32),
        cam_int=np.array(
            [[100.0, 0.0, 112.0], [0.0, 100.0, 112.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        image_size=np.asarray(image_size, dtype=np.int32),
    )


def _write_camera_json(dataset_root: Path, sequence_id: str) -> None:
    camera_path = dataset_root / sequence_id / "meta" / "cameras_kinect_cropped.json"
    camera_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kinect_1": {
            "K_color": [[100.0, 0.0, 112.0], [0.0, 100.0, 112.0], [0.0, 0.0, 1.0]],
            "extrinsic_world_to_color": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "extrinsic_world_to_color_unit": "m",
        },
        "kinect_2": {
            "K_color": [[100.0, 0.0, 112.0], [0.0, 100.0, 112.0], [0.0, 0.0, 1.0]],
            "extrinsic_world_to_color": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "extrinsic_world_to_color_unit": "m",
        },
    }
    camera_path.write_text(json.dumps(payload), encoding="utf-8")


def _write_gt3d(path: Path, *, x_cm: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joints = np.zeros((19, 4), dtype=np.float32)
    joints[:, 0] = x_cm
    joints[:, 2] = 200.0
    joints[:, 3] = 0.8
    np.save(path, joints)


def test_filter_samples_keeps_visible_views_and_drops_offscreen_views(tmp_path: Path) -> None:
    module = _load_filter_module()
    dataset_root = tmp_path / "panoptic"
    sequence_id = "170307_dance5"
    _write_camera_json(dataset_root, sequence_id)
    _write_gt3d(dataset_root / sequence_id / "gt3d" / "00000001.npy", x_cm=0.0)
    _write_gt3d(dataset_root / sequence_id / "gt3d" / "00000002.npy", x_cm=1000.0)

    predictions_dir = dataset_root / "sam3dbody"
    visible_view_a = predictions_dir / f"{sequence_id}_kinect_001_00000001.npz"
    visible_view_b = predictions_dir / f"{sequence_id}_kinect_002_00000001.npz"
    invisible_view_a = predictions_dir / f"{sequence_id}_kinect_001_00000002.npz"
    invisible_view_b = predictions_dir / f"{sequence_id}_kinect_002_00000002.npz"
    for path in (visible_view_a, visible_view_b, invisible_view_a, invisible_view_b):
        _write_prediction(path)

    raw_samples = [
        {
            "sample_id": "visible",
            "sequence_id": sequence_id,
            "frame_id": "00000001",
            "views": [
                {"camera_id": "kinect_001", "npz_path": str(visible_view_a)},
                {"camera_id": "kinect_002", "npz_path": str(visible_view_b)},
            ],
        },
        {
            "sample_id": "invisible",
            "sequence_id": sequence_id,
            "frame_id": "00000002",
            "views": [
                {"camera_id": "kinect_001", "npz_path": str(invisible_view_a)},
                {"camera_id": "kinect_002", "npz_path": str(invisible_view_b)},
            ],
        },
    ]

    filtered, stats = module.filter_samples(
        raw_samples,
        dataset_root=dataset_root,
        input_manifest_dir=tmp_path,
        output_manifest_dir=tmp_path,
        min_views=2,
        min_visible_joints=8,
        confidence_threshold=0.05,
        visibility_margin_px=0.0,
        min_depth=1.0e-4,
        require_root_visible=True,
        absolute_paths=True,
    )

    assert [sample["sample_id"] for sample in filtered] == ["visible"]
    assert len(filtered[0]["views"]) == 2
    assert stats.input_samples == 2
    assert stats.output_samples == 1
    assert stats.input_views == 4
    assert stats.output_views == 2
    assert stats.root_not_visible_views == 2
    assert stats.dropped_samples == 1


def test_filter_samples_drops_sample_when_too_few_views_remain(tmp_path: Path) -> None:
    module = _load_filter_module()
    dataset_root = tmp_path / "panoptic"
    sequence_id = "170307_dance5"
    _write_camera_json(dataset_root, sequence_id)
    _write_gt3d(dataset_root / sequence_id / "gt3d" / "00000001.npy", x_cm=0.0)

    predictions_dir = dataset_root / "sam3dbody"
    visible_view = predictions_dir / f"{sequence_id}_kinect_001_00000001.npz"
    missing_view = predictions_dir / f"{sequence_id}_kinect_002_00000001.npz"
    _write_prediction(visible_view)

    raw_samples = [
        {
            "sample_id": "one_valid_view",
            "sequence_id": sequence_id,
            "frame_id": "00000001",
            "views": [
                {"camera_id": "kinect_001", "npz_path": str(visible_view)},
                {"camera_id": "kinect_002", "npz_path": str(missing_view)},
            ],
        },
    ]

    filtered, stats = module.filter_samples(
        raw_samples,
        dataset_root=dataset_root,
        input_manifest_dir=tmp_path,
        output_manifest_dir=tmp_path,
        min_views=2,
        min_visible_joints=8,
        confidence_threshold=0.05,
        visibility_margin_px=0.0,
        min_depth=1.0e-4,
        require_root_visible=True,
        absolute_paths=True,
    )

    assert filtered == []
    assert stats.missing_prediction_views == 1
    assert stats.dropped_samples == 1
