from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_panoptic_script_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "save_panoptic_sam3dbody.py"
    spec = importlib.util.spec_from_file_location("mvhpe3d_save_panoptic_sam3dbody", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load Panoptic SAM3DBody script from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")


def test_discover_camera_tasks_matches_panoptic_layout_and_filters_padded_camera_ids(
    tmp_path: Path,
) -> None:
    panoptic = _load_panoptic_script_module()
    root = tmp_path / "panoptic"
    _touch(root / "170307_dance5" / "rgb" / "kinect_1" / "00001011.jpg")
    _touch(root / "170307_dance5" / "rgb" / "kinect_10" / "00001011.jpg")
    _touch(root / "170407_office2" / "rgb" / "kinect_1" / "00000001.jpg")
    (root / "170307_dance5" / "meta").mkdir(parents=True)

    tasks = panoptic.discover_camera_tasks(
        dataset_root=root,
        sequences=["170307_dance5"],
        cameras=["kinect_001"],
        camera_id_width=3,
    )

    assert [(task.sequence_id, task.camera_id) for task in tasks] == [
        ("170307_dance5", "kinect_1")
    ]


def test_final_output_path_flat_layout_is_manifest_parseable(tmp_path: Path) -> None:
    panoptic = _load_panoptic_script_module()

    path = panoptic.final_output_path(
        output_root=tmp_path / "sam3dbody",
        layout="flat",
        sequence_id="170307_dance5",
        camera_id="kinect_1",
        frame_stem="00001011",
        camera_id_width=3,
    )

    assert path.name == "170307_dance5_kinect_001_00001011.npz"


def test_final_output_path_nested_layout_preserves_sequence_and_camera_dirs(
    tmp_path: Path,
) -> None:
    panoptic = _load_panoptic_script_module()

    path = panoptic.final_output_path(
        output_root=tmp_path / "sam3dbody",
        layout="nested",
        sequence_id="170307_dance5",
        camera_id="kinect_10",
        frame_stem="00001011",
        camera_id_width=3,
    )

    assert path.relative_to(tmp_path).as_posix() == (
        "sam3dbody/170307_dance5/kinect_010/00001011.npz"
    )
