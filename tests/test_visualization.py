from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mvhpe3d.visualization import (
    camera_id_to_camera_key,
    load_camera_parameters,
    overlay_mask_on_image,
    render_projected_mesh_mask,
    resolve_camera_json_path,
    resolve_rgb_image_path,
)


def test_resolve_rgb_image_path_finds_expected_file(tmp_path: Path) -> None:
    rgb_dir = tmp_path / "rgb"
    rgb_dir.mkdir()
    image_path = rgb_dir / "seq_001_kinect_000_000001.jpg"
    image_path.write_bytes(b"test")

    resolved = resolve_rgb_image_path(
        rgb_dir,
        sequence_id="seq_001",
        camera_id="kinect_000",
        frame_id="000001",
    )

    assert resolved == image_path.resolve()


def test_camera_id_to_camera_key_maps_humman_names() -> None:
    assert camera_id_to_camera_key("iphone") == "iphone"
    assert camera_id_to_camera_key("kinect_007") == "kinect_color_007"


def test_load_camera_parameters_reads_json_payload(tmp_path: Path) -> None:
    cameras_dir = tmp_path / "cameras"
    cameras_dir.mkdir()
    camera_json = cameras_dir / "seq_001_cameras.json"
    camera_json.write_text(
        json.dumps(
            {
                "kinect_color_000": {
                    "K": [[100.0, 0.0, 50.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]],
                    "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "T": [0.0, 0.0, 0.0],
                }
            }
        ),
        encoding="utf-8",
    )

    resolved_json = resolve_camera_json_path(cameras_dir, sequence_id="seq_001")
    camera = load_camera_parameters(
        cameras_dir,
        sequence_id="seq_001",
        camera_id="kinect_000",
    )

    assert resolved_json == camera_json.resolve()
    assert camera.intrinsics.shape == (3, 3)
    assert camera.rotation.shape == (3, 3)
    assert camera.translation.shape == (3,)


def test_render_projected_mesh_mask_and_overlay_mask_on_image(tmp_path: Path) -> None:
    cameras_dir = tmp_path / "cameras"
    cameras_dir.mkdir()
    (cameras_dir / "seq_001_cameras.json").write_text(
        json.dumps(
            {
                "kinect_color_000": {
                    "K": [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
                    "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "T": [0.0, 0.0, 0.0],
                }
            }
        ),
        encoding="utf-8",
    )
    camera = load_camera_parameters(
        cameras_dir,
        sequence_id="seq_001",
        camera_id="kinect_000",
    )
    vertices_world = np.array(
        [
            [-0.2, -0.2, 2.0],
            [0.2, -0.2, 2.0],
            [0.0, 0.2, 2.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    mask = render_projected_mesh_mask(
        (100, 100),
        vertices_world=vertices_world,
        faces=faces,
        camera=camera,
    )
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    overlay = overlay_mask_on_image(
        image,
        mask,
        color=(0, 255, 0),
        alpha=0.5,
    )

    assert mask.any()
    assert overlay.shape == image.shape
    assert overlay[..., 1].max() > 0
