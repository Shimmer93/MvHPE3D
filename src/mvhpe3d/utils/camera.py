"""Camera calibration helpers shared by data loading and visualization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraParameters:
    """One calibrated camera used for HuMMan world-to-camera transforms."""

    intrinsics: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray


def resolve_camera_json_path(cameras_dir: str | Path, *, sequence_id: str) -> Path:
    """Resolve the HuMMan sequence-level camera JSON path."""
    cameras_path = Path(cameras_dir).resolve() / f"{sequence_id}_cameras.json"
    if not cameras_path.exists():
        raise FileNotFoundError(f"Camera JSON does not exist: {cameras_path}")
    return cameras_path


def camera_id_to_camera_key(camera_id: str) -> str:
    """Map manifest camera IDs to HuMMan camera JSON keys."""
    if camera_id == "iphone":
        return "iphone"
    if camera_id.startswith("kinect_"):
        suffix = camera_id.split("_", maxsplit=1)[1]
        return f"kinect_color_{suffix}"
    raise KeyError(f"Unsupported camera_id '{camera_id}'")


def load_camera_parameters(
    cameras_dir: str | Path,
    *,
    sequence_id: str,
    camera_id: str,
) -> CameraParameters:
    """Load one camera calibration entry from the HuMMan JSON file."""
    camera_json_path = resolve_camera_json_path(cameras_dir, sequence_id=sequence_id)
    payload = json.loads(camera_json_path.read_text(encoding="utf-8"))
    camera_key = camera_id_to_camera_key(camera_id)
    if camera_key not in payload:
        raise KeyError(f"Camera key '{camera_key}' was not found in {camera_json_path}")

    camera_payload = payload[camera_key]
    return CameraParameters(
        intrinsics=np.asarray(camera_payload["K"], dtype=np.float32),
        rotation=np.asarray(camera_payload["R"], dtype=np.float32),
        translation=np.asarray(camera_payload["T"], dtype=np.float32),
    )


def transform_smpl_world_to_camera(
    *,
    global_orient: np.ndarray,
    transl: np.ndarray,
    camera: CameraParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform world-frame SMPL root pose and translation into one camera frame."""
    world_rotation = axis_angle_to_matrix(np.asarray(global_orient, dtype=np.float32))
    camera_rotation = np.asarray(camera.rotation, dtype=np.float32) @ world_rotation
    camera_global_orient = matrix_to_axis_angle(camera_rotation)
    camera_transl = (
        np.asarray(camera.rotation, dtype=np.float32) @ np.asarray(transl, dtype=np.float32)
    ) + np.asarray(camera.translation, dtype=np.float32)
    return (
        np.ascontiguousarray(camera_global_orient.astype(np.float32)),
        np.ascontiguousarray(camera_transl.astype(np.float32)),
    )


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Convert an axis-angle rotation vector into a 3x3 rotation matrix."""
    rotation_matrix, _ = cv2.Rodrigues(np.asarray(axis_angle, dtype=np.float32).reshape(3, 1))
    return rotation_matrix.astype(np.float32, copy=False)


def matrix_to_axis_angle(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix into an axis-angle rotation vector."""
    axis_angle, _ = cv2.Rodrigues(np.asarray(rotation_matrix, dtype=np.float32))
    return axis_angle.reshape(3).astype(np.float32, copy=False)
