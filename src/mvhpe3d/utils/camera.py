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


def resolve_panoptic_camera_json_path(cameras_dir: str | Path, *, sequence_id: str) -> Path:
    """Resolve a Panoptic/Kinoptic cropped camera JSON path.

    Supports either the native dataset root layout:
    ``<root>/<sequence>/meta/cameras_kinect_cropped.json``
    or a pre-exported camera directory containing sequence-level JSON files.
    """
    cameras_path = Path(cameras_dir).resolve()
    candidates = [
        cameras_path / sequence_id / "meta" / "cameras_kinect_cropped.json",
        cameras_path / sequence_id / "cameras_kinect_cropped.json",
        cameras_path / "meta" / "cameras_kinect_cropped.json",
        cameras_path / "cameras_kinect_cropped.json",
        cameras_path / f"{sequence_id}_cameras_kinect_cropped.json",
        cameras_path / f"{sequence_id}_cameras.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Panoptic camera JSON does not exist. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def panoptic_camera_id_to_camera_key(camera_id: str) -> str:
    """Map manifest camera IDs such as ``kinect_001`` to Panoptic keys."""
    if not camera_id.startswith("kinect_"):
        raise KeyError(f"Unsupported Panoptic camera_id '{camera_id}'")
    suffix = camera_id.split("_", maxsplit=1)[1]
    return f"kinect_{int(suffix)}"


def load_panoptic_camera_parameters(
    cameras_dir: str | Path,
    *,
    sequence_id: str,
    camera_id: str,
) -> CameraParameters:
    """Load one cropped Panoptic/Kinoptic camera calibration entry."""
    camera_json_path = resolve_panoptic_camera_json_path(
        cameras_dir,
        sequence_id=sequence_id,
    )
    payload = json.loads(camera_json_path.read_text(encoding="utf-8"))

    camera_key = panoptic_camera_id_to_camera_key(camera_id)
    if camera_key not in payload and camera_id in payload:
        camera_key = camera_id
    if camera_key not in payload:
        raise KeyError(f"Camera key '{camera_key}' was not found in {camera_json_path}")

    camera_payload = payload[camera_key]
    if "K_color" in camera_payload:
        intrinsics = np.asarray(camera_payload["K_color"], dtype=np.float32)
    else:
        intrinsics = np.asarray(camera_payload["K"], dtype=np.float32)

    if "extrinsic_world_to_color" in camera_payload:
        extrinsic = np.asarray(camera_payload["extrinsic_world_to_color"], dtype=np.float32)
        rotation = extrinsic[:3, :3]
        translation = extrinsic[:3, 3]
        if str(camera_payload.get("extrinsic_world_to_color_unit", "")).lower() == "cm":
            translation = translation * 0.01
    else:
        rotation = np.asarray(camera_payload["R"], dtype=np.float32)
        translation = np.asarray(camera_payload["T"], dtype=np.float32)

    return CameraParameters(
        intrinsics=np.ascontiguousarray(intrinsics.astype(np.float32, copy=False)),
        rotation=np.ascontiguousarray(rotation.astype(np.float32, copy=False)),
        translation=np.ascontiguousarray(translation.astype(np.float32, copy=False)),
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
