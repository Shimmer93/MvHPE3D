"""Utilities for projecting SMPL meshes into HuMMan RGB images."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class CameraParameters:
    """One calibrated camera used for RGB overlay rendering."""

    intrinsics: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray


def resolve_rgb_image_path(
    rgb_dir: str | Path,
    *,
    sequence_id: str,
    camera_id: str,
    frame_id: str,
) -> Path:
    """Resolve one cropped HuMMan RGB image path from sample identifiers."""
    base_dir = Path(rgb_dir).resolve()
    stem = f"{sequence_id}_{camera_id}_{frame_id}"
    for extension in IMAGE_EXTENSIONS:
        candidate = base_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find RGB image for stem '{stem}' under {base_dir}"
    )


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
        raise KeyError(
            f"Camera key '{camera_key}' was not found in {camera_json_path}"
        )

    camera_payload = payload[camera_key]
    return CameraParameters(
        intrinsics=np.asarray(camera_payload["K"], dtype=np.float32),
        rotation=np.asarray(camera_payload["R"], dtype=np.float32),
        translation=np.asarray(camera_payload["T"], dtype=np.float32),
    )


def project_vertices_world_to_image(
    vertices_world: np.ndarray,
    camera: CameraParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world-space mesh vertices into image coordinates."""
    vertices_world = np.asarray(vertices_world, dtype=np.float32)
    vertices_camera = (camera.rotation @ vertices_world.T).T + camera.translation.reshape(1, 3)
    depths = vertices_camera[:, 2].copy()

    homogeneous = (camera.intrinsics @ vertices_camera.T).T
    projected = homogeneous[:, :2] / np.clip(homogeneous[:, 2:3], 1e-6, None)
    return projected.astype(np.float32, copy=False), depths.astype(np.float32, copy=False)


def render_projected_mesh_mask(
    image_shape: tuple[int, int],
    *,
    vertices_world: np.ndarray,
    faces: np.ndarray,
    camera: CameraParameters,
) -> np.ndarray:
    """Rasterize a coarse binary mesh mask with triangle filling."""
    height, width = image_shape
    projected_vertices, depths = project_vertices_world_to_image(vertices_world, camera)

    mask = np.zeros((height, width), dtype=np.uint8)
    visible_faces = np.asarray(faces, dtype=np.int32)
    face_depths = depths[visible_faces]
    valid_face_mask = np.all(face_depths > 1e-4, axis=1)
    visible_faces = visible_faces[valid_face_mask]
    if visible_faces.size == 0:
        return mask

    mean_depth = face_depths[valid_face_mask].mean(axis=1)
    draw_order = np.argsort(mean_depth)[::-1]

    for face_index in draw_order:
        polygon = projected_vertices[visible_faces[face_index]]
        polygon_int = np.round(polygon).astype(np.int32)
        if cv2.contourArea(polygon_int.astype(np.float32)) <= 0.5:
            continue
        cv2.fillConvexPoly(mask, polygon_int, 255)

    return mask


def overlay_mask_on_image(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int],
    alpha: float,
    edge_color: tuple[int, int, int] = (255, 255, 255),
    edge_thickness: int = 2,
) -> np.ndarray:
    """Blend one binary mesh mask onto a BGR image."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR image with shape [H, W, 3], got {image_bgr.shape}")
    if mask.shape[:2] != image_bgr.shape[:2]:
        raise ValueError(
            f"Mask shape {mask.shape} does not match image shape {image_bgr.shape[:2]}"
        )

    output = image_bgr.astype(np.float32, copy=True)
    mask_float = (mask.astype(np.float32) / 255.0)[..., None]
    color_array = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    output = output * (1.0 - alpha * mask_float) + color_array * (alpha * mask_float)
    output = np.clip(output, 0.0, 255.0).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(output, contours, -1, edge_color, edge_thickness, cv2.LINE_AA)
    return output
