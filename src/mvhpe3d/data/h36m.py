"""Human3.6M helpers following the HeatFormer multiview protocol."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


H36M_HEATFORMER_CAMERA_IDS: tuple[str, ...] = ("1", "2", "3", "4")
H36M_HEATFORMER_TRAIN_SUBJECTS: tuple[str, ...] = ("1", "5", "6", "7", "8")
H36M_HEATFORMER_VAL_SUBJECTS: tuple[str, ...] = ("9", "11")
H36M_HEATFORMER_TRAIN_SAMPLING = 10
H36M_HEATFORMER_VAL_SAMPLING = 10
H36M_HEATFORMER_JOINT_COUNT = 17
H36M_HEATFORMER_ROOT_INDEX = 0


def h36m_camera_id_to_index(camera_id: str | int) -> int:
    """Return the one-based HeatFormer camera id for ``1``/``ca_01`` style ids."""
    if isinstance(camera_id, int):
        return camera_id
    text = str(camera_id)
    if text.startswith("camera_"):
        text = text.split("_", maxsplit=1)[1]
    if text.startswith("ca_"):
        text = text.split("_", maxsplit=1)[1]
    return int(text)


def normalize_h36m_subject(subject_id: str | int) -> str:
    text = str(subject_id)
    if text.startswith("S"):
        text = text[1:]
    return str(int(text))


def h36m_protocol_extra_data_dir(root: str | Path) -> Path:
    """Resolve a HeatFormer ``extra_data`` directory from a protocol root."""
    root_path = Path(root).resolve()
    candidates = (
        root_path / "extra_data",
        root_path / "preprocessed_data_z" / "extra_data",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find HeatFormer Human3.6M extra_data under "
        f"{root_path}. Run scripts/prepare_h36m_stage2.py first."
    )


@lru_cache(maxsize=16)
def load_h36m_subject_joint_3d(extra_data_dir: str, subject_id: str) -> dict[str, Any]:
    path = (
        Path(extra_data_dir).resolve()
        / f"Human36M_subject{normalize_h36m_subject(subject_id)}_joint_3d.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"HeatFormer Human3.6M joint file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=16)
def load_h36m_subject_camera(extra_data_dir: str, subject_id: str) -> dict[str, Any]:
    path = (
        Path(extra_data_dir).resolve()
        / f"Human36M_subject{normalize_h36m_subject(subject_id)}_camera.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"HeatFormer Human3.6M camera file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_h36m_joint_target(
    root: str | Path,
    *,
    subject_id: str | int,
    action_id: str | int,
    subaction_id: str | int,
    frame_index: str | int,
    camera_ids: tuple[str, ...] | list[str] = H36M_HEATFORMER_CAMERA_IDS,
) -> dict[str, np.ndarray]:
    """Load HeatFormer-order Human3.6M joints in world meters.

    HeatFormer's H36M eval path reads ``Human36M_subject*_joint_3d.json`` and
    converts millimeters to meters with ``flip=False``. This helper mirrors
    that target side and also returns per-view camera-space joints computed
    from HeatFormer's camera JSON.
    """
    extra_data_dir = h36m_protocol_extra_data_dir(root)
    subject = normalize_h36m_subject(subject_id)
    action = str(int(action_id))
    subaction = str(int(subaction_id))
    frame = str(int(frame_index))

    subject_joints = load_h36m_subject_joint_3d(str(extra_data_dir), subject)
    try:
        joints_world = np.asarray(
            subject_joints[action][subaction][frame],
            dtype=np.float32,
        ) * 0.001
    except KeyError as exc:
        raise KeyError(
            "Missing Human3.6M HeatFormer target joints for "
            f"subject={subject}, action={action}, subaction={subaction}, frame={frame}"
        ) from exc

    camera_joints = []
    for camera_id in camera_ids:
        rotation, translation, _ = load_h36m_camera(
            root,
            subject_id=subject,
            camera_id=str(camera_id),
        )
        camera_joints.append(
            np.einsum("ij,kj->ki", rotation, joints_world) + translation[None, :]
        )
    camera_joints_array = np.stack(camera_joints, axis=0).astype(np.float32, copy=False)
    return {
        "joints": np.ascontiguousarray(joints_world.astype(np.float32, copy=False)),
        "confidence": np.ones((H36M_HEATFORMER_JOINT_COUNT,), dtype=np.float32),
        "camera_joints": np.ascontiguousarray(camera_joints_array),
        "camera_confidence": np.ones(
            (len(camera_ids), H36M_HEATFORMER_JOINT_COUNT),
            dtype=np.float32,
        ),
        "smpl_indices": np.arange(H36M_HEATFORMER_JOINT_COUNT, dtype=np.int64),
        "root_index": np.asarray(H36M_HEATFORMER_ROOT_INDEX, dtype=np.int64),
    }


def load_h36m_joint_2d_target(
    root: str | Path,
    *,
    subject_id: str | int,
    action_id: str | int,
    subaction_id: str | int,
    frame_index: str | int,
    camera_ids: tuple[str, ...] | list[str] = H36M_HEATFORMER_CAMERA_IDS,
) -> dict[str, np.ndarray]:
    joint_target = load_h36m_joint_target(
        root,
        subject_id=subject_id,
        action_id=action_id,
        subaction_id=subaction_id,
        frame_index=frame_index,
        camera_ids=camera_ids,
    )
    camera_joints = joint_target["camera_joints"]
    joints_2d = []
    confidence = []
    for view_index, camera_id in enumerate(camera_ids):
        _, _, intrinsics = load_h36m_camera(
            root,
            subject_id=subject_id,
            camera_id=str(camera_id),
        )
        view_camera_joints = camera_joints[view_index]
        depth = view_camera_joints[:, 2]
        projected = (intrinsics @ (view_camera_joints / depth[:, None]).T).T[:, :2]
        valid = np.isfinite(projected).all(axis=-1) & np.isfinite(depth) & (depth > 0.0)
        joints_2d.append(projected.astype(np.float32, copy=False))
        confidence.append(valid.astype(np.float32))
    return {
        "joints_2d": np.ascontiguousarray(np.stack(joints_2d, axis=0)),
        "confidence": np.ascontiguousarray(np.stack(confidence, axis=0)),
    }


def load_h36m_camera(
    root: str | Path,
    *,
    subject_id: str | int,
    camera_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load HeatFormer Human3.6M camera extrinsics/intrinsics.

    Rotation is world-to-camera. Translation is returned in meters to match the
    target joints. Intrinsics are in pixels.
    """
    extra_data_dir = h36m_protocol_extra_data_dir(root)
    subject = normalize_h36m_subject(subject_id)
    camera_key = str(h36m_camera_id_to_index(camera_id))
    subject_cameras = load_h36m_subject_camera(str(extra_data_dir), subject)
    try:
        payload = subject_cameras[camera_key]
    except KeyError as exc:
        raise KeyError(
            f"Missing Human3.6M camera {camera_key} for subject {subject}"
        ) from exc
    rotation = np.asarray(payload["R"], dtype=np.float32)
    translation = np.asarray(payload["t"], dtype=np.float32).reshape(3) * 0.001
    focal = np.asarray(payload["f"], dtype=np.float32).reshape(2)
    center = np.asarray(payload["c"], dtype=np.float32).reshape(2)
    intrinsics = np.array(
        [
            [focal[0], 0.0, center[0]],
            [0.0, focal[1], center[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return (
        np.ascontiguousarray(rotation),
        np.ascontiguousarray(translation),
        np.ascontiguousarray(intrinsics),
    )


def is_h36m_damaged_sequence(
    *,
    subject_id: str | int,
    action_id: str | int,
    subaction_id: str | int,
) -> bool:
    """Return HeatFormer's damaged-sequence filter for Human3.6M."""
    subject = int(normalize_h36m_subject(subject_id))
    action = int(action_id)
    subaction = int(subaction_id)
    if subject == 11 and action == 2 and subaction == 2:
        return True
    if subject == 9:
        return (
            (action == 5 and subaction == 2)
            or (action == 10 and subaction == 2)
            or (action == 13 and subaction == 1)
        )
    return False
