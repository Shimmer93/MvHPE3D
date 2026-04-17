"""Helpers for correcting input-camera placement using canonical torso alignment."""

from __future__ import annotations

import numpy as np
import torch

from mvhpe3d.utils import axis_angle_to_matrix, matrix_to_axis_angle

PELVIS_INDEX = 0
LEFT_HIP_INDEX = 1
RIGHT_HIP_INDEX = 2
NECK_INDEX = 12
LEFT_SHOULDER_INDEX = 16
RIGHT_SHOULDER_INDEX = 17


def correct_camera_global_orient_using_torso(
    *,
    input_canonical_joints: np.ndarray,
    pred_canonical_joints: np.ndarray,
    input_camera_global_orient: np.ndarray,
) -> np.ndarray:
    """Estimate a torso-frame rotation correction and compose it with input camera pose.

    The returned axis-angle orientation is suitable for placing the predicted
    canonical body into the same camera frame as the input view, while correcting
    large facing-direction mismatches between the input canonical fit and the
    fused predicted canonical pose.
    """
    input_frame = _estimate_torso_frame(input_canonical_joints)
    pred_frame = _estimate_torso_frame(pred_canonical_joints)
    rotation_delta = _project_to_rotation(pred_frame @ input_frame.T)

    input_rotation = (
        axis_angle_to_matrix(torch.as_tensor(input_camera_global_orient, dtype=torch.float32).view(1, 3))
        .squeeze(0)
        .cpu()
        .numpy()
    )
    corrected_rotation = _project_to_rotation(input_rotation @ rotation_delta.T)
    corrected_axis_angle = (
        matrix_to_axis_angle(torch.as_tensor(corrected_rotation, dtype=torch.float32).view(1, 3, 3))
        .squeeze(0)
        .cpu()
        .numpy()
    )
    return corrected_axis_angle.astype(np.float32, copy=False)


def _estimate_torso_frame(joints: np.ndarray) -> np.ndarray:
    torso_joints = np.asarray(joints, dtype=np.float32)
    pelvis = torso_joints[PELVIS_INDEX]
    neck = torso_joints[NECK_INDEX]
    shoulder_axis = torso_joints[RIGHT_SHOULDER_INDEX] - torso_joints[LEFT_SHOULDER_INDEX]
    hip_axis = torso_joints[RIGHT_HIP_INDEX] - torso_joints[LEFT_HIP_INDEX]
    lateral = shoulder_axis + hip_axis
    if np.linalg.norm(lateral) < 1e-6:
        lateral = hip_axis
    if np.linalg.norm(lateral) < 1e-6:
        lateral = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    lateral = _normalize(lateral)

    up = neck - pelvis
    if np.linalg.norm(up) < 1e-6:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    up = _normalize(up)

    facing = np.cross(lateral, up)
    if np.linalg.norm(facing) < 1e-6:
        facing = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    facing = _normalize(facing)
    up = _normalize(np.cross(facing, lateral))

    frame = np.stack((lateral, up, facing), axis=-1)
    return _project_to_rotation(frame)


def _project_to_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(matrix.astype(np.float32, copy=False))
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    return rotation.astype(np.float32, copy=False)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return vector.astype(np.float32, copy=False)
    return (vector / norm).astype(np.float32, copy=False)
