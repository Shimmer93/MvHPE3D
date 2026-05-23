"""Torch rotation conversion helpers used by Stage 2 parameter-space models."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _skew_symmetric(vectors: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(vectors[..., 0])
    x_coord = vectors[..., 0]
    y_coord = vectors[..., 1]
    z_coord = vectors[..., 2]
    return torch.stack(
        (
            torch.stack((zeros, -z_coord, y_coord), dim=-1),
            torch.stack((z_coord, zeros, -x_coord), dim=-1),
            torch.stack((-y_coord, x_coord, zeros), dim=-1),
        ),
        dim=-2,
    )


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle rotations with shape ``[..., 3]`` to matrices ``[..., 3, 3]``."""
    if axis_angle.shape[-1] != 3:
        raise ValueError(
            f"Expected axis_angle trailing dimension 3, got {tuple(axis_angle.shape)}"
        )
    return torch.matrix_exp(_skew_symmetric(axis_angle))


def matrix_to_rotation_6d(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices ``[..., 3, 3]`` to 6D rotation representation ``[..., 6]``."""
    if rotation_matrix.shape[-2:] != (3, 3):
        raise ValueError(
            "Expected rotation_matrix trailing shape (3, 3), "
            f"got {tuple(rotation_matrix.shape)}"
        )
    first_column = rotation_matrix[..., :, 0]
    second_column = rotation_matrix[..., :, 1]
    return torch.cat((first_column, second_column), dim=-1)


def rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotations with shape ``[..., 6]`` to rotation matrices ``[..., 3, 3]``."""
    if rotation_6d.shape[-1] != 6:
        raise ValueError(
            f"Expected rotation_6d trailing dimension 6, got {tuple(rotation_6d.shape)}"
        )
    original_dtype = rotation_6d.dtype
    rotation_6d = rotation_6d.float()
    first_raw = rotation_6d[..., 0:3]
    second_raw = rotation_6d[..., 3:6]

    first_basis = F.normalize(first_raw, dim=-1, eps=1e-6)
    second_basis = second_raw - (first_basis * second_raw).sum(dim=-1, keepdim=True) * first_basis
    second_basis = F.normalize(second_basis, dim=-1, eps=1e-6)
    third_basis = torch.cross(first_basis, second_basis, dim=-1)

    return torch.stack((first_basis, second_basis, third_basis), dim=-1).to(original_dtype)


def matrix_to_axis_angle(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices ``[..., 3, 3]`` to axis-angle vectors ``[..., 3]``."""
    if rotation_matrix.shape[-2:] != (3, 3):
        raise ValueError(
            "Expected rotation_matrix trailing shape (3, 3), "
            f"got {tuple(rotation_matrix.shape)}"
        )
    original_dtype = rotation_matrix.dtype
    rotation_matrix = rotation_matrix.float()
    quaternion = matrix_to_quaternion(rotation_matrix)
    axis_angle = quaternion_to_axis_angle(quaternion)
    return axis_angle.to(original_dtype)


def matrix_to_quaternion(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices ``[..., 3, 3]`` to normalized ``wxyz`` quaternions."""
    if rotation_matrix.shape[-2:] != (3, 3):
        raise ValueError(
            "Expected rotation_matrix trailing shape (3, 3), "
            f"got {tuple(rotation_matrix.shape)}"
        )
    matrix = rotation_matrix.float()
    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]

    q_abs = _sqrt_positive_part(
        torch.stack(
            (
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ),
            dim=-1,
        )
    )
    quat_by_component = torch.stack(
        (
            torch.stack((q_abs[..., 0].square(), m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[..., 1].square(), m01 + m10, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m01 + m10, q_abs[..., 2].square(), m12 + m21), dim=-1),
            torch.stack((m10 - m01, m02 + m20, m12 + m21, q_abs[..., 3].square()), dim=-1),
        ),
        dim=-2,
    )
    candidates = quat_by_component / (2.0 * q_abs[..., None].clamp_min(0.1))
    selector = F.one_hot(q_abs.argmax(dim=-1), num_classes=4).to(dtype=torch.bool)
    quaternion = candidates[selector, :].reshape(*q_abs.shape[:-1], 4)
    quaternion = F.normalize(quaternion, dim=-1, eps=1e-8)
    return torch.where(quaternion[..., :1] < 0.0, -quaternion, quaternion)


def quaternion_to_axis_angle(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert normalized ``wxyz`` quaternions to axis-angle vectors."""
    if quaternion.shape[-1] != 4:
        raise ValueError(
            f"Expected quaternion trailing dimension 4, got {tuple(quaternion.shape)}"
        )
    quaternion = F.normalize(quaternion.float(), dim=-1, eps=1e-8)
    quaternion = torch.where(quaternion[..., :1] < 0.0, -quaternion, quaternion)
    vector = quaternion[..., 1:]
    vector_norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    half_angle = torch.atan2(vector_norm, quaternion[..., :1])
    angle = 2.0 * half_angle
    scale = torch.where(
        vector_norm > 1e-6,
        angle / vector_norm.clamp_min(1e-8),
        2.0 + angle.square() / 12.0,
    )
    return vector * scale


def _sqrt_positive_part(value: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(value)
    positive = value > 0.0
    result[positive] = torch.sqrt(value[positive])
    return result


def axis_angle_to_rotation_6d(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle rotations ``[..., 3]`` to 6D rotations ``[..., 6]``."""
    return matrix_to_rotation_6d(axis_angle_to_matrix(axis_angle))


def rotation_6d_to_axis_angle(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotations ``[..., 6]`` to axis-angle vectors ``[..., 3]``."""
    return matrix_to_axis_angle(rotation_6d_to_matrix(rotation_6d))
