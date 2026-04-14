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
    first_raw = rotation_6d[..., 0:3]
    second_raw = rotation_6d[..., 3:6]

    first_basis = F.normalize(first_raw, dim=-1)
    second_basis = second_raw - (first_basis * second_raw).sum(dim=-1, keepdim=True) * first_basis
    second_basis = F.normalize(second_basis, dim=-1)
    third_basis = torch.cross(first_basis, second_basis, dim=-1)

    return torch.stack((first_basis, second_basis, third_basis), dim=-1)


def matrix_to_axis_angle(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices ``[..., 3, 3]`` to axis-angle vectors ``[..., 3]``."""
    if rotation_matrix.shape[-2:] != (3, 3):
        raise ValueError(
            "Expected rotation_matrix trailing shape (3, 3), "
            f"got {tuple(rotation_matrix.shape)}"
        )

    trace = rotation_matrix[..., 0, 0] + rotation_matrix[..., 1, 1] + rotation_matrix[..., 2, 2]
    cosine = ((trace - 1.0) * 0.5).clamp(min=-1.0, max=1.0)
    angle = torch.acos(cosine)

    vee = torch.stack(
        (
            rotation_matrix[..., 2, 1] - rotation_matrix[..., 1, 2],
            rotation_matrix[..., 0, 2] - rotation_matrix[..., 2, 0],
            rotation_matrix[..., 1, 0] - rotation_matrix[..., 0, 1],
        ),
        dim=-1,
    )

    sine = torch.sin(angle)
    scale = angle / (2.0 * sine.clamp_min(1e-6))
    axis_angle = vee * scale.unsqueeze(-1)

    small_angle = angle.abs() < 1e-4
    first_order = 0.5 * vee
    return torch.where(small_angle.unsqueeze(-1), first_order, axis_angle)


def axis_angle_to_rotation_6d(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle rotations ``[..., 3]`` to 6D rotations ``[..., 6]``."""
    return matrix_to_rotation_6d(axis_angle_to_matrix(axis_angle))


def rotation_6d_to_axis_angle(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotations ``[..., 6]`` to axis-angle vectors ``[..., 3]``."""
    return matrix_to_axis_angle(rotation_6d_to_matrix(rotation_6d))
