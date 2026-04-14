"""Canonicalization helpers for Stage 1 GT SMPL targets.

Stage 1 predicts the canonical SMPL body:
- ``smpl_body_pose`` (23 joints, axis-angle, 69-dim)
- ``smpl_betas`` (10-dim)

The target space is pelvis-centered with the SMPL root rotation removed.
For the current Stage 1 baseline, this means:
- drop ``smpl_global_orient``
- drop ``smpl_transl``
- supervise only ``smpl_body_pose`` and ``smpl_betas``
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from mvhpe3d.utils import axis_angle_to_rotation_6d


def _to_float32_array(value: Any, *, expected_last_dim: int | None = None) -> np.ndarray:
    """Convert a value to a contiguous float32 numpy array and validate shape."""
    array = np.asarray(value, dtype=np.float32)
    if expected_last_dim is not None:
        if array.ndim == 0 or array.shape[-1] != expected_last_dim:
            raise ValueError(
                f"Expected trailing dimension {expected_last_dim}, got shape {array.shape}"
            )
    return np.ascontiguousarray(array)


def canonicalize_stage1_target(
    *,
    smpl_body_pose: Any,
    smpl_betas: Any,
) -> dict[str, np.ndarray]:
    """Build the canonical Stage 1 target representation.

    Args:
        smpl_body_pose: SMPL body pose without the root joint (69-dim).
        smpl_betas: SMPL shape coefficients (10-dim).

    Returns:
        Dictionary with canonical target tensors.
    """
    canonical_body_pose = _to_float32_array(smpl_body_pose, expected_last_dim=69)
    canonical_betas = _to_float32_array(smpl_betas, expected_last_dim=10)

    return {
        "target_body_pose": canonical_body_pose,
        "target_betas": canonical_betas,
    }


def canonicalize_stage2_target(
    *,
    smpl_body_pose: Any,
    smpl_betas: Any,
) -> dict[str, np.ndarray]:
    """Build the canonical Stage 2 target representation.

    Stage 2 keeps the same canonical target as Stage 1 but also exposes the
    pose in 6D rotation form for parameter-space fusion and refinement.
    """
    canonical = canonicalize_stage1_target(
        smpl_body_pose=smpl_body_pose,
        smpl_betas=smpl_betas,
    )
    body_pose_axis_angle = torch.from_numpy(canonical["target_body_pose"].reshape(-1, 3))
    body_pose_6d = axis_angle_to_rotation_6d(body_pose_axis_angle).reshape(-1, 6)
    canonical["target_body_pose_6d"] = np.ascontiguousarray(
        body_pose_6d.cpu().numpy().astype(np.float32, copy=False)
    )
    return canonical
