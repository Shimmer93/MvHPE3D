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
