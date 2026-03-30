"""Canonicalization helpers for Stage 1 targets.

Stage 1 predicts only canonical ``smpl_betas`` and ``smpl_body_pose``.
The target space is pelvis-centered with the SMPL root rotation removed.

At this stage of the repository, the implementation is intentionally narrow:
- ``smpl_betas`` is passed through unchanged
- ``smpl_body_pose`` is treated as already root-relative
- optional root pose / translation inputs are recorded as removed metadata

Once the exact HuMMan-to-canonical transform is finalized, this module should be
the single place where that logic is made explicit and tested.
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
    smpl_betas: Any,
    smpl_body_pose: Any,
    smpl_global_orient: Any | None = None,
    smpl_transl: Any | None = None,
) -> dict[str, np.ndarray]:
    """Build the canonical Stage 1 target representation.

    Args:
        smpl_betas: SMPL shape coefficients.
        smpl_body_pose: SMPL body pose without the root joint.
        smpl_global_orient: Optional original root orientation retained only as
            metadata for later analysis or visualization.
        smpl_transl: Optional original global translation retained only as
            metadata for later analysis or visualization.

    Returns:
        Dictionary with canonical target tensors and removed-root metadata.
    """
    canonical_betas = _to_float32_array(smpl_betas)
    canonical_body_pose = _to_float32_array(smpl_body_pose, expected_last_dim=69)

    result = {
        "target_betas": canonical_betas,
        "target_body_pose": canonical_body_pose,
        # Stage 1 fixes the canonical root to identity rotation and zero
        # translation. The original values are retained for bookkeeping only.
        "canonical_root_orient": np.zeros(3, dtype=np.float32),
        "canonical_transl": np.zeros(3, dtype=np.float32),
    }

    if smpl_global_orient is not None:
        result["removed_global_orient"] = _to_float32_array(
            smpl_global_orient, expected_last_dim=3
        )

    if smpl_transl is not None:
        result["removed_transl"] = _to_float32_array(smpl_transl, expected_last_dim=3)

    return result
