"""Canonicalization helpers for Stage 1 targets.

Stage 1 predicts ``mhr_model_params`` (204-dim) and ``shape_params`` (45-dim).
The target space is pelvis-centered with the root rotation removed.

At this stage of the repository the implementation is intentionally narrow:
- ``mhr_model_params`` is passed through unchanged
- ``shape_params`` is passed through unchanged

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
    mhr_model_params: Any,
    shape_params: Any,
) -> dict[str, np.ndarray]:
    """Build the canonical Stage 1 target representation.

    Args:
        mhr_model_params: MHR model parameters (204-dim).
        shape_params: MHR shape parameters (45-dim).

    Returns:
        Dictionary with canonical target tensors.
    """
    canonical_mhr_params = _to_float32_array(mhr_model_params, expected_last_dim=204)
    canonical_shape_params = _to_float32_array(shape_params, expected_last_dim=45)

    return {
        "target_mhr_params": canonical_mhr_params,
        "target_shape_params": canonical_shape_params,
    }
