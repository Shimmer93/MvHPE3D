from __future__ import annotations

import numpy as np
import pytest

from mvhpe3d.data.canonicalization import canonicalize_stage1_target


def test_canonicalize_stage1_target_returns_expected_keys() -> None:
    result = canonicalize_stage1_target(
        mhr_model_params=np.ones(204, dtype=np.float32),
        shape_params=np.ones(45, dtype=np.float32),
    )

    assert result["target_mhr_params"].shape == (204,)
    assert result["target_shape_params"].shape == (45,)
    assert np.allclose(result["target_mhr_params"], np.ones(204, dtype=np.float32))
    assert np.allclose(result["target_shape_params"], np.ones(45, dtype=np.float32))


def test_canonicalize_stage1_target_rejects_invalid_shape_params_dim() -> None:
    with pytest.raises(ValueError, match="trailing dimension 45"):
        canonicalize_stage1_target(
            mhr_model_params=np.ones(204, dtype=np.float32),
            shape_params=np.ones(63, dtype=np.float32),
        )
