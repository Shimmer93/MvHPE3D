from __future__ import annotations

import numpy as np
import pytest

from mvhpe3d.data.canonicalization import canonicalize_stage1_target


def test_canonicalize_stage1_target_returns_expected_keys() -> None:
    result = canonicalize_stage1_target(
        smpl_body_pose=np.ones(69, dtype=np.float32),
        smpl_betas=np.ones(10, dtype=np.float32),
    )

    assert result["target_body_pose"].shape == (69,)
    assert result["target_betas"].shape == (10,)
    assert np.allclose(result["target_body_pose"], np.ones(69, dtype=np.float32))
    assert np.allclose(result["target_betas"], np.ones(10, dtype=np.float32))


def test_canonicalize_stage1_target_rejects_invalid_betas_dim() -> None:
    with pytest.raises(ValueError, match="trailing dimension 10"):
        canonicalize_stage1_target(
            smpl_body_pose=np.ones(69, dtype=np.float32),
            smpl_betas=np.ones(63, dtype=np.float32),
        )
