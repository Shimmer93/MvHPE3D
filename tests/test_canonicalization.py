from __future__ import annotations

import numpy as np
import pytest

from mvhpe3d.data.canonicalization import canonicalize_stage1_target


def test_canonicalize_stage1_target_returns_expected_keys() -> None:
    result = canonicalize_stage1_target(
        smpl_betas=np.ones(10, dtype=np.float32),
        smpl_body_pose=np.ones(69, dtype=np.float32),
        smpl_global_orient=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        smpl_transl=np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )

    assert result["target_betas"].shape == (10,)
    assert result["target_body_pose"].shape == (69,)
    assert np.allclose(result["canonical_root_orient"], np.zeros(3, dtype=np.float32))
    assert np.allclose(result["canonical_transl"], np.zeros(3, dtype=np.float32))
    assert result["removed_global_orient"].shape == (3,)
    assert result["removed_transl"].shape == (3,)


def test_canonicalize_stage1_target_rejects_invalid_body_pose_dim() -> None:
    with pytest.raises(ValueError, match="trailing dimension 69"):
        canonicalize_stage1_target(
            smpl_betas=np.ones(10, dtype=np.float32),
            smpl_body_pose=np.ones(63, dtype=np.float32),
        )
