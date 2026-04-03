from __future__ import annotations

import torch

from mvhpe3d.metrics import batch_mpjpe, batch_pa_mpjpe, root_center_joints


def test_root_center_joints_zeros_pelvis_joint() -> None:
    joints = torch.tensor(
        [[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]],
        dtype=torch.float32,
    )

    centered = root_center_joints(joints)

    assert torch.allclose(centered[:, 0], torch.zeros(1, 3))
    assert torch.allclose(centered[:, 1], torch.tensor([[1.0, 2.0, 3.0]]))


def test_batch_mpjpe_matches_expected_distance() -> None:
    pred = torch.tensor(
        [[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    gt = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    mpjpe = batch_mpjpe(pred, gt)

    assert torch.isclose(mpjpe, torch.tensor(1.0))


def test_batch_pa_mpjpe_is_zero_for_similarity_transform_equivalent_sets() -> None:
    gt = torch.tensor(
        [[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]],
        dtype=torch.float32,
    )
    scale = 2.5
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    translation = torch.tensor([[[5.0, -3.0, 7.0]]], dtype=torch.float32)
    pred = scale * (gt @ rotation) + translation

    pa_mpjpe = batch_pa_mpjpe(pred, gt)

    assert pa_mpjpe < 1e-5
