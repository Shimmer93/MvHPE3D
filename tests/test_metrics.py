from __future__ import annotations

import numpy as np
import torch

from mvhpe3d.metrics import batch_mpjpe, batch_pa_mpjpe, root_center_joints
from mvhpe3d.utils import axis_angle_to_matrix
from mvhpe3d.visualization import correct_camera_global_orient_using_torso


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


def test_correct_camera_global_orient_using_torso_recovers_yaw_rotation() -> None:
    input_joints = np.zeros((24, 3), dtype=np.float32)
    input_joints[0] = [0.0, 0.0, 0.0]
    input_joints[1] = [-1.0, 0.0, 0.0]
    input_joints[2] = [1.0, 0.0, 0.0]
    input_joints[12] = [0.0, 1.0, 0.0]
    input_joints[16] = [-1.0, 1.0, 0.0]
    input_joints[17] = [1.0, 1.0, 0.0]

    yaw_rotation = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    pred_joints = input_joints @ yaw_rotation.T

    corrected = correct_camera_global_orient_using_torso(
        input_canonical_joints=input_joints,
        pred_canonical_joints=pred_joints,
        input_camera_global_orient=np.zeros(3, dtype=np.float32),
    )

    corrected_matrix = axis_angle_to_matrix(torch.from_numpy(corrected).view(1, 3)).squeeze(0).cpu().numpy()
    assert np.allclose(corrected_matrix, yaw_rotation.T, atol=1e-4)
