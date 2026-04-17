"""Camera-space test metrics built from corrected input-view camera placement."""

from __future__ import annotations

import numpy as np
import torch

from mvhpe3d.utils import rotation_6d_to_axis_angle
from mvhpe3d.visualization.camera_correction import correct_camera_global_orient_using_torso

from .smpl_keypoints import batch_mpjpe, batch_pa_mpjpe, root_center_joints


def compute_input_corrected_camera_joint_metrics(
    *,
    build_smpl_joints,
    pred_body_pose: torch.Tensor,
    pred_betas: torch.Tensor,
    target_body_pose: torch.Tensor,
    target_betas: torch.Tensor,
    input_views_pose_6d: torch.Tensor,
    input_views_betas: torch.Tensor,
    input_camera_global_orient: torch.Tensor,
    input_transl: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute per-view camera-space metrics using torso-corrected input cameras."""
    batch_size, num_views, num_joints, _ = input_views_pose_6d.shape
    device = pred_body_pose.device
    dtype = pred_body_pose.dtype

    zero_root_batch = torch.zeros((batch_size, 3), dtype=dtype, device=device)
    zero_transl_batch = torch.zeros_like(zero_root_batch)
    zero_root_views = torch.zeros((batch_size * num_views, 3), dtype=dtype, device=device)
    zero_transl_views = torch.zeros_like(zero_root_views)

    pred_canonical_joints = root_center_joints(
        build_smpl_joints(
            body_pose=pred_body_pose,
            betas=pred_betas,
            global_orient=zero_root_batch,
            transl=zero_transl_batch,
        )
    )
    input_body_pose = rotation_6d_to_axis_angle(
        input_views_pose_6d.reshape(batch_size * num_views, num_joints, 6)
    ).reshape(batch_size * num_views, -1)
    input_canonical_joints = root_center_joints(
        build_smpl_joints(
            body_pose=input_body_pose,
            betas=input_views_betas.reshape(batch_size * num_views, -1),
            global_orient=zero_root_views,
            transl=zero_transl_views,
        )
    ).reshape(batch_size, num_views, -1, 3)

    corrected_global_orient_np = np.zeros((batch_size, num_views, 3), dtype=np.float32)
    input_canonical_joints_np = input_canonical_joints.detach().cpu().numpy()
    pred_canonical_joints_np = pred_canonical_joints.detach().cpu().numpy()
    input_camera_global_orient_np = input_camera_global_orient.detach().cpu().numpy()
    for batch_index in range(batch_size):
        for view_index in range(num_views):
            corrected_global_orient_np[batch_index, view_index] = correct_camera_global_orient_using_torso(
                input_canonical_joints=input_canonical_joints_np[batch_index, view_index],
                pred_canonical_joints=pred_canonical_joints_np[batch_index],
                input_camera_global_orient=input_camera_global_orient_np[batch_index, view_index],
            )
    corrected_global_orient = torch.from_numpy(corrected_global_orient_np).to(device=device, dtype=dtype)

    repeated_pred_body_pose = (
        pred_body_pose[:, None, :]
        .expand(batch_size, num_views, pred_body_pose.shape[-1])
        .reshape(batch_size * num_views, -1)
    )
    repeated_pred_betas = (
        pred_betas[:, None, :]
        .expand(batch_size, num_views, pred_betas.shape[-1])
        .reshape(batch_size * num_views, -1)
    )
    repeated_target_body_pose = (
        target_body_pose[:, None, :]
        .expand(batch_size, num_views, target_body_pose.shape[-1])
        .reshape(batch_size * num_views, -1)
    )
    repeated_target_betas = (
        target_betas[:, None, :]
        .expand(batch_size, num_views, target_betas.shape[-1])
        .reshape(batch_size * num_views, -1)
    )

    flat_corrected_global_orient = corrected_global_orient.reshape(batch_size * num_views, 3)
    flat_input_transl = input_transl.reshape(batch_size * num_views, 3).to(device=device, dtype=dtype)

    pred_joints = build_smpl_joints(
        body_pose=repeated_pred_body_pose,
        betas=repeated_pred_betas,
        global_orient=flat_corrected_global_orient,
        transl=flat_input_transl,
    )
    target_joints = build_smpl_joints(
        body_pose=repeated_target_body_pose,
        betas=repeated_target_betas,
        global_orient=flat_corrected_global_orient,
        transl=flat_input_transl,
    )
    return {
        "mpjpe": batch_mpjpe(pred_joints, target_joints),
        "pa_mpjpe": batch_pa_mpjpe(pred_joints, target_joints),
    }
