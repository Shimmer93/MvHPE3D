"""Keypoint metrics computed from SMPL joints."""

from __future__ import annotations

import torch

SMPL_EVAL_NUM_JOINTS = 24


def root_center_joints(joints: torch.Tensor) -> torch.Tensor:
    """Pelvis-center batched joints using joint 0 as the root."""
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"Expected joints with shape [B, J, 3], got {tuple(joints.shape)}")
    return joints - joints[:, :1, :]


def batch_mpjpe(pred_joints: torch.Tensor, gt_joints: torch.Tensor) -> torch.Tensor:
    """Return mean per-joint position error averaged over batch and joints."""
    _validate_joint_pair(pred_joints, gt_joints)
    return torch.linalg.norm(pred_joints - gt_joints, dim=-1).mean()


def batch_pa_mpjpe(pred_joints: torch.Tensor, gt_joints: torch.Tensor) -> torch.Tensor:
    """Return Procrustes-aligned MPJPE averaged over batch and joints."""
    _validate_joint_pair(pred_joints, gt_joints)
    aligned_pred = _batch_similarity_transform(pred_joints, gt_joints)
    return torch.linalg.norm(aligned_pred - gt_joints, dim=-1).mean()


def _validate_joint_pair(pred_joints: torch.Tensor, gt_joints: torch.Tensor) -> None:
    if pred_joints.shape != gt_joints.shape:
        raise ValueError(
            f"Expected matching joint shapes, got {tuple(pred_joints.shape)} and "
            f"{tuple(gt_joints.shape)}"
        )
    if pred_joints.ndim != 3 or pred_joints.shape[-1] != 3:
        raise ValueError(
            f"Expected joint tensors with shape [B, J, 3], got {tuple(pred_joints.shape)}"
        )


def _batch_similarity_transform(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Align source joints to target joints with a batched similarity transform."""
    mu_source = source.mean(dim=1, keepdim=True)
    mu_target = target.mean(dim=1, keepdim=True)
    source_centered = source - mu_source
    target_centered = target - mu_target

    covariance = source_centered.transpose(1, 2) @ target_centered
    u, singular_values, vh = torch.linalg.svd(covariance)
    v = vh.transpose(1, 2)

    sign_correction = torch.ones(
        (source.shape[0], 3),
        dtype=source.dtype,
        device=source.device,
    )
    det = torch.det(v @ u.transpose(1, 2))
    sign_correction[:, -1] = torch.where(det < 0, -1.0, 1.0).to(source.dtype)

    z = torch.diag_embed(sign_correction)
    rotation = u @ z @ v.transpose(1, 2)

    source_variance = (source_centered**2).sum(dim=(1, 2)).clamp_min(1e-8)
    trace = (singular_values * sign_correction).sum(dim=1)
    scale = (trace / source_variance).view(-1, 1, 1)
    translation = mu_target - scale * (mu_source @ rotation)
    return scale * (source @ rotation) + translation
