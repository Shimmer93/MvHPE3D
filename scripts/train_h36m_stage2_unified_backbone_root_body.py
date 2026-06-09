#!/usr/bin/env python
"""Jointly finetune Stage 2 backbone and image-conditioned root/body correction."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mvhpe3d.models.stage2.root_body_adapter import (  # noqa: E402
    Stage23RootBodyAdapterConfig,
    Stage23RootBodyAdapterModel,
)
from mvhpe3d.utils import axis_angle_to_matrix, load_experiment_config, matrix_to_axis_angle  # noqa: E402
from optimize_h36m_stage2_reprojection import move_to_device  # noqa: E402
from test import build_data_config, build_datamodule, build_model_config, load_eval_module  # noqa: E402
from train_h36m_stage2_1_root_adapter import (  # noqa: E402
    build_camera_projection,
    camera_joint_loss_and_error,
    projection_loss_and_error,
    run_stage2_forward,
)
from train_h36m_stage2_2_gated_body_adapter import (  # noqa: E402
    MetricAccumulator,
    canonical_preserve_loss,
    do_no_harm_loss_and_rate,
)


def zero_body_outputs(
    stage2_body_pose: torch.Tensor,
    *,
    num_body_joints: int,
) -> dict[str, torch.Tensor]:
    batch_size = stage2_body_pose.shape[0]
    return {
        "pred_body_pose_delta": torch.zeros_like(stage2_body_pose),
        "pred_body_pose_update": torch.zeros_like(stage2_body_pose),
        "pred_body_pose_gate": torch.zeros(
            (batch_size, num_body_joints),
            device=stage2_body_pose.device,
            dtype=stage2_body_pose.dtype,
        ),
    }


def pa_joint_loss_and_error(
    stage2_module,
    pred_camera_joints: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    target = batch["target_camera_joints"].to(device=pred_camera_joints.device, dtype=pred_camera_joints.dtype)
    confidence = batch["target_camera_joint_confidence"].to(device=pred_camera_joints.device, dtype=pred_camera_joints.dtype)
    root_index = int(batch["target_joint_root_index"].reshape(-1)[0].item())
    pred_root = pred_camera_joints - pred_camera_joints[:, :, root_index : root_index + 1, :]
    target_root = target - target[:, :, root_index : root_index + 1, :]
    valid = confidence > 0.05
    batch_size, num_views, joint_count, _ = pred_root.shape
    flat_pred = pred_root.reshape(batch_size * num_views, joint_count, 3)
    flat_target = target_root.reshape(batch_size * num_views, joint_count, 3)
    valid_float = valid.reshape(batch_size * num_views, joint_count).to(dtype=pred_camera_joints.dtype)
    valid_weight = valid_float * (valid_float > 0.05).to(valid_float.dtype)
    aligned = stage2_module._weighted_similarity_align(
        flat_pred.float(),
        flat_target.float(),
        valid_weight.float(),
    ).to(dtype=pred_camera_joints.dtype)
    error_m = torch.linalg.norm(aligned - flat_target, dim=-1)
    valid_count = valid_float.sum().clamp_min(1.0)
    loss = (error_m * valid_float).sum() / valid_count
    return loss, loss.detach() * 1000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage2-checkpoint-path", required=True)
    parser.add_argument(
        "--resume-unified-checkpoint",
        default=None,
        help="Optional checkpoint from this unified script to continue training.",
    )
    parser.add_argument(
        "--resume-allow-head-expansion",
        action="store_true",
        help=(
            "Allow loading a unified adapter checkpoint when a final linear head "
            "has been expanded; matching leading rows are copied and new rows keep "
            "their initialization."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smpl-model-path", default="data/weights/SMPL_NEUTRAL.pkl")
    parser.add_argument("--input-smpl-cache-dir", default="data/h36m/sam3dbody_fitted_smpl")
    parser.add_argument(
        "--image-measurement-cache-dir",
        default="data/h36m/image_measurements_keypointrcnn_heatmap_grid5_uv_h36m_lrfix_direct12_sam3mask_compact_gated",
    )
    parser.add_argument("--rgb-feature-cache-dir", default=None)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--gt-smpl-dir", default=None)
    parser.add_argument("--cameras-dir", default=None)
    parser.add_argument("--split-config-path", default=None)
    parser.add_argument("--split-name", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--body-start-epoch", type=int, default=0)
    parser.add_argument(
        "--body-extra-start-epoch",
        type=int,
        default=0,
        help="Delay the optional extra body branch until this epoch.",
    )
    parser.add_argument("--train-max-batches", type=int, default=None)
    parser.add_argument("--val-max-batches", type=int, default=None)
    parser.add_argument("--log-every-n-batches", type=int, default=100)
    parser.add_argument("--freeze-stage2", action="store_true")
    parser.add_argument("--stage2-start-epoch", type=int, default=0)
    parser.add_argument("--stage2-lr", type=float, default=1.0e-6)
    parser.add_argument("--root-lr", type=float, default=3.0e-5)
    parser.add_argument("--body-lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--root-hidden-dim", type=int, default=256)
    parser.add_argument("--body-hidden-dim", type=int, default=512)
    parser.add_argument("--root-num-layers", type=int, default=3)
    parser.add_argument("--body-num-layers", type=int, default=3)
    parser.add_argument("--body-image-feature-dim", type=int, default=250)
    parser.add_argument("--body-mask-feature-dim", type=int, default=20)
    parser.add_argument("--body-evidence-token-dim", type=int, default=32)
    parser.add_argument("--body-evidence-hidden-dim", type=int, default=128)
    parser.add_argument("--body-evidence-layers", type=int, default=2)
    parser.add_argument("--use-body-image-joint-feature", action="store_true")
    parser.add_argument("--use-body-image-mask-feature", action="store_true")
    parser.add_argument("--use-body-evidence-gate", action="store_true")
    parser.add_argument(
        "--disable-root-input-global-orient",
        "--no-root-input-global-orient",
        dest="disable_root_input_global_orient",
        action="store_true",
        help="Ablation: remove root-head input_global_orient features.",
    )
    parser.add_argument(
        "--disable-root-input-transl",
        "--no-root-input-transl",
        dest="disable_root_input_transl",
        action="store_true",
        help="Ablation: remove root-head input_transl features.",
    )
    parser.add_argument(
        "--disable-root-measurement-residual",
        "--no-root-measurement-residual",
        dest="disable_root_measurement_residual",
        action="store_true",
        help="Ablation: remove root-head 2D measurement residual features.",
    )
    parser.add_argument(
        "--disable-root-measurement-confidence",
        "--no-root-measurement-confidence",
        dest="disable_root_measurement_confidence",
        action="store_true",
        help="Ablation: remove root-head 2D measurement confidence features.",
    )
    parser.add_argument(
        "--disable-root-measurement-valid",
        "--no-root-measurement-valid",
        dest="disable_root_measurement_valid",
        action="store_true",
        help="Ablation: ignore root-head measurement validity mask.",
    )
    parser.add_argument(
        "--disable-root-image-size",
        "--no-root-image-size",
        dest="disable_root_image_size",
        action="store_true",
        help="Ablation: remove root-head image-size normalization.",
    )
    parser.add_argument(
        "--disable-body-stage2-pose",
        "--no-body-stage2-pose",
        dest="disable_body_stage2_pose",
        action="store_true",
        help="Ablation: remove frozen Stage2 body-pose feature from the body head.",
    )
    parser.add_argument(
        "--disable-body-input-pose-mean",
        "--no-body-input-pose-mean",
        dest="disable_body_input_pose_mean",
        action="store_true",
        help="Ablation: remove decoded views_input body-pose mean feature.",
    )
    parser.add_argument(
        "--disable-body-input-pose-dispersion",
        "--no-body-input-pose-dispersion",
        dest="disable_body_input_pose_dispersion",
        action="store_true",
        help="Ablation: remove decoded views_input body-pose dispersion feature.",
    )
    parser.add_argument(
        "--disable-body-input-betas",
        "--no-body-input-betas",
        dest="disable_body_input_betas",
        action="store_true",
        help="Ablation: remove decoded views_input betas mean feature.",
    )
    parser.add_argument(
        "--disable-body-image-residual",
        "--no-body-image-residual",
        dest="disable_body_image_residual",
        action="store_true",
        help="Ablation: remove body-head root-corrected 2D residual features.",
    )
    parser.add_argument(
        "--disable-body-image-confidence",
        "--no-body-image-confidence",
        dest="disable_body_image_confidence",
        action="store_true",
        help="Ablation: remove body-head 2D measurement confidence features.",
    )
    parser.add_argument(
        "--disable-body-image-valid",
        "--no-body-image-valid",
        dest="disable_body_image_valid",
        action="store_true",
        help="Ablation: ignore body-head measurement validity mask.",
    )
    parser.add_argument(
        "--disable-body-image-size",
        "--no-body-image-size",
        dest="disable_body_image_size",
        action="store_true",
        help="Ablation: remove body-head image-size normalization.",
    )
    parser.add_argument(
        "--body-evidence-gate-only",
        action="store_true",
        help=(
            "Use local image/mask evidence only to gate body updates instead of "
            "also appending evidence tokens to the global body MLP input."
        ),
    )
    parser.add_argument(
        "--use-body-evidence-weighted-pose-fusion",
        action="store_true",
        help=(
            "Use per-joint image/measurement evidence to form a reliability-weighted "
            "average of the per-view input body poses, then feed it to the body MLP."
        ),
    )
    parser.add_argument(
        "--evidence-weighted-pose-project-so3",
        action="store_true",
        help=(
            "When evidence-weighted pose fusion is enabled, average per-view "
            "body rotations in matrix space and project back to SO(3) instead "
            "of linearly averaging axis-angle vectors."
        ),
    )
    parser.add_argument(
        "--use-body-evidence-weighted-betas-fusion",
        action="store_true",
        help=(
            "Use per-joint image/measurement evidence to form reliability-weighted "
            "per-view betas features for the body MLP."
        ),
    )
    parser.add_argument(
        "--body-evidence-weighted-pose-joint-policy",
        choices=("all", "lower_body", "distal_legs", "ankle_feet"),
        default="all",
        help=(
            "Choose which SMPL body joints receive the evidence-weighted pose prior; "
            "other joints fall back to the simple per-view mean pose."
        ),
    )
    parser.add_argument("--body-evidence-gate-bias", type=float, default=2.0)
    parser.add_argument("--body-extra-candidate-count", type=int, default=0)
    parser.add_argument("--body-extra-hidden-dim", type=int, default=512)
    parser.add_argument("--body-extra-num-layers", type=int, default=3)
    parser.add_argument("--body-extra-selector-bias", type=float, default=2.0)
    parser.add_argument(
        "--body-extra-update-joint-policy",
        choices=("all", "lower_body", "distal_legs", "ankle_feet"),
        default="all",
        help="Restrict the optional extra body-update branch to a body-joint subset.",
    )
    parser.add_argument("--use-body-local-joint-update-head", action="store_true")
    parser.add_argument("--body-local-joint-hidden-dim", type=int, default=256)
    parser.add_argument("--body-local-joint-layers", type=int, default=3)
    parser.add_argument("--body-local-joint-embedding-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--global-orient-delta-scale", type=float, default=0.05)
    parser.add_argument("--transl-delta-scale", type=float, default=0.05)
    parser.add_argument(
        "--compose-global-orient-delta",
        action="store_true",
        help="Apply root/global-orient updates by SO(3) composition instead of axis-angle addition.",
    )
    parser.add_argument("--body-delta-scale", type=float, default=0.15)
    parser.add_argument("--betas-delta-scale", type=float, default=0.05)
    parser.add_argument("--use-body-betas-update", action="store_true")
    parser.add_argument(
        "--compose-body-update",
        action="store_true",
        help="Apply body-pose updates by SO(3) composition instead of axis-angle addition.",
    )
    parser.add_argument(
        "--detach-base-update-in-final-loss",
        action="store_true",
        help=(
            "Keep the final body-pose value unchanged but stop final-output losses "
            "from backpropagating through the base body update when an extra branch exists."
        ),
    )
    parser.add_argument("--gate-bias", type=float, default=-2.0)
    parser.add_argument("--camera-joint-weight", type=float, default=1.0)
    parser.add_argument(
        "--camera-joint-loss-weights",
        default=None,
        help=(
            "Optional comma-separated H36M joint loss weights, or 'h36m_limb2'. "
            "Metrics remain unweighted."
        ),
    )
    parser.add_argument(
        "--camera-joint-loss-weights-end-epoch",
        type=int,
        default=None,
        help="Disable camera-joint loss weights at this epoch and later.",
    )
    parser.add_argument("--euclidean-joint-weight", type=float, default=0.0)
    parser.add_argument(
        "--euclidean-joint-loss-weights",
        default=None,
        help=(
            "Optional comma-separated H36M joint weights for Euclidean MPJPE-style "
            "training loss, or 'h36m_limb2'. Metrics remain unweighted."
        ),
    )
    parser.add_argument("--pa-joint-weight", type=float, default=0.2)
    parser.add_argument(
        "--body-losses-start-with-body",
        action="store_true",
        help="Apply PA, preserve, do-no-harm, and body regularizers only after body_start_epoch.",
    )
    parser.add_argument(
        "--pa-joint-loss-weights",
        default=None,
        help=(
            "Optional comma-separated H36M joint weights for the PA training loss, "
            "or 'h36m_limb2'. Reported PA-MPJPE remains unweighted."
        ),
    )
    parser.add_argument(
        "--pa-joint-loss-weights-end-epoch",
        type=int,
        default=None,
        help="Disable PA-joint loss weights at this epoch and later.",
    )
    parser.add_argument("--root-camera-weight", type=float, default=0.0)
    parser.add_argument(
        "--body-base-camera-joint-weight",
        type=float,
        default=0.0,
        help="Auxiliary camera-joint loss on the base body branch before optional extra visual update.",
    )
    parser.add_argument(
        "--body-base-pa-joint-weight",
        type=float,
        default=0.0,
        help="Auxiliary PA-joint loss on the base body branch before optional extra visual update.",
    )
    parser.add_argument("--gt-projection-weight", type=float, default=0.03)
    parser.add_argument("--image-projection-weight", type=float, default=0.0)
    parser.add_argument(
        "--projection-losses-start-with-body",
        action="store_true",
        help="Apply GT/image projection losses only after body_start_epoch.",
    )
    parser.add_argument("--preserve-joint-weight", type=float, default=0.5)
    parser.add_argument("--do-no-harm-weight", type=float, default=0.1)
    parser.add_argument("--do-no-harm-margin-mm", type=float, default=0.0)
    parser.add_argument("--gate-sparsity-weight", type=float, default=0.0005)
    parser.add_argument(
        "--body-extra-selector-sparsity-weight",
        type=float,
        default=0.0,
        help="Penalty on the optional extra body branch selector mean.",
    )
    parser.add_argument(
        "--body-extra-update-l1-weight",
        type=float,
        default=0.0,
        help="Penalty on the applied optional extra body branch update magnitude.",
    )
    parser.add_argument(
        "--body-extra-counterfactual-preserve-weight",
        type=float,
        default=0.0,
        help=(
            "Preserve selected H36M camera joints from the base-only body output "
            "when the optional extra body branch is active."
        ),
    )
    parser.add_argument(
        "--body-extra-counterfactual-joint-policy",
        choices=("all", "upper_body", "non_lower_body", "non_distal_legs", "non_ankle_feet"),
        default="upper_body",
        help="H36M joint subset preserved against the base-only counterfactual.",
    )
    parser.add_argument(
        "--body-extra-counterfactual-do-no-harm-weight",
        type=float,
        default=0.0,
        help=(
            "Penalize the optional extra body branch when selected H36M joints "
            "become worse than the base-only body output against GT."
        ),
    )
    parser.add_argument(
        "--body-extra-counterfactual-do-no-harm-joint-policy",
        choices=("all", "upper_body", "non_lower_body", "non_distal_legs", "non_ankle_feet"),
        default="all",
        help="H36M joint subset protected by base-counterfactual do-no-harm.",
    )
    parser.add_argument(
        "--body-extra-counterfactual-do-no-harm-margin-mm",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--body-extra-selector-counterfactual-harm-weight",
        type=float,
        default=0.0,
        help=(
            "Penalize optional extra-branch selector permission on samples where "
            "the final output is worse than the base-only counterfactual."
        ),
    )
    parser.add_argument(
        "--body-extra-selector-counterfactual-harm-joint-policy",
        choices=("all", "upper_body", "non_lower_body", "non_distal_legs", "non_ankle_feet"),
        default="all",
    )
    parser.add_argument(
        "--body-extra-selector-counterfactual-harm-margin-mm",
        type=float,
        default=0.0,
    )
    parser.add_argument("--body-delta-weight", type=float, default=0.005)
    parser.add_argument("--betas-delta-weight", type=float, default=0.01)
    parser.add_argument("--global-orient-delta-weight", type=float, default=1.0)
    parser.add_argument("--transl-delta-weight", type=float, default=10.0)
    parser.add_argument("--camera-huber-beta-m", type=float, default=0.02)
    parser.add_argument("--projection-charbonnier-eps", type=float, default=1.0e-3)
    parser.add_argument("--projection-min-depth", type=float, default=0.1)
    parser.add_argument("--projection-border-px", type=float, default=0.0)
    parser.add_argument("--image-confidence-threshold", type=float, default=0.2)
    parser.add_argument("--allow-nonstandard-regressor", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment = load_experiment_config(args.config)
    helper_args = build_helper_args(args, stage="val")
    data_config = build_data_config(experiment["data"], helper_args)
    data_config.batch_size = int(args.batch_size)
    data_config.eval_batch_size = int(args.batch_size)
    data_config.num_workers = int(args.num_workers)
    data_config.drop_last_train = False
    data_config.drop_last_eval = False
    data_config.shuffle_train_views = True
    validate_fair_h36m_regressor(data_config, args=args)

    model_config = build_model_config(
        experiment["model"],
        checkpoint_path=args.stage2_checkpoint_path,
    )
    datamodule = build_datamodule(data_config)
    datamodule.prepare_data()
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    stage2_module = load_eval_module(
        checkpoint_path=args.stage2_checkpoint_path,
        model_config=model_config,
        data_config=data_config,
        args=helper_args,
    )
    stage2_module.to(device)
    stage2_can_train = (not bool(args.freeze_stage2)) and float(args.stage2_lr) > 0.0
    stage2_module.train(False)
    stage2_module.requires_grad_(False)

    adapter_config = Stage23RootBodyAdapterConfig(
        view_input_dim=int(model_config.input_dim),
        image_feature_dim=int(args.body_image_feature_dim),
        use_root_input_global_orient=not bool(args.disable_root_input_global_orient),
        use_root_input_transl=not bool(args.disable_root_input_transl),
        use_root_measurement_residual=not bool(args.disable_root_measurement_residual),
        use_root_measurement_confidence=not bool(args.disable_root_measurement_confidence),
        use_root_measurement_valid=not bool(args.disable_root_measurement_valid),
        use_root_image_size=not bool(args.disable_root_image_size),
        use_body_stage2_pose=not bool(args.disable_body_stage2_pose),
        use_body_input_pose_mean=not bool(args.disable_body_input_pose_mean),
        use_body_input_pose_dispersion=not bool(args.disable_body_input_pose_dispersion),
        use_body_input_betas=not bool(args.disable_body_input_betas),
        use_body_image_residual=not bool(args.disable_body_image_residual),
        use_body_image_confidence=not bool(args.disable_body_image_confidence),
        use_body_image_valid=not bool(args.disable_body_image_valid),
        use_body_image_size=not bool(args.disable_body_image_size),
        mask_feature_dim=int(args.body_mask_feature_dim),
        evidence_token_dim=int(args.body_evidence_token_dim),
        evidence_hidden_dim=int(args.body_evidence_hidden_dim),
        evidence_layers=int(args.body_evidence_layers),
        use_body_local_joint_update_head=bool(args.use_body_local_joint_update_head),
        body_local_joint_hidden_dim=int(args.body_local_joint_hidden_dim),
        body_local_joint_layers=int(args.body_local_joint_layers),
        body_local_joint_embedding_dim=int(args.body_local_joint_embedding_dim),
        root_hidden_dim=int(args.root_hidden_dim),
        body_hidden_dim=int(args.body_hidden_dim),
        root_num_layers=int(args.root_num_layers),
        body_num_layers=int(args.body_num_layers),
        dropout=float(args.dropout),
        global_orient_delta_scale=float(args.global_orient_delta_scale),
        transl_delta_scale=float(args.transl_delta_scale),
        body_delta_scale=float(args.body_delta_scale),
        betas_delta_scale=float(args.betas_delta_scale),
        gate_bias=float(args.gate_bias),
        use_body_image_joint_feature=bool(args.use_body_image_joint_feature),
        use_body_image_mask_feature=bool(args.use_body_image_mask_feature),
        use_body_evidence_gate=bool(args.use_body_evidence_gate),
        use_body_evidence_gate_only=bool(args.body_evidence_gate_only),
        use_body_evidence_weighted_pose_fusion=bool(
            args.use_body_evidence_weighted_pose_fusion
        ),
        evidence_weighted_pose_project_so3=bool(args.evidence_weighted_pose_project_so3),
        use_body_evidence_weighted_betas_fusion=bool(
            args.use_body_evidence_weighted_betas_fusion
        ),
        body_weighted_pose_joint_policy=str(args.body_evidence_weighted_pose_joint_policy),
        body_evidence_gate_bias=float(args.body_evidence_gate_bias),
        body_extra_candidate_count=int(args.body_extra_candidate_count),
        body_extra_hidden_dim=int(args.body_extra_hidden_dim),
        body_extra_num_layers=int(args.body_extra_num_layers),
        body_extra_selector_bias=float(args.body_extra_selector_bias),
        body_extra_update_joint_policy=str(args.body_extra_update_joint_policy),
        use_body_betas_update=bool(args.use_body_betas_update),
    )
    adapter = Stage23RootBodyAdapterModel(adapter_config).to(device)
    if args.resume_unified_checkpoint:
        resume_path = Path(args.resume_unified_checkpoint)
        checkpoint = torch.load(resume_path, map_location=device)
        stage2_state = checkpoint.get("stage2_state_dict")
        if stage2_state is not None:
            stage2_module.load_state_dict(stage2_state)
        load_adapter_resume_state(
            adapter,
            checkpoint["adapter_state_dict"],
            allow_head_expansion=bool(args.resume_allow_head_expansion),
        )

    param_groups = [
        {"params": adapter.root_adapter.parameters(), "lr": float(args.root_lr)},
        {"params": adapter.body_adapter.parameters(), "lr": float(args.body_lr)},
    ]
    if stage2_can_train:
        param_groups.insert(0, {"params": stage2_module.parameters(), "lr": float(args.stage2_lr)})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(args.weight_decay))

    history: list[dict[str, Any]] = []
    best_score = float("inf")
    for epoch in range(int(args.max_epochs)):
        stage2_trainable = stage2_can_train and epoch >= int(args.stage2_start_epoch)
        stage2_module.requires_grad_(stage2_trainable)
        body_enabled = epoch >= int(args.body_start_epoch)
        train_stats = run_epoch(
            stage2_module=stage2_module,
            adapter=adapter,
            dataloader=train_loader,
            args=args,
            device=device,
            optimizer=optimizer,
            train=True,
            max_batches=args.train_max_batches,
            body_enabled=body_enabled,
            stage2_trainable=stage2_trainable,
            epoch=epoch,
            phase="train",
        )
        val_stats = run_epoch(
            stage2_module=stage2_module,
            adapter=adapter,
            dataloader=val_loader,
            args=args,
            device=device,
            optimizer=None,
            train=False,
            max_batches=args.val_max_batches,
            body_enabled=body_enabled,
            stage2_trainable=stage2_trainable,
            epoch=epoch,
            phase="val",
        )
        row = {
            "epoch": epoch,
            "body_enabled": body_enabled,
            "train": train_stats,
            "val": val_stats,
        }
        history.append(row)
        print(json.dumps(row, indent=2, sort_keys=True), flush=True)
        score = goal_selection_score(val_stats)
        if score < best_score:
            best_score = score
            save_checkpoint(
                output_dir / "model_best.pt",
                stage2_module=stage2_module,
                adapter=adapter,
                adapter_config=adapter_config,
                args=args,
                epoch=epoch,
                stats=val_stats,
            )

    save_checkpoint(
        output_dir / "model_last.pt",
        stage2_module=stage2_module,
        adapter=adapter,
        adapter_config=adapter_config,
        args=args,
        epoch=int(args.max_epochs) - 1,
        stats=history[-1]["val"] if history else {},
    )
    summary = {
        "config": str(Path(args.config).resolve()),
        "stage2_checkpoint_path": str(Path(args.stage2_checkpoint_path).resolve()),
        "resume_unified_checkpoint": (
            str(Path(args.resume_unified_checkpoint).resolve())
            if args.resume_unified_checkpoint
            else None
        ),
        "adapter_config": asdict(adapter_config),
        "args": vars(args),
        "history": history,
        "best_score": best_score,
        "pipeline_note": (
            "One training loop jointly finetunes the Stage 2 backbone initialized "
            "from stage2-checkpoint-path and a fresh image-conditioned root/body "
            "adapter. No Stage 2.3 checkpoint is loaded."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    export_best_eval_json(output_dir)


def build_helper_args(args: argparse.Namespace, *, stage: str) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint_path=args.stage2_checkpoint_path,
        config=args.config,
        manifest_path=args.manifest_path,
        gt_smpl_dir=args.gt_smpl_dir,
        cameras_dir=args.cameras_dir,
        split_config_path=args.split_config_path,
        split_name=args.split_name,
        stage=stage,
        default_root_dir=None,
        accelerator=None,
        devices="1",
        strategy=None,
        num_nodes=None,
        seed=None,
        smpl_model_path=args.smpl_model_path,
        mhr_assets_dir=None,
        input_smpl_cache_dir=args.input_smpl_cache_dir,
        rgb_feature_cache_dir=args.rgb_feature_cache_dir,
        image_measurement_cache_dir=args.image_measurement_cache_dir,
        stage2_checkpoint_path=args.stage2_checkpoint_path,
        pred_camera_mode="input_corrected",
        output_path=None,
    )


def load_adapter_resume_state(
    adapter: Stage23RootBodyAdapterModel,
    state_dict: dict[str, torch.Tensor],
    *,
    allow_head_expansion: bool,
) -> None:
    if not allow_head_expansion:
        adapter.load_state_dict(state_dict)
        return
    current = adapter.state_dict()
    patched: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, target in current.items():
        source = state_dict.get(key)
        if source is None:
            skipped.append(key)
            continue
        if tuple(source.shape) == tuple(target.shape):
            patched[key] = source
            continue
        can_copy_rows = (
            source.ndim == target.ndim
            and source.ndim >= 1
            and source.shape[0] <= target.shape[0]
            and tuple(source.shape[1:]) == tuple(target.shape[1:])
        )
        if can_copy_rows:
            expanded = target.clone()
            expanded[: source.shape[0]] = source.to(device=target.device, dtype=target.dtype)
            patched[key] = expanded
        else:
            skipped.append(key)
    missing, unexpected = adapter.load_state_dict(patched, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected adapter resume keys: {unexpected}")
    if skipped:
        print(
            json.dumps(
                {
                    "event": "resume_head_expansion",
                    "loaded": len(patched),
                    "missing": list(missing),
                    "skipped": skipped,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def validate_fair_h36m_regressor(data_config, *, args: argparse.Namespace) -> None:
    regressor_path = getattr(data_config, "joint_target_joint_regressor_path", None)
    if bool(args.allow_nonstandard_regressor):
        return
    if regressor_path is None:
        raise ValueError("H36M fair run requires joint_target_joint_regressor_path")
    if Path(str(regressor_path)).name != "J_regressor_h36m_correct.npy":
        raise ValueError(
            "Refusing nonstandard H36M regressor for fair unified training: "
            f"{regressor_path}"
        )


def apply_body_pose_update(
    *,
    base_body_pose: torch.Tensor,
    body_pose_update: torch.Tensor,
    compose: bool,
) -> torch.Tensor:
    if not compose:
        return base_body_pose + body_pose_update
    batch_size = base_body_pose.shape[0]
    base = base_body_pose.reshape(batch_size, -1, 3)
    update = body_pose_update.reshape(batch_size, -1, 3)
    composed = torch.matmul(axis_angle_to_matrix(update), axis_angle_to_matrix(base))
    return matrix_to_axis_angle(composed).reshape_as(base_body_pose)


def run_epoch(
    *,
    stage2_module,
    adapter: Stage23RootBodyAdapterModel,
    dataloader,
    args: argparse.Namespace,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    train: bool,
    max_batches: int | None,
    body_enabled: bool,
    stage2_trainable: bool,
    epoch: int,
    phase: str,
) -> dict[str, float | int]:
    stage2_module.train(train and stage2_trainable)
    adapter.train(train)
    accum = MetricAccumulator()
    start_time = time.monotonic()
    total_batches = _safe_len(dataloader)
    if max_batches is not None and total_batches is not None:
        total_batches = min(total_batches, int(max_batches))
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(dataloader):
            batch = move_to_device(batch, device)
            predictions = run_stage2_forward(stage2_module, batch)
            stage2_projection = build_camera_projection(
                stage2_module=stage2_module,
                predictions=predictions,
                batch=batch,
                global_orient_delta=None,
                transl_delta=None,
                compose_global_orient_delta=bool(args.compose_global_orient_delta),
            )
            adapter_outputs = adapter.forward_root(
                views_input=batch["views_input"],
                view_aux=batch["view_aux"],
                stage2_uv=stage2_projection["uv"].detach(),
                measured_uv=batch.get("view_image_joint_uv"),
                measured_valid=batch.get("view_image_joint_valid"),
                measured_confidence=batch.get("view_image_joint_confidence"),
            )
            root_projection = build_camera_projection(
                stage2_module=stage2_module,
                predictions=predictions,
                batch=batch,
                global_orient_delta=adapter_outputs["pred_view_global_orient_delta"],
                transl_delta=adapter_outputs["pred_view_transl_delta"],
                compose_global_orient_delta=bool(args.compose_global_orient_delta),
            )
            if body_enabled:
                body_outputs = adapter.forward_body(
                    views_input=batch["views_input"],
                    stage2_body_pose=predictions["pred_body_pose"],
                    corrected_stage2_uv=root_projection["uv"].detach(),
                    measured_uv=batch.get("view_image_joint_uv"),
                    measured_valid=batch.get("view_image_joint_valid"),
                    measured_confidence=batch.get("view_image_joint_confidence"),
                    image_size=batch["view_aux"]["image_size"],
                    image_joint_feature=batch.get("view_image_joint_feature"),
                    image_mask_feature=batch.get("view_image_mask_feature"),
                )
                if epoch < int(args.body_extra_start_epoch):
                    base_update = body_outputs.get("pred_body_pose_base_update")
                    if base_update is not None:
                        body_outputs["pred_body_pose_update"] = base_update
                    extra_update = body_outputs.get("pred_body_pose_extra_update")
                    if extra_update is not None:
                        body_outputs["pred_body_pose_extra_update"] = torch.zeros_like(extra_update)
                    extra_selector = body_outputs.get("pred_body_pose_extra_selector")
                    if extra_selector is not None:
                        body_outputs["pred_body_pose_extra_selector"] = torch.zeros_like(extra_selector)
            else:
                body_outputs = zero_body_outputs(
                    predictions["pred_body_pose"],
                    num_body_joints=adapter.config.num_body_joints,
                )
            body_outputs.setdefault(
                "pred_betas_update",
                torch.zeros_like(predictions["pred_betas"]),
            )
            adapter_outputs.update(body_outputs)
            loss, stats = compute_losses_and_metrics(
                stage2_module=stage2_module,
                predictions=predictions,
                batch=batch,
                stage2_projection=stage2_projection,
                root_projection=root_projection,
                adapter_outputs=adapter_outputs,
                args=args,
                body_enabled=body_enabled,
                epoch=epoch,
            )
            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in list(stage2_module.parameters()) + list(adapter.parameters())
                        if parameter.requires_grad
                    ],
                    float(args.gradient_clip_norm),
                )
                optimizer.step()
            accum.update(stats, batch_size=int(batch["views_input"].shape[0]))
            maybe_log_progress(stats, args, epoch, phase, batch_index, total_batches, start_time)
            if max_batches is not None and batch_index + 1 >= int(max_batches):
                break
    return accum.finalize()


def compute_losses_and_metrics(
    *,
    stage2_module,
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    stage2_projection: dict[str, torch.Tensor],
    root_projection: dict[str, torch.Tensor],
    adapter_outputs: dict[str, torch.Tensor],
    args: argparse.Namespace,
    body_enabled: bool,
    epoch: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    body_pose_update_for_final = adapter_outputs["pred_body_pose_update"]
    base_update_for_final = adapter_outputs.get("pred_body_pose_base_update")
    if (
        bool(args.detach_base_update_in_final_loss)
        and base_update_for_final is not None
        and base_update_for_final.numel() > 0
    ):
        body_pose_update_for_final = body_pose_update_for_final + (
            base_update_for_final.detach() - base_update_for_final
        )
    corrected_body_pose = apply_body_pose_update(
        base_body_pose=predictions["pred_body_pose"],
        body_pose_update=body_pose_update_for_final,
        compose=bool(args.compose_body_update),
    )
    corrected_betas = predictions["pred_betas"] + adapter_outputs.get(
        "pred_betas_update",
        torch.zeros_like(predictions["pred_betas"]),
    )
    corrected_predictions = dict(predictions)
    corrected_predictions["pred_body_pose"] = corrected_body_pose
    corrected_predictions["pred_betas"] = corrected_betas
    corrected_projection = build_camera_projection(
        stage2_module=stage2_module,
        predictions=corrected_predictions,
        batch=batch,
        global_orient_delta=adapter_outputs["pred_view_global_orient_delta"],
        transl_delta=adapter_outputs["pred_view_transl_delta"],
        compose_global_orient_delta=bool(args.compose_global_orient_delta),
    )
    camera_loss, corrected_mpjpe = camera_joint_loss_and_error(
        pred_camera_joints=corrected_projection["camera_joints"],
        batch=batch,
        huber_beta_m=float(args.camera_huber_beta_m),
    )
    camera_joint_loss_weight_spec = active_loss_weight_spec(
        spec=args.camera_joint_loss_weights,
        end_epoch=args.camera_joint_loss_weights_end_epoch,
        epoch=epoch,
    )
    joint_loss_weights = build_joint_loss_weights(
        spec=camera_joint_loss_weight_spec,
        joint_count=int(corrected_projection["camera_joints"].shape[-2]),
        device=corrected_projection["camera_joints"].device,
        dtype=corrected_projection["camera_joints"].dtype,
    )
    if joint_loss_weights is not None:
        camera_loss = weighted_camera_joint_loss(
            pred_camera_joints=corrected_projection["camera_joints"],
            batch=batch,
            huber_beta_m=float(args.camera_huber_beta_m),
            joint_weights=joint_loss_weights,
        )
    euclidean_joint_loss_weights = build_joint_loss_weights(
        spec=args.euclidean_joint_loss_weights,
        joint_count=int(corrected_projection["camera_joints"].shape[-2]),
        device=corrected_projection["camera_joints"].device,
        dtype=corrected_projection["camera_joints"].dtype,
    )
    euclidean_loss = euclidean_camera_joint_loss(
        pred_camera_joints=corrected_projection["camera_joints"],
        batch=batch,
        joint_weights=euclidean_joint_loss_weights,
    )
    root_camera_loss, root_mpjpe = camera_joint_loss_and_error(
        pred_camera_joints=root_projection["camera_joints"],
        batch=batch,
        huber_beta_m=float(args.camera_huber_beta_m),
    )
    _, stage2_mpjpe = camera_joint_loss_and_error(
        pred_camera_joints=stage2_projection["camera_joints"],
        batch=batch,
        huber_beta_m=float(args.camera_huber_beta_m),
    )
    pa_loss, corrected_pa = pa_joint_loss_and_error(
        stage2_module,
        corrected_projection["camera_joints"],
        batch,
    )
    pa_joint_loss_weight_spec = active_loss_weight_spec(
        spec=args.pa_joint_loss_weights,
        end_epoch=args.pa_joint_loss_weights_end_epoch,
        epoch=epoch,
    )
    pa_joint_loss_weights = build_joint_loss_weights(
        spec=pa_joint_loss_weight_spec,
        joint_count=int(corrected_projection["camera_joints"].shape[-2]),
        device=corrected_projection["camera_joints"].device,
        dtype=corrected_projection["camera_joints"].dtype,
    )
    if pa_joint_loss_weights is not None:
        pa_loss = weighted_pa_joint_loss(
            stage2_module=stage2_module,
            pred_camera_joints=corrected_projection["camera_joints"],
            batch=batch,
            joint_weights=pa_joint_loss_weights,
        )
    _, stage2_pa = pa_joint_loss_and_error(
        stage2_module,
        stage2_projection["camera_joints"],
        batch,
    )
    preserve_loss = canonical_preserve_loss(
        stage2_module=stage2_module,
        stage2_body_pose=predictions["pred_body_pose"],
        corrected_body_pose=corrected_body_pose,
        betas=corrected_betas,
        batch=batch,
    )
    harm_loss, harm_rate = do_no_harm_loss_and_rate(
        stage2_camera_joints=stage2_projection["camera_joints"],
        corrected_camera_joints=corrected_projection["camera_joints"],
        batch=batch,
        margin_mm=float(args.do_no_harm_margin_mm),
    )
    gt_projection_loss, gt_projection_error = projection_loss_and_error(
        pred_uv=corrected_projection["uv"],
        pred_depth=corrected_projection["depth"],
        batch=batch,
        target="gt",
        confidence_threshold=0.05,
        args=args,
    )
    image_projection_loss, image_projection_error = projection_loss_and_error(
        pred_uv=corrected_projection["uv"],
        pred_depth=corrected_projection["depth"],
        batch=batch,
        target="image",
        confidence_threshold=float(args.image_confidence_threshold),
        args=args,
    )
    global_delta_l2 = adapter_outputs["pred_view_global_orient_delta"].square().mean()
    transl_delta_l2 = adapter_outputs["pred_view_transl_delta"].square().mean()
    gate = adapter_outputs["pred_body_pose_gate"]
    delta = adapter_outputs["pred_body_pose_delta"]
    update = adapter_outputs["pred_body_pose_update"]
    betas_update = adapter_outputs.get(
        "pred_betas_update",
        torch.zeros_like(predictions["pred_betas"]),
    )
    evidence_gate = adapter_outputs.get("pred_body_pose_evidence_gate")
    if evidence_gate is None:
        evidence_gate = torch.ones_like(gate)
    base_update = adapter_outputs.get("pred_body_pose_base_update")
    extra_selector = adapter_outputs.get("pred_body_pose_extra_selector")
    extra_update = adapter_outputs.get("pred_body_pose_extra_update")
    gate_sparsity = gate.mean()
    if extra_selector is not None and extra_selector.numel() > 0:
        extra_selector_sparsity = extra_selector.mean()
    else:
        extra_selector_sparsity = gate.new_zeros(())
    if extra_update is not None and extra_update.numel() > 0:
        extra_update_l1 = extra_update.abs().mean()
    else:
        extra_update_l1 = gate.new_zeros(())
    extra_counterfactual_preserve_loss = gate.new_zeros(())
    extra_counterfactual_harm_loss = gate.new_zeros(())
    extra_counterfactual_harm_rate = gate.new_zeros(())
    extra_selector_counterfactual_harm_loss = gate.new_zeros(())
    base_camera_loss = camera_loss.new_zeros(())
    base_mpjpe = corrected_mpjpe.new_zeros(())
    base_pa_loss = pa_loss.new_zeros(())
    base_pa = corrected_pa.new_zeros(())
    base_aux_requested = bool(body_enabled) and (
        float(args.body_base_camera_joint_weight) > 0.0
        or float(args.body_base_pa_joint_weight) > 0.0
    )
    need_base_only_projection = (
        base_aux_requested
        or float(args.body_extra_counterfactual_preserve_weight) > 0.0
        or float(args.body_extra_counterfactual_do_no_harm_weight) > 0.0
        or float(args.body_extra_selector_counterfactual_harm_weight) > 0.0
    )
    if (
        base_update is not None
        and base_update.numel() > 0
        and need_base_only_projection
    ):
        base_only_predictions = dict(predictions)
        base_only_predictions["pred_body_pose"] = apply_body_pose_update(
            base_body_pose=predictions["pred_body_pose"],
            body_pose_update=base_update,
            compose=bool(args.compose_body_update),
        )
        base_only_predictions["pred_betas"] = corrected_betas
        base_only_projection = build_camera_projection(
            stage2_module=stage2_module,
            predictions=base_only_predictions,
            batch=batch,
            global_orient_delta=adapter_outputs["pred_view_global_orient_delta"],
            transl_delta=adapter_outputs["pred_view_transl_delta"],
            compose_global_orient_delta=bool(args.compose_global_orient_delta),
        )
        if base_aux_requested:
            base_camera_loss, base_mpjpe = camera_joint_loss_and_error(
                pred_camera_joints=base_only_projection["camera_joints"],
                batch=batch,
                huber_beta_m=float(args.camera_huber_beta_m),
            )
            if joint_loss_weights is not None:
                base_camera_loss = weighted_camera_joint_loss(
                    pred_camera_joints=base_only_projection["camera_joints"],
                    batch=batch,
                    huber_beta_m=float(args.camera_huber_beta_m),
                    joint_weights=joint_loss_weights,
                )
            base_pa_loss, base_pa = pa_joint_loss_and_error(
                stage2_module,
                base_only_projection["camera_joints"],
                batch,
            )
            if pa_joint_loss_weights is not None:
                base_pa_loss = weighted_pa_joint_loss(
                    stage2_module=stage2_module,
                    pred_camera_joints=base_only_projection["camera_joints"],
                    batch=batch,
                    joint_weights=pa_joint_loss_weights,
                )
        if float(args.body_extra_counterfactual_preserve_weight) > 0.0:
            extra_counterfactual_preserve_loss = counterfactual_camera_joint_preserve_loss(
                pred_camera_joints=corrected_projection["camera_joints"],
                reference_camera_joints=base_only_projection["camera_joints"].detach(),
                batch=batch,
                joint_policy=str(args.body_extra_counterfactual_joint_policy),
            )
        if float(args.body_extra_counterfactual_do_no_harm_weight) > 0.0:
            extra_counterfactual_harm_loss, extra_counterfactual_harm_rate = (
                counterfactual_camera_joint_do_no_harm_loss_and_rate(
                    pred_camera_joints=corrected_projection["camera_joints"],
                    reference_camera_joints=base_only_projection["camera_joints"].detach(),
                    batch=batch,
                    joint_policy=str(args.body_extra_counterfactual_do_no_harm_joint_policy),
                    margin_mm=float(args.body_extra_counterfactual_do_no_harm_margin_mm),
                )
            )
        if (
            extra_selector is not None
            and extra_selector.numel() > 0
            and float(args.body_extra_selector_counterfactual_harm_weight) > 0.0
        ):
            extra_selector_counterfactual_harm_loss = counterfactual_selector_harm_loss(
                pred_camera_joints=corrected_projection["camera_joints"],
                reference_camera_joints=base_only_projection["camera_joints"].detach(),
                batch=batch,
                joint_policy=str(args.body_extra_selector_counterfactual_harm_joint_policy),
                margin_mm=float(args.body_extra_selector_counterfactual_harm_margin_mm),
                extra_selector=extra_selector,
            )
    body_delta_l2 = delta.square().mean()
    betas_delta_l2 = betas_update.square().mean()
    body_loss_enabled = bool(body_enabled) or not bool(args.body_losses_start_with_body)
    projection_loss_enabled = bool(body_enabled) or not bool(
        args.projection_losses_start_with_body
    )
    pa_weight = float(args.pa_joint_weight) if body_loss_enabled else 0.0
    preserve_weight = float(args.preserve_joint_weight) if body_loss_enabled else 0.0
    do_no_harm_weight = float(args.do_no_harm_weight) if body_loss_enabled else 0.0
    gate_sparsity_weight = float(args.gate_sparsity_weight) if body_loss_enabled else 0.0
    extra_selector_sparsity_weight = (
        float(args.body_extra_selector_sparsity_weight) if body_loss_enabled else 0.0
    )
    extra_update_l1_weight = (
        float(args.body_extra_update_l1_weight) if body_loss_enabled else 0.0
    )
    extra_counterfactual_preserve_weight = (
        float(args.body_extra_counterfactual_preserve_weight) if body_loss_enabled else 0.0
    )
    extra_counterfactual_harm_weight = (
        float(args.body_extra_counterfactual_do_no_harm_weight) if body_loss_enabled else 0.0
    )
    extra_selector_counterfactual_harm_weight = (
        float(args.body_extra_selector_counterfactual_harm_weight) if body_loss_enabled else 0.0
    )
    base_camera_weight = float(args.body_base_camera_joint_weight) if body_loss_enabled else 0.0
    base_pa_weight = float(args.body_base_pa_joint_weight) if body_loss_enabled else 0.0
    body_delta_weight = float(args.body_delta_weight) if body_loss_enabled else 0.0
    betas_delta_weight = float(args.betas_delta_weight) if body_loss_enabled else 0.0
    gt_projection_weight = (
        float(args.gt_projection_weight) if projection_loss_enabled else 0.0
    )
    image_projection_weight = (
        float(args.image_projection_weight) if projection_loss_enabled else 0.0
    )
    loss = (
        float(args.camera_joint_weight) * camera_loss
        + float(args.euclidean_joint_weight) * euclidean_loss
        + pa_weight * pa_loss
        + float(args.root_camera_weight) * root_camera_loss
        + base_camera_weight * base_camera_loss
        + base_pa_weight * base_pa_loss
        + gt_projection_weight * gt_projection_loss
        + image_projection_weight * image_projection_loss
        + preserve_weight * preserve_loss
        + do_no_harm_weight * harm_loss
        + gate_sparsity_weight * gate_sparsity
        + extra_selector_sparsity_weight * extra_selector_sparsity
        + extra_update_l1_weight * extra_update_l1
        + extra_counterfactual_preserve_weight * extra_counterfactual_preserve_loss
        + extra_counterfactual_harm_weight * extra_counterfactual_harm_loss
        + extra_selector_counterfactual_harm_weight * extra_selector_counterfactual_harm_loss
        + body_delta_weight * body_delta_l2
        + betas_delta_weight * betas_delta_l2
        + float(args.global_orient_delta_weight) * global_delta_l2
        + float(args.transl_delta_weight) * transl_delta_l2
    )
    stats = {
        "body_enabled": int(body_enabled),
        "effective/camera_joint_loss_weights_active": int(joint_loss_weights is not None),
        "effective/pa_joint_loss_weights_active": int(pa_joint_loss_weights is not None),
        "effective/pa_joint_weight": pa_weight,
        "effective/gt_projection_weight": gt_projection_weight,
        "effective/image_projection_weight": image_projection_weight,
        "effective/do_no_harm_weight": do_no_harm_weight,
        "effective/body_extra_selector_sparsity_weight": extra_selector_sparsity_weight,
        "effective/body_extra_update_l1_weight": extra_update_l1_weight,
        "effective/body_extra_counterfactual_preserve_weight": extra_counterfactual_preserve_weight,
        "effective/body_extra_counterfactual_do_no_harm_weight": extra_counterfactual_harm_weight,
        "effective/body_extra_selector_counterfactual_harm_weight": extra_selector_counterfactual_harm_weight,
        "effective/body_base_camera_joint_weight": base_camera_weight,
        "effective/body_base_pa_joint_weight": base_pa_weight,
        "effective/detach_base_update_in_final_loss": int(
            bool(args.detach_base_update_in_final_loss)
        ),
        "effective/evidence_weighted_pose_project_so3": int(
            bool(args.evidence_weighted_pose_project_so3)
        ),
        "loss": float(loss.detach().cpu().item()),
        "loss/camera_joint": float(camera_loss.detach().cpu().item()),
        "loss/euclidean_joint": float(euclidean_loss.detach().cpu().item()),
        "loss/pa_joint": float(pa_loss.detach().cpu().item()),
        "loss/root_camera_joint": float(root_camera_loss.detach().cpu().item()),
        "loss/body_base_camera_joint": float(base_camera_loss.detach().cpu().item()),
        "loss/body_base_pa_joint": float(base_pa_loss.detach().cpu().item()),
        "loss/preserve_joint": float(preserve_loss.detach().cpu().item()),
        "loss/do_no_harm": float(harm_loss.detach().cpu().item()),
        "loss/gt_projection": float(gt_projection_loss.detach().cpu().item()),
        "loss/image_projection": float(image_projection_loss.detach().cpu().item()),
        "loss/global_orient_delta_l2": float(global_delta_l2.detach().cpu().item()),
        "loss/transl_delta_l2": float(transl_delta_l2.detach().cpu().item()),
        "loss/gate_sparsity": float(gate_sparsity.detach().cpu().item()),
        "loss/body_extra_selector_sparsity": float(
            extra_selector_sparsity.detach().cpu().item()
        ),
        "loss/body_extra_update_l1": float(extra_update_l1.detach().cpu().item()),
        "loss/body_extra_counterfactual_preserve": float(
            extra_counterfactual_preserve_loss.detach().cpu().item()
        ),
        "loss/body_extra_counterfactual_do_no_harm": float(
            extra_counterfactual_harm_loss.detach().cpu().item()
        ),
        "loss/body_extra_selector_counterfactual_harm": float(
            extra_selector_counterfactual_harm_loss.detach().cpu().item()
        ),
        "loss/body_delta_l2": float(body_delta_l2.detach().cpu().item()),
        "loss/betas_delta_l2": float(betas_delta_l2.detach().cpu().item()),
        "stage2/camera_mpjpe_mm": float(stage2_mpjpe.detach().cpu().item()),
        "root/camera_mpjpe_mm": float(root_mpjpe.detach().cpu().item()),
        "base/camera_mpjpe_mm": float(base_mpjpe.detach().cpu().item()),
        "corrected/camera_mpjpe_mm": float(corrected_mpjpe.detach().cpu().item()),
        "stage2/camera_pa_mpjpe_mm": float(stage2_pa.detach().cpu().item()),
        "base/camera_pa_mpjpe_mm": float(base_pa.detach().cpu().item()),
        "corrected/camera_pa_mpjpe_mm": float(corrected_pa.detach().cpu().item()),
        "delta/camera_mpjpe_mm": float((corrected_mpjpe - stage2_mpjpe).detach().cpu().item()),
        "delta/camera_pa_mpjpe_mm": float((corrected_pa - stage2_pa).detach().cpu().item()),
        "corrected/gt_projection_error_px": float(gt_projection_error.detach().cpu().item()),
        "corrected/image_projection_error_px": float(image_projection_error.detach().cpu().item()),
        "adapter/global_orient_abs_mean": float(
            adapter_outputs["pred_view_global_orient_delta"].detach().abs().mean().cpu().item()
        ),
        "adapter/transl_abs_mean": float(
            adapter_outputs["pred_view_transl_delta"].detach().abs().mean().cpu().item()
        ),
        "adapter/gate_mean": float(gate.detach().mean().cpu().item()),
        "adapter/gate_max": float(gate.detach().amax().cpu().item()),
        "adapter/evidence_gate_mean": float(evidence_gate.detach().mean().cpu().item()),
        "adapter/evidence_gate_min": float(evidence_gate.detach().amin().cpu().item()),
        "adapter/update_abs_mean": float(update.detach().abs().mean().cpu().item()),
        "adapter/delta_abs_mean": float(delta.detach().abs().mean().cpu().item()),
        "adapter/base_update_abs_mean": float(
            base_update.detach().abs().mean().cpu().item()
        )
        if base_update is not None and base_update.numel() > 0
        else 0.0,
        "adapter/extra_selector_mean": float(
            extra_selector.detach().mean().cpu().item()
        )
        if extra_selector is not None and extra_selector.numel() > 0
        else 0.0,
        "adapter/extra_update_abs_mean": float(
            extra_update.detach().abs().mean().cpu().item()
        )
        if extra_update is not None and extra_update.numel() > 0
        else 0.0,
        "adapter/betas_update_abs_mean": float(
            betas_update.detach().abs().mean().cpu().item()
        ),
        "adapter/harm_rate": float(harm_rate.detach().cpu().item()),
        "adapter/extra_counterfactual_harm_rate": float(
            extra_counterfactual_harm_rate.detach().cpu().item()
        ),
    }
    return loss, stats


def build_camera_joint_loss_weights(
    *,
    args: argparse.Namespace,
    pred_camera_joints: torch.Tensor,
) -> torch.Tensor | None:
    return build_joint_loss_weights(
        spec=args.camera_joint_loss_weights,
        joint_count=int(pred_camera_joints.shape[-2]),
        device=pred_camera_joints.device,
        dtype=pred_camera_joints.dtype,
    )


def active_loss_weight_spec(
    *,
    spec: str | None,
    end_epoch: int | None,
    epoch: int,
) -> str | None:
    if end_epoch is not None and int(epoch) >= int(end_epoch):
        return None
    return spec


def build_joint_loss_weights(
    *,
    spec: str | None,
    joint_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if spec is None or str(spec).strip() == "":
        return None
    if spec == "h36m_limb2":
        values = [
            0.2,
            1.0,
            1.5,
            2.0,
            1.0,
            1.5,
            2.0,
            1.0,
            1.0,
            1.2,
            1.2,
            1.0,
            1.5,
            2.0,
            1.0,
            1.5,
            2.0,
        ]
    else:
        values = [float(part) for part in str(spec).split(",") if part.strip()]
    if len(values) != joint_count:
        raise ValueError(
            f"Expected {joint_count} camera joint loss weights, got {len(values)}"
        )
    return torch.tensor(
        values,
        device=device,
        dtype=dtype,
    )


def counterfactual_camera_joint_preserve_loss(
    *,
    pred_camera_joints: torch.Tensor,
    reference_camera_joints: torch.Tensor,
    batch: dict[str, torch.Tensor],
    joint_policy: str,
) -> torch.Tensor:
    confidence = batch["target_camera_joint_confidence"].to(
        device=pred_camera_joints.device,
        dtype=pred_camera_joints.dtype,
    )
    root_index = int(batch["target_joint_root_index"].reshape(-1)[0].item())
    pred_root = pred_camera_joints - pred_camera_joints[:, :, root_index : root_index + 1, :]
    reference_root = reference_camera_joints.to(
        device=pred_camera_joints.device,
        dtype=pred_camera_joints.dtype,
    )
    reference_root = reference_root - reference_root[:, :, root_index : root_index + 1, :]
    joint_mask = build_counterfactual_h36m_joint_mask(
        policy=joint_policy,
        joint_count=int(pred_camera_joints.shape[-2]),
        device=pred_camera_joints.device,
        dtype=pred_camera_joints.dtype,
    )
    valid = confidence > 0.05
    weight = confidence * valid.to(dtype=confidence.dtype) * joint_mask.reshape(1, 1, -1)
    error_m = torch.linalg.norm(pred_root - reference_root, dim=-1)
    return (error_m * weight).sum() / weight.sum().clamp_min(1.0)


def counterfactual_camera_joint_do_no_harm_loss_and_rate(
    *,
    pred_camera_joints: torch.Tensor,
    reference_camera_joints: torch.Tensor,
    batch: dict[str, torch.Tensor],
    joint_policy: str,
    margin_mm: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = batch["target_camera_joints"].to(
        device=pred_camera_joints.device,
        dtype=pred_camera_joints.dtype,
    )
    confidence = batch["target_camera_joint_confidence"].to(
        device=pred_camera_joints.device,
        dtype=pred_camera_joints.dtype,
    )
    root_index = int(batch["target_joint_root_index"].reshape(-1)[0].item())
    pred_root = pred_camera_joints - pred_camera_joints[:, :, root_index : root_index + 1, :]
    reference_root = reference_camera_joints.to(
        device=pred_camera_joints.device,
        dtype=pred_camera_joints.dtype,
    )
    reference_root = reference_root - reference_root[:, :, root_index : root_index + 1, :]
    target_root = target - target[:, :, root_index : root_index + 1, :]
    joint_mask = build_counterfactual_h36m_joint_mask(
        policy=joint_policy,
        joint_count=int(pred_camera_joints.shape[-2]),
        device=pred_camera_joints.device,
        dtype=pred_camera_joints.dtype,
    )
    valid = confidence > 0.05
    weight = confidence * valid.to(dtype=confidence.dtype) * joint_mask.reshape(1, 1, -1)
    pred_error = torch.linalg.norm(pred_root - target_root, dim=-1)
    reference_error = torch.linalg.norm(reference_root - target_root, dim=-1)
    harm = torch.relu(pred_error - reference_error + float(margin_mm) * 0.001)
    denom = weight.sum().clamp_min(1.0)
    harm_loss = (harm * weight).sum() / denom
    harm_rate = ((pred_error > reference_error + 1.0e-8).to(dtype=weight.dtype) * weight).sum()
    harm_rate = harm_rate / denom
    return harm_loss, harm_rate


def counterfactual_selector_harm_loss(
    *,
    pred_camera_joints: torch.Tensor,
    reference_camera_joints: torch.Tensor,
    batch: dict[str, torch.Tensor],
    joint_policy: str,
    margin_mm: float,
    extra_selector: torch.Tensor,
) -> torch.Tensor:
    target = batch["target_camera_joints"].to(
        device=pred_camera_joints.device,
        dtype=pred_camera_joints.dtype,
    )
    confidence = batch["target_camera_joint_confidence"].to(
        device=pred_camera_joints.device,
        dtype=pred_camera_joints.dtype,
    )
    root_index = int(batch["target_joint_root_index"].reshape(-1)[0].item())
    pred_root = pred_camera_joints - pred_camera_joints[:, :, root_index : root_index + 1, :]
    reference_root = reference_camera_joints.to(
        device=pred_camera_joints.device,
        dtype=pred_camera_joints.dtype,
    )
    reference_root = reference_root - reference_root[:, :, root_index : root_index + 1, :]
    target_root = target - target[:, :, root_index : root_index + 1, :]
    joint_mask = build_counterfactual_h36m_joint_mask(
        policy=joint_policy,
        joint_count=int(pred_camera_joints.shape[-2]),
        device=pred_camera_joints.device,
        dtype=pred_camera_joints.dtype,
    )
    valid = confidence > 0.05
    weight = confidence * valid.to(dtype=confidence.dtype) * joint_mask.reshape(1, 1, -1)
    pred_error = torch.linalg.norm(pred_root - target_root, dim=-1)
    reference_error = torch.linalg.norm(reference_root - target_root, dim=-1)
    harm = torch.relu(pred_error - reference_error + float(margin_mm) * 0.001)
    reduce_dims = tuple(range(1, harm.ndim))
    sample_weight = weight.sum(dim=reduce_dims).clamp_min(1.0)
    sample_harm = (harm * weight).sum(dim=reduce_dims) / sample_weight
    selector_mean = extra_selector.reshape(extra_selector.shape[0], -1).mean(dim=-1)
    return (selector_mean * sample_harm.detach()).mean()


def build_counterfactual_h36m_joint_mask(
    *,
    policy: str,
    joint_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    all_indices = set(range(joint_count))
    lower_body = {1, 2, 3, 4, 5, 6, 7}
    distal_legs = {2, 3, 5, 6, 7}
    ankle_feet = {3, 6, 7}
    upper_body = {index for index in range(8, joint_count)}
    if policy == "all":
        indices = all_indices
    elif policy == "upper_body":
        indices = upper_body
    elif policy == "non_lower_body":
        indices = all_indices - lower_body
    elif policy == "non_distal_legs":
        indices = all_indices - distal_legs
    elif policy == "non_ankle_feet":
        indices = all_indices - ankle_feet
    else:
        raise ValueError(f"Unknown counterfactual H36M joint policy: {policy}")
    mask = torch.zeros(joint_count, device=device, dtype=dtype)
    valid_indices = sorted(index for index in indices if 0 <= index < joint_count)
    if valid_indices:
        mask[torch.tensor(valid_indices, device=device)] = 1.0
    return mask


def weighted_camera_joint_loss(
    *,
    pred_camera_joints: torch.Tensor,
    batch: dict[str, torch.Tensor],
    huber_beta_m: float,
    joint_weights: torch.Tensor,
) -> torch.Tensor:
    target = batch["target_camera_joints"].to(
        device=pred_camera_joints.device,
        dtype=pred_camera_joints.dtype,
    )
    confidence = batch["target_camera_joint_confidence"].to(
        device=pred_camera_joints.device,
        dtype=pred_camera_joints.dtype,
    )
    root_index = int(batch["target_joint_root_index"].reshape(-1)[0].item())
    pred_root = pred_camera_joints - pred_camera_joints[:, :, root_index : root_index + 1, :]
    target_root = target - target[:, :, root_index : root_index + 1, :]
    valid = confidence > 0.05
    weight = confidence * valid.to(dtype=confidence.dtype)
    joint_weights = joint_weights.reshape(1, 1, -1)
    diff = pred_root - target_root
    per_coord = torch.nn.functional.smooth_l1_loss(
        diff,
        torch.zeros_like(diff),
        beta=float(huber_beta_m),
        reduction="none",
    )
    per_joint_loss = per_coord.sum(dim=-1)
    weighted = weight * joint_weights
    return (per_joint_loss * weighted).sum() / weighted.sum().clamp_min(1.0)


def euclidean_camera_joint_loss(
    *,
    pred_camera_joints: torch.Tensor,
    batch: dict[str, torch.Tensor],
    joint_weights: torch.Tensor | None,
) -> torch.Tensor:
    target = batch["target_camera_joints"].to(
        device=pred_camera_joints.device,
        dtype=pred_camera_joints.dtype,
    )
    confidence = batch["target_camera_joint_confidence"].to(
        device=pred_camera_joints.device,
        dtype=pred_camera_joints.dtype,
    )
    root_index = int(batch["target_joint_root_index"].reshape(-1)[0].item())
    pred_root = pred_camera_joints - pred_camera_joints[:, :, root_index : root_index + 1, :]
    target_root = target - target[:, :, root_index : root_index + 1, :]
    valid = confidence > 0.05
    weight = confidence * valid.to(dtype=confidence.dtype)
    if joint_weights is not None:
        weight = weight * joint_weights.reshape(1, 1, -1)
    error_m = torch.linalg.norm(pred_root - target_root, dim=-1)
    return (error_m * weight).sum() / weight.sum().clamp_min(1.0)


def weighted_pa_joint_loss(
    *,
    stage2_module,
    pred_camera_joints: torch.Tensor,
    batch: dict[str, torch.Tensor],
    joint_weights: torch.Tensor,
) -> torch.Tensor:
    target = batch["target_camera_joints"].to(
        device=pred_camera_joints.device,
        dtype=pred_camera_joints.dtype,
    )
    confidence = batch["target_camera_joint_confidence"].to(
        device=pred_camera_joints.device,
        dtype=pred_camera_joints.dtype,
    )
    root_index = int(batch["target_joint_root_index"].reshape(-1)[0].item())
    pred_root = pred_camera_joints - pred_camera_joints[:, :, root_index : root_index + 1, :]
    target_root = target - target[:, :, root_index : root_index + 1, :]
    batch_size, num_views, joint_count, _ = pred_root.shape
    flat_pred = pred_root.reshape(batch_size * num_views, joint_count, 3)
    flat_target = target_root.reshape(batch_size * num_views, joint_count, 3)
    flat_confidence = confidence.reshape(batch_size * num_views, joint_count)
    valid_weight = flat_confidence * (flat_confidence > 0.05).to(flat_confidence.dtype)
    aligned = stage2_module._weighted_similarity_align(
        flat_pred.float(),
        flat_target.float(),
        valid_weight.float(),
    ).to(dtype=pred_camera_joints.dtype)
    error_m = torch.linalg.norm(aligned - flat_target, dim=-1)
    joint_weights = joint_weights.reshape(1, joint_count)
    weighted = valid_weight * joint_weights
    return (error_m * weighted).sum() / weighted.sum().clamp_min(1.0)


def save_checkpoint(
    path: Path,
    *,
    stage2_module,
    adapter: Stage23RootBodyAdapterModel,
    adapter_config: Stage23RootBodyAdapterConfig,
    args: argparse.Namespace,
    epoch: int,
    stats: dict[str, float | int],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "stage2_state_dict": stage2_module.state_dict(),
            "adapter_state_dict": adapter.state_dict(),
            "adapter_config": asdict(adapter_config),
            "args": vars(args),
            "stats": stats,
        },
        path,
    )


def export_best_eval_json(output_dir: Path) -> None:
    best_path = output_dir / "model_best.pt"
    if not best_path.exists():
        return
    checkpoint = torch.load(best_path, map_location="cpu")
    payload = {
        "artifact": str(best_path),
        "epoch": checkpoint.get("epoch"),
        "samples": checkpoint.get("stats", {}).get("sample_weight"),
        "metrics": checkpoint.get("stats", {}),
        "mesh_faithful": True,
        "joint_path": (
            "final SMPL parameters -> SMPL vertices -> "
            "data/weights/J_regressor_h36m_correct.npy -> "
            "H36M camera joints -> root-centered metrics"
        ),
        "protocol_note": (
            "Fair H36M protocol; no Stage 2.3 checkpoint, no train-fitted "
            "regressor, and no post-SMPL H36M joint residual."
        ),
    }
    (output_dir / "eval_val_full.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def maybe_log_progress(stats, args, epoch, phase, batch_index, total_batches, start_time) -> None:
    if (batch_index + 1) % int(args.log_every_n_batches) != 0:
        if total_batches is None or batch_index + 1 != total_batches:
            return
    payload = {
        "event": "progress",
        "phase": phase,
        "epoch": epoch,
        "batch": batch_index + 1,
        "total_batches": total_batches,
        "elapsed_sec": round(time.monotonic() - start_time, 3),
        "stage2_mpjpe_mm": stats.get("stage2/camera_mpjpe_mm"),
        "corrected_mpjpe_mm": stats.get("corrected/camera_mpjpe_mm"),
        "corrected_pa_mpjpe_mm": stats.get("corrected/camera_pa_mpjpe_mm"),
        "delta_mpjpe_mm": stats.get("delta/camera_mpjpe_mm"),
        "delta_pa_mpjpe_mm": stats.get("delta/camera_pa_mpjpe_mm"),
        "gate_mean": stats.get("adapter/gate_mean"),
    }
    print(json.dumps(payload, sort_keys=True), flush=True)


def goal_selection_score(stats: dict[str, float | int]) -> float:
    mpjpe = float(stats.get("corrected/camera_mpjpe_mm", float("inf")))
    pa = float(stats.get("corrected/camera_pa_mpjpe_mm", float("inf")))
    return 100.0 * max(0.0, mpjpe - 32.0) + 100.0 * max(0.0, pa - 22.0) + 0.01 * (mpjpe + pa)


def _safe_len(value) -> int | None:
    try:
        return len(value)
    except TypeError:
        return None


if __name__ == "__main__":
    main()
