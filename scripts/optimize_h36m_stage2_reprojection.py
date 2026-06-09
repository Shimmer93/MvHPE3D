#!/usr/bin/env python
"""Offline H36M Stage 2 reprojection optimizer diagnostic."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mvhpe3d.utils import load_experiment_config  # noqa: E402
from test import build_data_config, build_datamodule, build_model_config, load_eval_module  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--stage", choices=("val", "test"), default="val")
    parser.add_argument("--smpl-model-path", default="data/weights/SMPL_NEUTRAL.pkl")
    parser.add_argument("--input-smpl-cache-dir", default="data/h36m/sam3dbody_fitted_smpl")
    parser.add_argument(
        "--image-measurement-cache-dir",
        default="data/h36m/image_measurements_keypointrcnn_heatmap_grid5_uv_h36m_lrfix_direct12",
    )
    parser.add_argument("--stage2-checkpoint-path", default=None)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--gt-smpl-dir", default=None)
    parser.add_argument("--cameras-dir", default=None)
    parser.add_argument("--split-config-path", default=None)
    parser.add_argument("--split-name", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2.0e-2)
    parser.add_argument("--betas-lr", type=float, default=2.0e-3)
    parser.add_argument("--freeze-body-pose", action="store_true")
    parser.add_argument("--optimize-betas", action="store_true")
    parser.add_argument("--optimize-global-orient", action="store_true")
    parser.add_argument("--global-orient-lr", type=float, default=2.0e-3)
    parser.add_argument("--optimize-transl", action="store_true")
    parser.add_argument("--transl-lr", type=float, default=1.0e-3)
    parser.add_argument("--projection-weight", type=float, default=1.0)
    parser.add_argument(
        "--projection-target",
        choices=("image", "gt"),
        default="image",
        help="Use cached image measurements or GT 2D joints as reprojection targets.",
    )
    parser.add_argument("--pose-delta-weight", type=float, default=2.0e-2)
    parser.add_argument("--betas-delta-weight", type=float, default=1.0e-2)
    parser.add_argument("--global-orient-delta-weight", type=float, default=1.0e2)
    parser.add_argument("--transl-delta-weight", type=float, default=1.0e2)
    parser.add_argument("--joint-anchor-weight", type=float, default=2.0)
    parser.add_argument("--huber-beta-px", type=float, default=10.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.2)
    parser.add_argument("--min-depth", type=float, default=1.0e-3)
    parser.add_argument("--border-px", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    experiment = load_experiment_config(args.config)
    helper_args = build_helper_args(args)
    data_config = build_data_config(experiment["data"], helper_args)
    data_config.eval_batch_size = int(args.batch_size)
    data_config.num_workers = int(args.num_workers)
    data_config.drop_last_eval = False
    data_config.shuffle_train_views = False
    model_config = build_model_config(experiment["model"], checkpoint_path=args.checkpoint_path)

    datamodule = build_datamodule(data_config)
    datamodule.prepare_data()
    datamodule.setup("test" if args.stage == "test" else "validate")
    dataloader = datamodule.test_dataloader() if args.stage == "test" else datamodule.val_dataloader()

    module = load_eval_module(
        checkpoint_path=args.checkpoint_path,
        model_config=model_config,
        data_config=data_config,
        args=helper_args,
    )
    module.to(device)
    module.eval()
    module.requires_grad_(False)

    accum = MetricAccumulator()
    sample_count = 0
    batch_count = 0
    for batch in dataloader:
        batch = move_to_device(batch, device)
        with torch.no_grad():
            predictions = run_module_forward(module, batch)
        stats = optimize_batch(module=module, predictions=predictions, batch=batch, args=args)
        accum.update(stats, batch_size=int(batch["views_input"].shape[0]))
        sample_count += int(batch["views_input"].shape[0])
        batch_count += 1
        if args.max_batches is not None and batch_count >= args.max_batches:
            break

    payload = {
        "config": str(Path(args.config).resolve()),
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "image_measurement_cache_dir": str(Path(args.image_measurement_cache_dir).resolve()),
        "stage": args.stage,
        "samples": sample_count,
        "batches": batch_count,
        "optimizer": {
            "steps": int(args.steps),
            "lr": float(args.lr),
            "betas_lr": float(args.betas_lr),
            "freeze_body_pose": bool(args.freeze_body_pose),
            "optimize_betas": bool(args.optimize_betas),
            "optimize_global_orient": bool(args.optimize_global_orient),
            "global_orient_lr": float(args.global_orient_lr),
            "optimize_transl": bool(args.optimize_transl),
            "transl_lr": float(args.transl_lr),
            "projection_weight": float(args.projection_weight),
            "projection_target": str(args.projection_target),
            "pose_delta_weight": float(args.pose_delta_weight),
            "betas_delta_weight": float(args.betas_delta_weight),
            "global_orient_delta_weight": float(args.global_orient_delta_weight),
            "transl_delta_weight": float(args.transl_delta_weight),
            "joint_anchor_weight": float(args.joint_anchor_weight),
            "huber_beta_px": float(args.huber_beta_px),
            "confidence_threshold": float(args.confidence_threshold),
            "min_depth": float(args.min_depth),
            "border_px": float(args.border_px),
        },
        "metrics": accum.finalize(),
        "notes": {
            "purpose": "Offline diagnostic only. Uses frozen Stage2 output as initialization and optimizes per-sample SMPL residuals.",
            "model_side": "Projection uses SAM3DBody input_global_orient/input_transl/intrinsics, matching Stage2 image projection.",
            "target_side": "3D metrics use the Stage2 H36M/HeatFormer external-joint metric path.",
            "gt_2d_error": "GT 2D is used by the optimizer only when projection_target='gt'.",
        },
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True), flush=True)
    print(f"Saved diagnostics to {output_path}", flush=True)


def build_helper_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint_path=args.checkpoint_path,
        config=args.config,
        manifest_path=args.manifest_path,
        gt_smpl_dir=args.gt_smpl_dir,
        cameras_dir=args.cameras_dir,
        split_config_path=args.split_config_path,
        split_name=args.split_name,
        stage=args.stage,
        default_root_dir=None,
        accelerator=None,
        devices="1",
        strategy=None,
        num_nodes=None,
        seed=None,
        smpl_model_path=args.smpl_model_path,
        mhr_assets_dir=None,
        input_smpl_cache_dir=args.input_smpl_cache_dir,
        rgb_feature_cache_dir=None,
        image_measurement_cache_dir=args.image_measurement_cache_dir,
        stage2_checkpoint_path=args.stage2_checkpoint_path,
        pred_camera_mode="input_corrected",
        output_path=None,
    )


def run_module_forward(module, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    view_aux = batch.get("view_aux")
    return module(
        batch["views_input"],
        view_rgb_feature=batch.get("view_rgb_feature"),
        view_image_joint_feature=batch.get("view_image_joint_feature"),
        view_image_joint_valid=batch.get("view_image_joint_valid"),
        view_image_joint_confidence=batch.get("view_image_joint_confidence"),
        view_image_joint_uv=batch.get("view_image_joint_uv"),
        view_image_joint_projected_uv=batch.get("view_image_joint_projected_uv"),
        view_image_mask_feature=batch.get("view_image_mask_feature"),
        view_image_size=view_aux.get("image_size") if view_aux is not None else None,
        view_aux=view_aux,
        target_joint_smpl_indices=batch.get("target_joint_smpl_indices"),
        target_joint_root_index=batch.get("target_joint_root_index"),
    )


def optimize_batch(
    *,
    module,
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, float | int]:
    init_body_pose = predictions["pred_body_pose"].detach()
    init_betas = predictions["pred_betas"].detach()
    init_global_orient = batch["view_aux"]["input_global_orient"].detach()
    init_transl = batch["view_aux"]["input_transl"].detach()
    pose_delta = torch.zeros_like(init_body_pose, requires_grad=not args.freeze_body_pose)
    parameters: list[dict[str, Any]] = []
    if not args.freeze_body_pose:
        parameters.append({"params": [pose_delta], "lr": float(args.lr)})
    if args.optimize_betas:
        betas_delta = torch.zeros_like(init_betas, requires_grad=True)
        parameters.append({"params": [betas_delta], "lr": float(args.betas_lr)})
    else:
        betas_delta = torch.zeros_like(init_betas)
    if args.optimize_global_orient:
        global_orient_delta = torch.zeros_like(init_global_orient, requires_grad=True)
        parameters.append(
            {"params": [global_orient_delta], "lr": float(args.global_orient_lr)}
        )
    else:
        global_orient_delta = torch.zeros_like(init_global_orient)
    if args.optimize_transl:
        transl_delta = torch.zeros_like(init_transl, requires_grad=True)
        parameters.append({"params": [transl_delta], "lr": float(args.transl_lr)})
    else:
        transl_delta = torch.zeros_like(init_transl)
    if not parameters:
        raise ValueError("No optimization variables are enabled.")
    optimizer = torch.optim.Adam(parameters)

    with torch.no_grad():
        init_eval_joints = build_eval_joints(module, init_body_pose, init_betas, batch)
        stage2_projection = project_body(module, init_body_pose, init_betas, batch)
        stage2_metrics = evaluate_state(
            module=module,
            batch=batch,
            body_pose=init_body_pose,
            betas=init_betas,
            projection=stage2_projection,
            prefix="stage2",
        )

    final_loss = None
    final_projection_loss = None
    final_pose_loss = None
    final_betas_loss = None
    final_global_orient_loss = None
    final_transl_loss = None
    final_anchor_loss = None
    for _ in range(int(args.steps)):
        optimizer.zero_grad(set_to_none=True)
        body_pose = init_body_pose + pose_delta
        betas = init_betas + betas_delta
        projection = project_body(
            module,
            body_pose,
            betas,
            batch,
            global_orient_delta=global_orient_delta,
            transl_delta=transl_delta,
        )
        projection_loss = compute_heatmap_projection_loss(
            pred_uv=projection["uv"],
            pred_depth=projection["depth"],
            batch=batch,
            confidence_threshold=float(args.confidence_threshold),
            min_depth=float(args.min_depth),
            border_px=float(args.border_px),
            huber_beta_px=float(args.huber_beta_px),
            projection_target=str(args.projection_target),
        )
        pose_loss = pose_delta.square().mean()
        betas_loss = betas_delta.square().mean()
        global_orient_loss = global_orient_delta.square().mean()
        transl_loss = transl_delta.square().mean()
        eval_joints = build_eval_joints(module, body_pose, betas, batch)
        anchor_loss = (eval_joints - init_eval_joints).square().mean()
        loss = (
            float(args.projection_weight) * projection_loss
            + float(args.pose_delta_weight) * pose_loss
            + float(args.betas_delta_weight) * betas_loss
            + float(args.global_orient_delta_weight) * global_orient_loss
            + float(args.transl_delta_weight) * transl_loss
            + float(args.joint_anchor_weight) * anchor_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [item for group in parameters for item in group["params"]],
            5.0,
        )
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())
        final_projection_loss = float(projection_loss.detach().cpu().item())
        final_pose_loss = float(pose_loss.detach().cpu().item())
        final_betas_loss = float(betas_loss.detach().cpu().item())
        final_global_orient_loss = float(global_orient_loss.detach().cpu().item())
        final_transl_loss = float(transl_loss.detach().cpu().item())
        final_anchor_loss = float(anchor_loss.detach().cpu().item())

    with torch.no_grad():
        opt_body_pose = init_body_pose + pose_delta.detach()
        opt_betas = init_betas + betas_delta.detach()
        opt_global_orient_delta = global_orient_delta.detach()
        opt_transl_delta = transl_delta.detach()
        opt_projection = project_body(
            module,
            opt_body_pose,
            opt_betas,
            batch,
            global_orient_delta=opt_global_orient_delta,
            transl_delta=opt_transl_delta,
        )
        opt_metrics = evaluate_state(
            module=module,
            batch=batch,
            body_pose=opt_body_pose,
            betas=opt_betas,
            projection=opt_projection,
            prefix="optimized",
            global_orient_delta=opt_global_orient_delta,
            transl_delta=opt_transl_delta,
        )
        delta_pose_abs = pose_delta.detach().abs().mean()
        delta_betas_abs = betas_delta.detach().abs().mean()
        delta_global_orient_abs = opt_global_orient_delta.abs().mean()
        delta_transl_abs = opt_transl_delta.abs().mean()

    result: dict[str, float | int] = {}
    result.update(stage2_metrics)
    result.update(opt_metrics)
    result["optimization/loss"] = final_loss if final_loss is not None else 0.0
    result["optimization/projection_loss"] = (
        final_projection_loss if final_projection_loss is not None else 0.0
    )
    result["optimization/pose_delta_l2"] = final_pose_loss if final_pose_loss is not None else 0.0
    result["optimization/betas_delta_l2"] = (
        final_betas_loss if final_betas_loss is not None else 0.0
    )
    result["optimization/global_orient_delta_l2"] = (
        final_global_orient_loss if final_global_orient_loss is not None else 0.0
    )
    result["optimization/transl_delta_l2"] = (
        final_transl_loss if final_transl_loss is not None else 0.0
    )
    result["optimization/joint_anchor_l2"] = (
        final_anchor_loss if final_anchor_loss is not None else 0.0
    )
    result["optimization/mean_abs_pose_delta"] = float(delta_pose_abs.detach().cpu().item())
    result["optimization/mean_abs_betas_delta"] = float(delta_betas_abs.detach().cpu().item())
    result["optimization/mean_abs_global_orient_delta"] = float(
        delta_global_orient_abs.detach().cpu().item()
    )
    result["optimization/mean_abs_transl_delta"] = float(
        delta_transl_abs.detach().cpu().item()
    )
    return result


def build_eval_joints(
    module,
    body_pose: torch.Tensor,
    betas: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    batch_size = body_pose.shape[0]
    zero_root = torch.zeros((batch_size, 3), device=body_pose.device, dtype=body_pose.dtype)
    smpl_output = module._build_smpl_output(
        body_pose=body_pose,
        betas=betas,
        global_orient=zero_root,
        transl=torch.zeros_like(zero_root),
    )
    smpl_indices = batch["target_joint_smpl_indices"][0].to(device=body_pose.device, dtype=torch.long)
    joints = module._select_external_pred_joints(
        smpl_output=smpl_output,
        smpl_indices=smpl_indices,
    )
    root_index = int(batch["target_joint_root_index"].reshape(-1)[0].item())
    return joints - joints[:, root_index : root_index + 1, :]


def project_body(
    module,
    body_pose: torch.Tensor,
    betas: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    global_orient_delta: torch.Tensor | None = None,
    transl_delta: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    batch_size, num_views = batch["views_input"].shape[:2]
    view_aux = batch["view_aux"]
    camera_global_orient = view_aux["input_global_orient"].to(
        device=body_pose.device,
        dtype=body_pose.dtype,
    )
    if global_orient_delta is not None:
        camera_global_orient = camera_global_orient + global_orient_delta.to(
            device=body_pose.device,
            dtype=body_pose.dtype,
        )
    camera_global_orient = camera_global_orient.reshape(batch_size * num_views, 3)
    camera_transl = view_aux["input_transl"].to(
        device=body_pose.device,
        dtype=body_pose.dtype,
    )
    if transl_delta is not None:
        camera_transl = camera_transl + transl_delta.to(
            device=body_pose.device,
            dtype=body_pose.dtype,
        )
    camera_transl = camera_transl.reshape(batch_size * num_views, 3)
    smpl_output = module._build_smpl_output(
        body_pose=body_pose[:, None, :].expand(batch_size, num_views, -1).reshape(batch_size * num_views, -1),
        betas=betas[:, None, :].expand(batch_size, num_views, -1).reshape(batch_size * num_views, -1),
        global_orient=camera_global_orient,
        transl=camera_transl,
    )
    smpl_indices = batch["target_joint_smpl_indices"][0].to(device=body_pose.device, dtype=torch.long)
    camera_joints = module._select_external_pred_joints(
        smpl_output=smpl_output,
        smpl_indices=smpl_indices,
    )
    camera_joints = camera_joints.reshape(batch_size, num_views, camera_joints.shape[1], 3)
    uv, depth = module._project_camera_joints(
        camera_joints,
        intrinsics=view_aux["cam_int"].to(device=body_pose.device, dtype=body_pose.dtype),
    )
    return {"camera_joints": camera_joints, "uv": uv, "depth": depth}


def compute_heatmap_projection_loss(
    *,
    pred_uv: torch.Tensor,
    pred_depth: torch.Tensor,
    batch: dict[str, torch.Tensor],
    confidence_threshold: float,
    min_depth: float,
    border_px: float,
    huber_beta_px: float,
    projection_target: str,
) -> torch.Tensor:
    if projection_target == "gt":
        heat_uv = batch["target_joints_2d"].to(device=pred_uv.device, dtype=pred_uv.dtype)
        heat_conf = batch.get("target_joints_2d_confidence")
        if heat_conf is None:
            heat_conf = torch.ones(
                heat_uv.shape[:-1],
                device=pred_uv.device,
                dtype=pred_uv.dtype,
            )
        else:
            heat_conf = heat_conf.to(device=pred_uv.device, dtype=pred_uv.dtype)
        heat_valid = torch.isfinite(heat_uv).all(dim=-1)
    else:
        heat_uv = batch["view_image_joint_uv"].to(device=pred_uv.device, dtype=pred_uv.dtype)
        heat_valid = batch["view_image_joint_valid"].to(device=pred_uv.device, dtype=torch.bool)
        heat_conf = batch["view_image_joint_confidence"].to(device=pred_uv.device, dtype=pred_uv.dtype)
    image_size = batch["view_aux"]["image_size"].to(device=pred_uv.device, dtype=pred_uv.dtype)
    valid = build_image_valid_mask(
        pred_uv=pred_uv,
        pred_depth=pred_depth,
        target_uv=heat_uv,
        target_valid=heat_valid,
        target_confidence=heat_conf,
        image_size=image_size,
        confidence_threshold=confidence_threshold,
        min_depth=min_depth,
        border_px=border_px,
    )
    if not bool(valid.any()):
        return pred_uv.sum() * 0.0
    diff = pred_uv - heat_uv
    per_coord = F.smooth_l1_loss(
        diff,
        torch.zeros_like(diff),
        beta=float(huber_beta_px),
        reduction="none",
    )
    per_joint = per_coord.sum(dim=-1)
    return per_joint[valid].mean()


def evaluate_state(
    *,
    module,
    batch: dict[str, torch.Tensor],
    body_pose: torch.Tensor,
    betas: torch.Tensor,
    projection: dict[str, torch.Tensor],
    prefix: str,
    global_orient_delta: torch.Tensor | None = None,
    transl_delta: torch.Tensor | None = None,
) -> dict[str, float | int]:
    pred_joints = build_eval_joints(module, body_pose, betas, batch)
    view_aux = build_metric_view_aux(
        batch,
        global_orient_delta=global_orient_delta,
        transl_delta=transl_delta,
    )
    metrics = module._compute_external_joint_metrics(
        pred_joints=pred_joints,
        target_joints=batch["target_joints"],
        joint_weight=batch["target_joint_confidence"],
        target_joint_root_index=batch.get("target_joint_root_index"),
        target_camera_joints=batch.get("target_camera_joints"),
        target_camera_joint_confidence=batch.get("target_camera_joint_confidence"),
        view_aux=view_aux,
    )
    result = {
        f"{prefix}/mpjpe": float(metrics["mpjpe"].detach().cpu().item()),
        f"{prefix}/pa_mpjpe": float(metrics["pa_mpjpe"].detach().cpu().item()),
        f"{prefix}/pck_150": float(metrics.get("pck_150", torch.zeros(())).detach().cpu().item()),
        f"{prefix}/auc": float(metrics.get("auc", torch.zeros(())).detach().cpu().item()),
    }
    result.update(pixel_error_metrics(prefix=prefix, projection=projection, batch=batch))
    return result


def build_metric_view_aux(
    batch: dict[str, torch.Tensor],
    *,
    global_orient_delta: torch.Tensor | None,
    transl_delta: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    view_aux = dict(batch["view_aux"])
    if global_orient_delta is not None:
        view_aux["input_global_orient"] = (
            view_aux["input_global_orient"].to(
                device=global_orient_delta.device,
                dtype=global_orient_delta.dtype,
            )
            + global_orient_delta
        )
    if transl_delta is not None:
        view_aux["input_transl"] = (
            view_aux["input_transl"].to(device=transl_delta.device, dtype=transl_delta.dtype)
            + transl_delta
        )
    return view_aux


def pixel_error_metrics(
    *,
    prefix: str,
    projection: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, float | int]:
    pred_uv = projection["uv"]
    pred_depth = projection["depth"]
    image_size = batch["view_aux"]["image_size"].to(device=pred_uv.device, dtype=pred_uv.dtype)
    heat_uv = batch["view_image_joint_uv"].to(device=pred_uv.device, dtype=pred_uv.dtype)
    heat_valid = batch["view_image_joint_valid"].to(device=pred_uv.device, dtype=torch.bool)
    heat_conf = batch["view_image_joint_confidence"].to(device=pred_uv.device, dtype=pred_uv.dtype)
    target_uv = batch["target_joints_2d"].to(device=pred_uv.device, dtype=pred_uv.dtype)
    target_conf = batch["target_joints_2d_confidence"].to(device=pred_uv.device, dtype=pred_uv.dtype)
    pred_finite = torch.isfinite(pred_uv).all(dim=-1) & (pred_depth > 1.0e-3)
    heat_mask = build_image_valid_mask(
        pred_uv=pred_uv,
        pred_depth=pred_depth,
        target_uv=heat_uv,
        target_valid=heat_valid,
        target_confidence=heat_conf,
        image_size=image_size,
        confidence_threshold=0.0,
        min_depth=1.0e-3,
        border_px=0.0,
    )
    gt_mask = build_image_valid_mask(
        pred_uv=pred_uv,
        pred_depth=pred_depth,
        target_uv=target_uv,
        target_valid=target_conf > 0.05,
        target_confidence=target_conf,
        image_size=image_size,
        confidence_threshold=0.05,
        min_depth=1.0e-3,
        border_px=0.0,
    )
    heat_error = torch.linalg.norm(pred_uv - heat_uv, dim=-1)
    gt_error = torch.linalg.norm(pred_uv - target_uv, dim=-1)
    return {
        f"{prefix}/heatmap_error_px": masked_mean(heat_error, heat_mask),
        f"{prefix}/gt_2d_error_px": masked_mean(gt_error, gt_mask),
        f"{prefix}/heatmap_valid_count": int(heat_mask.sum().detach().cpu().item()),
        f"{prefix}/gt_2d_valid_count": int(gt_mask.sum().detach().cpu().item()),
        f"{prefix}/finite_projection_count": int(pred_finite.sum().detach().cpu().item()),
    }


def build_image_valid_mask(
    *,
    pred_uv: torch.Tensor,
    pred_depth: torch.Tensor,
    target_uv: torch.Tensor,
    target_valid: torch.Tensor,
    target_confidence: torch.Tensor,
    image_size: torch.Tensor,
    confidence_threshold: float,
    min_depth: float,
    border_px: float,
) -> torch.Tensor:
    width = image_size[..., 0].clamp_min(1.0)[..., None]
    height = image_size[..., 1].clamp_min(1.0)[..., None]
    return (
        torch.isfinite(pred_uv).all(dim=-1)
        & torch.isfinite(target_uv).all(dim=-1)
        & (pred_depth > min_depth)
        & target_valid
        & (target_confidence > confidence_threshold)
        & (target_uv[..., 0] >= border_px)
        & (target_uv[..., 0] <= width - 1.0 - border_px)
        & (target_uv[..., 1] >= border_px)
        & (target_uv[..., 1] <= height - 1.0 - border_px)
    )


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return 0.0
    return float(values[mask].detach().float().mean().cpu().item())


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


class MetricAccumulator:
    def __init__(self) -> None:
        self.weighted_sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.total_batch_weight = 0

    def update(self, stats: dict[str, float | int], *, batch_size: int) -> None:
        self.total_batch_weight += int(batch_size)
        for key, value in stats.items():
            if key.endswith("_count"):
                self.counts[key] = self.counts.get(key, 0) + int(value)
                continue
            self.weighted_sums[key] = self.weighted_sums.get(key, 0.0) + float(value) * batch_size

    def finalize(self) -> dict[str, float | int]:
        denom = max(1, self.total_batch_weight)
        result = {
            key: value / denom for key, value in sorted(self.weighted_sums.items())
        }
        result.update(dict(sorted(self.counts.items())))
        result["sample_weight"] = self.total_batch_weight
        if "stage2/mpjpe" in result and "optimized/mpjpe" in result:
            result["delta/mpjpe"] = result["optimized/mpjpe"] - result["stage2/mpjpe"]
        if "stage2/pa_mpjpe" in result and "optimized/pa_mpjpe" in result:
            result["delta/pa_mpjpe"] = result["optimized/pa_mpjpe"] - result["stage2/pa_mpjpe"]
        if "stage2/heatmap_error_px" in result and "optimized/heatmap_error_px" in result:
            result["delta/heatmap_error_px"] = (
                result["optimized/heatmap_error_px"] - result["stage2/heatmap_error_px"]
            )
        if "stage2/gt_2d_error_px" in result and "optimized/gt_2d_error_px" in result:
            result["delta/gt_2d_error_px"] = (
                result["optimized/gt_2d_error_px"] - result["stage2/gt_2d_error_px"]
            )
        return result


if __name__ == "__main__":
    main()
