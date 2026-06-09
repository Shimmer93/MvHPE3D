#!/usr/bin/env python
"""Train a frozen-body H36M Stage 2.1 root correction adapter."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
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

from mvhpe3d.models.stage2.root_correction_adapter import (  # noqa: E402
    Stage21RootCorrectionAdapterConfig,
    Stage21RootCorrectionAdapterModel,
)
from mvhpe3d.metrics import batch_similarity_align  # noqa: E402
from mvhpe3d.utils import axis_angle_to_matrix, load_experiment_config, matrix_to_axis_angle  # noqa: E402
from optimize_h36m_stage2_reprojection import move_to_device  # noqa: E402
from test import build_data_config, build_datamodule, build_model_config, load_eval_module  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage2-checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smpl-model-path", default="data/weights/SMPL_NEUTRAL.pkl")
    parser.add_argument("--input-smpl-cache-dir", default="data/h36m/sam3dbody_fitted_smpl")
    parser.add_argument(
        "--image-measurement-cache-dir",
        default="data/h36m/image_measurements_keypointrcnn_heatmap_grid5_uv_h36m_lrfix_direct12",
    )
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--gt-smpl-dir", default=None)
    parser.add_argument("--cameras-dir", default=None)
    parser.add_argument("--split-config-path", default=None)
    parser.add_argument("--split-name", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--train-max-batches", type=int, default=None)
    parser.add_argument("--val-max-batches", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--global-orient-delta-scale", type=float, default=0.05)
    parser.add_argument("--transl-delta-scale", type=float, default=0.05)
    parser.add_argument("--compose-global-orient-delta", action="store_true")
    parser.add_argument("--disable-measurement-residual", action="store_true")
    parser.add_argument("--disable-measurement-confidence", action="store_true")
    parser.add_argument("--camera-joint-weight", type=float, default=1.0)
    parser.add_argument("--gt-projection-weight", type=float, default=0.05)
    parser.add_argument("--image-projection-weight", type=float, default=0.0)
    parser.add_argument("--global-orient-delta-weight", type=float, default=1.0)
    parser.add_argument("--transl-delta-weight", type=float, default=10.0)
    parser.add_argument("--camera-huber-beta-m", type=float, default=0.02)
    parser.add_argument("--projection-charbonnier-eps", type=float, default=1.0e-3)
    parser.add_argument("--projection-min-depth", type=float, default=0.1)
    parser.add_argument("--projection-border-px", type=float, default=0.0)
    parser.add_argument("--image-confidence-threshold", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment = load_experiment_config(args.config)
    helper_args = build_helper_args(args)
    data_config = build_data_config(experiment["data"], helper_args)
    data_config.batch_size = int(args.batch_size)
    data_config.eval_batch_size = int(args.batch_size)
    data_config.num_workers = int(args.num_workers)
    data_config.drop_last_train = False
    data_config.drop_last_eval = False
    data_config.shuffle_train_views = True

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
    stage2_module.eval()
    stage2_module.requires_grad_(False)

    adapter_config = Stage21RootCorrectionAdapterConfig(
        view_input_dim=int(model_config.input_dim),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        global_orient_delta_scale=float(args.global_orient_delta_scale),
        transl_delta_scale=float(args.transl_delta_scale),
        use_measurement_residual=not bool(args.disable_measurement_residual),
        use_measurement_confidence=not bool(args.disable_measurement_confidence),
    )
    adapter = Stage21RootCorrectionAdapterModel(adapter_config).to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    history = []
    best_val_mpjpe = float("inf")
    for epoch in range(int(args.max_epochs)):
        train_stats = run_epoch(
            stage2_module=stage2_module,
            adapter=adapter,
            dataloader=train_loader,
            args=args,
            device=device,
            optimizer=optimizer,
            train=True,
            max_batches=args.train_max_batches,
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
        )
        row = {"epoch": epoch, "train": train_stats, "val": val_stats}
        history.append(row)
        print(json.dumps(row, indent=2, sort_keys=True), flush=True)
        val_mpjpe = float(val_stats.get("corrected/camera_mpjpe_mm", float("inf")))
        if val_mpjpe < best_val_mpjpe:
            best_val_mpjpe = val_mpjpe
            save_checkpoint(
                output_dir / "adapter_best.pt",
                adapter=adapter,
                adapter_config=adapter_config,
                args=args,
                epoch=epoch,
                stats=val_stats,
            )

    save_checkpoint(
        output_dir / "adapter_last.pt",
        adapter=adapter,
        adapter_config=adapter_config,
        args=args,
        epoch=int(args.max_epochs) - 1,
        stats=history[-1]["val"] if history else {},
    )
    summary = {
        "config": str(Path(args.config).resolve()),
        "stage2_checkpoint_path": str(Path(args.stage2_checkpoint_path).resolve()),
        "adapter_config": asdict(adapter_config),
        "args": vars(args),
        "history": history,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(f"Saved Stage 2.1 adapter outputs to {output_dir}", flush=True)


def build_helper_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint_path=args.stage2_checkpoint_path,
        config=args.config,
        manifest_path=args.manifest_path,
        gt_smpl_dir=args.gt_smpl_dir,
        cameras_dir=args.cameras_dir,
        split_config_path=args.split_config_path,
        split_name=args.split_name,
        stage="val",
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


def run_epoch(
    *,
    stage2_module,
    adapter: Stage21RootCorrectionAdapterModel,
    dataloader,
    args: argparse.Namespace,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    train: bool,
    max_batches: int | None,
) -> dict[str, float | int]:
    adapter.train(train)
    accum = MetricAccumulator()
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(dataloader):
            batch = move_to_device(batch, device)
            with torch.no_grad():
                predictions = run_stage2_forward(stage2_module, batch)
                stage2_projection = build_camera_projection(
                    stage2_module=stage2_module,
                    predictions=predictions,
                    batch=batch,
                    global_orient_delta=None,
                    transl_delta=None,
                    compose_global_orient_delta=bool(args.compose_global_orient_delta),
                )
            adapter_outputs = adapter(
                views_input=batch["views_input"],
                view_aux=batch["view_aux"],
                stage2_uv=stage2_projection["uv"].detach(),
                measured_uv=batch.get("view_image_joint_uv"),
                measured_valid=batch.get("view_image_joint_valid"),
                measured_confidence=batch.get("view_image_joint_confidence"),
            )
            loss, stats = compute_losses_and_metrics(
                stage2_module=stage2_module,
                predictions=predictions,
                batch=batch,
                stage2_projection=stage2_projection,
                adapter_outputs=adapter_outputs,
                args=args,
            )
            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    adapter.parameters(),
                    float(args.gradient_clip_norm),
                )
                optimizer.step()
            accum.update(stats, batch_size=int(batch["views_input"].shape[0]))
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    return accum.finalize()


def run_stage2_forward(stage2_module, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    view_aux = batch.get("view_aux")
    return stage2_module(
        batch["views_input"],
        view_rgb_feature=batch.get("view_rgb_feature"),
        view_image_joint_feature=batch.get("view_image_joint_feature"),
        view_image_joint_valid=batch.get("view_image_joint_valid"),
        view_image_joint_confidence=batch.get("view_image_joint_confidence"),
        view_image_joint_uv=batch.get("view_image_joint_uv"),
        view_image_joint_projected_uv=batch.get("view_image_joint_projected_uv"),
        view_image_mask_feature=batch.get("view_image_mask_feature"),
        view_segmentation_mask=batch.get("view_segmentation_mask"),
        view_segmentation_distance=batch.get("view_segmentation_distance"),
        view_segmentation_valid=batch.get("view_segmentation_valid"),
        view_image_size=view_aux.get("image_size") if view_aux is not None else None,
        view_aux=view_aux,
        target_joint_smpl_indices=batch.get("target_joint_smpl_indices"),
        target_joint_root_index=batch.get("target_joint_root_index"),
    )


def compute_losses_and_metrics(
    *,
    stage2_module,
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    stage2_projection: dict[str, torch.Tensor],
    adapter_outputs: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    corrected_projection = build_camera_projection(
        stage2_module=stage2_module,
        predictions=predictions,
        batch=batch,
        global_orient_delta=adapter_outputs["pred_view_global_orient_delta"],
        transl_delta=adapter_outputs["pred_view_transl_delta"],
        compose_global_orient_delta=bool(args.compose_global_orient_delta),
    )
    camera_loss, corrected_camera_mpjpe = camera_joint_loss_and_error(
        pred_camera_joints=corrected_projection["camera_joints"],
        batch=batch,
        huber_beta_m=float(args.camera_huber_beta_m),
    )
    _, stage2_camera_mpjpe = camera_joint_loss_and_error(
        pred_camera_joints=stage2_projection["camera_joints"],
        batch=batch,
        huber_beta_m=float(args.camera_huber_beta_m),
    )
    corrected_camera_pa_mpjpe = camera_pa_mpjpe(
        pred_camera_joints=corrected_projection["camera_joints"],
        batch=batch,
    )
    stage2_camera_pa_mpjpe = camera_pa_mpjpe(
        pred_camera_joints=stage2_projection["camera_joints"],
        batch=batch,
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
    stage2_gt_projection_loss, stage2_gt_projection_error = projection_loss_and_error(
        pred_uv=stage2_projection["uv"],
        pred_depth=stage2_projection["depth"],
        batch=batch,
        target="gt",
        confidence_threshold=0.05,
        args=args,
    )
    stage2_image_projection_loss, stage2_image_projection_error = projection_loss_and_error(
        pred_uv=stage2_projection["uv"],
        pred_depth=stage2_projection["depth"],
        batch=batch,
        target="image",
        confidence_threshold=float(args.image_confidence_threshold),
        args=args,
    )
    global_delta_l2 = adapter_outputs["pred_view_global_orient_delta"].square().mean()
    transl_delta_l2 = adapter_outputs["pred_view_transl_delta"].square().mean()
    loss = (
        float(args.camera_joint_weight) * camera_loss
        + float(args.gt_projection_weight) * gt_projection_loss
        + float(args.image_projection_weight) * image_projection_loss
        + float(args.global_orient_delta_weight) * global_delta_l2
        + float(args.transl_delta_weight) * transl_delta_l2
    )
    stats = {
        "loss": float(loss.detach().cpu().item()),
        "loss/camera_joint": float(camera_loss.detach().cpu().item()),
        "loss/gt_projection": float(gt_projection_loss.detach().cpu().item()),
        "loss/image_projection": float(image_projection_loss.detach().cpu().item()),
        "loss/global_orient_delta_l2": float(global_delta_l2.detach().cpu().item()),
        "loss/transl_delta_l2": float(transl_delta_l2.detach().cpu().item()),
        "stage2/camera_mpjpe_mm": float(stage2_camera_mpjpe.detach().cpu().item()),
        "corrected/camera_mpjpe_mm": float(corrected_camera_mpjpe.detach().cpu().item()),
        "delta/camera_mpjpe_mm": float(
            (corrected_camera_mpjpe - stage2_camera_mpjpe).detach().cpu().item()
        ),
        "stage2/camera_pa_mpjpe_mm": float(stage2_camera_pa_mpjpe.detach().cpu().item()),
        "corrected/camera_pa_mpjpe_mm": float(
            corrected_camera_pa_mpjpe.detach().cpu().item()
        ),
        "delta/camera_pa_mpjpe_mm": float(
            (corrected_camera_pa_mpjpe - stage2_camera_pa_mpjpe).detach().cpu().item()
        ),
        "stage2/gt_projection_error_px": float(stage2_gt_projection_error.detach().cpu().item()),
        "corrected/gt_projection_error_px": float(gt_projection_error.detach().cpu().item()),
        "delta/gt_projection_error_px": float(
            (gt_projection_error - stage2_gt_projection_error).detach().cpu().item()
        ),
        "stage2/image_projection_error_px": float(
            stage2_image_projection_error.detach().cpu().item()
        ),
        "corrected/image_projection_error_px": float(
            image_projection_error.detach().cpu().item()
        ),
        "delta/image_projection_error_px": float(
            (image_projection_error - stage2_image_projection_error).detach().cpu().item()
        ),
        "delta/global_orient_abs_mean": float(
            adapter_outputs["pred_view_global_orient_delta"].detach().abs().mean().cpu().item()
        ),
        "delta/transl_abs_mean": float(
            adapter_outputs["pred_view_transl_delta"].detach().abs().mean().cpu().item()
        ),
    }
    return loss, stats


def build_camera_projection(
    *,
    stage2_module,
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    global_orient_delta: torch.Tensor | None,
    transl_delta: torch.Tensor | None,
    compose_global_orient_delta: bool = False,
) -> dict[str, torch.Tensor]:
    body_pose = predictions["pred_body_pose"]
    betas = predictions["pred_betas"]
    batch_size, num_views = batch["views_input"].shape[:2]
    view_aux = batch["view_aux"]
    camera_global_orient = view_aux["input_global_orient"].to(
        device=body_pose.device,
        dtype=body_pose.dtype,
    )
    if global_orient_delta is not None:
        update = global_orient_delta.to(
            device=body_pose.device,
            dtype=body_pose.dtype,
        )
        if compose_global_orient_delta:
            camera_global_orient = compose_global_orient_update(
                base_global_orient=camera_global_orient,
                global_orient_update=update,
            )
        else:
            camera_global_orient = camera_global_orient + update
    camera_transl = view_aux["input_transl"].to(device=body_pose.device, dtype=body_pose.dtype)
    if transl_delta is not None:
        camera_transl = camera_transl + transl_delta.to(
            device=body_pose.device,
            dtype=body_pose.dtype,
        )
    smpl_output = stage2_module._build_smpl_output(
        body_pose=body_pose[:, None, :].expand(batch_size, num_views, -1).reshape(
            batch_size * num_views,
            -1,
        ),
        betas=betas[:, None, :].expand(batch_size, num_views, -1).reshape(
            batch_size * num_views,
            -1,
        ),
        global_orient=camera_global_orient.reshape(batch_size * num_views, 3),
        transl=camera_transl.reshape(batch_size * num_views, 3),
    )
    smpl_indices = batch["target_joint_smpl_indices"][0].to(
        device=body_pose.device,
        dtype=torch.long,
    )
    camera_joints = stage2_module._select_external_pred_joints(
        smpl_output=smpl_output,
        smpl_indices=smpl_indices,
    ).reshape(batch_size, num_views, -1, 3)
    uv, depth = stage2_module._project_camera_joints(
        camera_joints,
        intrinsics=view_aux["cam_int"].to(device=body_pose.device, dtype=body_pose.dtype),
    )
    return {"camera_joints": camera_joints, "uv": uv, "depth": depth}


def compose_global_orient_update(
    *,
    base_global_orient: torch.Tensor,
    global_orient_update: torch.Tensor,
) -> torch.Tensor:
    composed = torch.matmul(
        axis_angle_to_matrix(global_orient_update),
        axis_angle_to_matrix(base_global_orient),
    )
    return matrix_to_axis_angle(composed).reshape_as(base_global_orient)


def camera_joint_loss_and_error(
    *,
    pred_camera_joints: torch.Tensor,
    batch: dict[str, torch.Tensor],
    huber_beta_m: float,
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
    target_root = target - target[:, :, root_index : root_index + 1, :]
    valid = confidence > 0.05
    weight = confidence * valid.to(dtype=confidence.dtype)
    diff = pred_root - target_root
    per_coord = F.smooth_l1_loss(
        diff,
        torch.zeros_like(diff),
        beta=float(huber_beta_m),
        reduction="none",
    )
    per_joint_loss = per_coord.sum(dim=-1)
    denom = weight.sum().clamp_min(1.0)
    loss = (per_joint_loss * weight).sum() / denom
    error_mm = torch.linalg.norm(diff.detach(), dim=-1) * 1000.0
    mean_error = (error_mm * valid.to(dtype=error_mm.dtype)).sum() / valid.sum().clamp_min(1)
    return loss, mean_error


def camera_pa_mpjpe(
    *,
    pred_camera_joints: torch.Tensor,
    batch: dict[str, torch.Tensor],
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
    valid = confidence.reshape(batch_size * num_views, joint_count) > 0.05
    weight = confidence.reshape(batch_size * num_views, joint_count) * valid.to(
        dtype=confidence.dtype
    )
    with torch.autocast(device_type=pred_camera_joints.device.type, enabled=False):
        aligned = batch_similarity_align(flat_pred.float(), flat_target.float()).to(
            dtype=pred_camera_joints.dtype
        )
    error_mm = torch.linalg.norm(aligned - flat_target, dim=-1) * 1000.0
    return (error_mm * weight).sum() / weight.sum().clamp_min(1.0)


def projection_loss_and_error(
    *,
    pred_uv: torch.Tensor,
    pred_depth: torch.Tensor,
    batch: dict[str, torch.Tensor],
    target: str,
    confidence_threshold: float,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target == "gt":
        target_uv = batch["target_joints_2d"].to(device=pred_uv.device, dtype=pred_uv.dtype)
        target_conf = batch["target_joints_2d_confidence"].to(
            device=pred_uv.device,
            dtype=pred_uv.dtype,
        )
        target_valid = target_conf > 0.05
    elif target == "image":
        target_uv = batch["view_image_joint_uv"].to(device=pred_uv.device, dtype=pred_uv.dtype)
        target_conf = batch["view_image_joint_confidence"].to(
            device=pred_uv.device,
            dtype=pred_uv.dtype,
        )
        target_valid = batch["view_image_joint_valid"].to(device=pred_uv.device, dtype=torch.bool)
    else:
        raise ValueError(f"Unknown projection target: {target!r}")
    view_aux = batch["view_aux"]
    image_size = view_aux["image_size"].to(device=pred_uv.device, dtype=pred_uv.dtype)
    border = float(args.projection_border_px)
    width = image_size[..., 0].clamp_min(1.0)[..., None]
    height = image_size[..., 1].clamp_min(1.0)[..., None]
    valid = (
        torch.isfinite(pred_uv).all(dim=-1)
        & torch.isfinite(target_uv).all(dim=-1)
        & (pred_depth > float(args.projection_min_depth))
        & target_valid
        & (target_conf > float(confidence_threshold))
        & (target_uv[..., 0] >= border)
        & (target_uv[..., 0] <= width - 1.0 - border)
        & (target_uv[..., 1] >= border)
        & (target_uv[..., 1] <= height - 1.0 - border)
    )
    valid_float = valid.to(dtype=pred_uv.dtype)
    focal_scale = (
        0.5
        * (
            view_aux["cam_int"].to(device=pred_uv.device, dtype=pred_uv.dtype)[..., 0, 0].abs()
            + view_aux["cam_int"].to(device=pred_uv.device, dtype=pred_uv.dtype)[..., 1, 1].abs()
        )
    ).clamp_min(1.0)
    normalized_delta = (pred_uv - target_uv) / focal_scale[..., None, None]
    normalized_delta = torch.where(
        valid[..., None],
        normalized_delta,
        torch.zeros_like(normalized_delta),
    )
    eps = float(args.projection_charbonnier_eps)
    per_joint_loss = torch.sqrt(normalized_delta.square().sum(dim=-1) + eps * eps) - eps
    denom = valid_float.sum().clamp_min(1.0)
    loss = (per_joint_loss * valid_float).sum() / denom
    pixel_error = torch.linalg.norm(pred_uv.detach() - target_uv.detach(), dim=-1)
    error = (pixel_error * valid_float).sum() / denom
    return loss, error


def save_checkpoint(
    path: Path,
    *,
    adapter: Stage21RootCorrectionAdapterModel,
    adapter_config: Stage21RootCorrectionAdapterConfig,
    args: argparse.Namespace,
    epoch: int,
    stats: dict[str, float | int],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "state_dict": adapter.state_dict(),
            "adapter_config": asdict(adapter_config),
            "args": vars(args),
            "stats": stats,
        },
        path,
    )


class MetricAccumulator:
    def __init__(self) -> None:
        self.weighted_sums: dict[str, float] = {}
        self.count = 0

    def update(self, stats: dict[str, float | int], *, batch_size: int) -> None:
        self.count += int(batch_size)
        for key, value in stats.items():
            self.weighted_sums[key] = self.weighted_sums.get(key, 0.0) + float(value) * batch_size

    def finalize(self) -> dict[str, float | int]:
        denom = max(1, self.count)
        result = {key: value / denom for key, value in sorted(self.weighted_sums.items())}
        result["sample_weight"] = self.count
        return result


if __name__ == "__main__":
    main()
