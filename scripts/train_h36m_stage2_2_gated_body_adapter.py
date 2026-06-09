#!/usr/bin/env python
"""Train a frozen-Stage2 H36M gated body adapter."""
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

from mvhpe3d.models.stage2.gated_body_adapter import (  # noqa: E402
    Stage22GatedBodyAdapterConfig,
    Stage22GatedBodyAdapterModel,
)
from mvhpe3d.utils import load_experiment_config  # noqa: E402
from optimize_h36m_stage2_reprojection import build_eval_joints, move_to_device  # noqa: E402
from test import build_data_config, build_datamodule, build_model_config, load_eval_module  # noqa: E402
from train_h36m_stage2_1_root_adapter import (  # noqa: E402
    build_camera_projection,
    camera_joint_loss_and_error,
    projection_loss_and_error,
    run_stage2_forward,
)


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
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--body-delta-scale", type=float, default=0.05)
    parser.add_argument("--gate-bias", type=float, default=-4.0)
    parser.add_argument("--camera-joint-weight", type=float, default=1.0)
    parser.add_argument("--gt-projection-weight", type=float, default=0.0)
    parser.add_argument("--image-projection-weight", type=float, default=0.0)
    parser.add_argument("--preserve-joint-weight", type=float, default=20.0)
    parser.add_argument("--do-no-harm-weight", type=float, default=2.0)
    parser.add_argument("--do-no-harm-margin-mm", type=float, default=0.0)
    parser.add_argument("--gate-sparsity-weight", type=float, default=0.05)
    parser.add_argument("--body-delta-weight", type=float, default=1.0)
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

    adapter_config = Stage22GatedBodyAdapterConfig(
        view_input_dim=int(model_config.input_dim),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        body_delta_scale=float(args.body_delta_scale),
        gate_bias=float(args.gate_bias),
    )
    adapter = Stage22GatedBodyAdapterModel(adapter_config).to(device)
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
    print(f"Saved Stage 2.2 adapter outputs to {output_dir}", flush=True)


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
    adapter: Stage22GatedBodyAdapterModel,
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
                )
            adapter_outputs = adapter(
                views_input=batch["views_input"],
                stage2_body_pose=predictions["pred_body_pose"],
                stage2_uv=stage2_projection["uv"].detach(),
                measured_uv=batch.get("view_image_joint_uv"),
                measured_valid=batch.get("view_image_joint_valid"),
                measured_confidence=batch.get("view_image_joint_confidence"),
                image_size=batch["view_aux"]["image_size"],
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


def compute_losses_and_metrics(
    *,
    stage2_module,
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    stage2_projection: dict[str, torch.Tensor],
    adapter_outputs: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    corrected_body_pose = predictions["pred_body_pose"] + adapter_outputs["pred_body_pose_update"]
    corrected_predictions = dict(predictions)
    corrected_predictions["pred_body_pose"] = corrected_body_pose
    corrected_projection = build_camera_projection(
        stage2_module=stage2_module,
        predictions=corrected_predictions,
        batch=batch,
        global_orient_delta=None,
        transl_delta=None,
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
    preserve_loss = canonical_preserve_loss(
        stage2_module=stage2_module,
        stage2_body_pose=predictions["pred_body_pose"],
        corrected_body_pose=corrected_body_pose,
        betas=predictions["pred_betas"],
        batch=batch,
    )
    do_no_harm_loss, harm_rate = do_no_harm_loss_and_rate(
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
    _, stage2_gt_projection_error = projection_loss_and_error(
        pred_uv=stage2_projection["uv"],
        pred_depth=stage2_projection["depth"],
        batch=batch,
        target="gt",
        confidence_threshold=0.05,
        args=args,
    )
    _, stage2_image_projection_error = projection_loss_and_error(
        pred_uv=stage2_projection["uv"],
        pred_depth=stage2_projection["depth"],
        batch=batch,
        target="image",
        confidence_threshold=float(args.image_confidence_threshold),
        args=args,
    )
    gate = adapter_outputs["pred_body_pose_gate"]
    delta = adapter_outputs["pred_body_pose_delta"]
    update = adapter_outputs["pred_body_pose_update"]
    gate_sparsity = gate.mean()
    body_delta_l2 = delta.square().mean()
    loss = (
        float(args.camera_joint_weight) * camera_loss
        + float(args.gt_projection_weight) * gt_projection_loss
        + float(args.image_projection_weight) * image_projection_loss
        + float(args.preserve_joint_weight) * preserve_loss
        + float(args.do_no_harm_weight) * do_no_harm_loss
        + float(args.gate_sparsity_weight) * gate_sparsity
        + float(args.body_delta_weight) * body_delta_l2
    )
    stats = {
        "loss": float(loss.detach().cpu().item()),
        "loss/camera_joint": float(camera_loss.detach().cpu().item()),
        "loss/preserve_joint": float(preserve_loss.detach().cpu().item()),
        "loss/do_no_harm": float(do_no_harm_loss.detach().cpu().item()),
        "loss/gt_projection": float(gt_projection_loss.detach().cpu().item()),
        "loss/image_projection": float(image_projection_loss.detach().cpu().item()),
        "loss/gate_sparsity": float(gate_sparsity.detach().cpu().item()),
        "loss/body_delta_l2": float(body_delta_l2.detach().cpu().item()),
        "stage2/camera_mpjpe_mm": float(stage2_camera_mpjpe.detach().cpu().item()),
        "corrected/camera_mpjpe_mm": float(corrected_camera_mpjpe.detach().cpu().item()),
        "delta/camera_mpjpe_mm": float(
            (corrected_camera_mpjpe - stage2_camera_mpjpe).detach().cpu().item()
        ),
        "stage2/gt_projection_error_px": float(stage2_gt_projection_error.detach().cpu().item()),
        "corrected/gt_projection_error_px": float(gt_projection_error.detach().cpu().item()),
        "delta/gt_projection_error_px": float(
            (gt_projection_error - stage2_gt_projection_error).detach().cpu().item()
        ),
        "stage2/image_projection_error_px": float(
            stage2_image_projection_error.detach().cpu().item()
        ),
        "corrected/image_projection_error_px": float(image_projection_error.detach().cpu().item()),
        "delta/image_projection_error_px": float(
            (image_projection_error - stage2_image_projection_error).detach().cpu().item()
        ),
        "adapter/gate_mean": float(gate.detach().mean().cpu().item()),
        "adapter/gate_max": float(gate.detach().amax().cpu().item()),
        "adapter/update_abs_mean": float(update.detach().abs().mean().cpu().item()),
        "adapter/delta_abs_mean": float(delta.detach().abs().mean().cpu().item()),
        "adapter/harm_rate": float(harm_rate.detach().cpu().item()),
    }
    return loss, stats


def canonical_preserve_loss(
    *,
    stage2_module,
    stage2_body_pose: torch.Tensor,
    corrected_body_pose: torch.Tensor,
    betas: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    stage2_joints = build_eval_joints(stage2_module, stage2_body_pose, betas, batch)
    corrected_joints = build_eval_joints(stage2_module, corrected_body_pose, betas, batch)
    return (corrected_joints - stage2_joints).square().mean()


def do_no_harm_loss_and_rate(
    *,
    stage2_camera_joints: torch.Tensor,
    corrected_camera_joints: torch.Tensor,
    batch: dict[str, torch.Tensor],
    margin_mm: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = batch["target_camera_joints"].to(
        device=stage2_camera_joints.device,
        dtype=stage2_camera_joints.dtype,
    )
    confidence = batch["target_camera_joint_confidence"].to(
        device=stage2_camera_joints.device,
        dtype=stage2_camera_joints.dtype,
    )
    root_index = int(batch["target_joint_root_index"].reshape(-1)[0].item())
    target_root = target - target[:, :, root_index : root_index + 1, :]
    stage2_root = stage2_camera_joints - stage2_camera_joints[:, :, root_index : root_index + 1, :]
    corrected_root = corrected_camera_joints - corrected_camera_joints[:, :, root_index : root_index + 1, :]
    valid = confidence > 0.05
    valid_float = valid.to(dtype=stage2_camera_joints.dtype)
    stage2_error = torch.linalg.norm(stage2_root - target_root, dim=-1)
    corrected_error = torch.linalg.norm(corrected_root - target_root, dim=-1)
    denom = valid_float.sum(dim=1).clamp_min(1.0)
    stage2_joint_error = (stage2_error * valid_float).sum(dim=1) / denom
    corrected_joint_error = (corrected_error * valid_float).sum(dim=1) / denom
    joint_valid = valid.any(dim=1)
    harm = corrected_joint_error - stage2_joint_error + float(margin_mm) * 0.001
    harm = torch.relu(harm)
    if not bool(joint_valid.any()):
        return harm.sum() * 0.0, harm.sum() * 0.0
    harm_loss = harm[joint_valid].mean()
    harm_rate = (
        corrected_joint_error[joint_valid] > stage2_joint_error[joint_valid] + 1.0e-8
    ).to(dtype=stage2_camera_joints.dtype).mean()
    return harm_loss, harm_rate


def save_checkpoint(
    path: Path,
    *,
    adapter: Stage22GatedBodyAdapterModel,
    adapter_config: Stage22GatedBodyAdapterConfig,
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
