#!/usr/bin/env python
"""Visualize predicted and GT SMPL meshes overlaid on RGB images."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import trimesh
from torch.utils.data import DataLoader

if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import pyrender

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mvhpe3d.data import Stage1DataConfig, Stage1HuMManDataModule, multiview_collate
from mvhpe3d.lightning import Stage1FusionLightningModule
from mvhpe3d.models import Stage1MLPFusionConfig, Stage1ResidualFusionConfig
from mvhpe3d.utils import (
    build_smpl_model,
    load_experiment_config,
    resolve_smpl_model_path as resolve_smpl_model_path_impl,
    validate_mhr_asset_folder,
)
from mvhpe3d.visualization import (
    load_camera_parameters,
    resolve_rgb_image_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save predicted-vs-GT SMPL mesh overlays for a trained Stage 1 checkpoint"
    )
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment/stage1_cross_camera.yaml",
        help="Path to the experiment YAML file",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Optional override for the data manifest path",
    )
    parser.add_argument(
        "--gt-smpl-dir",
        type=str,
        default=None,
        help="Optional override for the HuMMan GT SMPL directory",
    )
    parser.add_argument(
        "--split-config-path",
        type=str,
        default=None,
        help="Optional override for the split policy YAML path",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default=None,
        help="Optional override for the named split policy in the split config YAML",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=("train", "val", "test"),
        default="test",
        help="Which split to visualize",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/stage1_visualizations",
        help="Directory to save the visualizations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for inference, e.g. cuda:0 or cpu",
    )
    parser.add_argument(
        "--rgb-dir",
        type=str,
        default=None,
        help="Optional override for the HuMMan cropped RGB directory",
    )
    parser.add_argument(
        "--cameras-dir",
        type=str,
        default=None,
        help="Optional override for the HuMMan camera JSON directory used by the dataset loader",
    )
    parser.add_argument(
        "--mhr-assets-dir",
        type=str,
        default=None,
        help="Optional override for the MHR asset directory used to place fused predictions",
    )
    parser.add_argument(
        "--input-smpl-cache-dir",
        type=str,
        default=None,
        help="Optional directory for caching fitted SMPL parameters converted from input views",
    )
    parser.add_argument(
        "--smpl-model-path",
        type=str,
        default=None,
        help="Path to the neutral SMPL model .pkl file",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.55,
        help="Alpha used for semi-transparent mesh overlays",
    )
    parser.add_argument(
        "--pred-camera-mode",
        type=str,
        choices=("input", "gt"),
        default="input",
        help="Camera/root placement used for prediction overlays",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        choices=("first", "best_improvement"),
        default="first",
        help="How to choose which samples to visualize",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        choices=("mpjpe", "pa_mpjpe"),
        default="mpjpe",
        help="Metric used when --selection-mode=best_improvement",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=None,
        help="Optional minimum input-minus-pred improvement required for selection",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=1,
        help="Centered temporal smoothing window for predicted SMPL params; 1 disables smoothing",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional override for experiment seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = load_experiment_config(args.config)
    if args.smoothing_window < 1:
        raise ValueError(f"--smoothing-window must be >= 1, got {args.smoothing_window}")
    validate_mhr_asset_folder(args.mhr_assets_dir or "/opt/data/assets")
    data_config = build_data_config(experiment["data"], args)
    model_config = build_model_config(
        experiment["model"],
        checkpoint_path=args.checkpoint_path,
    )
    data_config.batch_size = 1
    data_config.drop_last_train = False

    if args.seed is not None:
        data_config.seed = args.seed

    datamodule = Stage1HuMManDataModule(data_config)
    datamodule.setup(None)
    dataset = select_dataset(datamodule, args.stage)
    device = resolve_device(args.device)
    module = Stage1FusionLightningModule.load_from_checkpoint(
        args.checkpoint_path,
        map_location="cpu",
        model_config=model_config,
        mhr_assets_dir=args.mhr_assets_dir,
        smpl_model_path=args.smpl_model_path,
        input_smpl_cache_dir=resolve_input_smpl_cache_dir(args, data_config),
        strict=False,
    )
    module.eval()
    module.to(device)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(data_config.manifest_path).resolve().parent
    rgb_dir = resolve_required_dir(args.rgb_dir, fallback=data_root / "rgb", name="rgb")
    smpl_model_path = resolve_smpl_model_path(args.smpl_model_path)
    smpl_model = build_smpl_model(
        device=device,
        smpl_model_path=str(smpl_model_path),
        batch_size=1,
    )
    faces = np.asarray(smpl_model.faces, dtype=np.int32)
    overlay_renderer = CameraMeshOverlayRenderer()
    grouped_indices = build_sequence_index(dataset)

    selected_entries = select_sample_entries(
        dataset=dataset,
        module=module,
        device=device,
        num_samples=args.num_samples,
        selection_mode=args.selection_mode,
        selection_metric=args.selection_metric,
        min_improvement=args.min_improvement,
    )

    written = 0
    summaries = []
    try:
        with torch.no_grad():
            for selected_entry in selected_entries:
                batch = multiview_collate([dataset[selected_entry["dataset_index"]]])
                views_input = batch["views_input"].to(device)
                predictions = module(views_input)

                sample_summary = save_sample_outputs(
                    output_dir=output_dir,
                    sample_index=written,
                    batch=batch,
                    predictions=predictions,
                    smpl_model=smpl_model,
                    faces=faces,
                    device=device,
                    rgb_dir=rgb_dir,
                    cameras_dir=Path(data_config.cameras_dir).resolve(),
                    overlay_alpha=args.overlay_alpha,
                    pred_camera_mode=args.pred_camera_mode,
                    module=module,
                    overlay_renderer=overlay_renderer,
                    dataset=dataset,
                    grouped_indices=grouped_indices,
                    dataset_index=int(selected_entry["dataset_index"]),
                    smoothing_window=args.smoothing_window,
                )
                sample_summary["selection"] = {
                    "dataset_index": selected_entry["dataset_index"],
                    "selection_mode": args.selection_mode,
                    "selection_metric": args.selection_metric,
                    "improvement": selected_entry["improvement"],
                    "pred_metric": selected_entry["pred_metric"],
                    "input_metric": selected_entry["input_metric"],
                }
                summaries.append(sample_summary)
                written += 1
    finally:
        overlay_renderer.close()

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({"samples": summaries}, indent=2), encoding="utf-8")
    print(f"Saved {written} visualization samples to {output_dir}")
    print(f"Saved summary to {summary_path}")


def build_data_config(config: dict[str, Any], args: argparse.Namespace) -> Stage1DataConfig:
    data_kwargs = dict(config)
    data_kwargs.pop("name", None)
    data_kwargs.pop("_config_path", None)

    if args.manifest_path is not None:
        data_kwargs["manifest_path"] = args.manifest_path
    if args.gt_smpl_dir is not None:
        data_kwargs["gt_smpl_dir"] = args.gt_smpl_dir
    if args.cameras_dir is not None:
        data_kwargs["cameras_dir"] = args.cameras_dir
    if args.split_config_path is not None:
        data_kwargs["split_config_path"] = args.split_config_path
    if args.split_name is not None:
        data_kwargs["split_name"] = args.split_name
    if args.seed is not None:
        data_kwargs["seed"] = args.seed

    return Stage1DataConfig(**data_kwargs)


def build_model_config(
    config: dict[str, Any],
    *,
    checkpoint_path: str,
) -> Stage1MLPFusionConfig | Stage1ResidualFusionConfig:
    model_kwargs = dict(config)
    model_name = str(model_kwargs.pop("name", "stage1_mlp_fusion"))
    model_kwargs.pop("_config_path", None)
    model_kwargs.update(load_checkpoint_model_overrides(checkpoint_path))
    if model_name == "stage1_residual_fusion":
        return Stage1ResidualFusionConfig(**model_kwargs)
    return Stage1MLPFusionConfig(**model_kwargs)


def load_checkpoint_model_overrides(checkpoint_path: str) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint.get("hyper_parameters", {}).get("model_config", {})
    if not isinstance(model_config, dict):
        return {}
    return dict(model_config)


def select_dataset(datamodule: Stage1HuMManDataModule, stage: str):
    if stage == "train":
        if datamodule.train_dataset is None:
            raise RuntimeError("train_dataset was not initialized")
        return datamodule.train_dataset
    if stage == "val":
        if datamodule.val_dataset is None:
            raise RuntimeError("val_dataset was not initialized")
        return datamodule.val_dataset
    if datamodule.test_dataset is None:
        raise RuntimeError("test_dataset was not initialized")
    return datamodule.test_dataset


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_required_dir(path_arg: str | None, *, fallback: Path, name: str) -> Path:
    resolved = Path(path_arg).resolve() if path_arg is not None else fallback.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{name} directory does not exist: {resolved}")
    return resolved


def resolve_smpl_model_path(path_arg: str | None) -> Path:
    return resolve_smpl_model_path_impl(path_arg)


def resolve_input_smpl_cache_dir(args: argparse.Namespace, data_config: Stage1DataConfig) -> str:
    if args.input_smpl_cache_dir is not None:
        return str(Path(args.input_smpl_cache_dir).resolve())
    manifest_parent = Path(data_config.manifest_path).resolve().parent
    return str((manifest_parent / "sam3dbody_fitted_smpl").resolve())


def build_sequence_index(dataset) -> dict[str, list[int]]:
    grouped: dict[str, list[tuple[int, int]]] = {}
    for index, record in enumerate(dataset.records):
        grouped.setdefault(record.sequence_id, []).append((int(record.frame_id), index))
    return {
        sequence_id: [index for _, index in sorted(items)]
        for sequence_id, items in grouped.items()
    }


def select_sample_entries(
    *,
    dataset,
    module: Stage1FusionLightningModule,
    device: torch.device,
    num_samples: int,
    selection_mode: str,
    selection_metric: str,
    min_improvement: float | None,
) -> list[dict[str, float | int]]:
    if selection_mode == "first":
        return [
            {
                "dataset_index": dataset_index,
                "improvement": 0.0,
                "pred_metric": 0.0,
                "input_metric": 0.0,
            }
            for dataset_index in range(min(num_samples, len(dataset)))
        ]

    candidates = []
    with torch.no_grad():
        for dataset_index in range(len(dataset)):
            batch = multiview_collate([dataset[dataset_index]])
            score = compute_selection_score(
                batch=batch,
                module=module,
                device=device,
                metric_name=selection_metric,
            )
            if min_improvement is not None and score["improvement"] < min_improvement:
                continue
            candidates.append({"dataset_index": dataset_index, **score})

    candidates.sort(key=lambda item: item["improvement"], reverse=True)
    return candidates[:num_samples]


def compute_selection_score(
    *,
    batch: dict[str, Any],
    module: Stage1FusionLightningModule,
    device: torch.device,
    metric_name: str,
) -> dict[str, float]:
    views_input = batch["views_input"].to(device)
    pred_cam_t = batch["view_aux"]["pred_cam_t"].to(device)
    predictions = module(views_input)
    input_view_smpl_params = module.convert_input_views_to_smpl(
        views_input=views_input,
        pred_cam_t=pred_cam_t,
        batch_meta=batch["meta"],
    )
    pred_metrics = module._compute_test_joint_metrics(
        pred_body_pose=predictions["pred_body_pose"],
        pred_betas=predictions["pred_betas"],
        target_body_pose=batch["target_body_pose"].to(device),
        target_betas=batch["target_betas"].to(device),
        target_aux={key: value.to(device) for key, value in batch["target_aux"].items()},
        pred_cam_t=pred_cam_t,
        input_view_smpl_params=input_view_smpl_params,
    )
    input_metrics = module._compute_input_view_joint_metrics(
        input_view_smpl_params=input_view_smpl_params,
        target_body_pose=batch["target_body_pose"].to(device),
        target_betas=batch["target_betas"].to(device),
        target_aux={key: value.to(device) for key, value in batch["target_aux"].items()},
        pred_cam_t=pred_cam_t,
    )
    pred_value = float(pred_metrics[metric_name].detach().cpu().item())
    input_value = float(input_metrics[metric_name].detach().cpu().item())
    return {
        "pred_metric": pred_value,
        "input_metric": input_value,
        "improvement": input_value - pred_value,
    }


def save_sample_outputs(
    *,
    output_dir: Path,
    sample_index: int,
    batch: dict[str, Any],
    predictions: dict[str, torch.Tensor],
    smpl_model,
    faces: np.ndarray,
    device: torch.device,
    rgb_dir: Path,
    cameras_dir: Path,
    overlay_alpha: float,
    pred_camera_mode: str,
    module: Stage1FusionLightningModule,
    overlay_renderer: "CameraMeshOverlayRenderer",
    dataset,
    grouped_indices: dict[str, list[int]],
    dataset_index: int,
    smoothing_window: int,
) -> dict[str, Any]:
    meta = batch["meta"][0]
    sample_id = str(meta["sample_id"])
    sample_dir = output_dir / f"{sample_index:03d}_{sample_id}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    pred_body_pose, pred_betas = get_visualization_prediction(
        batch=batch,
        predictions=predictions,
        module=module,
        dataset=dataset,
        grouped_indices=grouped_indices,
        dataset_index=dataset_index,
        device=device,
        smoothing_window=smoothing_window,
    )
    gt_body_pose = batch["target_body_pose"][0].detach().cpu().numpy()
    gt_betas = batch["target_betas"][0].detach().cpu().numpy()
    gt_world_global_orient = batch["target_aux"]["global_orient"][0].detach().cpu().numpy()
    gt_world_transl = batch["target_aux"]["transl"][0].detach().cpu().numpy()

    body_pose_abs_error = np.abs(pred_body_pose - gt_body_pose)
    betas_abs_error = np.abs(pred_betas - gt_betas)
    metrics = {
        "sample_id": sample_id,
        "sequence_id": str(meta["sequence_id"]),
        "frame_id": str(meta["frame_id"]),
        "camera_ids": [str(camera_id) for camera_id in meta["camera_ids"]],
        "smoothing_window": smoothing_window,
        "placement_note": (
            "Predicted fused meshes are canonical bodies placed per view using "
            f"{'the input-view fitted SMPL root pose and input intrinsics' if pred_camera_mode == 'input' else 'the HuMMan GT camera-frame root pose and GT intrinsics'}. "
            "GT meshes are built from HuMMan world-frame SMPL and projected with HuMMan GT cameras."
        ),
        "pred_camera_mode": pred_camera_mode,
        "body_pose_mse": float(np.mean((pred_body_pose - gt_body_pose) ** 2)),
        "betas_mse": float(np.mean((pred_betas - gt_betas) ** 2)),
        "body_pose_mae": float(np.mean(body_pose_abs_error)),
        "betas_mae": float(np.mean(betas_abs_error)),
        "body_pose_max_abs_error": float(np.max(body_pose_abs_error)),
        "betas_max_abs_error": float(np.max(betas_abs_error)),
    }

    view_panels: list[np.ndarray] = []
    view_summaries: list[dict[str, Any]] = []
    pred_view_smpl = module.convert_input_views_to_smpl(
        views_input=batch["views_input"].to(device),
        pred_cam_t=batch["view_aux"]["pred_cam_t"].to(device),
        batch_meta=batch["meta"],
    )
    pred_view_global_orient = pred_view_smpl["global_orient"].detach().cpu().numpy()
    pred_view_transl = pred_view_smpl["transl"].detach().cpu().numpy()
    gt_camera_global_orient = batch["target_aux"]["camera_global_orient"][0].detach().cpu().numpy()
    gt_camera_transl = batch["target_aux"]["camera_transl"][0].detach().cpu().numpy()
    gt_world_vertices, _ = build_smpl_outputs(
        smpl_model=smpl_model,
        device=device,
        body_pose=gt_body_pose,
        betas=gt_betas,
        global_orient=gt_world_global_orient,
        transl=gt_world_transl,
    )
    pred_world_vertices = build_smpl_vertices(
        smpl_model=smpl_model,
        device=device,
        body_pose=pred_body_pose,
        betas=pred_betas,
        global_orient=gt_world_global_orient,
        transl=gt_world_transl,
    )
    pred_view_vertices = []
    gt_view_vertices = []
    for view_index, camera_id in enumerate(meta["camera_ids"]):
        image_path = resolve_rgb_image_path(
            rgb_dir,
            sequence_id=str(meta["sequence_id"]),
            camera_id=str(camera_id),
            frame_id=str(meta["frame_id"]),
        )
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read RGB image: {image_path}")
        pred_cam_int = batch["view_aux"]["cam_int"][0, view_index].detach().cpu().numpy()
        gt_camera = load_camera_parameters(
            cameras_dir,
            sequence_id=str(meta["sequence_id"]),
            camera_id=str(camera_id),
        )
        gt_cam_int = gt_camera.intrinsics
        input_vertices_camera, input_joints_camera = build_smpl_outputs(
            smpl_model=smpl_model,
            device=device,
            body_pose=pred_view_smpl["body_pose"][view_index].detach().cpu().numpy(),
            betas=pred_view_smpl["betas"][view_index].detach().cpu().numpy(),
            global_orient=pred_view_global_orient[view_index],
            transl=pred_view_transl[view_index],
        )
        gt_vertices_camera = transform_world_vertices_to_camera(
            vertices_world=gt_world_vertices,
            rotation=gt_camera.rotation,
            translation=gt_camera.translation,
        )
        if pred_camera_mode == "gt":
            pred_vertices_camera = transform_world_vertices_to_camera(
                vertices_world=pred_world_vertices,
                rotation=gt_camera.rotation,
                translation=gt_camera.translation,
            )
            pred_overlay_intrinsics = gt_cam_int
            pred_translation_offset = np.zeros(3, dtype=np.float32)
        else:
            pred_vertices_oriented, pred_joints_oriented = build_smpl_outputs(
                smpl_model=smpl_model,
                device=device,
                body_pose=pred_body_pose,
                betas=pred_betas,
                global_orient=pred_view_global_orient[view_index],
                transl=np.zeros(3, dtype=np.float32),
            )
            pred_translation_offset = input_joints_camera[0] - pred_joints_oriented[0]
            pred_vertices_camera = np.ascontiguousarray(
                pred_vertices_oriented + pred_translation_offset.reshape(1, 3)
            )
            pred_overlay_intrinsics = pred_cam_int
        pred_view_vertices.append(pred_vertices_camera)
        gt_view_vertices.append(gt_vertices_camera)

        input_overlay, input_mask = overlay_renderer.render_overlay(
            image_bgr=image_bgr,
            vertices_camera=input_vertices_camera,
            faces=faces,
            intrinsics=pred_cam_int,
            color_bgr=(60, 180, 245),
            alpha=overlay_alpha,
        )
        pred_overlay, pred_mask = overlay_renderer.render_overlay(
            image_bgr=image_bgr,
            vertices_camera=pred_vertices_camera,
            faces=faces,
            intrinsics=pred_overlay_intrinsics,
            color_bgr=(40, 70, 230),
            alpha=overlay_alpha,
        )
        gt_overlay, gt_mask = overlay_renderer.render_overlay(
            image_bgr=image_bgr,
            vertices_camera=gt_vertices_camera,
            faces=faces,
            intrinsics=gt_cam_int,
            color_bgr=(70, 210, 70),
            alpha=overlay_alpha,
        )
        panel = build_view_panel(
            [
                ("RGB", image_bgr),
                ("Input Overlay", input_overlay),
                ("Pred Overlay", pred_overlay),
                ("GT Overlay", gt_overlay),
            ],
            footer="",
        )
        view_panels.append(panel)
        view_summaries.append(
            {
                "camera_id": str(camera_id),
                "image_path": str(image_path),
                "input_visible": bool(input_mask.any()),
                "pred_visible": bool(pred_mask.any()),
                "gt_visible": bool(gt_mask.any()),
                "input_mask_pixels": int(np.count_nonzero(input_mask)),
                "pred_mask_pixels": int(np.count_nonzero(pred_mask)),
                "gt_mask_pixels": int(np.count_nonzero(gt_mask)),
                "pred_camera_mode": pred_camera_mode,
                "pred_cam_int": pred_cam_int.astype(np.float32).tolist(),
                "gt_cam_int": gt_cam_int.astype(np.float32).tolist(),
                "pred_translation_offset": pred_translation_offset.astype(np.float32).tolist(),
                "gt_vertices_source": "smpl_rebuild",
            }
        )
        cv2.imwrite(str(sample_dir / f"{camera_id}_input_overlay.png"), input_overlay)
        cv2.imwrite(str(sample_dir / f"{camera_id}_pred_overlay.png"), pred_overlay)
        cv2.imwrite(str(sample_dir / f"{camera_id}_gt_overlay.png"), gt_overlay)

    summary_image = build_contact_sheet(
        title_lines=[
            f"sequence_id: {meta['sequence_id']}",
            f"pred_camera_mode: {pred_camera_mode}",
            f"smoothing_window: {smoothing_window}",
        ],
        view_panels=view_panels,
    )

    cv2.imwrite(str(sample_dir / "comparison.png"), summary_image)
    np.savez_compressed(
        sample_dir / "arrays.npz",
        pred_body_pose=pred_body_pose.astype(np.float32),
        gt_body_pose=gt_body_pose.astype(np.float32),
        pred_betas=pred_betas.astype(np.float32),
        gt_betas=gt_betas.astype(np.float32),
        pred_view_global_orient=pred_view_global_orient.astype(np.float32),
        pred_view_transl=pred_view_transl.astype(np.float32),
        gt_camera_global_orient=gt_camera_global_orient.astype(np.float32),
        gt_camera_transl=gt_camera_transl.astype(np.float32),
        gt_world_global_orient=gt_world_global_orient.astype(np.float32),
        gt_world_transl=gt_world_transl.astype(np.float32),
        pred_view_vertices=np.stack(pred_view_vertices, axis=0).astype(np.float32),
        gt_view_vertices=np.stack(gt_view_vertices, axis=0).astype(np.float32),
    )
    metrics["views"] = view_summaries
    (sample_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def get_visualization_prediction(
    *,
    batch: dict[str, Any],
    predictions: dict[str, torch.Tensor],
    module: Stage1FusionLightningModule,
    dataset,
    grouped_indices: dict[str, list[int]],
    dataset_index: int,
    device: torch.device,
    smoothing_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    if smoothing_window <= 1:
        return (
            predictions["pred_body_pose"][0].detach().cpu().numpy(),
            predictions["pred_betas"][0].detach().cpu().numpy(),
        )

    meta = batch["meta"][0]
    sequence_indices = grouped_indices[str(meta["sequence_id"])]
    center_position = sequence_indices.index(dataset_index)
    smoothed_batches = select_centered_window_indices(
        sequence_indices=sequence_indices,
        center_position=center_position,
        window_size=smoothing_window,
    )

    pred_body_pose_values = []
    pred_betas_values = []
    with torch.no_grad():
        for neighbor_index in smoothed_batches:
            neighbor_batch = multiview_collate([dataset[neighbor_index]])
            neighbor_predictions = module(neighbor_batch["views_input"].to(device))
            pred_body_pose_values.append(neighbor_predictions["pred_body_pose"][0].detach().cpu().numpy())
            pred_betas_values.append(neighbor_predictions["pred_betas"][0].detach().cpu().numpy())

    return (
        np.mean(np.stack(pred_body_pose_values, axis=0), axis=0).astype(np.float32, copy=False),
        np.mean(np.stack(pred_betas_values, axis=0), axis=0).astype(np.float32, copy=False),
    )


def select_centered_window_indices(
    *,
    sequence_indices: list[int],
    center_position: int,
    window_size: int,
) -> list[int]:
    half = window_size // 2
    start = max(0, center_position - half)
    end = min(len(sequence_indices), start + window_size)
    start = max(0, end - window_size)
    return sequence_indices[start:end]


def build_smpl_vertices(
    *,
    smpl_model,
    device: torch.device,
    body_pose: np.ndarray,
    betas: np.ndarray,
    global_orient: np.ndarray,
    transl: np.ndarray,
) -> np.ndarray:
    body_pose_tensor = torch.as_tensor(body_pose, dtype=torch.float32, device=device).view(1, -1)
    betas_tensor = torch.as_tensor(betas, dtype=torch.float32, device=device).view(1, -1)
    global_orient_tensor = torch.as_tensor(
        global_orient, dtype=torch.float32, device=device
    ).view(1, -1)
    transl_tensor = torch.as_tensor(transl, dtype=torch.float32, device=device).view(1, -1)
    output = smpl_model(
        body_pose=body_pose_tensor,
        betas=betas_tensor,
        global_orient=global_orient_tensor,
        transl=transl_tensor,
    )
    return output.vertices[0].detach().cpu().numpy()


def build_smpl_outputs(
    *,
    smpl_model,
    device: torch.device,
    body_pose: np.ndarray,
    betas: np.ndarray,
    global_orient: np.ndarray,
    transl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    body_pose_tensor = torch.as_tensor(body_pose, dtype=torch.float32, device=device).view(1, -1)
    betas_tensor = torch.as_tensor(betas, dtype=torch.float32, device=device).view(1, -1)
    global_orient_tensor = torch.as_tensor(
        global_orient, dtype=torch.float32, device=device
    ).view(1, -1)
    transl_tensor = torch.as_tensor(transl, dtype=torch.float32, device=device).view(1, -1)
    output = smpl_model(
        body_pose=body_pose_tensor,
        betas=betas_tensor,
        global_orient=global_orient_tensor,
        transl=transl_tensor,
    )
    return (
        output.vertices[0].detach().cpu().numpy(),
        output.joints[0].detach().cpu().numpy(),
    )

def transform_world_vertices_to_camera(
    *,
    vertices_world: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    vertices_world = np.asarray(vertices_world, dtype=np.float32)
    rotation = np.asarray(rotation, dtype=np.float32)
    translation = np.asarray(translation, dtype=np.float32).reshape(1, 3)
    return np.ascontiguousarray((rotation @ vertices_world.T).T + translation)


class CameraMeshOverlayRenderer:
    """Offscreen renderer for shaded camera-space mesh overlays."""

    def __init__(self) -> None:
        self._renderers: dict[tuple[int, int], pyrender.OffscreenRenderer] = {}

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.delete()
        self._renderers.clear()

    def render_overlay(
        self,
        *,
        image_bgr: np.ndarray,
        vertices_camera: np.ndarray,
        faces: np.ndarray,
        intrinsics: np.ndarray,
        color_bgr: tuple[int, int, int],
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = image_bgr.shape[:2]
        renderer = self._get_renderer(width=width, height=height)
        mesh = trimesh.Trimesh(
            np.asarray(vertices_camera, dtype=np.float32).copy(),
            np.asarray(faces, dtype=np.int32),
            process=False,
        )
        mesh.apply_transform(opencv_to_opengl_transform())
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.1,
            roughnessFactor=0.75,
            baseColorFactor=bgr_to_rgba(color_bgr),
            alphaMode="OPAQUE",
        )
        render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)

        scene = pyrender.Scene(
            bg_color=np.array([0, 0, 0, 0], dtype=np.uint8),
            ambient_light=np.array([0.35, 0.35, 0.35], dtype=np.float32),
        )
        scene.add(render_mesh)
        add_camera_lights(scene)
        camera = pyrender.IntrinsicsCamera(
            fx=float(intrinsics[0, 0]),
            fy=float(intrinsics[1, 1]),
            cx=float(intrinsics[0, 2]),
            cy=float(intrinsics[1, 2]),
        )
        scene.add(camera, pose=np.eye(4, dtype=np.float32))

        color_rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        return composite_rgba_on_image(image_bgr, color_rgba, alpha=alpha)

    def _get_renderer(self, *, width: int, height: int) -> pyrender.OffscreenRenderer:
        key = (width, height)
        renderer = self._renderers.get(key)
        if renderer is None:
            renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
            self._renderers[key] = renderer
        return renderer


def opencv_to_opengl_transform() -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[1, 1] = -1.0
    transform[2, 2] = -1.0
    return transform


def add_camera_lights(scene: pyrender.Scene) -> None:
    light = pyrender.DirectionalLight(color=np.ones(3, dtype=np.float32), intensity=2.6)
    light_poses = (
        np.eye(4, dtype=np.float32),
        np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.8660254, -0.5, 0.0],
                [0.0, 0.5, 0.8660254, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        np.asarray(
            [
                [0.8660254, 0.0, 0.5, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-0.5, 0.0, 0.8660254, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    for pose in light_poses:
        scene.add(light, pose=pose)


def composite_rgba_on_image(
    image_bgr: np.ndarray,
    color_rgba: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    render_rgb = color_rgba[..., :3].astype(np.float32)
    render_bgr = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR)
    render_alpha = (color_rgba[..., 3].astype(np.float32) / 255.0) * float(alpha)
    render_alpha = np.clip(render_alpha, 0.0, 1.0)
    output = image_bgr.astype(np.float32).copy()
    output *= 1.0 - render_alpha[..., None]
    output += render_bgr * render_alpha[..., None]
    return np.clip(output, 0.0, 255.0).astype(np.uint8), (render_alpha > 1e-4).astype(np.uint8)


def bgr_to_rgba(color_bgr: tuple[int, int, int]) -> tuple[float, float, float, float]:
    b, g, r = color_bgr
    return (r / 255.0, g / 255.0, b / 255.0, 1.0)

def build_view_panel(items: list[tuple[str, np.ndarray]], *, footer: str) -> np.ndarray:
    tile_height, tile_width = items[0][1].shape[:2]
    header_height = 34
    footer_height = 28 if footer else 0
    tile_gap = 12
    panel_width = tile_width * len(items) + tile_gap * max(len(items) - 1, 0)
    panel = np.full(
        (tile_height + header_height + footer_height, panel_width, 3),
        255,
        dtype=np.uint8,
    )

    for item_index, (title, image) in enumerate(items):
        x0 = item_index * (tile_width + tile_gap)
        panel[header_height : header_height + tile_height, x0 : x0 + tile_width] = image
        cv2.putText(
            panel,
            title,
            (x0 + 12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    if footer:
        cv2.putText(
            panel,
            footer,
            (12, header_height + tile_height + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (50, 50, 50),
            1,
            cv2.LINE_AA,
        )
    return panel


def build_contact_sheet(*, title_lines: list[str], view_panels: list[np.ndarray]) -> np.ndarray:
    margin = 24
    text_height = 28
    panel_gap = 18
    width = max(panel.shape[1] for panel in view_panels) + 2 * margin
    title_height = margin + len(title_lines) * text_height + 8
    total_height = title_height + sum(panel.shape[0] for panel in view_panels)
    total_height += panel_gap * max(len(view_panels) - 1, 0) + margin

    canvas = np.full((total_height, width, 3), 255, dtype=np.uint8)
    y = margin + 4
    for line in title_lines:
        cv2.putText(
            canvas,
            line,
            (margin, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        y += text_height

    y += 8
    for panel in view_panels:
        x = max((width - panel.shape[1]) // 2, 0)
        canvas[y : y + panel.shape[0], x : x + panel.shape[1]] = panel
        y += panel.shape[0] + panel_gap
    return canvas


if __name__ == "__main__":
    main()
