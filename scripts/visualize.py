#!/usr/bin/env python
"""Visualize predicted and GT SMPL meshes overlaid on RGB images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mvhpe3d.data import Stage1DataConfig, Stage1HuMManDataModule, multiview_collate
from mvhpe3d.lightning import Stage1FusionLightningModule
from mvhpe3d.utils import (
    build_smpl_model,
    load_experiment_config,
    resolve_smpl_model_path as resolve_smpl_model_path_impl,
)
from mvhpe3d.visualization import (
    load_camera_parameters,
    overlay_mask_on_image,
    render_projected_mesh_mask,
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
        help="Optional override for the HuMMan camera JSON directory",
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
    parser.add_argument("--seed", type=int, default=None, help="Optional override for experiment seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = load_experiment_config(args.config)
    data_config = build_data_config(experiment["data"], args)
    data_config.batch_size = 1
    data_config.drop_last_train = False

    if args.seed is not None:
        data_config.seed = args.seed

    datamodule = Stage1HuMManDataModule(data_config)
    datamodule.setup(None)
    dataset = select_dataset(datamodule, args.stage)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=multiview_collate,
    )

    device = resolve_device(args.device)
    module = Stage1FusionLightningModule.load_from_checkpoint(
        args.checkpoint_path,
        map_location="cpu",
    )
    module.eval()
    module.to(device)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(data_config.manifest_path).resolve().parent
    rgb_dir = resolve_required_dir(args.rgb_dir, fallback=data_root / "rgb", name="rgb")
    cameras_dir = resolve_required_dir(
        args.cameras_dir,
        fallback=data_root / "cameras",
        name="cameras",
    )
    smpl_model_path = resolve_smpl_model_path(args.smpl_model_path)
    smpl_model = build_smpl_model(
        device=device,
        smpl_model_path=str(smpl_model_path),
        batch_size=1,
    )
    faces = np.asarray(smpl_model.faces, dtype=np.int32)

    written = 0
    summaries = []
    with torch.no_grad():
        for batch in dataloader:
            if written >= args.num_samples:
                break

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
                cameras_dir=cameras_dir,
                overlay_alpha=args.overlay_alpha,
            )
            summaries.append(sample_summary)
            written += 1

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
    if args.split_config_path is not None:
        data_kwargs["split_config_path"] = args.split_config_path
    if args.split_name is not None:
        data_kwargs["split_name"] = args.split_name
    if args.seed is not None:
        data_kwargs["seed"] = args.seed

    return Stage1DataConfig(**data_kwargs)


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
) -> dict[str, Any]:
    meta = batch["meta"][0]
    sample_id = str(meta["sample_id"])
    sample_dir = output_dir / f"{sample_index:03d}_{sample_id}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    pred_body_pose = predictions["pred_body_pose"][0].detach().cpu().numpy()
    pred_betas = predictions["pred_betas"][0].detach().cpu().numpy()
    gt_body_pose = batch["target_body_pose"][0].detach().cpu().numpy()
    gt_betas = batch["target_betas"][0].detach().cpu().numpy()
    gt_global_orient = batch["target_aux"]["global_orient"][0].detach().cpu().numpy()
    gt_transl = batch["target_aux"]["transl"][0].detach().cpu().numpy()

    body_pose_abs_error = np.abs(pred_body_pose - gt_body_pose)
    betas_abs_error = np.abs(pred_betas - gt_betas)
    metrics = {
        "sample_id": sample_id,
        "sequence_id": str(meta["sequence_id"]),
        "frame_id": str(meta["frame_id"]),
        "camera_ids": [str(camera_id) for camera_id in meta["camera_ids"]],
        "placement_note": (
            "Predicted and GT meshes are both rendered with HuMMan GT global_orient/transl "
            "and HuMMan camera extrinsics. The prediction differs only in canonical "
            "SMPL body_pose/betas."
        ),
        "body_pose_mse": float(np.mean((pred_body_pose - gt_body_pose) ** 2)),
        "betas_mse": float(np.mean((pred_betas - gt_betas) ** 2)),
        "body_pose_mae": float(np.mean(body_pose_abs_error)),
        "betas_mae": float(np.mean(betas_abs_error)),
        "body_pose_max_abs_error": float(np.max(body_pose_abs_error)),
        "betas_max_abs_error": float(np.max(betas_abs_error)),
    }

    pred_vertices_world = build_smpl_vertices(
        smpl_model=smpl_model,
        device=device,
        body_pose=pred_body_pose,
        betas=pred_betas,
        global_orient=gt_global_orient,
        transl=gt_transl,
    )
    gt_vertices_world = build_smpl_vertices(
        smpl_model=smpl_model,
        device=device,
        body_pose=gt_body_pose,
        betas=gt_betas,
        global_orient=gt_global_orient,
        transl=gt_transl,
    )

    view_panels: list[np.ndarray] = []
    view_summaries: list[dict[str, Any]] = []
    for camera_id in meta["camera_ids"]:
        image_path = resolve_rgb_image_path(
            rgb_dir,
            sequence_id=str(meta["sequence_id"]),
            camera_id=str(camera_id),
            frame_id=str(meta["frame_id"]),
        )
        camera = load_camera_parameters(
            cameras_dir,
            sequence_id=str(meta["sequence_id"]),
            camera_id=str(camera_id),
        )
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read RGB image: {image_path}")

        pred_mask = render_projected_mesh_mask(
            image_bgr.shape[:2],
            vertices_world=pred_vertices_world,
            faces=faces,
            camera=camera,
        )
        gt_mask = render_projected_mesh_mask(
            image_bgr.shape[:2],
            vertices_world=gt_vertices_world,
            faces=faces,
            camera=camera,
        )

        pred_overlay = overlay_mask_on_image(
            image_bgr,
            pred_mask,
            color=(40, 70, 230),
            alpha=overlay_alpha,
        )
        gt_overlay = overlay_mask_on_image(
            image_bgr,
            gt_mask,
            color=(70, 210, 70),
            alpha=overlay_alpha,
        )
        combined_overlay = overlay_mask_on_image(
            overlay_mask_on_image(
                image_bgr,
                gt_mask,
                color=(70, 210, 70),
                alpha=overlay_alpha,
            ),
            pred_mask,
            color=(40, 70, 230),
            alpha=overlay_alpha,
        )

        panel = build_view_panel(
            [
                ("RGB", image_bgr),
                ("Pred Overlay", pred_overlay),
                ("GT Overlay", gt_overlay),
                ("Combined", combined_overlay),
            ],
            footer=f"{camera_id}  pred_px={int(pred_mask.sum() > 0)}  gt_px={int(gt_mask.sum() > 0)}",
        )
        view_panels.append(panel)
        view_summaries.append(
            {
                "camera_id": str(camera_id),
                "image_path": str(image_path),
                "pred_visible": bool(pred_mask.any()),
                "gt_visible": bool(gt_mask.any()),
                "pred_mask_pixels": int(np.count_nonzero(pred_mask)),
                "gt_mask_pixels": int(np.count_nonzero(gt_mask)),
            }
        )
        cv2.imwrite(str(sample_dir / f"{camera_id}_pred_overlay.png"), pred_overlay)
        cv2.imwrite(str(sample_dir / f"{camera_id}_gt_overlay.png"), gt_overlay)
        cv2.imwrite(str(sample_dir / f"{camera_id}_combined_overlay.png"), combined_overlay)

    summary_image = build_contact_sheet(
        title_lines=[
            f"sample_id: {sample_id}",
            f"sequence_id: {meta['sequence_id']}  frame_id: {meta['frame_id']}",
            (
                f"body_pose_mse={metrics['body_pose_mse']:.6f}  betas_mse={metrics['betas_mse']:.6f}  "
                f"body_pose_mae={metrics['body_pose_mae']:.6f}  betas_mae={metrics['betas_mae']:.6f}"
            ),
            "Placement: both meshes use HuMMan GT global pose and camera calibration.",
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
        gt_global_orient=gt_global_orient.astype(np.float32),
        gt_transl=gt_transl.astype(np.float32),
        pred_vertices_world=pred_vertices_world.astype(np.float32),
        gt_vertices_world=gt_vertices_world.astype(np.float32),
    )
    metrics["views"] = view_summaries
    (sample_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


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


def build_view_panel(items: list[tuple[str, np.ndarray]], *, footer: str) -> np.ndarray:
    tile_height, tile_width = items[0][1].shape[:2]
    header_height = 34
    footer_height = 28
    panel = np.full(
        (tile_height + header_height + footer_height, tile_width * len(items), 3),
        255,
        dtype=np.uint8,
    )

    for item_index, (title, image) in enumerate(items):
        x0 = item_index * tile_width
        panel[header_height : header_height + tile_height, x0 : x0 + tile_width] = image
        cv2.putText(
            panel,
            title,
            (x0 + 12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        panel,
        footer,
        (12, header_height + tile_height + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (50, 50, 50),
        2,
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
            2,
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
