#!/usr/bin/env python
"""Render per-sequence MP4 videos with input, fused prediction, and GT SMPL overlays."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

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
    validate_mhr_asset_folder,
)
from mvhpe3d.visualization import (
    load_camera_parameters,
    overlay_mask_on_image,
    render_projected_mesh_mask_camera,
    resolve_rgb_image_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save per-sequence MP4 videos with input, predicted, and GT SMPL overlays"
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
        "--output-dir",
        type=str,
        default="outputs/stage1_videos",
        help="Directory to save the MP4 videos",
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
        help="Optional override for the MHR asset directory used for input-view conversion",
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
        "--fps",
        type=float,
        default=10.0,
        help="Frames per second for the output MP4",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="FourCC codec for cv2.VideoWriter, e.g. mp4v or avc1",
    )
    parser.add_argument(
        "--sequence-id",
        type=str,
        default=None,
        help="Optional single sequence_id to visualize",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Optional cap on the number of output videos",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on the number of frames per sequence",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Use every Nth frame in each sequence",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional override for experiment seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_step < 1:
        raise ValueError(f"--frame-step must be >= 1, got {args.frame_step}")
    if args.fps <= 0:
        raise ValueError(f"--fps must be > 0, got {args.fps}")

    experiment = load_experiment_config(args.config)
    validate_mhr_asset_folder(args.mhr_assets_dir or "/opt/data/assets")

    data_config = build_data_config(experiment["data"], args)
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
    cameras_dir = Path(data_config.cameras_dir).resolve()
    smpl_model_path = resolve_smpl_model_path(args.smpl_model_path)
    smpl_model = build_smpl_model(
        device=device,
        smpl_model_path=str(smpl_model_path),
        batch_size=1,
    )
    faces = np.asarray(smpl_model.faces, dtype=np.int32)

    grouped_indices = build_sequence_index(dataset)
    selected_sequence_ids = sorted(grouped_indices)
    if args.sequence_id is not None:
        if args.sequence_id not in grouped_indices:
            raise KeyError(
                f"sequence_id '{args.sequence_id}' was not found in the selected {args.stage} split"
            )
        selected_sequence_ids = [args.sequence_id]
    if args.max_sequences is not None:
        selected_sequence_ids = selected_sequence_ids[: args.max_sequences]

    summaries = []
    for sequence_id in selected_sequence_ids:
        frame_indices = grouped_indices[sequence_id][:: args.frame_step]
        if args.max_frames is not None:
            frame_indices = frame_indices[: args.max_frames]
        if not frame_indices:
            continue

        video_path = output_dir / f"{sequence_id}.mp4"
        writer: cv2.VideoWriter | None = None
        frame_width: int | None = None
        frame_height: int | None = None
        written_frames = 0

        for dataset_index in frame_indices:
            batch = multiview_collate([dataset[dataset_index]])
            frame_image, frame_summary = build_video_frame(
                batch=batch,
                module=module,
                smpl_model=smpl_model,
                faces=faces,
                device=device,
                rgb_dir=rgb_dir,
                cameras_dir=cameras_dir,
                overlay_alpha=args.overlay_alpha,
            )
            if writer is None:
                frame_height, frame_width = frame_image.shape[:2]
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*args.codec),
                    args.fps,
                    (frame_width, frame_height),
                )
                if not writer.isOpened():
                    raise RuntimeError(
                        f"Failed to open video writer for {video_path} with codec '{args.codec}'"
                    )
            else:
                assert frame_width is not None and frame_height is not None
                if frame_image.shape[1] != frame_width or frame_image.shape[0] != frame_height:
                    frame_image = cv2.resize(
                        frame_image,
                        (frame_width, frame_height),
                        interpolation=cv2.INTER_LINEAR,
                    )
            writer.write(frame_image)
            written_frames += 1

        if writer is not None:
            writer.release()

        summaries.append(
            {
                "sequence_id": sequence_id,
                "video_path": str(video_path),
                "num_frames": written_frames,
                "stage": args.stage,
                "fps": args.fps,
            }
        )
        print(f"Saved {written_frames} frames to {video_path}")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({"videos": summaries}, indent=2), encoding="utf-8")
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
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for index, record in enumerate(dataset.records):
        grouped[record.sequence_id].append((int(record.frame_id), index))
    return {
        sequence_id: [index for _, index in sorted(items)]
        for sequence_id, items in grouped.items()
    }


def build_video_frame(
    *,
    batch: dict[str, Any],
    module: Stage1FusionLightningModule,
    smpl_model,
    faces: np.ndarray,
    device: torch.device,
    rgb_dir: Path,
    cameras_dir: Path,
    overlay_alpha: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    meta = batch["meta"][0]
    sample_id = str(meta["sample_id"])
    sequence_id = str(meta["sequence_id"])
    frame_id = str(meta["frame_id"])

    views_input = batch["views_input"].to(device)
    pred_cam_t = batch["view_aux"]["pred_cam_t"].to(device)
    with torch.no_grad():
        predictions = module(views_input)
    input_view_smpl = module.convert_input_views_to_smpl(
        views_input=views_input,
        pred_cam_t=pred_cam_t,
        batch_meta=batch["meta"],
    )

    pred_body_pose = predictions["pred_body_pose"][0].detach().cpu().numpy()
    pred_betas = predictions["pred_betas"][0].detach().cpu().numpy()
    gt_body_pose = batch["target_body_pose"][0].detach().cpu().numpy()
    gt_betas = batch["target_betas"][0].detach().cpu().numpy()
    gt_world_global_orient = batch["target_aux"]["global_orient"][0].detach().cpu().numpy()
    gt_world_transl = batch["target_aux"]["transl"][0].detach().cpu().numpy()

    input_body_pose = input_view_smpl["body_pose"].detach().cpu().numpy()
    input_betas = input_view_smpl["betas"].detach().cpu().numpy()
    input_global_orient = input_view_smpl["global_orient"].detach().cpu().numpy()
    input_transl = input_view_smpl["transl"].detach().cpu().numpy()
    zero_root = np.zeros(3, dtype=np.float32)

    gt_world_vertices, _ = build_smpl_outputs(
        smpl_model=smpl_model,
        device=device,
        body_pose=gt_body_pose,
        betas=gt_betas,
        global_orient=gt_world_global_orient,
        transl=gt_world_transl,
    )

    view_panels: list[np.ndarray] = []
    per_view_summary: list[dict[str, Any]] = []

    for view_index, camera_id in enumerate(meta["camera_ids"]):
        image_path = resolve_rgb_image_path(
            rgb_dir,
            sequence_id=sequence_id,
            camera_id=str(camera_id),
            frame_id=frame_id,
        )
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read RGB image: {image_path}")

        pred_cam_int = batch["view_aux"]["cam_int"][0, view_index].detach().cpu().numpy()
        gt_camera = load_camera_parameters(
            cameras_dir,
            sequence_id=sequence_id,
            camera_id=str(camera_id),
        )

        input_vertices_camera, input_joints_camera = build_smpl_outputs(
            smpl_model=smpl_model,
            device=device,
            body_pose=input_body_pose[view_index],
            betas=input_betas[view_index],
            global_orient=input_global_orient[view_index],
            transl=input_transl[view_index],
        )
        _, input_joints_canonical = build_smpl_outputs(
            smpl_model=smpl_model,
            device=device,
            body_pose=input_body_pose[view_index],
            betas=input_betas[view_index],
            global_orient=zero_root,
            transl=zero_root,
        )
        pred_vertices_canonical, _ = build_smpl_outputs(
            smpl_model=smpl_model,
            device=device,
            body_pose=pred_body_pose,
            betas=pred_betas,
            global_orient=zero_root,
            transl=zero_root,
        )
        pred_vertices_camera = map_canonical_vertices_to_input_frame(
            canonical_vertices=pred_vertices_canonical,
            input_canonical_joints=input_joints_canonical,
            input_camera_joints=input_joints_camera,
        )
        gt_vertices_camera = transform_world_vertices_to_camera(
            vertices_world=gt_world_vertices,
            rotation=gt_camera.rotation,
            translation=gt_camera.translation,
        )

        input_mask = render_projected_mesh_mask_camera(
            image_bgr.shape[:2],
            vertices_camera=input_vertices_camera,
            faces=faces,
            intrinsics=pred_cam_int,
        )
        pred_mask = render_projected_mesh_mask_camera(
            image_bgr.shape[:2],
            vertices_camera=pred_vertices_camera,
            faces=faces,
            intrinsics=pred_cam_int,
        )
        gt_mask = render_projected_mesh_mask_camera(
            image_bgr.shape[:2],
            vertices_camera=gt_vertices_camera,
            faces=faces,
            intrinsics=gt_camera.intrinsics,
        )

        input_overlay = overlay_mask_on_image(
            image_bgr,
            input_mask,
            color=(0, 190, 255),
            alpha=overlay_alpha,
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
                alpha=overlay_alpha * 0.8,
            ),
            pred_mask,
            color=(40, 70, 230),
            alpha=overlay_alpha * 0.8,
        )

        view_panels.append(
            build_view_panel(
                [
                    ("RGB", image_bgr),
                    ("Input", input_overlay),
                    ("Pred", pred_overlay),
                    ("GT", gt_overlay),
                    ("Combined", combined_overlay),
                ],
                footer=(
                    f"{camera_id}  "
                    f"in={int(input_mask.any())}  pred={int(pred_mask.any())}  gt={int(gt_mask.any())}"
                ),
            )
        )
        per_view_summary.append(
            {
                "camera_id": str(camera_id),
                "image_path": str(image_path),
                "input_visible": bool(input_mask.any()),
                "pred_visible": bool(pred_mask.any()),
                "gt_visible": bool(gt_mask.any()),
            }
        )

    body_pose_mae = float(np.mean(np.abs(pred_body_pose - gt_body_pose)))
    betas_mae = float(np.mean(np.abs(pred_betas - gt_betas)))
    frame_image = build_contact_sheet(
        title_lines=[
            f"sequence_id: {sequence_id}  frame_id: {frame_id}  sample_id: {sample_id}",
            f"body_pose_mae={body_pose_mae:.6f}  betas_mae={betas_mae:.6f}",
            "Panels: RGB, input SMPL, fused prediction, GT, combined",
        ],
        view_panels=view_panels,
    )
    return frame_image, {
        "sample_id": sample_id,
        "sequence_id": sequence_id,
        "frame_id": frame_id,
        "views": per_view_summary,
    }


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
    global_orient_tensor = torch.as_tensor(global_orient, dtype=torch.float32, device=device).view(
        1, -1
    )
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
    global_orient_tensor = torch.as_tensor(global_orient, dtype=torch.float32, device=device).view(
        1, -1
    )
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


def map_canonical_vertices_to_input_frame(
    *,
    canonical_vertices: np.ndarray,
    input_canonical_joints: np.ndarray,
    input_camera_joints: np.ndarray,
) -> np.ndarray:
    rotation_matrix, translation = estimate_rigid_transform(
        source_points=np.asarray(input_canonical_joints, dtype=np.float32),
        target_points=np.asarray(input_camera_joints, dtype=np.float32),
    )
    canonical_vertices = np.asarray(canonical_vertices, dtype=np.float32)
    transformed = (rotation_matrix @ canonical_vertices.T).T + translation.reshape(1, 3)
    return np.ascontiguousarray(transformed)


def estimate_rigid_transform(
    *,
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_points, dtype=np.float32)
    target = np.asarray(target_points, dtype=np.float32)
    source_mean = source.mean(axis=0, keepdims=True)
    target_mean = target.mean(axis=0, keepdims=True)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = source_centered.T @ target_centered
    u, _, vh = np.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0:
        vh[-1, :] *= -1.0
        rotation = vh.T @ u.T
    translation = target_mean.reshape(3) - rotation @ source_mean.reshape(3)
    return (
        rotation.astype(np.float32, copy=False),
        translation.astype(np.float32, copy=False),
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
