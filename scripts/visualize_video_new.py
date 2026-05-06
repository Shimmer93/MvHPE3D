#!/usr/bin/env python
"""Render per-sequence MP4 videos in a 3x2 grid with view-quality ordering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from visualize_video import (
    CameraMeshOverlayRenderer,
    build_data_config,
    build_datamodule,
    build_model_config,
    build_sequence_fixed_view_scene,
    build_sequence_index,
    build_smpl_outputs,
    build_view_panel,
    get_visualization_prediction,
    is_stage2_experiment,
    load_camera_parameters,
    load_experiment_config,
    load_visualization_module,
    multiview_collate,
    resolve_device,
    resolve_input_smpl_cache_dir,
    resolve_input_view_smpl_params,
    resolve_required_dir,
    resolve_rgb_image_path,
    resolve_smpl_model_path,
    select_sequence_ids,
    select_dataset,
    transform_camera_vertices_to_world,
)
from mvhpe3d.metrics import batch_mpjpe
from mvhpe3d.utils import build_smpl_model, validate_mhr_asset_folder
from mvhpe3d.visualization import correct_camera_global_orient_using_torso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save per-sequence MP4 videos in a 3x2 comparison grid"
    )
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment/stage1_cross_camera.yaml",
        help="Path to the experiment YAML file",
    )
    parser.add_argument("--manifest-path", type=str, default=None, help="Optional override for manifest")
    parser.add_argument("--gt-smpl-dir", type=str, default=None, help="Optional GT SMPL override")
    parser.add_argument("--split-config-path", type=str, default=None, help="Optional split config override")
    parser.add_argument("--split-name", type=str, default=None, help="Optional split name override")
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
        default="outputs/stage1_videos_newgrid",
        help="Directory to save the MP4 videos",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device for inference")
    parser.add_argument("--rgb-dir", type=str, default=None, help="Optional cropped RGB override")
    parser.add_argument("--cameras-dir", type=str, default=None, help="Optional camera JSON override")
    parser.add_argument("--mhr-assets-dir", type=str, default=None, help="Optional MHR asset override")
    parser.add_argument(
        "--input-smpl-cache-dir",
        type=str,
        default=None,
        help="Optional fitted SMPL cache directory",
    )
    parser.add_argument("--smpl-model-path", type=str, default=None, help="Path to SMPL .pkl")
    parser.add_argument("--fps", type=float, default=10.0, help="Frames per second for the MP4")
    parser.add_argument("--codec", type=str, default="mp4v", help="FourCC codec for cv2.VideoWriter")
    parser.add_argument("--sequence-id", type=str, default=None, help="Optional single sequence_id")
    parser.add_argument("--max-sequences", type=int, default=None, help="Optional cap on output videos")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on frames per sequence")
    parser.add_argument("--frame-step", type=int, default=1, help="Use every Nth frame in each sequence")
    parser.add_argument(
        "--pred-camera-mode",
        type=str,
        choices=("input", "input_corrected", "gt"),
        default="gt",
        help="Camera/root placement used for predicted RGB overlays",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        choices=("first", "best_improvement"),
        default="first",
        help="How to choose which sequences to visualize",
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
        help="Optional minimum average input-minus-pred improvement required for selection",
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
    if args.frame_step < 1:
        raise ValueError(f"--frame-step must be >= 1, got {args.frame_step}")
    if args.fps <= 0:
        raise ValueError(f"--fps must be > 0, got {args.fps}")
    if args.smoothing_window < 1:
        raise ValueError(f"--smoothing-window must be >= 1, got {args.smoothing_window}")

    experiment = load_experiment_config(args.config)
    if not is_stage2_experiment(experiment):
        validate_mhr_asset_folder(args.mhr_assets_dir or "/opt/data/assets")

    data_config = build_data_config(experiment["data"], args)
    model_config = build_model_config(experiment["model"], checkpoint_path=args.checkpoint_path)
    data_config.batch_size = 1
    data_config.drop_last_train = False
    if args.seed is not None:
        data_config.seed = args.seed

    datamodule = build_datamodule(data_config)
    datamodule.setup(None)
    dataset = select_dataset(datamodule, args.stage)

    device = resolve_device(args.device)
    module = load_visualization_module(
        checkpoint_path=args.checkpoint_path,
        model_config=model_config,
        data_config=data_config,
        args=args,
    )
    module.eval()
    module.to(device)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(data_config.manifest_path).resolve().parent
    rgb_dir = resolve_required_dir(args.rgb_dir, fallback=data_root / "rgb", name="rgb")
    cameras_dir = Path(data_config.cameras_dir).resolve()
    input_smpl_cache_dir = Path(resolve_input_smpl_cache_dir(args, data_config))
    smpl_model_path = resolve_smpl_model_path(args.smpl_model_path)
    smpl_model = build_smpl_model(
        device=device,
        smpl_model_path=str(smpl_model_path),
        batch_size=1,
    )
    faces = np.asarray(smpl_model.faces, dtype=np.int32)
    overlay_renderer = CameraMeshOverlayRenderer()

    try:
        grouped_indices = build_sequence_index(dataset)
        selected_sequence_ids = sorted(grouped_indices)
        if args.sequence_id is not None:
            if args.sequence_id not in grouped_indices:
                raise KeyError(
                    f"sequence_id '{args.sequence_id}' was not found in the selected {args.stage} split"
                )
            selected_sequence_ids = [args.sequence_id]
        else:
            selected_sequence_ids = select_sequence_ids(
                dataset=dataset,
                grouped_indices=grouped_indices,
                selected_sequence_ids=selected_sequence_ids,
                module=module,
                device=device,
                selection_mode=args.selection_mode,
                selection_metric=args.selection_metric,
                min_improvement=args.min_improvement,
                frame_step=args.frame_step,
                max_frames=args.max_frames,
                max_sequences=args.max_sequences,
                input_smpl_cache_dir=input_smpl_cache_dir,
            )

        summaries = []
        for sequence_id in selected_sequence_ids:
            frame_indices = grouped_indices[sequence_id][:: args.frame_step]
            if args.max_frames is not None:
                frame_indices = frame_indices[: args.max_frames]
            if not frame_indices:
                continue

            reference_batch = [dataset[frame_indices[0]]]
            reference_meta = reference_batch[0]["meta"]
            reference_image_path = resolve_rgb_image_path(
                rgb_dir,
                sequence_id=sequence_id,
                camera_id=str(reference_meta["camera_ids"][0]),
                frame_id=str(reference_meta["frame_id"]),
            )
            reference_image = cv2.imread(str(reference_image_path), cv2.IMREAD_COLOR)
            if reference_image is None:
                raise FileNotFoundError(f"Failed to read RGB image: {reference_image_path}")
            panel_height, panel_width = reference_image.shape[:2]
            fixed_view_scene = build_sequence_fixed_view_scene(
                dataset=dataset,
                sequence_frame_indices=frame_indices,
                module=module,
                smpl_model=smpl_model,
                device=device,
                cameras_dir=cameras_dir,
                width=panel_width,
                height=panel_height,
                smoothing_window=args.smoothing_window,
                input_smpl_cache_dir=input_smpl_cache_dir,
            )
            fixed_view_scene = zoom_fixed_view_scene(fixed_view_scene, zoom_factor=1.15)
            ordered_view_indices = rank_sequence_views(
                dataset=dataset,
                sequence_frame_indices=frame_indices,
                module=module,
                device=device,
                input_smpl_cache_dir=input_smpl_cache_dir,
            )

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
                    overlay_renderer=overlay_renderer,
                    dataset=dataset,
                    sequence_frame_indices=frame_indices,
                    dataset_index=dataset_index,
                    smoothing_window=args.smoothing_window,
                    input_smpl_cache_dir=input_smpl_cache_dir,
                    fixed_view_scene=fixed_view_scene,
                    ordered_view_indices=ordered_view_indices,
                    pred_camera_mode=args.pred_camera_mode,
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
                        frame_image = cv2.resize(frame_image, (frame_width, frame_height))
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
                    "view_order": [int(index) for index in ordered_view_indices],
                }
            )
            print(f"Saved {written_frames} frames to {video_path}")
    finally:
        overlay_renderer.close()

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({"videos": summaries}, indent=2), encoding="utf-8")
    print(f"Saved summary to {summary_path}")


def rank_sequence_views(
    *,
    dataset,
    sequence_frame_indices: list[int],
    module,
    device: torch.device,
    input_smpl_cache_dir: Path,
) -> list[int]:
    if not sequence_frame_indices:
        return [0, 1]
    first_sample = dataset[sequence_frame_indices[0]]
    num_views = len(first_sample["meta"]["camera_ids"])
    if num_views != 2:
        raise ValueError(f"visualize_video_new.py expects exactly 2 selected views, got {num_views}")

    per_view_scores = np.zeros(num_views, dtype=np.float64)
    count = 0
    with torch.no_grad():
        for dataset_index in sequence_frame_indices:
            batch = multiview_collate([dataset[dataset_index]])
            scores = compute_frame_input_view_mpjpe(
                batch=batch,
                module=module,
                device=device,
                input_smpl_cache_dir=input_smpl_cache_dir,
            )
            per_view_scores += np.asarray(scores, dtype=np.float64)
            count += 1
    if count == 0:
        return [0, 1]
    mean_scores = per_view_scores / float(count)
    return list(np.argsort(-mean_scores))


def compute_frame_input_view_mpjpe(
    *,
    batch: dict[str, Any],
    module,
    device: torch.device,
    input_smpl_cache_dir: Path,
) -> list[float]:
    views_input = batch["views_input"].to(device)
    pred_cam_t = batch["view_aux"]["pred_cam_t"].to(device)
    target_body_pose = batch["target_body_pose"].to(device)
    target_betas = batch["target_betas"].to(device)
    input_view_smpl = resolve_input_view_smpl_params(
        module=module,
        views_input=views_input,
        pred_cam_t=pred_cam_t,
        batch_meta=batch["meta"],
        input_smpl_cache_dir=input_smpl_cache_dir,
        device=device,
    )
    num_views = pred_cam_t.shape[1]
    scores: list[float] = []
    for view_index in range(num_views):
        input_body_pose = input_view_smpl["body_pose"][view_index : view_index + 1]
        input_betas = input_view_smpl["betas"][view_index : view_index + 1]
        input_global_orient = input_view_smpl["global_orient"][view_index : view_index + 1]
        input_transl = input_view_smpl["transl"][view_index : view_index + 1]
        pred_joints = module._build_smpl_joints(
            body_pose=input_body_pose,
            betas=input_betas,
            global_orient=input_global_orient,
            transl=input_transl,
        )
        target_joints = module._build_smpl_joints(
            body_pose=target_body_pose,
            betas=target_betas,
            global_orient=input_global_orient,
            transl=input_transl,
        )
        scores.append(float(batch_mpjpe(pred_joints, target_joints).detach().cpu().item()))
    return scores


def build_video_frame(
    *,
    batch: dict[str, Any],
    module,
    smpl_model,
    faces: np.ndarray,
    device: torch.device,
    rgb_dir: Path,
    cameras_dir: Path,
    overlay_renderer: CameraMeshOverlayRenderer,
    dataset,
    sequence_frame_indices: list[int],
    dataset_index: int,
    smoothing_window: int,
    input_smpl_cache_dir: Path,
    fixed_view_scene: dict[str, np.ndarray | float],
    ordered_view_indices: list[int],
    pred_camera_mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    meta = batch["meta"][0]
    sequence_id = str(meta["sequence_id"])
    frame_id = str(meta["frame_id"])

    views_input = batch["views_input"].to(device)
    pred_cam_t = batch["view_aux"]["pred_cam_t"].to(device)
    with torch.no_grad():
        predictions = module(views_input)
    pred_body_pose, pred_betas = get_visualization_prediction(
        batch=batch,
        predictions=predictions,
        module=module,
        dataset=dataset,
        sequence_frame_indices=sequence_frame_indices,
        dataset_index=dataset_index,
        device=device,
        smoothing_window=smoothing_window,
    )
    input_view_smpl = resolve_input_view_smpl_params(
        module=module,
        views_input=views_input,
        pred_cam_t=pred_cam_t,
        batch_meta=batch["meta"],
        input_smpl_cache_dir=input_smpl_cache_dir,
        device=device,
    )

    gt_body_pose = batch["target_body_pose"][0].detach().cpu().numpy()
    gt_betas = batch["target_betas"][0].detach().cpu().numpy()
    gt_global_orient = batch["target_aux"]["global_orient"][0].detach().cpu().numpy()
    gt_transl = batch["target_aux"]["transl"][0].detach().cpu().numpy()
    pred_world_vertices, pred_world_joints = build_smpl_outputs(
        smpl_model=smpl_model,
        device=device,
        body_pose=pred_body_pose,
        betas=pred_betas,
        global_orient=gt_global_orient,
        transl=gt_transl,
    )
    gt_world_vertices, gt_world_joints = build_smpl_outputs(
        smpl_model=smpl_model,
        device=device,
        body_pose=gt_body_pose,
        betas=gt_betas,
        global_orient=gt_global_orient,
        transl=gt_transl,
    )

    worse_view_index, better_view_index = ordered_view_indices
    frame_input_scores = compute_frame_input_view_mpjpe(
        batch=batch,
        module=module,
        device=device,
        input_smpl_cache_dir=input_smpl_cache_dir,
    )

    worse_rgb = load_view_rgb(
        rgb_dir=rgb_dir,
        sequence_id=sequence_id,
        camera_id=str(meta["camera_ids"][worse_view_index]),
        frame_id=frame_id,
    )
    better_rgb = load_view_rgb(
        rgb_dir=rgb_dir,
        sequence_id=sequence_id,
        camera_id=str(meta["camera_ids"][better_view_index]),
        frame_id=frame_id,
    )
    panel_height, panel_width = worse_rgb.shape[:2]

    worse_input_world_vertices, worse_input_world_joints = build_input_world_mesh(
        smpl_model=smpl_model,
        device=device,
        input_view_smpl=input_view_smpl,
        sequence_id=sequence_id,
        camera_id=str(meta["camera_ids"][worse_view_index]),
        cameras_dir=cameras_dir,
        view_index=worse_view_index,
    )
    worse_input_camera_vertices, _ = build_smpl_outputs(
        smpl_model=smpl_model,
        device=device,
        body_pose=input_view_smpl["body_pose"][worse_view_index].detach().cpu().numpy(),
        betas=input_view_smpl["betas"][worse_view_index].detach().cpu().numpy(),
        global_orient=input_view_smpl["global_orient"][worse_view_index].detach().cpu().numpy(),
        transl=input_view_smpl["transl"][worse_view_index].detach().cpu().numpy(),
    )
    worse_input_canonical_joints = build_smpl_outputs(
        smpl_model=smpl_model,
        device=device,
        body_pose=input_view_smpl["body_pose"][worse_view_index].detach().cpu().numpy(),
        betas=input_view_smpl["betas"][worse_view_index].detach().cpu().numpy(),
        global_orient=np.zeros(3, dtype=np.float32),
        transl=np.zeros(3, dtype=np.float32),
    )[1]
    canonical_pred_joints = build_smpl_outputs(
        smpl_model=smpl_model,
        device=device,
        body_pose=pred_body_pose,
        betas=pred_betas,
        global_orient=np.zeros(3, dtype=np.float32),
        transl=np.zeros(3, dtype=np.float32),
    )[1]
    gt_view_a, gt_view_b = build_mesh_views(
        overlay_renderer=overlay_renderer,
        vertices=gt_world_vertices,
        joints=gt_world_joints,
        faces=faces,
        width=panel_width,
        height=panel_height,
        scene_spec=fixed_view_scene,
        color_bgr=(85, 200, 95),
    )
    input_view_a, input_view_b = build_mesh_views(
        overlay_renderer=overlay_renderer,
        vertices=worse_input_world_vertices,
        joints=worse_input_world_joints,
        faces=faces,
        width=panel_width,
        height=panel_height,
        scene_spec=fixed_view_scene,
        color_bgr=(0, 150, 225),
    )
    pred_view_a, pred_view_b = build_mesh_views(
        overlay_renderer=overlay_renderer,
        vertices=pred_world_vertices,
        joints=pred_world_joints,
        faces=faces,
        width=panel_width,
        height=panel_height,
        scene_spec=fixed_view_scene,
        color_bgr=(55, 85, 235),
    )

    worse_camera = load_camera_parameters(
        cameras_dir,
        sequence_id=sequence_id,
        camera_id=str(meta["camera_ids"][worse_view_index]),
    )
    gt_overlay = build_world_mesh_overlay(
        overlay_renderer=overlay_renderer,
        image_bgr=worse_rgb,
        vertices_world=gt_world_vertices,
        faces=faces,
        camera=worse_camera,
        color_bgr=(85, 200, 95),
    )
    input_overlay = build_camera_mesh_overlay(
        overlay_renderer=overlay_renderer,
        image_bgr=worse_rgb,
        vertices_camera=worse_input_camera_vertices,
        intrinsics=batch["view_aux"]["cam_int"][0, worse_view_index].detach().cpu().numpy(),
        faces=faces,
        color_bgr=(0, 150, 225),
    )
    if pred_camera_mode == "gt":
        pred_overlay = build_world_mesh_overlay(
            overlay_renderer=overlay_renderer,
            image_bgr=worse_rgb,
            vertices_world=pred_world_vertices,
            faces=faces,
            camera=worse_camera,
            color_bgr=(55, 85, 235),
        )
    else:
        pred_cam_int = batch["view_aux"]["cam_int"][0, worse_view_index].detach().cpu().numpy()
        pred_global_orient = input_view_smpl["global_orient"][worse_view_index].detach().cpu().numpy()
        if pred_camera_mode == "input_corrected":
            pred_global_orient = correct_camera_global_orient_using_torso(
                input_canonical_joints=worse_input_canonical_joints,
                pred_canonical_joints=canonical_pred_joints,
                input_camera_global_orient=pred_global_orient,
            )
        pred_vertices_camera, pred_joints_camera = build_smpl_outputs(
            smpl_model=smpl_model,
            device=device,
            body_pose=pred_body_pose,
            betas=pred_betas,
            global_orient=pred_global_orient,
            transl=np.zeros(3, dtype=np.float32),
        )
        _, input_joints_camera = build_smpl_outputs(
            smpl_model=smpl_model,
            device=device,
            body_pose=input_view_smpl["body_pose"][worse_view_index].detach().cpu().numpy(),
            betas=input_view_smpl["betas"][worse_view_index].detach().cpu().numpy(),
            global_orient=input_view_smpl["global_orient"][worse_view_index].detach().cpu().numpy(),
            transl=input_view_smpl["transl"][worse_view_index].detach().cpu().numpy(),
        )
        pred_translation_offset = input_joints_camera[0] - pred_joints_camera[0]
        pred_vertices_camera = np.ascontiguousarray(
            pred_vertices_camera + pred_translation_offset.reshape(1, 3)
        )
        pred_overlay = build_camera_mesh_overlay(
            overlay_renderer=overlay_renderer,
            image_bgr=worse_rgb,
            vertices_camera=pred_vertices_camera,
            intrinsics=pred_cam_int,
            faces=faces,
            color_bgr=(55, 85, 235),
        )

    info_image = build_info_panel(width=panel_width, height=panel_height)

    panels = [
        build_view_panel([("View 1 RGB", worse_rgb)], footer=""),
        build_view_panel([("View 2 RGB", better_rgb)], footer=""),
        info_image,
        build_view_panel([("Ground Truth", gt_overlay)], footer=""),
        build_view_panel([("Ground Truth", gt_view_a)], footer=""),
        build_view_panel([("Ground Truth", gt_view_b)], footer=""),
        build_view_panel([("View 1 SAM3DBody", input_overlay)], footer=""),
        build_view_panel([("View 1 SAM3DBody", input_view_a)], footer=""),
        build_view_panel([("View 1 SAM3DBody", input_view_b)], footer=""),
        build_view_panel([("Predicted (Ours)", pred_overlay)], footer=""),
        build_view_panel([("Predicted (Ours)", pred_view_a)], footer=""),
        build_view_panel([("Predicted (Ours)", pred_view_b)], footer=""),
    ]
    frame_image = build_grid_contact_sheet(
        title_lines=[
            f"sequence_id: {sequence_id}",
        ],
        grid_panels=panels,
        num_rows=4,
        num_cols=3,
    )
    return frame_image, {
        "sequence_id": sequence_id,
        "frame_id": frame_id,
        "ordered_view_indices": ordered_view_indices,
        "pred_camera_mode": pred_camera_mode,
    }


def build_input_world_mesh(
    *,
    smpl_model,
    device: torch.device,
    input_view_smpl: dict[str, torch.Tensor],
    sequence_id: str,
    camera_id: str,
    cameras_dir: Path,
    view_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    input_vertices_camera, input_joints_camera = build_smpl_outputs(
        smpl_model=smpl_model,
        device=device,
        body_pose=input_view_smpl["body_pose"][view_index].detach().cpu().numpy(),
        betas=input_view_smpl["betas"][view_index].detach().cpu().numpy(),
        global_orient=input_view_smpl["global_orient"][view_index].detach().cpu().numpy(),
        transl=input_view_smpl["transl"][view_index].detach().cpu().numpy(),
    )
    camera = load_camera_parameters(
        cameras_dir,
        sequence_id=sequence_id,
        camera_id=camera_id,
    )
    return (
        transform_camera_vertices_to_world(
            vertices_camera=input_vertices_camera,
            rotation=camera.rotation,
            translation=camera.translation,
        ),
        transform_camera_vertices_to_world(
            vertices_camera=input_joints_camera,
            rotation=camera.rotation,
            translation=camera.translation,
        ),
    )


def build_mesh_views(
    *,
    overlay_renderer: CameraMeshOverlayRenderer,
    vertices: np.ndarray,
    joints: np.ndarray,
    faces: np.ndarray,
    width: int,
    height: int,
    scene_spec: dict[str, np.ndarray | float],
    color_bgr: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    scene_view_b = scene_spec
    scene_view_a = rotate_scene_spec(scene_spec, yaw_delta_degrees=90.0)
    image_a = overlay_renderer.render_fixed_view(
        vertices=vertices,
        joints=joints,
        faces=faces,
        color_bgr=color_bgr,
        width=width,
        height=height,
        scene_spec=scene_view_a,
    )
    image_b = overlay_renderer.render_fixed_view(
        vertices=vertices,
        joints=joints,
        faces=faces,
        color_bgr=color_bgr,
        width=width,
        height=height,
        scene_spec=scene_view_b,
    )
    return image_a, image_b


def build_world_mesh_overlay(
    *,
    overlay_renderer: CameraMeshOverlayRenderer,
    image_bgr: np.ndarray,
    vertices_world: np.ndarray,
    faces: np.ndarray,
    camera,
    color_bgr: tuple[int, int, int],
) -> np.ndarray:
    vertices_camera = (camera.rotation @ np.asarray(vertices_world, dtype=np.float32).T).T
    vertices_camera = vertices_camera + np.asarray(camera.translation, dtype=np.float32).reshape(1, 3)
    overlay_image, _ = overlay_renderer.render_overlay(
        image_bgr=image_bgr,
        vertices_camera=vertices_camera,
        faces=faces,
        intrinsics=np.asarray(camera.intrinsics, dtype=np.float32),
        color_bgr=color_bgr,
        alpha=0.55,
    )
    return overlay_image


def build_camera_mesh_overlay(
    *,
    overlay_renderer: CameraMeshOverlayRenderer,
    image_bgr: np.ndarray,
    vertices_camera: np.ndarray,
    intrinsics: np.ndarray,
    faces: np.ndarray,
    color_bgr: tuple[int, int, int],
) -> np.ndarray:
    overlay_image, _ = overlay_renderer.render_overlay(
        image_bgr=image_bgr,
        vertices_camera=np.asarray(vertices_camera, dtype=np.float32),
        faces=faces,
        intrinsics=np.asarray(intrinsics, dtype=np.float32),
        color_bgr=color_bgr,
        alpha=0.55,
    )
    return overlay_image


def load_view_rgb(*, rgb_dir: Path, sequence_id: str, camera_id: str, frame_id: str) -> np.ndarray:
    image_path = resolve_rgb_image_path(
        rgb_dir,
        sequence_id=sequence_id,
        camera_id=camera_id,
        frame_id=frame_id,
    )
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read RGB image: {image_path}")
    return image_bgr


def build_info_panel(*, width: int, height: int) -> np.ndarray:
    return np.full((height + 34, width, 3), 255, dtype=np.uint8)


def rotate_scene_spec(
    scene_spec: dict[str, np.ndarray | float],
    *,
    yaw_delta_degrees: float,
) -> dict[str, np.ndarray | float]:
    yaw = np.deg2rad(float(yaw_delta_degrees))
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rotation_delta = np.array(
        [
            [cos_yaw, 0.0, sin_yaw],
            [0.0, 1.0, 0.0],
            [-sin_yaw, 0.0, cos_yaw],
        ],
        dtype=np.float32,
    )
    rotated = dict(scene_spec)
    rotated["rotation"] = rotation_delta @ np.asarray(scene_spec["rotation"], dtype=np.float32)
    return rotated


def build_grid_contact_sheet(
    *,
    title_lines: list[str],
    grid_panels: list[np.ndarray],
    num_rows: int,
    num_cols: int,
) -> np.ndarray:
    if len(grid_panels) != num_rows * num_cols:
        raise ValueError(
            f"Expected {num_rows * num_cols} panels, got {len(grid_panels)}"
        )
    margin = 24
    text_height = 28
    row_gap = 18
    col_gap = 18
    row_heights = [max(grid_panels[row * num_cols + col].shape[0] for col in range(num_cols)) for row in range(num_rows)]
    col_widths = [max(grid_panels[row * num_cols + col].shape[1] for row in range(num_rows)) for col in range(num_cols)]
    title_height = margin + len(title_lines) * text_height + 8
    total_width = 2 * margin + sum(col_widths) + col_gap * (num_cols - 1)
    total_height = title_height + sum(row_heights) + row_gap * (num_rows - 1) + margin
    canvas = np.full((total_height, total_width, 3), 255, dtype=np.uint8)

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

    for row in range(num_rows):
        x = margin
        for col in range(num_cols):
            panel = grid_panels[row * num_cols + col]
            cell_width = col_widths[col]
            cell_height = row_heights[row]
            x_offset = x + max((cell_width - panel.shape[1]) // 2, 0)
            y_offset = y + max((cell_height - panel.shape[0]) // 2, 0)
            canvas[y_offset : y_offset + panel.shape[0], x_offset : x_offset + panel.shape[1]] = panel
            x += cell_width + col_gap
        y += row_heights[row] + row_gap
    return canvas


def zoom_fixed_view_scene(
    scene_spec: dict[str, np.ndarray | float],
    *,
    zoom_factor: float,
) -> dict[str, np.ndarray | float]:
    if zoom_factor <= 0:
        raise ValueError(f"zoom_factor must be > 0, got {zoom_factor}")
    zoomed = dict(scene_spec)
    zoomed["depth"] = float(scene_spec["depth"]) / float(zoom_factor)
    return zoomed


if __name__ == "__main__":
    main()
