#!/usr/bin/env python
"""Render a 2x2 comparison demo for two samples and their rotating 3D meshes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mvhpe3d.utils import build_smpl_model, validate_mhr_asset_folder
from visualize_video import (
    CameraMeshOverlayRenderer,
    build_data_config,
    build_datamodule,
    build_model_config,
    build_sequence_index,
    get_visualization_prediction,
    is_stage2_experiment,
    load_camera_parameters,
    load_experiment_config,
    load_visualization_module,
    multiview_collate,
    resolve_device,
    resolve_required_dir,
    resolve_rgb_image_path,
    resolve_smpl_model_path,
    select_dataset,
)

from demo_video_2d_3d_compare import (
    build_smpl_outputs,
    build_view_panel,
    render_mesh_panel,
    resolve_reference_dataset_index,
    rotate_geometry_about_world_vertical,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a 2x2 demo with two static RGB samples and two rotating 3D mesh panels"
    )
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment/stage2_cross_camera_joint_graph_refiner.yaml",
        help="Path to the experiment YAML file",
    )
    parser.add_argument("--manifest-path", type=str, default=None, help="Optional manifest override")
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
    parser.add_argument("--output-dir", type=str, default="outputs/demo", help="Output directory")
    parser.add_argument("--output-path", type=str, default=None, help="Optional explicit MP4 path")
    parser.add_argument("--fps", type=float, default=10.0, help="Frames per second for MP4")
    parser.add_argument("--codec", type=str, default="mp4v", help="FourCC codec for cv2.VideoWriter")
    parser.add_argument("--sequence-a", type=str, required=True, help="Sequence ID for sample A")
    parser.add_argument("--view-a", type=str, required=True, help="Camera ID for sample A static view")
    parser.add_argument("--sequence-b", type=str, required=True, help="Sequence ID for sample B")
    parser.add_argument("--view-b", type=str, required=True, help="Camera ID for sample B static view")
    parser.add_argument("--frame-step", type=int, default=1, help="Use every Nth frame in each sequence")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on frames per sequence")
    parser.add_argument(
        "--reference-frame-a",
        type=str,
        choices=("first", "middle", "random", "index"),
        default="first",
        help="How to choose sample A reference frame",
    )
    parser.add_argument(
        "--reference-frame-index-a",
        type=int,
        default=None,
        help="0-based frame index for sample A when --reference-frame-a=index",
    )
    parser.add_argument(
        "--reference-frame-b",
        type=str,
        choices=("first", "middle", "random", "index"),
        default="first",
        help="How to choose sample B reference frame",
    )
    parser.add_argument(
        "--reference-frame-index-b",
        type=int,
        default=None,
        help="0-based frame index for sample B when --reference-frame-b=index",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reference frame sampling")
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=1,
        help="Centered temporal smoothing window for predicted SMPL params; 1 disables smoothing",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=1.0,
        help="Initial static hold duration before rotation starts",
    )
    parser.add_argument(
        "--rotate-seconds",
        type=float,
        default=3.0,
        help="Rotation duration after the static hold",
    )
    parser.add_argument(
        "--rotation-degrees",
        type=float,
        default=360.0,
        help="Total yaw rotation angle for the 3D mesh panels",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_step < 1:
        raise ValueError(f"--frame-step must be >= 1, got {args.frame_step}")
    if args.fps <= 0:
        raise ValueError(f"--fps must be > 0, got {args.fps}")
    if args.smoothing_window < 1:
        raise ValueError(f"--smoothing-window must be >= 1, got {args.smoothing_window}")
    if args.hold_seconds < 0:
        raise ValueError(f"--hold-seconds must be >= 0, got {args.hold_seconds}")
    if args.rotate_seconds <= 0:
        raise ValueError(f"--rotate-seconds must be > 0, got {args.rotate_seconds}")

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
    grouped_indices = build_sequence_index(dataset)

    device = resolve_device(args.device)
    module = load_visualization_module(
        checkpoint_path=args.checkpoint_path,
        model_config=model_config,
        data_config=data_config,
        args=args,
    )
    module.eval()
    module.to(device)

    data_root = Path(data_config.manifest_path).resolve().parent
    rgb_dir = resolve_required_dir(args.rgb_dir, fallback=data_root / "rgb", name="rgb")
    cameras_dir = resolve_required_dir(args.cameras_dir, fallback=data_root / "cameras", name="cameras")
    smpl_model = build_smpl_model(
        device=device,
        smpl_model_path=str(resolve_smpl_model_path(args.smpl_model_path)),
        batch_size=1,
    )
    faces = np.asarray(smpl_model.faces, dtype=np.int32)
    overlay_renderer = CameraMeshOverlayRenderer()

    try:
        sample_a = build_demo_sample(
            dataset=dataset,
            grouped_indices=grouped_indices,
            sequence_id=args.sequence_a,
            camera_id=args.view_a,
            reference_frame=args.reference_frame_a,
            reference_frame_index=args.reference_frame_index_a,
            frame_step=args.frame_step,
            max_frames=args.max_frames,
            seed=args.seed,
            module=module,
            smpl_model=smpl_model,
            device=device,
            rgb_dir=rgb_dir,
            cameras_dir=cameras_dir,
            smoothing_window=args.smoothing_window,
        )
        sample_b = build_demo_sample(
            dataset=dataset,
            grouped_indices=grouped_indices,
            sequence_id=args.sequence_b,
            camera_id=args.view_b,
            reference_frame=args.reference_frame_b,
            reference_frame_index=args.reference_frame_index_b,
            frame_step=args.frame_step,
            max_frames=args.max_frames,
            seed=args.seed,
            module=module,
            smpl_model=smpl_model,
            device=device,
            rgb_dir=rgb_dir,
            cameras_dir=cameras_dir,
            smoothing_window=args.smoothing_window,
        )

        hold_frames = max(int(round(args.hold_seconds * args.fps)), 1)
        rotate_frames = max(int(round(args.rotate_seconds * args.fps)), 1)
        initial_frame = build_2x2_frame(
            sample_a_rgb=sample_a["rgb"],
            sample_a_mesh=render_mesh_panel(
                overlay_renderer=overlay_renderer,
                vertices_world=sample_a["vertices"],
                joints_world=sample_a["joints"],
                faces=faces,
                width=sample_a["rgb"].shape[1],
                height=sample_a["rgb"].shape[0],
                camera=sample_a["camera"],
            ),
            sample_b_rgb=sample_b["rgb"],
            sample_b_mesh=render_mesh_panel(
                overlay_renderer=overlay_renderer,
                vertices_world=sample_b["vertices"],
                joints_world=sample_b["joints"],
                faces=faces,
                width=sample_b["rgb"].shape[1],
                height=sample_b["rgb"].shape[0],
                camera=sample_b["camera"],
            ),
            title_a=args.sequence_a,
            title_b=args.sequence_b,
        )

        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = (
            Path(args.output_path).resolve()
            if args.output_path is not None
            else (output_dir / f"{args.sequence_a}__{args.sequence_b}.mp4").resolve()
        )
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*args.codec),
            args.fps,
            (initial_frame.shape[1], initial_frame.shape[0]),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path} with codec '{args.codec}'")
        try:
            for frame_index in range(hold_frames + rotate_frames):
                if frame_index < hold_frames:
                    current_a_vertices = sample_a["vertices"]
                    current_a_joints = sample_a["joints"]
                    current_b_vertices = sample_b["vertices"]
                    current_b_joints = sample_b["joints"]
                else:
                    progress = (frame_index - hold_frames) / max(rotate_frames - 1, 1)
                    angle = float(progress * args.rotation_degrees)
                    current_a_vertices, current_a_joints = rotate_geometry_about_world_vertical(
                        vertices=sample_a["vertices"],
                        joints=sample_a["joints"],
                        angle_degrees=angle,
                        pivot=sample_a["pivot"],
                    )
                    current_b_vertices, current_b_joints = rotate_geometry_about_world_vertical(
                        vertices=sample_b["vertices"],
                        joints=sample_b["joints"],
                        angle_degrees=angle,
                        pivot=sample_b["pivot"],
                    )

                frame = build_2x2_frame(
                    sample_a_rgb=sample_a["rgb"],
                    sample_a_mesh=render_mesh_panel(
                        overlay_renderer=overlay_renderer,
                        vertices_world=current_a_vertices,
                        joints_world=current_a_joints,
                        faces=faces,
                        width=sample_a["rgb"].shape[1],
                        height=sample_a["rgb"].shape[0],
                        camera=sample_a["camera"],
                    ),
                    sample_b_rgb=sample_b["rgb"],
                    sample_b_mesh=render_mesh_panel(
                        overlay_renderer=overlay_renderer,
                        vertices_world=current_b_vertices,
                        joints_world=current_b_joints,
                        faces=faces,
                        width=sample_b["rgb"].shape[1],
                        height=sample_b["rgb"].shape[0],
                        camera=sample_b["camera"],
                    ),
                    title_a=args.sequence_a,
                    title_b=args.sequence_b,
                )
                writer.write(frame)
        finally:
            writer.release()

        print(f"Saved {hold_frames + rotate_frames} frames to {output_path}")
    finally:
        overlay_renderer.close()


def build_demo_sample(
    *,
    dataset,
    grouped_indices: dict[str, list[int]],
    sequence_id: str,
    camera_id: str,
    reference_frame: str,
    reference_frame_index: int | None,
    frame_step: int,
    max_frames: int | None,
    seed: int,
    module,
    smpl_model,
    device: torch.device,
    rgb_dir: Path,
    cameras_dir: Path,
    smoothing_window: int,
) -> dict[str, object]:
    try:
        frame_indices = grouped_indices[sequence_id][::frame_step]
    except KeyError as exc:
        raise KeyError(f"sequence_id '{sequence_id}' was not found in the selected split") from exc
    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]
    if not frame_indices:
        raise ValueError(f"sequence_id '{sequence_id}' has no frames after selection")

    reference_dataset_index = resolve_reference_dataset_index(
        frame_indices=frame_indices,
        reference_frame=reference_frame,
        reference_frame_index=reference_frame_index,
        seed=seed,
        sequence_id=sequence_id,
    )
    batch = multiview_collate([dataset[reference_dataset_index]])
    meta = batch["meta"][0]
    camera_ids = [str(item) for item in meta["camera_ids"]]
    try:
        view_index = camera_ids.index(camera_id)
    except ValueError as exc:
        raise KeyError(
            f"Requested camera '{camera_id}' is not present in selected views {camera_ids} for sequence {sequence_id}"
        ) from exc

    with torch.no_grad():
        predictions = module(batch["views_input"].to(device))
    pred_body_pose, pred_betas = get_visualization_prediction(
        batch=batch,
        predictions=predictions,
        module=module,
        dataset=dataset,
        sequence_frame_indices=frame_indices,
        dataset_index=reference_dataset_index,
        device=device,
        smoothing_window=smoothing_window,
    )
    gt_global_orient = batch["target_aux"]["global_orient"][0].detach().cpu().numpy()
    gt_transl = batch["target_aux"]["transl"][0].detach().cpu().numpy()
    vertices, joints = build_smpl_outputs(
        smpl_model=smpl_model,
        device=device,
        body_pose=pred_body_pose,
        betas=pred_betas,
        global_orient=gt_global_orient,
        transl=gt_transl,
    )

    frame_id = str(meta["frame_id"])
    image_path = resolve_rgb_image_path(
        rgb_dir,
        sequence_id=sequence_id,
        camera_id=camera_id,
        frame_id=frame_id,
    )
    rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"Failed to read RGB image: {image_path}")
    camera = load_camera_parameters(
        cameras_dir,
        sequence_id=sequence_id,
        camera_id=camera_id,
    )
    pivot = 0.5 * (np.min(vertices, axis=0) + np.max(vertices, axis=0))
    return {
        "rgb": rgb,
        "vertices": vertices,
        "joints": joints,
        "camera": camera,
        "pivot": pivot.astype(np.float32, copy=False),
    }


def build_2x2_frame(
    *,
    sample_a_rgb: np.ndarray,
    sample_a_mesh: np.ndarray,
    sample_b_rgb: np.ndarray,
    sample_b_mesh: np.ndarray,
    title_a: str,
    title_b: str,
) -> np.ndarray:
    top_panel = build_view_panel(
        [
            (title_a, sample_a_rgb),
            (f"{title_a} 3D", sample_a_mesh),
        ],
        footer="",
    )
    bottom_panel = build_view_panel(
        [
            (title_b, sample_b_rgb),
            (f"{title_b} 3D", sample_b_mesh),
        ],
        footer="",
    )
    width = max(top_panel.shape[1], bottom_panel.shape[1])
    gap = 16
    canvas = np.full(
        (top_panel.shape[0] + bottom_panel.shape[0] + gap, width, 3),
        255,
        dtype=np.uint8,
    )
    top_x = (width - top_panel.shape[1]) // 2
    bottom_x = (width - bottom_panel.shape[1]) // 2
    canvas[: top_panel.shape[0], top_x : top_x + top_panel.shape[1]] = top_panel
    start_y = top_panel.shape[0] + gap
    canvas[start_y : start_y + bottom_panel.shape[0], bottom_x : bottom_x + bottom_panel.shape[1]] = bottom_panel
    return canvas


if __name__ == "__main__":
    main()
