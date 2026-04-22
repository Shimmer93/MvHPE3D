#!/usr/bin/env python
"""Render a simple three-panel demo with static RGB, 2D pose, and rotating 3D mesh."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

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
    build_smpl_outputs,
    build_view_panel,
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
    transform_world_vertices_to_camera,
)

COCO_KEYPOINT_EDGES = (
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
)

SMPL_SKELETON_EDGES = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (2, 5),
    (3, 6),
    (4, 7),
    (5, 8),
    (6, 9),
    (7, 10),
    (8, 11),
    (9, 12),
    (12, 13),
    (12, 14),
    (12, 15),
    (13, 16),
    (14, 17),
    (16, 18),
    (17, 19),
    (18, 20),
    (19, 21),
    (20, 22),
    (21, 23),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a demo video with static RGB, static 2D pose, and a rotating 3D mesh"
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
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/demo_videos",
        help="Directory to save MP4 videos",
    )
    parser.add_argument(
        "--asset-dir",
        type=str,
        default="outputs/demo",
        help="Directory to save exported demo meshes and copied reference RGB images",
    )
    parser.add_argument("--output-path", type=str, default=None, help="Optional single-sequence MP4 path")
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
    parser.add_argument("--fps", type=float, default=10.0, help="Frames per second for MP4")
    parser.add_argument("--codec", type=str, default="mp4v", help="FourCC codec for cv2.VideoWriter")
    parser.add_argument("--sequence-id", type=str, default=None, help="Optional single sequence_id")
    parser.add_argument("--max-sequences", type=int, default=None, help="Optional cap on output videos")
    parser.add_argument(
        "--random-selection",
        action="store_true",
        help="Randomly sample sequences when --max-sequences is used",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sequence sampling")
    parser.add_argument("--frame-step", type=int, default=1, help="Use every Nth frame in each sequence")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on frames per sequence")
    parser.add_argument(
        "--reference-frame",
        type=str,
        choices=("first", "middle", "random", "index"),
        default="first",
        help="How to choose the static reference frame used for the demo panels",
    )
    parser.add_argument(
        "--reference-frame-index",
        type=int,
        default=None,
        help="0-based index into the selected sequence frames when --reference-frame=index",
    )
    parser.add_argument("--view-a", type=str, default=None, help="Optional camera id to use for the static frame")
    parser.add_argument(
        "--pred-camera-mode",
        type=str,
        choices=("input", "input_corrected", "gt"),
        default="gt",
        help="Accepted for CLI compatibility; ignored in this non-overlay demo",
    )
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
        help="Total yaw rotation angle for the 3D mesh panel",
    )
    parser.add_argument(
        "--detector-config",
        type=str,
        default="COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml",
        help="Detectron2 model-zoo config name",
    )
    parser.add_argument(
        "--detector-weights",
        type=str,
        default=None,
        help="Optional explicit Detectron2 weights path or URL",
    )
    parser.add_argument(
        "--detector-device",
        type=str,
        default=None,
        help="Device for Detectron2, e.g. cuda or cpu",
    )
    parser.add_argument(
        "--detector-score-threshold",
        type=float,
        default=0.7,
        help="Person detection score threshold",
    )
    parser.add_argument(
        "--keypoint-threshold",
        type=float,
        default=0.05,
        help="Keypoint score threshold used for drawing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_step < 1:
        raise ValueError(f"--frame-step must be >= 1, got {args.frame_step}")
    if args.fps <= 0:
        raise ValueError(f"--fps must be > 0, got {args.fps}")
    if args.max_sequences is not None and args.max_sequences < 1:
        raise ValueError(f"--max-sequences must be >= 1, got {args.max_sequences}")
    if args.reference_frame_index is not None and args.reference_frame_index < 0:
        raise ValueError(
            f"--reference-frame-index must be >= 0, got {args.reference_frame_index}"
        )
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
    asset_dir = Path(args.asset_dir).resolve()
    asset_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(data_config.manifest_path).resolve().parent
    rgb_dir = resolve_required_dir(args.rgb_dir, fallback=data_root / "rgb", name="rgb")
    cameras_dir = resolve_required_dir(args.cameras_dir, fallback=data_root / "cameras", name="cameras")
    smpl_model = build_smpl_model(
        device=device,
        smpl_model_path=str(resolve_smpl_model_path(args.smpl_model_path)),
        batch_size=1,
    )
    predictor = build_keypoint_predictor(args)
    overlay_renderer = CameraMeshOverlayRenderer()
    faces = np.asarray(smpl_model.faces, dtype=np.int32)

    grouped_indices = build_sequence_index(dataset)
    selected_sequence_ids = select_demo_sequence_ids(grouped_indices=grouped_indices, args=args)
    if len(selected_sequence_ids) > 1 and args.output_path is not None:
        raise ValueError("--output-path can only be used with a single sequence")

    try:
        for sequence_id in selected_sequence_ids:
            frame_indices = grouped_indices[sequence_id][:: args.frame_step]
            if args.max_frames is not None:
                frame_indices = frame_indices[: args.max_frames]
            if not frame_indices:
                continue

            reference_dataset_index = resolve_reference_dataset_index(
                frame_indices=frame_indices,
                reference_frame=args.reference_frame,
                reference_frame_index=args.reference_frame_index,
                seed=args.seed,
                sequence_id=sequence_id,
            )
            batch = multiview_collate([dataset[reference_dataset_index]])
            (
                static_rgb,
                static_pose2d,
                pred_vertices,
                pred_joints,
                footer,
                selected_camera,
                reference_image_path,
                reference_stem,
            ) = build_static_assets(
                batch=batch,
                module=module,
                smpl_model=smpl_model,
                device=device,
                rgb_dir=rgb_dir,
                cameras_dir=cameras_dir,
                dataset=dataset,
                sequence_frame_indices=frame_indices,
                dataset_index=reference_dataset_index,
                smoothing_window=args.smoothing_window,
                predictor=predictor,
                keypoint_threshold=args.keypoint_threshold,
                view_a_camera_id=args.view_a,
            )
            save_demo_assets(
                asset_dir=asset_dir,
                stem=reference_stem,
                pred_vertices=pred_vertices,
                faces=faces,
                reference_image_path=reference_image_path,
            )

            panel_height, panel_width = static_rgb.shape[:2]
            rotation_pivot = 0.5 * (
                np.min(pred_vertices, axis=0) + np.max(pred_vertices, axis=0)
            ).astype(np.float32, copy=False)
            initial_mesh_panel = render_mesh_panel(
                overlay_renderer=overlay_renderer,
                vertices_world=pred_vertices,
                joints_world=pred_joints,
                faces=faces,
                width=panel_width,
                height=panel_height,
                camera=selected_camera,
            )
            hold_frames = max(int(round(args.hold_seconds * args.fps)), 1)
            rotate_frames = max(int(round(args.rotate_seconds * args.fps)), 1)

            output_path = resolve_output_path(args=args, output_dir=output_dir, sequence_id=sequence_id)
            initial_contact_frame = build_contact_frame(
                static_rgb=static_rgb,
                static_pose2d=static_pose2d,
                mesh_panel=initial_mesh_panel,
                footer=footer,
            )
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*args.codec),
                args.fps,
                (initial_contact_frame.shape[1], initial_contact_frame.shape[0]),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer for {output_path} with codec '{args.codec}'")
            try:
                for frame_index in range(hold_frames + rotate_frames):
                    if frame_index < hold_frames:
                        current_vertices = pred_vertices
                        current_joints = pred_joints
                    else:
                        progress = (frame_index - hold_frames) / max(rotate_frames - 1, 1)
                        current_vertices, current_joints = rotate_geometry_about_world_vertical(
                            vertices=pred_vertices,
                            joints=pred_joints,
                            angle_degrees=float(progress * args.rotation_degrees),
                            pivot=rotation_pivot,
                        )
                    mesh_panel = render_mesh_panel(
                        overlay_renderer=overlay_renderer,
                        vertices_world=current_vertices,
                        joints_world=current_joints,
                        faces=faces,
                        width=panel_width,
                        height=panel_height,
                        camera=selected_camera,
                    )
                    writer.write(
                        build_contact_frame(
                            static_rgb=static_rgb,
                            static_pose2d=static_pose2d,
                            mesh_panel=mesh_panel,
                            footer=footer,
                        )
                    )
            finally:
                writer.release()

            print(f"Saved {hold_frames + rotate_frames} frames to {output_path}")
    finally:
        overlay_renderer.close()


def select_demo_sequence_ids(
    *,
    grouped_indices: dict[str, list[int]],
    args: argparse.Namespace,
) -> list[str]:
    available_sequence_ids = sorted(grouped_indices)
    if args.sequence_id is not None:
        if args.sequence_id not in grouped_indices:
            raise KeyError(f"sequence_id '{args.sequence_id}' was not found in the selected {args.stage} split")
        return [args.sequence_id]
    if args.max_sequences is None:
        return available_sequence_ids
    if args.random_selection:
        rng = random.Random(args.seed)
        sample_size = min(args.max_sequences, len(available_sequence_ids))
        return sorted(rng.sample(available_sequence_ids, sample_size))
    return available_sequence_ids[: args.max_sequences]


def resolve_output_path(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    sequence_id: str,
) -> Path:
    if args.output_path is not None:
        return Path(args.output_path).resolve()
    return (output_dir / f"{sequence_id}.mp4").resolve()


def resolve_reference_dataset_index(
    *,
    frame_indices: list[int],
    reference_frame: str,
    reference_frame_index: int | None,
    seed: int,
    sequence_id: str,
) -> int:
    if reference_frame == "first":
        return frame_indices[0]
    if reference_frame == "middle":
        return frame_indices[len(frame_indices) // 2]
    if reference_frame == "random":
        rng = random.Random(f"{seed}:{sequence_id}")
        return frame_indices[rng.randrange(len(frame_indices))]
    if reference_frame == "index":
        if reference_frame_index is None:
            raise ValueError(
                "--reference-frame=index requires --reference-frame-index"
            )
        if reference_frame_index >= len(frame_indices):
            raise ValueError(
                f"--reference-frame-index {reference_frame_index} is out of range for "
                f"{len(frame_indices)} selected frames"
            )
        return frame_indices[reference_frame_index]
    raise ValueError(f"Unsupported reference frame mode: {reference_frame}")


def build_keypoint_predictor(args: argparse.Namespace) -> DefaultPredictor:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.detector_config))
    cfg.MODEL.WEIGHTS = (
        args.detector_weights
        if args.detector_weights is not None
        else model_zoo.get_checkpoint_url(args.detector_config)
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.detector_score_threshold
    cfg.MODEL.DEVICE = args.detector_device or ("cuda" if torch.cuda.is_available() else "cpu")
    return DefaultPredictor(cfg)


def build_static_assets(
    *,
    batch: dict,
    module,
    smpl_model,
    device: torch.device,
    rgb_dir: Path,
    cameras_dir: Path,
    dataset,
    sequence_frame_indices: list[int],
    dataset_index: int,
    smoothing_window: int,
    predictor: DefaultPredictor,
    keypoint_threshold: float,
    view_a_camera_id: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, object, Path, str]:
    meta = batch["meta"][0]
    camera_ids = [str(camera_id) for camera_id in meta["camera_ids"]]
    view_index = resolve_view_index(camera_ids=camera_ids, view_a_camera_id=view_a_camera_id)
    sequence_id = str(meta["sequence_id"])
    frame_id = str(meta["frame_id"])

    image_bgr = load_view_rgb(
        rgb_dir=rgb_dir,
        sequence_id=sequence_id,
        camera_id=camera_ids[view_index],
        frame_id=frame_id,
    )
    reference_image_path = resolve_rgb_image_path(
        rgb_dir,
        sequence_id=sequence_id,
        camera_id=camera_ids[view_index],
        frame_id=frame_id,
    ).resolve()
    selected_camera = load_camera_parameters(
        cameras_dir,
        sequence_id=sequence_id,
        camera_id=camera_ids[view_index],
    )
    pose2d_panel = render_pose2d_panel(
        image_shape=image_bgr.shape,
        predictor=predictor,
        image_bgr=image_bgr,
        keypoint_threshold=keypoint_threshold,
    )

    with torch.no_grad():
        predictions = module(batch["views_input"].to(device))
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
    gt_global_orient = batch["target_aux"]["global_orient"][0].detach().cpu().numpy()
    gt_transl = batch["target_aux"]["transl"][0].detach().cpu().numpy()
    pred_vertices, pred_joints = build_smpl_outputs(
        smpl_model=smpl_model,
        device=device,
        body_pose=pred_body_pose,
        betas=pred_betas,
        global_orient=gt_global_orient,
        transl=gt_transl,
    )
    footer = f"{sequence_id} | frame {frame_id} | view {camera_ids[view_index]}"
    reference_stem = f"{sequence_id}_frame{frame_id}_{camera_ids[view_index]}"
    return (
        image_bgr,
        pose2d_panel,
        pred_vertices,
        pred_joints,
        footer,
        selected_camera,
        reference_image_path,
        reference_stem,
    )


def resolve_view_index(*, camera_ids: list[str], view_a_camera_id: str | None) -> int:
    if view_a_camera_id is None:
        return 0
    try:
        return camera_ids.index(view_a_camera_id)
    except ValueError as exc:
        raise KeyError(
            f"Requested --view-a '{view_a_camera_id}' is not present in selected views {camera_ids}"
        ) from exc


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


def save_demo_assets(
    *,
    asset_dir: Path,
    stem: str,
    pred_vertices: np.ndarray,
    faces: np.ndarray,
    reference_image_path: Path,
) -> None:
    mesh_path = asset_dir / f"{stem}_pred_mesh.ply"
    image_path = asset_dir / f"{stem}_image{reference_image_path.suffix.lower()}"

    mesh = trimesh.Trimesh(
        vertices=np.asarray(pred_vertices, dtype=np.float32).copy(),
        faces=np.asarray(faces, dtype=np.int32),
        process=False,
    )
    mesh.export(mesh_path)
    shutil.copy2(reference_image_path, image_path)


def render_pose2d_panel(
    *,
    image_shape: tuple[int, int, int],
    predictor: DefaultPredictor,
    image_bgr: np.ndarray,
    keypoint_threshold: float,
) -> np.ndarray:
    canvas = np.full(image_shape, 255, dtype=np.uint8)
    output = predictor(image_bgr)
    instances = output["instances"].to("cpu")
    if len(instances) == 0 or not hasattr(instances, "pred_keypoints"):
        cv2.putText(
            canvas,
            "No 2D pose detected",
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas

    best_index = int(np.argmax(instances.scores.numpy()))
    keypoints = instances.pred_keypoints[best_index].numpy()
    adaptive_threshold = keypoint_threshold
    if int((keypoints[:, 2] >= adaptive_threshold).sum()) < 5:
        adaptive_threshold = min(adaptive_threshold, 0.02)
    if int((keypoints[:, 2] >= adaptive_threshold).sum()) < 5:
        adaptive_threshold = 0.0
    return draw_coco_keypoints(canvas, keypoints=keypoints, threshold=adaptive_threshold)


def draw_coco_keypoints(
    image_bgr: np.ndarray,
    *,
    keypoints: np.ndarray,
    threshold: float,
) -> np.ndarray:
    output = image_bgr.copy()
    for joint_a, joint_b in COCO_KEYPOINT_EDGES:
        if float(keypoints[joint_a, 2]) < threshold or float(keypoints[joint_b, 2]) < threshold:
            continue
        point_a = tuple(np.round(keypoints[joint_a, :2]).astype(np.int32))
        point_b = tuple(np.round(keypoints[joint_b, :2]).astype(np.int32))
        cv2.line(output, point_a, point_b, (255, 210, 60), 2, cv2.LINE_AA)

    for joint_index in range(keypoints.shape[0]):
        if float(keypoints[joint_index, 2]) < threshold:
            continue
        center = tuple(np.round(keypoints[joint_index, :2]).astype(np.int32))
        cv2.circle(output, center, 3, (170, 70, 10), -1, cv2.LINE_AA)
    return output


def render_mesh_panel(
    *,
    overlay_renderer: CameraMeshOverlayRenderer,
    vertices_world: np.ndarray,
    joints_world: np.ndarray,
    faces: np.ndarray,
    width: int,
    height: int,
    camera,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    vertices_camera = transform_world_vertices_to_camera(
        vertices_world=vertices_world,
        rotation=np.asarray(camera.rotation, dtype=np.float32),
        translation=np.asarray(camera.translation, dtype=np.float32),
    )
    joints_camera = transform_world_vertices_to_camera(
        vertices_world=joints_world[:24],
        rotation=np.asarray(camera.rotation, dtype=np.float32),
        translation=np.asarray(camera.translation, dtype=np.float32),
    )
    mesh_panel, _ = overlay_renderer.render_overlay(
        image_bgr=canvas,
        vertices_camera=vertices_camera,
        faces=faces,
        color_bgr=(55, 85, 235),
        intrinsics=np.asarray(camera.intrinsics, dtype=np.float32),
        alpha=0.92,
    )
    return draw_projected_smpl_skeleton(
        image_bgr=mesh_panel,
        joints_camera=joints_camera,
        intrinsics=np.asarray(camera.intrinsics, dtype=np.float32),
        color_bgr=(115, 135, 245),
    )


def rotate_geometry_about_world_vertical(
    *,
    vertices: np.ndarray,
    joints: np.ndarray,
    angle_degrees: float,
    pivot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    angle_radians = np.deg2rad(float(angle_degrees))
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    rotation = np.array(
        [
            [cos_angle, 0.0, sin_angle],
            [0.0, 1.0, 0.0],
            [-sin_angle, 0.0, cos_angle],
        ],
        dtype=np.float32,
    )
    pivot = np.asarray(pivot, dtype=np.float32).reshape(1, 3)
    centered_vertices = np.asarray(vertices, dtype=np.float32) - pivot
    centered_joints = np.asarray(joints, dtype=np.float32) - pivot
    rotated_vertices = (rotation @ centered_vertices.T).T + pivot
    rotated_joints = (rotation @ centered_joints.T).T + pivot
    return (
        np.ascontiguousarray(rotated_vertices),
        np.ascontiguousarray(rotated_joints),
    )


def build_contact_frame(
    *,
    static_rgb: np.ndarray,
    static_pose2d: np.ndarray,
    mesh_panel: np.ndarray,
    footer: str,
) -> np.ndarray:
    return build_view_panel(
        [
            ("Frame", static_rgb),
            ("2D Pred", static_pose2d),
            ("3D Pred", mesh_panel),
        ],
        footer=footer,
    )


def draw_projected_smpl_skeleton(
    *,
    image_bgr: np.ndarray,
    joints_camera: np.ndarray,
    intrinsics: np.ndarray,
    color_bgr: tuple[int, int, int],
) -> np.ndarray:
    output = image_bgr.copy()
    projected = project_points_to_image(
        joints_camera=np.asarray(joints_camera, dtype=np.float32),
        intrinsics=np.asarray(intrinsics, dtype=np.float32),
    )
    for joint_start, joint_end in SMPL_SKELETON_EDGES:
        start = projected[joint_start]
        end = projected[joint_end]
        if start is None or end is None:
            continue
        cv2.line(output, start, end, color_bgr, 2, cv2.LINE_AA)
    for point in projected:
        if point is None:
            continue
        cv2.circle(output, point, 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(output, point, 2, color_bgr, -1, cv2.LINE_AA)
    return output


def project_points_to_image(
    *,
    joints_camera: np.ndarray,
    intrinsics: np.ndarray,
) -> list[tuple[int, int] | None]:
    projected: list[tuple[int, int] | None] = []
    for joint in joints_camera[:24]:
        depth = float(joint[2])
        if depth <= 1e-4:
            projected.append(None)
            continue
        x = float(joint[0] / joint[2])
        y = float(joint[1] / joint[2])
        pixel_x = intrinsics[0, 0] * x + intrinsics[0, 2]
        pixel_y = intrinsics[1, 1] * y + intrinsics[1, 2]
        projected.append((int(round(pixel_x)), int(round(pixel_y))))
    return projected


if __name__ == "__main__":
    main()
