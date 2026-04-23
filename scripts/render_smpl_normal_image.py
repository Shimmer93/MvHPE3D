#!/usr/bin/env python
"""Render one SMPL mesh as a surface-normal image with transparent background."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mvhpe3d.utils import build_smpl_model, validate_mhr_asset_folder
from visualize_video import (
    build_data_config,
    build_datamodule,
    build_model_config,
    build_sequence_index,
    build_smpl_outputs,
    get_visualization_prediction,
    is_stage2_experiment,
    load_experiment_config,
    load_visualization_module,
    multiview_collate,
    resolve_device,
    resolve_smpl_model_path,
    select_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one SMPL mesh as a surface-normal image with transparent background"
    )
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Optional path to .ckpt file")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment/stage2_cross_camera_joint_graph_refiner.yaml",
        help="Path to the experiment YAML file",
    )
    parser.add_argument("--manifest-path", type=str, default=None, help="Optional manifest override")
    parser.add_argument("--gt-smpl-dir", type=str, default=None, help="Optional GT SMPL override")
    parser.add_argument("--cameras-dir", type=str, default=None, help="Optional camera JSON override")
    parser.add_argument("--split-config-path", type=str, default=None, help="Optional split config override")
    parser.add_argument("--split-name", type=str, default=None, help="Optional split name override")
    parser.add_argument(
        "--stage",
        type=str,
        choices=("train", "val", "test"),
        default="test",
        help="Which split to sample from",
    )
    parser.add_argument("--sequence-id", type=str, default=None, help="Optional sequence_id to render")
    parser.add_argument("--max-sequences", type=int, default=None, help="Optional cap on rendered sequences")
    parser.add_argument(
        "--random-selection",
        action="store_true",
        help="Randomly sample sequences when --max-sequences is used",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sequence/frame/view selection")
    parser.add_argument(
        "--reference-frame",
        type=str,
        choices=("first", "middle", "random", "index"),
        default="first",
        help="How to choose the frame within the selected sequence",
    )
    parser.add_argument(
        "--reference-frame-index",
        type=int,
        default=None,
        help="0-based index into the selected sequence frames when --reference-frame=index",
    )
    parser.add_argument("--frame-step", type=int, default=1, help="Use every Nth frame in the sequence")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on usable frames per sequence")
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=1,
        help="Centered temporal smoothing window for predicted SMPL params; 1 disables smoothing",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device for inference")
    parser.add_argument(
        "--input-smpl-cache-dir",
        type=str,
        default=None,
        help="Optional fitted SMPL cache directory",
    )
    parser.add_argument("--smpl-model-path", type=str, default=None, help="Path to SMPL .pkl")
    parser.add_argument("--mhr-assets-dir", type=str, default=None, help="Optional MHR asset override")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/normal_images",
        help="Directory to save rendered normal images",
    )
    parser.add_argument("--output-path", type=str, default=None, help="Optional explicit output PNG path")
    parser.add_argument("--image-size", type=int, default=768, help="Output image size in pixels")
    parser.add_argument("--fov-deg", type=float, default=35.0, help="Virtual camera field of view in degrees")
    parser.add_argument(
        "--distance-scale",
        type=float,
        default=2.2,
        help="Virtual camera distance scale relative to mesh extent",
    )
    parser.add_argument("--pitch-min-deg", type=float, default=-18.0, help="Minimum random pitch in degrees")
    parser.add_argument("--pitch-max-deg", type=float, default=18.0, help="Maximum random pitch in degrees")
    parser.add_argument(
        "--mesh-source",
        type=str,
        choices=("pred", "gt"),
        default="gt",
        help="Which SMPL mesh to render",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_step < 1:
        raise ValueError(f"--frame-step must be >= 1, got {args.frame_step}")
    if args.max_sequences is not None and args.max_sequences < 1:
        raise ValueError(f"--max-sequences must be >= 1, got {args.max_sequences}")
    if args.reference_frame_index is not None and args.reference_frame_index < 0:
        raise ValueError(
            f"--reference-frame-index must be >= 0, got {args.reference_frame_index}"
        )
    if args.smoothing_window < 1:
        raise ValueError(f"--smoothing-window must be >= 1, got {args.smoothing_window}")
    if args.image_size < 64:
        raise ValueError(f"--image-size must be >= 64, got {args.image_size}")
    if args.fov_deg <= 1.0 or args.fov_deg >= 179.0:
        raise ValueError(f"--fov-deg must be in (1, 179), got {args.fov_deg}")
    if args.distance_scale <= 0:
        raise ValueError(f"--distance-scale must be > 0, got {args.distance_scale}")
    if args.pitch_min_deg > args.pitch_max_deg:
        raise ValueError(
            f"--pitch-min-deg must be <= --pitch-max-deg, got {args.pitch_min_deg} > {args.pitch_max_deg}"
        )

    experiment = load_experiment_config(args.config)
    need_module = args.mesh_source == "pred"
    if need_module and args.checkpoint_path is None:
        raise ValueError("--checkpoint-path is required when --mesh-source=pred")
    if need_module and not is_stage2_experiment(experiment):
        validate_mhr_asset_folder(args.mhr_assets_dir or "/opt/data/assets")

    data_config = build_data_config(experiment["data"], args)
    data_config.batch_size = 1
    data_config.drop_last_train = False
    if args.seed is not None:
        data_config.seed = args.seed

    datamodule = build_datamodule(data_config)
    datamodule.setup(None)
    dataset = select_dataset(datamodule, args.stage)

    device = resolve_device(args.device)
    module = None
    if need_module:
        assert args.checkpoint_path is not None
        model_config = build_model_config(experiment["model"], checkpoint_path=args.checkpoint_path)
        module = load_visualization_module(
            checkpoint_path=args.checkpoint_path,
            model_config=model_config,
            data_config=data_config,
            args=args,
        )
        module.eval()
        module.to(device)

    smpl_model = build_smpl_model(
        device=device,
        smpl_model_path=str(resolve_smpl_model_path(args.smpl_model_path)),
        batch_size=1,
    )
    faces = np.asarray(smpl_model.faces, dtype=np.int32)

    grouped_indices = build_sequence_index(dataset)
    sequence_ids = select_sequence_ids(grouped_indices=grouped_indices, args=args)
    if len(sequence_ids) > 1 and args.output_path is not None:
        raise ValueError("--output-path can only be used with a single rendered sequence")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for sequence_id in sequence_ids:
        frame_indices = grouped_indices[sequence_id][:: args.frame_step]
        if args.max_frames is not None:
            frame_indices = frame_indices[: args.max_frames]
        if not frame_indices:
            raise ValueError(f"sequence_id '{sequence_id}' has no frames after selection")
        dataset_index = resolve_reference_dataset_index(
            frame_indices=frame_indices,
            reference_frame=args.reference_frame,
            reference_frame_index=args.reference_frame_index,
            seed=args.seed,
            sequence_id=sequence_id,
        )

        batch = multiview_collate([dataset[dataset_index]])
        meta = batch["meta"][0]
        frame_id = str(meta["frame_id"])
        pred_body_pose = None
        pred_betas = None
        if need_module:
            assert module is not None
            with torch.no_grad():
                predictions = module(batch["views_input"].to(device))
            pred_body_pose, pred_betas = get_visualization_prediction(
                batch=batch,
                predictions=predictions,
                module=module,
                dataset=dataset,
                sequence_frame_indices=frame_indices,
                dataset_index=dataset_index,
                device=device,
                smoothing_window=args.smoothing_window,
            )

        if args.mesh_source == "pred":
            assert pred_body_pose is not None
            assert pred_betas is not None
            body_pose = pred_body_pose
            betas = pred_betas
        else:
            body_pose = batch["target_body_pose"][0].detach().cpu().numpy()
            betas = batch["target_betas"][0].detach().cpu().numpy()
        global_orient = batch["target_aux"]["global_orient"][0].detach().cpu().numpy()
        transl = batch["target_aux"]["transl"][0].detach().cpu().numpy()
        vertices_world, _ = build_smpl_outputs(
            smpl_model=smpl_model,
            device=device,
            body_pose=body_pose,
            betas=betas,
            global_orient=global_orient,
            transl=transl,
        )

        yaw_deg, pitch_deg = sample_random_view(
            seed=args.seed,
            sequence_id=sequence_id,
            frame_id=frame_id,
            pitch_min_deg=args.pitch_min_deg,
            pitch_max_deg=args.pitch_max_deg,
        )
        normal_image = render_normal_image(
            vertices_world=vertices_world,
            faces=faces,
            image_size=args.image_size,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            fov_deg=args.fov_deg,
            distance_scale=args.distance_scale,
        )

        output_path = resolve_output_path(
            args=args,
            sequence_id=sequence_id,
            frame_id=frame_id,
            mesh_source=args.mesh_source,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), normal_image):
            raise RuntimeError(f"Failed to write normal image to {output_path}")
        print(
            "Saved normal image to "
            f"{output_path} | sequence_id={sequence_id} frame_id={frame_id} "
            f"mesh_source={args.mesh_source} yaw_deg={yaw_deg:.2f} pitch_deg={pitch_deg:.2f}"
        )


def select_sequence_ids(
    *,
    grouped_indices: dict[str, list[int]],
    args: argparse.Namespace,
) -> list[str]:
    if args.sequence_id is not None:
        if args.sequence_id not in grouped_indices:
            raise KeyError(f"sequence_id '{args.sequence_id}' was not found in the selected {args.stage} split")
        return [args.sequence_id]
    available_sequence_ids = sorted(grouped_indices)
    if not available_sequence_ids:
        raise RuntimeError(f"No sequences available in the selected {args.stage} split")
    if args.max_sequences is None:
        rng = random.Random(args.seed)
        return [available_sequence_ids[rng.randrange(len(available_sequence_ids))]]
    sample_size = min(args.max_sequences, len(available_sequence_ids))
    if args.random_selection:
        rng = random.Random(args.seed)
        return sorted(rng.sample(available_sequence_ids, sample_size))
    return available_sequence_ids[:sample_size]


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
            raise ValueError("--reference-frame=index requires --reference-frame-index")
        if reference_frame_index >= len(frame_indices):
            raise ValueError(
                f"--reference-frame-index {reference_frame_index} is out of range for "
                f"{len(frame_indices)} selected frames"
            )
        return frame_indices[reference_frame_index]
    raise ValueError(f"Unsupported reference frame mode: {reference_frame}")


def resolve_output_path(
    *,
    args: argparse.Namespace,
    sequence_id: str,
    frame_id: str,
    mesh_source: str,
) -> Path:
    if args.output_path is not None:
        return Path(args.output_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    return (output_dir / f"{sequence_id}_frame{frame_id}_{mesh_source}_normal.png").resolve()


def sample_random_view(
    *,
    seed: int,
    sequence_id: str,
    frame_id: str,
    pitch_min_deg: float,
    pitch_max_deg: float,
) -> tuple[float, float]:
    rng = random.Random(f"{seed}:{sequence_id}:{frame_id}:normal-view")
    return rng.uniform(0.0, 360.0), rng.uniform(pitch_min_deg, pitch_max_deg)


def render_normal_image(
    *,
    vertices_world: np.ndarray,
    faces: np.ndarray,
    image_size: int,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    distance_scale: float,
) -> np.ndarray:
    vertices_world = np.asarray(vertices_world, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    mesh = trimesh.Trimesh(vertices_world, faces, process=False)
    normals_world = np.asarray(mesh.vertex_normals, dtype=np.float32)

    center = 0.5 * (
        np.min(vertices_world, axis=0) + np.max(vertices_world, axis=0)
    ).astype(np.float32, copy=False)
    extent = np.max(vertices_world, axis=0) - np.min(vertices_world, axis=0)
    radius = max(float(np.max(extent)), 1e-3)
    distance = distance_scale * radius

    camera_position = orbit_camera_position(
        center=center,
        radius=distance,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
    )
    rotation = build_opencv_look_at_rotation(
        camera_position=camera_position,
        target=center,
    )
    vertices_camera = ((rotation @ (vertices_world - camera_position.reshape(1, 3)).T).T).astype(
        np.float32,
        copy=False,
    )
    normals_camera = (rotation @ normals_world.T).T.astype(np.float32, copy=False)

    focal = 0.5 * image_size / np.tan(0.5 * np.deg2rad(fov_deg))
    intrinsics = np.array(
        [
            [focal, 0.0, image_size / 2.0],
            [0.0, focal, image_size / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return rasterize_normal_image(
        vertices_camera=vertices_camera,
        vertex_normals_camera=normals_camera,
        faces=faces,
        intrinsics=intrinsics,
        width=image_size,
        height=image_size,
    )


def orbit_camera_position(
    *,
    center: np.ndarray,
    radius: float,
    yaw_deg: float,
    pitch_deg: float,
) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    horizontal_radius = radius * np.cos(pitch)
    offset = np.array(
        [
            horizontal_radius * np.sin(yaw),
            -radius * np.sin(pitch),
            horizontal_radius * np.cos(yaw),
        ],
        dtype=np.float32,
    )
    return center.astype(np.float32, copy=False) + offset


def build_opencv_look_at_rotation(
    *,
    camera_position: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    forward = target.astype(np.float32) - camera_position.astype(np.float32)
    forward /= np.linalg.norm(forward) + 1e-8
    world_up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right) + 1e-8
    down = np.cross(forward, right)
    down /= np.linalg.norm(down) + 1e-8
    return np.stack((right, down, forward), axis=0).astype(np.float32, copy=False)


def rasterize_normal_image(
    *,
    vertices_camera: np.ndarray,
    vertex_normals_camera: np.ndarray,
    faces: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    image = np.zeros((height, width, 4), dtype=np.uint8)
    z_buffer = np.full((height, width), np.inf, dtype=np.float32)

    vertices_camera = np.asarray(vertices_camera, dtype=np.float32)
    vertex_normals_camera = np.asarray(vertex_normals_camera, dtype=np.float32)
    z_values = vertices_camera[:, 2]
    projected = (intrinsics @ vertices_camera.T).T
    projected = projected[:, :2] / np.clip(projected[:, 2:3], 1e-6, None)

    for face in np.asarray(faces, dtype=np.int32):
        tri_z = z_values[face]
        if np.any(tri_z <= 1e-4):
            continue
        tri_2d = projected[face]
        tri_n = vertex_normals_camera[face]
        rasterize_triangle(
            image=image,
            z_buffer=z_buffer,
            tri_2d=tri_2d,
            tri_z=tri_z,
            tri_normals=tri_n,
            width=width,
            height=height,
        )
    return image


def rasterize_triangle(
    *,
    image: np.ndarray,
    z_buffer: np.ndarray,
    tri_2d: np.ndarray,
    tri_z: np.ndarray,
    tri_normals: np.ndarray,
    width: int,
    height: int,
) -> None:
    min_x = max(int(np.floor(np.min(tri_2d[:, 0]))), 0)
    max_x = min(int(np.ceil(np.max(tri_2d[:, 0]))), width - 1)
    min_y = max(int(np.floor(np.min(tri_2d[:, 1]))), 0)
    max_y = min(int(np.ceil(np.max(tri_2d[:, 1]))), height - 1)
    if min_x > max_x or min_y > max_y:
        return

    a = tri_2d[0]
    b = tri_2d[1]
    c = tri_2d[2]
    denominator = ((b[1] - c[1]) * (a[0] - c[0])) + ((c[0] - b[0]) * (a[1] - c[1]))
    if abs(float(denominator)) < 1e-8:
        return

    xs = np.arange(min_x, max_x + 1, dtype=np.float32)
    ys = np.arange(min_y, max_y + 1, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    w0 = (((b[1] - c[1]) * (grid_x - c[0])) + ((c[0] - b[0]) * (grid_y - c[1]))) / denominator
    w1 = (((c[1] - a[1]) * (grid_x - c[0])) + ((a[0] - c[0]) * (grid_y - c[1]))) / denominator
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
    if not np.any(inside):
        return

    depth = (w0 * tri_z[0]) + (w1 * tri_z[1]) + (w2 * tri_z[2])
    current_depth = z_buffer[min_y : max_y + 1, min_x : max_x + 1]
    update_mask = inside & (depth < current_depth)
    if not np.any(update_mask):
        return

    normal = (
        w0[..., None] * tri_normals[0]
        + w1[..., None] * tri_normals[1]
        + w2[..., None] * tri_normals[2]
    )
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / np.clip(norm, 1e-8, None)
    normal_rgb = np.clip((0.5 * (normal + 1.0)) * 255.0, 0.0, 255.0).astype(np.uint8)
    normal_bgra = np.concatenate(
        (
            normal_rgb[..., ::-1],
            np.full(normal_rgb.shape[:2] + (1,), 255, dtype=np.uint8),
        ),
        axis=-1,
    )

    patch = image[min_y : max_y + 1, min_x : max_x + 1]
    patch[update_mask] = normal_bgra[update_mask]
    current_depth[update_mask] = depth[update_mask]


if __name__ == "__main__":
    main()
