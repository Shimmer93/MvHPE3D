#!/usr/bin/env python
"""Render one SMPL mesh as a depth image from the normal-render virtual camera."""

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
from render_smpl_normal_image import (
    build_data_config,
    build_datamodule,
    build_model_config,
    build_opencv_look_at_rotation,
    build_sequence_index,
    build_smpl_outputs,
    get_visualization_prediction,
    is_stage2_experiment,
    load_experiment_config,
    load_visualization_module,
    multiview_collate,
    orbit_camera_position,
    resolve_device,
    resolve_reference_dataset_index,
    resolve_smpl_model_path,
    sample_random_view,
    select_dataset,
    select_sequence_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one SMPL mesh as a depth image from the same virtual camera as render_smpl_normal_image.py"
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
        default="outputs/depth_images",
        help="Directory to save rendered depth images",
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
    parser.add_argument(
        "--depth-format",
        type=str,
        choices=("uint16", "uint8", "rgba8"),
        default="uint16",
        help=(
            "PNG encoding. uint16/uint8 write one channel with 0 as background; "
            "rgba8 writes displayable grayscale BGRA with transparent background."
        ),
    )
    parser.add_argument(
        "--depth-min",
        type=float,
        default=None,
        help="Camera-space depth mapped to the minimum visible pixel value; defaults to visible min",
    )
    parser.add_argument(
        "--depth-max",
        type=float,
        default=None,
        help="Camera-space depth mapped to the maximum visible pixel value; defaults to visible max",
    )
    parser.add_argument(
        "--inverse-depth",
        action="store_true",
        help="Map nearer surfaces to brighter values instead of farther surfaces",
    )
    parser.add_argument(
        "--save-raw-depth-npy",
        action="store_true",
        help="Also save the raw camera-space z-buffer as .npy with NaN background",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)

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
        depth_image, raw_depth = render_depth_image(
            vertices_world=vertices_world,
            faces=faces,
            image_size=args.image_size,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            fov_deg=args.fov_deg,
            distance_scale=args.distance_scale,
            depth_format=args.depth_format,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            inverse_depth=args.inverse_depth,
        )

        output_path = resolve_depth_output_path(
            args=args,
            sequence_id=sequence_id,
            frame_id=frame_id,
            mesh_source=args.mesh_source,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), depth_image):
            raise RuntimeError(f"Failed to write depth image to {output_path}")
        if args.save_raw_depth_npy:
            raw_output_path = resolve_raw_depth_output_path(output_path)
            np.save(raw_output_path, raw_depth.astype(np.float32, copy=False))
        valid_depth = raw_depth[np.isfinite(raw_depth)]
        depth_range = (
            (float(valid_depth.min()), float(valid_depth.max()))
            if valid_depth.size
            else (float("nan"), float("nan"))
        )
        print(
            "Saved depth image to "
            f"{output_path} | sequence_id={sequence_id} frame_id={frame_id} "
            f"mesh_source={args.mesh_source} yaw_deg={yaw_deg:.2f} pitch_deg={pitch_deg:.2f} "
            f"depth_range=({depth_range[0]:.4f}, {depth_range[1]:.4f})"
        )


def validate_args(args: argparse.Namespace) -> None:
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
    if args.depth_min is not None and args.depth_max is not None and args.depth_min >= args.depth_max:
        raise ValueError(
            f"--depth-min must be smaller than --depth-max, got {args.depth_min} >= {args.depth_max}"
        )


def resolve_depth_output_path(
    *,
    args: argparse.Namespace,
    sequence_id: str,
    frame_id: str,
    mesh_source: str,
) -> Path:
    if args.output_path is not None:
        return Path(args.output_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    return (output_dir / f"{sequence_id}_frame{frame_id}_{mesh_source}_depth.png").resolve()


def resolve_raw_depth_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_raw_depth.npy")


def render_depth_image(
    *,
    vertices_world: np.ndarray,
    faces: np.ndarray,
    image_size: int,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    distance_scale: float,
    depth_format: str = "uint16",
    depth_min: float | None = None,
    depth_max: float | None = None,
    inverse_depth: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    vertices_world = np.asarray(vertices_world, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)

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

    focal = 0.5 * image_size / np.tan(0.5 * np.deg2rad(fov_deg))
    intrinsics = np.array(
        [
            [focal, 0.0, image_size / 2.0],
            [0.0, focal, image_size / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    raw_depth = rasterize_depth_map(
        vertices_camera=vertices_camera,
        faces=faces,
        intrinsics=intrinsics,
        width=image_size,
        height=image_size,
    )
    depth_image = encode_depth_image(
        raw_depth,
        depth_format=depth_format,
        depth_min=depth_min,
        depth_max=depth_max,
        inverse_depth=inverse_depth,
    )
    return depth_image, raw_depth


def rasterize_depth_map(
    *,
    vertices_camera: np.ndarray,
    faces: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    z_buffer = np.full((height, width), np.inf, dtype=np.float32)
    depth_map = np.full((height, width), np.nan, dtype=np.float32)

    vertices_camera = np.asarray(vertices_camera, dtype=np.float32)
    z_values = vertices_camera[:, 2]
    projected = (intrinsics @ vertices_camera.T).T
    projected = projected[:, :2] / np.clip(projected[:, 2:3], 1e-6, None)

    for face in np.asarray(faces, dtype=np.int32):
        tri_z = z_values[face]
        if np.any(tri_z <= 1e-4):
            continue
        tri_2d = projected[face]
        rasterize_depth_triangle(
            depth_map=depth_map,
            z_buffer=z_buffer,
            tri_2d=tri_2d,
            tri_z=tri_z,
            width=width,
            height=height,
        )
    return depth_map


def rasterize_depth_triangle(
    *,
    depth_map: np.ndarray,
    z_buffer: np.ndarray,
    tri_2d: np.ndarray,
    tri_z: np.ndarray,
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

    inv_depth = (w0 / tri_z[0]) + (w1 / tri_z[1]) + (w2 / tri_z[2])
    depth = 1.0 / np.clip(inv_depth, 1e-8, None)
    current_depth = z_buffer[min_y : max_y + 1, min_x : max_x + 1]
    update_mask = inside & (depth < current_depth)
    if not np.any(update_mask):
        return

    patch = depth_map[min_y : max_y + 1, min_x : max_x + 1]
    patch[update_mask] = depth[update_mask]
    current_depth[update_mask] = depth[update_mask]


def encode_depth_image(
    depth_map: np.ndarray,
    *,
    depth_format: str,
    depth_min: float | None,
    depth_max: float | None,
    inverse_depth: bool,
) -> np.ndarray:
    valid_mask = np.isfinite(depth_map)
    if not np.any(valid_mask):
        if depth_format == "rgba8":
            return np.zeros(depth_map.shape + (4,), dtype=np.uint8)
        if depth_format == "uint16":
            return np.zeros(depth_map.shape, dtype=np.uint16)
        if depth_format == "uint8":
            return np.zeros(depth_map.shape, dtype=np.uint8)
        raise ValueError(f"Unsupported depth format: {depth_format}")

    visible_depth = depth_map[valid_mask].astype(np.float32, copy=False)
    near = float(np.min(visible_depth) if depth_min is None else depth_min)
    far = float(np.max(visible_depth) if depth_max is None else depth_max)
    if far <= near:
        far = near + 1e-6

    normalized = np.zeros_like(depth_map, dtype=np.float32)
    normalized[valid_mask] = np.clip((depth_map[valid_mask] - near) / (far - near), 0.0, 1.0)
    if inverse_depth:
        normalized[valid_mask] = 1.0 - normalized[valid_mask]

    if depth_format == "uint16":
        image = np.zeros(depth_map.shape, dtype=np.uint16)
        image[valid_mask] = np.round(1.0 + normalized[valid_mask] * 65534.0).astype(np.uint16)
        return image
    if depth_format == "uint8":
        image = np.zeros(depth_map.shape, dtype=np.uint8)
        image[valid_mask] = np.round(1.0 + normalized[valid_mask] * 254.0).astype(np.uint8)
        return image
    if depth_format == "rgba8":
        gray = np.zeros(depth_map.shape, dtype=np.uint8)
        gray[valid_mask] = np.round(1.0 + normalized[valid_mask] * 254.0).astype(np.uint8)
        alpha = np.zeros(depth_map.shape, dtype=np.uint8)
        alpha[valid_mask] = 255
        return np.stack((gray, gray, gray, alpha), axis=-1)
    raise ValueError(f"Unsupported depth format: {depth_format}")


if __name__ == "__main__":
    main()
