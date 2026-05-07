#!/usr/bin/env python
"""Render one SMPL mesh as a body-part segmentation map."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mvhpe3d.data import Stage1DataConfig, Stage2DataConfig
from mvhpe3d.utils import build_smpl_model, validate_mhr_asset_folder
from render_smpl_normal_image import (
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


PART_NAMES: tuple[str, ...] = (
    "background",
    "head_neck",
    "torso",
    "left_upper_arm",
    "left_lower_arm",
    "left_hand",
    "right_upper_arm",
    "right_lower_arm",
    "right_hand",
    "left_upper_leg",
    "left_lower_leg",
    "left_foot",
    "right_upper_leg",
    "right_lower_leg",
    "right_foot",
)

SMPL_JOINT_TO_PART_ID: dict[int, int] = {
    0: 2,  # pelvis
    1: 9,  # left hip
    2: 12,  # right hip
    3: 2,  # spine1
    4: 10,  # left knee
    5: 13,  # right knee
    6: 2,  # spine2
    7: 11,  # left ankle
    8: 14,  # right ankle
    9: 2,  # spine3
    10: 11,  # left foot
    11: 14,  # right foot
    12: 1,  # neck
    13: 3,  # left collar
    14: 6,  # right collar
    15: 1,  # head
    16: 3,  # left shoulder
    17: 6,  # right shoulder
    18: 4,  # left elbow
    19: 7,  # right elbow
    20: 5,  # left wrist
    21: 8,  # right wrist
    22: 5,  # left hand
    23: 8,  # right hand
}

PART_PALETTE_RGB = np.array(
    [
        [0, 0, 0],
        [244, 67, 54],
        [255, 193, 7],
        [33, 150, 243],
        [3, 169, 244],
        [0, 188, 212],
        [156, 39, 176],
        [103, 58, 183],
        [63, 81, 181],
        [76, 175, 80],
        [139, 195, 74],
        [205, 220, 57],
        [255, 112, 67],
        [255, 152, 0],
        [121, 85, 72],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one SMPL mesh as a body-part segmentation map from the normal-render virtual camera"
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
    parser.add_argument(
        "--frame-id",
        type=str,
        default=None,
        help="Optional 1-based frame_id to render inside --sequence-id, e.g. 000039",
    )
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
        default="outputs/part_segmentation_images",
        help="Directory to save rendered part segmentation images",
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
        "--output-format",
        type=str,
        choices=("palette", "label", "rgba"),
        default="palette",
        help="palette saves RGB part colors; label saves uint8 part IDs; rgba saves palette with transparent background",
    )
    parser.add_argument(
        "--save-label-map",
        action="store_true",
        help="Also save the raw uint8 part-ID map with _labels.png suffix",
    )
    parser.add_argument(
        "--save-legend",
        action="store_true",
        help="Save a JSON legend mapping part IDs to names and RGB colors",
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

    data_config = build_part_data_config(experiment["data"], args)
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
    vertex_part_ids = build_vertex_part_ids(smpl_model)
    face_part_ids = build_face_part_ids(faces=faces, vertex_part_ids=vertex_part_ids)

    grouped_indices = build_sequence_index(dataset)
    sequence_ids = select_sequence_ids(grouped_indices=grouped_indices, args=args)
    if len(sequence_ids) > 1 and args.output_path is not None:
        raise ValueError("--output-path can only be used with a single rendered sequence")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_legend:
        save_legend_json(output_dir / "part_segmentation_legend.json")

    for sequence_id in sequence_ids:
        sequence_frame_indices = grouped_indices[sequence_id]
        frame_indices = sequence_frame_indices[:: args.frame_step]
        if args.max_frames is not None and args.frame_id is None:
            frame_indices = frame_indices[: args.max_frames]
        if not sequence_frame_indices:
            raise ValueError(f"sequence_id '{sequence_id}' has no frames after selection")
        if args.frame_id is not None:
            dataset_index = resolve_dataset_index_by_frame_id(
                dataset=dataset,
                frame_indices=sequence_frame_indices,
                frame_id=args.frame_id,
                sequence_id=sequence_id,
            )
            smoothing_frame_indices = sequence_frame_indices
        else:
            if not frame_indices:
                raise ValueError(f"sequence_id '{sequence_id}' has no frames after selection")
            dataset_index = resolve_reference_dataset_index(
                frame_indices=frame_indices,
                reference_frame=args.reference_frame,
                reference_frame_index=args.reference_frame_index,
                seed=args.seed,
                sequence_id=sequence_id,
            )
            smoothing_frame_indices = frame_indices

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
                sequence_frame_indices=smoothing_frame_indices,
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
        label_map = render_part_label_map(
            vertices_world=vertices_world,
            faces=faces,
            face_part_ids=face_part_ids,
            image_size=args.image_size,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            fov_deg=args.fov_deg,
            distance_scale=args.distance_scale,
        )
        output_image = encode_part_segmentation(label_map, output_format=args.output_format)

        output_path = resolve_part_output_path(
            args=args,
            sequence_id=sequence_id,
            frame_id=frame_id,
            mesh_source=args.mesh_source,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), output_image):
            raise RuntimeError(f"Failed to write part segmentation image to {output_path}")
        if args.save_label_map:
            label_output_path = resolve_label_output_path(output_path)
            if not cv2.imwrite(str(label_output_path), label_map):
                raise RuntimeError(f"Failed to write label map to {label_output_path}")
        visible_labels = sorted(int(value) for value in np.unique(label_map) if int(value) != 0)
        print(
            "Saved part segmentation image to "
            f"{output_path} | sequence_id={sequence_id} frame_id={frame_id} "
            f"mesh_source={args.mesh_source} yaw_deg={yaw_deg:.2f} pitch_deg={pitch_deg:.2f} "
            f"visible_parts={visible_labels}"
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
    if args.frame_id is not None and args.sequence_id is None:
        raise ValueError("--frame-id requires --sequence-id")


def build_part_data_config(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> Stage1DataConfig | Stage2DataConfig:
    data_kwargs = dict(config)
    data_name = str(data_kwargs.pop("name", "humman_stage1"))
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

    if data_name == "humman_stage2":
        if args.input_smpl_cache_dir is not None:
            data_kwargs["input_smpl_cache_dir"] = args.input_smpl_cache_dir
        return Stage2DataConfig(**data_kwargs)
    if data_name != "humman_stage1":
        raise ValueError(
            f"Part segmentation rendering supports humman_stage1 and humman_stage2 data, got {data_name!r}"
        )
    return Stage1DataConfig(**data_kwargs)


def resolve_dataset_index_by_frame_id(
    *,
    dataset,
    frame_indices: list[int],
    frame_id: str,
    sequence_id: str,
) -> int:
    normalized_frame_id = str(frame_id).zfill(6)
    for dataset_index in frame_indices:
        record = dataset.records[dataset_index]
        if str(record.frame_id).zfill(6) == normalized_frame_id:
            return dataset_index
    raise KeyError(
        f"frame_id '{normalized_frame_id}' was not found for sequence_id '{sequence_id}' "
        "in the selected split"
    )


def resolve_part_output_path(
    *,
    args: argparse.Namespace,
    sequence_id: str,
    frame_id: str,
    mesh_source: str,
) -> Path:
    if args.output_path is not None:
        return Path(args.output_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    return (output_dir / f"{sequence_id}_frame{frame_id}_{mesh_source}_parts.png").resolve()


def resolve_label_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_labels.png")


def build_vertex_part_ids(smpl_model) -> np.ndarray:
    if not hasattr(smpl_model, "lbs_weights"):
        raise AttributeError("SMPL model does not expose lbs_weights; cannot derive body parts")
    lbs_weights = smpl_model.lbs_weights.detach().cpu().numpy()
    dominant_joint_ids = np.argmax(lbs_weights, axis=1).astype(np.int32, copy=False)
    vertex_part_ids = np.zeros(dominant_joint_ids.shape, dtype=np.uint8)
    for joint_id, part_id in SMPL_JOINT_TO_PART_ID.items():
        vertex_part_ids[dominant_joint_ids == joint_id] = part_id
    return vertex_part_ids


def build_face_part_ids(*, faces: np.ndarray, vertex_part_ids: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int32)
    face_vertex_parts = vertex_part_ids[faces]
    face_part_ids = np.zeros(faces.shape[0], dtype=np.uint8)
    for index, labels in enumerate(face_vertex_parts):
        counts = np.bincount(labels, minlength=len(PART_NAMES))
        face_part_ids[index] = int(np.argmax(counts))
    return face_part_ids


def render_part_label_map(
    *,
    vertices_world: np.ndarray,
    faces: np.ndarray,
    face_part_ids: np.ndarray,
    image_size: int,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    distance_scale: float,
) -> np.ndarray:
    vertices_world = np.asarray(vertices_world, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    face_part_ids = np.asarray(face_part_ids, dtype=np.uint8)

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
    return rasterize_part_label_map(
        vertices_camera=vertices_camera,
        faces=faces,
        face_part_ids=face_part_ids,
        intrinsics=intrinsics,
        width=image_size,
        height=image_size,
    )


def rasterize_part_label_map(
    *,
    vertices_camera: np.ndarray,
    faces: np.ndarray,
    face_part_ids: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    label_map = np.zeros((height, width), dtype=np.uint8)
    z_buffer = np.full((height, width), np.inf, dtype=np.float32)

    vertices_camera = np.asarray(vertices_camera, dtype=np.float32)
    z_values = vertices_camera[:, 2]
    projected = (intrinsics @ vertices_camera.T).T
    projected = projected[:, :2] / np.clip(projected[:, 2:3], 1e-6, None)

    for face, part_id in zip(np.asarray(faces, dtype=np.int32), face_part_ids, strict=True):
        tri_z = z_values[face]
        if np.any(tri_z <= 1e-4):
            continue
        tri_2d = projected[face]
        rasterize_part_triangle(
            label_map=label_map,
            z_buffer=z_buffer,
            tri_2d=tri_2d,
            tri_z=tri_z,
            part_id=int(part_id),
            width=width,
            height=height,
        )
    return label_map


def rasterize_part_triangle(
    *,
    label_map: np.ndarray,
    z_buffer: np.ndarray,
    tri_2d: np.ndarray,
    tri_z: np.ndarray,
    part_id: int,
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

    patch = label_map[min_y : max_y + 1, min_x : max_x + 1]
    patch[update_mask] = part_id
    current_depth[update_mask] = depth[update_mask]


def encode_part_segmentation(label_map: np.ndarray, *, output_format: str) -> np.ndarray:
    label_map = np.asarray(label_map, dtype=np.uint8)
    clipped_labels = np.clip(label_map, 0, len(PART_PALETTE_RGB) - 1)
    rgb = PART_PALETTE_RGB[clipped_labels]
    if output_format == "label":
        return label_map
    if output_format == "palette":
        return rgb[..., ::-1]
    if output_format == "rgba":
        alpha = np.where(label_map > 0, 255, 0).astype(np.uint8)
        bgr = rgb[..., ::-1]
        return np.concatenate((bgr, alpha[..., None]), axis=-1)
    raise ValueError(f"Unsupported output format: {output_format}")


def save_legend_json(path: Path) -> None:
    payload = {
        str(index): {
            "name": name,
            "rgb": PART_PALETTE_RGB[index].astype(int).tolist(),
        }
        for index, name in enumerate(PART_NAMES)
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
