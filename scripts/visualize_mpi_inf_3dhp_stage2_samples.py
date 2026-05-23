#!/usr/bin/env python
"""Visualize MPI-INF-3DHP Stage 2 predictions on cropped test images."""

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

from mvhpe3d.data.datasets.humman_stage2_multiview import HuMManStage2Dataset  # noqa: E402
from mvhpe3d.data.mpi_inf_3dhp import (  # noqa: E402
    MPII3D_HEATFORMER_EVAL_JOINTS,
    camera_id_to_index,
    load_annotation_mat,
    sequence_dir,
)
from mvhpe3d.data.splits import load_sample_records  # noqa: E402
from mvhpe3d.lightning import Stage2FusionLightningModule  # noqa: E402
from mvhpe3d.metrics import root_center_joints  # noqa: E402
from mvhpe3d.utils import (  # noqa: E402
    axis_angle_to_matrix,
    build_smpl_model,
    cache_path_for_source_npz,
    matrix_to_axis_angle,
    rotation_6d_to_axis_angle,
)
from mvhpe3d.visualization.camera_correction import correct_camera_global_orient_using_torso  # noqa: E402
from mvhpe3d.visualization.smpl_overlay import (  # noqa: E402
    overlay_mask_on_image,
    project_vertices_camera_to_image,
    render_projected_mesh_mask_from_projection,
)


H36M17_BONES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (11, 12),
    (12, 13),
    (8, 14),
    (14, 15),
    (15, 16),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--manifest-path", default="data/mpi_inf_3dhp/mpi_inf_3dhp_stage2_manifest.json")
    parser.add_argument("--dataset-root", default="/dysData/shimmer/datasets/mpi_inf_3dhp")
    parser.add_argument("--frames-dir", default="data/mpi_inf_3dhp/frames")
    parser.add_argument("--gt-smpl-dir", default="data/mpi_inf_3dhp/gt_smpl_fit_input_root_h36m_regressor_headtop_proxy")
    parser.add_argument("--input-smpl-cache-dir", default="data/mpi_inf_3dhp/sam3dbody_fitted_smpl")
    parser.add_argument("--smpl-model-path", default="data/weights/SMPL_NEUTRAL.pkl")
    parser.add_argument("--output-dir", default="outputs/visualizations/mpi_inf_3dhp_stage2_samples")
    parser.add_argument("--split", default="val")
    parser.add_argument("--camera-id", default="video_0")
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--sample-stride", type=int, default=120)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--crop-scale", type=float, default=1.35)
    parser.add_argument("--panel-height", type=int, default=360)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    manifest_path = Path(args.manifest_path).resolve()
    records = [
        record
        for record in load_sample_records(manifest_path)
        if record.split == args.split and any(view.camera_id == args.camera_id for view in record.views)
    ]
    records = records[args.start_index :: max(args.sample_stride, 1)]
    records = records[: args.num_samples]
    if not records:
        raise RuntimeError("No records selected for visualization")

    dataset = HuMManStage2Dataset(
        records,
        num_views=args.num_views,
        train=False,
        gt_smpl_dir=args.gt_smpl_dir,
        cameras_dir=args.dataset_root,
        input_smpl_cache_dir=args.input_smpl_cache_dir,
        joint_target_dataset="mpi_inf_3dhp",
        joint_target_root=args.dataset_root,
        joint_target_use_smpl_targets=True,
        seed=0,
    )

    module = Stage2FusionLightningModule.load_from_checkpoint(
        args.checkpoint_path,
        map_location=device,
        smpl_model_path=args.smpl_model_path,
        strict=False,
    ).to(device)
    module.eval()
    smpl = build_smpl_model(
        device=device,
        smpl_model_path=args.smpl_model_path,
        batch_size=1,
    ).eval()
    faces = np.asarray(smpl.faces, dtype=np.int32)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for dataset_index, record in enumerate(records):
        sample = dataset[dataset_index]
        camera_ids = list(sample["meta"]["camera_ids"])
        try:
            view_index = camera_ids.index(args.camera_id)
        except ValueError:
            view_index = 0
        image_path = resolve_mpi_frame_path(
            Path(args.frames_dir),
            sequence_id=record.sequence_id,
            camera_id=camera_ids[view_index],
            frame_id=record.frame_id,
        )
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image_height, image_width = image_bgr.shape[:2]

        h36m_joints_2d = load_h36m17_2d_joints(
            dataset_root=Path(args.dataset_root),
            sequence_id=record.sequence_id,
            camera_id=camera_ids[view_index],
            frame_id=record.frame_id,
        )
        crop_box = bbox_from_points(
            h36m_joints_2d,
            image_width=image_width,
            image_height=image_height,
            crop_scale=args.crop_scale,
        )

        batch = batchify_sample(sample, device=device)
        with torch.no_grad():
            predictions = module(
                batch["views_input"],
                view_rgb_feature=batch.get("view_rgb_feature"),
            )

        input_vertices = smpl_vertices_from_params(
            smpl=smpl,
            body_pose=load_view_input_smpl(sample, view_index, args.input_smpl_cache_dir)["body_pose"],
            betas=load_view_input_smpl(sample, view_index, args.input_smpl_cache_dir)["betas"],
            global_orient=load_view_input_smpl(sample, view_index, args.input_smpl_cache_dir)["global_orient"],
            transl=load_view_input_smpl(sample, view_index, args.input_smpl_cache_dir)["transl"],
            device=device,
        )
        gt_vertices = build_fitted_gt_vertices_camera(
            smpl=smpl,
            record=record,
            gt_smpl_dir=Path(args.gt_smpl_dir),
            view_input_global_orient=sample["view_aux"]["input_global_orient"][view_index].numpy(),
            view_input_transl=sample["view_aux"]["input_transl"][view_index].numpy(),
            device=device,
        )
        output_vertices = build_stage2_output_vertices_camera(
            smpl=smpl,
            predictions=predictions,
            sample=sample,
            view_index=view_index,
            device=device,
        )
        intrinsics = sample["view_aux"]["cam_int"][view_index].numpy()

        panels = [
            crop_and_label(image_bgr, crop_box, "image crop", panel_height=args.panel_height),
            crop_and_label(
                draw_skeleton(image_bgr.copy(), h36m_joints_2d),
                crop_box,
                "GT H36M skeleton",
                panel_height=args.panel_height,
            ),
            crop_and_label(
                overlay_mesh(image_bgr, gt_vertices, faces, intrinsics, color=(40, 190, 60)),
                crop_box,
                "fitted GT SMPL",
                panel_height=args.panel_height,
            ),
            crop_and_label(
                overlay_mesh(image_bgr, input_vertices, faces, intrinsics, color=(30, 130, 240)),
                crop_box,
                "input SAM3DBody",
                panel_height=args.panel_height,
            ),
            crop_and_label(
                overlay_mesh(image_bgr, output_vertices, faces, intrinsics, color=(220, 70, 220)),
                crop_box,
                "Stage2 output",
                panel_height=args.panel_height,
            ),
        ]
        grid = np.concatenate(panels, axis=1)
        output_path = output_dir / f"{record.sequence_id}_{camera_ids[view_index]}_{record.frame_id}.jpg"
        cv2.imwrite(str(output_path), grid)
        written.append(output_path)
        print(f"Wrote {output_path}")

    print("Summary:")
    for path in written:
        print(path)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def batchify_sample(sample: dict, *, device: torch.device) -> dict:
    batch = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0).to(device)
        elif isinstance(value, dict):
            batch[key] = {
                nested_key: (
                    nested_value.unsqueeze(0).to(device)
                    if torch.is_tensor(nested_value)
                    else nested_value
                )
                for nested_key, nested_value in value.items()
            }
        else:
            batch[key] = value
    return batch


def resolve_mpi_frame_path(
    frames_dir: Path,
    *,
    sequence_id: str,
    camera_id: str,
    frame_id: str,
) -> Path:
    base = frames_dir.resolve() / sequence_id / camera_id
    for extension in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = base / f"frame_{frame_id}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve frame_{frame_id} under {base}")


def load_h36m17_2d_joints(
    *,
    dataset_root: Path,
    sequence_id: str,
    camera_id: str,
    frame_id: str,
) -> np.ndarray:
    payload = load_annotation_mat(str(sequence_dir(dataset_root, sequence_id)))
    camera_index = camera_id_to_index(camera_id)
    annot2 = np.asarray(payload["annot2"][camera_index, 0], dtype=np.float32)
    joints_2d = annot2[int(frame_id)].reshape(-1, 2)
    return np.ascontiguousarray(joints_2d[list(MPII3D_HEATFORMER_EVAL_JOINTS)])


def bbox_from_points(
    points: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    crop_scale: float,
) -> np.ndarray:
    finite = np.isfinite(points).all(axis=1)
    inside = (
        finite
        & (points[:, 0] >= 0)
        & (points[:, 0] < image_width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < image_height)
    )
    selected = points[inside] if int(inside.sum()) >= 4 else points[finite]
    if selected.shape[0] < 4:
        return np.asarray([0, 0, image_width, image_height], dtype=np.int32)
    x1, y1 = selected.min(axis=0)
    x2, y2 = selected.max(axis=0)
    width = max(float(x2 - x1), 32.0)
    height = max(float(y2 - y1), 32.0)
    side = max(width, height) * crop_scale
    cx = (float(x1) + float(x2)) * 0.5
    cy = (float(y1) + float(y2)) * 0.5
    return np.asarray(
        [
            max(0, int(np.floor(cx - side * 0.5))),
            max(0, int(np.floor(cy - side * 0.5))),
            min(image_width, int(np.ceil(cx + side * 0.5))),
            min(image_height, int(np.ceil(cy + side * 0.5))),
        ],
        dtype=np.int32,
    )


def draw_skeleton(image_bgr: np.ndarray, joints_2d: np.ndarray) -> np.ndarray:
    points = np.asarray(joints_2d, dtype=np.float32)
    for start, end in H36M17_BONES:
        if not np.isfinite(points[[start, end]]).all():
            continue
        p1 = tuple(np.round(points[start]).astype(np.int32).tolist())
        p2 = tuple(np.round(points[end]).astype(np.int32).tolist())
        cv2.line(image_bgr, p1, p2, (0, 255, 255), 4, cv2.LINE_AA)
    for point in points:
        if not np.isfinite(point).all():
            continue
        center = tuple(np.round(point).astype(np.int32).tolist())
        cv2.circle(image_bgr, center, 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(image_bgr, center, 7, (255, 255, 255), 1, cv2.LINE_AA)
    return image_bgr


def overlay_mesh(
    image_bgr: np.ndarray,
    vertices_camera: np.ndarray,
    faces: np.ndarray,
    intrinsics: np.ndarray,
    *,
    color: tuple[int, int, int],
) -> np.ndarray:
    projected, depths = project_vertices_camera_to_image(
        vertices_camera=vertices_camera,
        intrinsics=intrinsics,
    )
    mask = render_projected_mesh_mask_from_projection(
        image_bgr.shape[:2],
        projected_vertices=projected,
        depths=depths,
        faces=faces,
    )
    return overlay_mask_on_image(
        image_bgr.copy(),
        mask,
        color=color,
        alpha=0.42,
        edge_color=(255, 255, 255),
        edge_thickness=2,
    )


def crop_and_label(
    image_bgr: np.ndarray,
    crop_box: np.ndarray,
    label: str,
    *,
    panel_height: int,
) -> np.ndarray:
    x1, y1, x2, y2 = crop_box.tolist()
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = image_bgr
    scale = panel_height / max(crop.shape[0], 1)
    panel_width = max(1, int(round(crop.shape[1] * scale)))
    panel = cv2.resize(crop, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
    label_bar = np.zeros((36, panel_width, 3), dtype=np.uint8)
    cv2.putText(
        label_bar,
        label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([label_bar, panel])


def smpl_vertices_from_params(
    *,
    smpl,
    body_pose: np.ndarray,
    betas: np.ndarray,
    global_orient: np.ndarray,
    transl: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        output = smpl(
            body_pose=torch.as_tensor(body_pose, dtype=torch.float32, device=device).reshape(1, -1),
            betas=torch.as_tensor(betas, dtype=torch.float32, device=device).reshape(1, -1),
            global_orient=torch.as_tensor(global_orient, dtype=torch.float32, device=device).reshape(1, 3),
            transl=torch.as_tensor(transl, dtype=torch.float32, device=device).reshape(1, 3),
        )
    return output.vertices[0].detach().cpu().numpy().astype(np.float32, copy=False)


def load_view_input_smpl(sample: dict, view_index: int, input_smpl_cache_dir: str | Path) -> dict[str, np.ndarray]:
    source_npz = Path(sample["meta"]["view_npz_paths"][view_index])
    cache_path = cache_path_for_source_npz(Path(input_smpl_cache_dir), source_npz)
    with np.load(cache_path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key], dtype=np.float32) for key in ("global_orient", "body_pose", "betas", "transl")}


def build_fitted_gt_vertices_camera(
    *,
    smpl,
    record,
    gt_smpl_dir: Path,
    view_input_global_orient: np.ndarray,
    view_input_transl: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    with np.load(gt_smpl_dir / f"{record.sequence_id}_smpl_params.npz", allow_pickle=False) as payload:
        frame_index = int(record.frame_id)
        body_pose = np.asarray(payload["body_pose"][frame_index], dtype=np.float32)
        betas = np.asarray(payload["betas"][frame_index], dtype=np.float32)
        local_orient = np.asarray(payload["global_orient"][frame_index], dtype=np.float32)
        local_transl = np.asarray(payload["transl"][frame_index], dtype=np.float32)
        fit_space_value = payload["fit_space"] if "fit_space" in payload else np.asarray("input_root")
        fit_space = str(np.asarray(fit_space_value).item())
    if fit_space == "input_root":
        global_orient, transl = compose_input_root_to_camera(
            local_global_orient=local_orient,
            local_transl=local_transl,
            input_global_orient=view_input_global_orient,
            input_transl=view_input_transl,
        )
    else:
        global_orient, transl = local_orient, local_transl
    return smpl_vertices_from_params(
        smpl=smpl,
        body_pose=body_pose,
        betas=betas,
        global_orient=global_orient,
        transl=transl,
        device=device,
    )


def build_stage2_output_vertices_camera(
    *,
    smpl,
    predictions: dict[str, torch.Tensor],
    sample: dict,
    view_index: int,
    device: torch.device,
) -> np.ndarray:
    pred_body_pose = predictions["pred_body_pose"][0].detach()
    pred_betas = predictions["pred_betas"][0].detach()
    views_input = sample["views_input"]
    input_pose_6d = views_input[view_index, :138].reshape(23, 6)
    input_betas = views_input[view_index, 138:]
    with torch.no_grad():
        input_body_pose = rotation_6d_to_axis_angle(
            input_pose_6d.to(device=device, dtype=torch.float32).reshape(23, 6)
        ).reshape(1, -1)
        input_joints = root_center_joints(
            smpl(
                body_pose=input_body_pose,
                betas=input_betas.to(device=device, dtype=torch.float32).reshape(1, -1),
                global_orient=torch.zeros((1, 3), device=device),
                transl=torch.zeros((1, 3), device=device),
            ).joints[:, :24, :]
        )[0]
        pred_joints = root_center_joints(
            smpl(
                body_pose=pred_body_pose.reshape(1, -1).to(device=device),
                betas=pred_betas.reshape(1, -1).to(device=device),
                global_orient=torch.zeros((1, 3), device=device),
                transl=torch.zeros((1, 3), device=device),
            ).joints[:, :24, :]
        )[0]
    corrected_orient = correct_camera_global_orient_using_torso(
        input_canonical_joints=input_joints.detach().cpu().numpy(),
        pred_canonical_joints=pred_joints.detach().cpu().numpy(),
        input_camera_global_orient=sample["view_aux"]["input_global_orient"][view_index].numpy(),
    )
    return smpl_vertices_from_params(
        smpl=smpl,
        body_pose=pred_body_pose.detach().cpu().numpy(),
        betas=pred_betas.detach().cpu().numpy(),
        global_orient=corrected_orient,
        transl=sample["view_aux"]["input_transl"][view_index].numpy(),
        device=device,
    )


def compose_input_root_to_camera(
    *,
    local_global_orient: np.ndarray,
    local_transl: np.ndarray,
    input_global_orient: np.ndarray,
    input_transl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    input_rotation = axis_angle_to_matrix(
        torch.as_tensor(input_global_orient, dtype=torch.float32).reshape(1, 3)
    )[0]
    local_rotation = axis_angle_to_matrix(
        torch.as_tensor(local_global_orient, dtype=torch.float32).reshape(1, 3)
    )[0]
    camera_rotation = torch.matmul(input_rotation, local_rotation)
    camera_orient = matrix_to_axis_angle(camera_rotation.reshape(1, 3, 3))[0].cpu().numpy()
    camera_transl = (
        torch.matmul(
            input_rotation,
            torch.as_tensor(local_transl, dtype=torch.float32).reshape(3, 1),
        ).reshape(3)
        + torch.as_tensor(input_transl, dtype=torch.float32).reshape(3)
    ).cpu().numpy()
    return camera_orient.astype(np.float32, copy=False), camera_transl.astype(np.float32, copy=False)


if __name__ == "__main__":
    main()
