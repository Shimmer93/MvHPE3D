#!/usr/bin/env python
"""Fit pseudo-GT SMPL parameters from MPI-INF-3DHP GT skeletons."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mvhpe3d.data.mpi_inf_3dhp import (
    MPII3D_HEATFORMER_CAMERA_IDS,
    MPII3D_HEATFORMER_EVAL_JOINTS,
    MPII3D_HEATFORMER_ROOT_INDEX,
    MPII3D_HEATFORMER_TO_SMPL24,
    MPII3D_HEATFORMER_TRAIN_SAMPLING,
    MPII3D_HEATFORMER_VAL_SAMPLING,
    MPII3D_SEQUENCES,
    MPII3D_TRAIN_SUBJECTS,
    MPII3D_VAL_SUBJECTS,
    annotation_frame_count,
    camera_id_to_index,
    load_annotation_mat,
    sequence_dir,
)
from mvhpe3d.data.splits import SampleRecord, ViewRecord, load_sample_records
from mvhpe3d.utils import (
    axis_angle_to_matrix,
    build_smpl_model,
    cache_path_for_source_npz,
    load_camera_parameters,
)


MPII3D_HEATFORMER_BONE_PAIRS: tuple[tuple[int, int], ...] = (
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
    parser.add_argument("--dataset-root", default="/dysData/shimmer/datasets/mpi_inf_3dhp")
    parser.add_argument(
        "--manifest-path",
        default="data/mpi_inf_3dhp/mpi_inf_3dhp_stage2_manifest.json",
        help="Manifest whose sampled frame ids should be fitted",
    )
    parser.add_argument("--output-dir", default="data/mpi_inf_3dhp/gt_smpl_fit")
    parser.add_argument("--smpl-model-path", required=True)
    parser.add_argument("--annotation-key", choices=("univ_annot3", "annot3"), default="univ_annot3")
    parser.add_argument("--splits", nargs="*", default=("train", "val"))
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--cameras", nargs="*", default=list(MPII3D_HEATFORMER_CAMERA_IDS))
    parser.add_argument(
        "--fit-space",
        choices=("world", "input_root"),
        default="world",
        help=(
            "Coordinate frame used for SMPL fitting. Use input_root for stage-2 "
            "pseudo-SMPL supervision because stage-2 predicts body pose in the "
            "SAM3DBody input root frame."
        ),
    )
    parser.add_argument(
        "--input-smpl-cache-dir",
        default="data/mpi_inf_3dhp/sam3dbody_fitted_smpl",
        help="Cached fitted SAM3DBody SMPL directory, required by --fit-space input_root",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=4,
        help="Number of sorted manifest views used to define the input-root frame",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-iters", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument(
        "--fit-joint-source",
        choices=("smpl24", "smpl24_headtop_proxy", "regressor", "regressor_headtop_proxy"),
        default="smpl24_headtop_proxy",
        help=(
            "Joint source used while fitting. smpl24 preserves the original direct "
            "SMPL24 mapping. smpl24_headtop_proxy replaces the duplicated headtop "
            "joint with a template top-vertex proxy. regressor uses --joint-regressor-path. "
            "regressor_headtop_proxy uses the regressor and replaces only headtop with "
            "the template top-vertex proxy."
        ),
    )
    parser.add_argument(
        "--joint-regressor-path",
        default=None,
        help=(
            "Optional SMPL-to-H36M17 regressor .npy/.npz path. Supported shapes are "
            "[17, 6890] for vertices, [17, N] for SMPL joints, or their transposes."
        ),
    )
    parser.add_argument(
        "--headtop-topk",
        type=int,
        default=32,
        help="Number of highest neutral-template vertices averaged for the headtop proxy",
    )
    parser.add_argument(
        "--headtop-axis",
        choices=("x", "y", "z"),
        default="y",
        help="Neutral-template axis used to select the headtop proxy vertices",
    )
    parser.add_argument("--joint-weight", type=float, default=1.0)
    parser.add_argument("--pose-reg", type=float, default=0.02)
    parser.add_argument("--shape-reg", type=float, default=0.1)
    parser.add_argument("--bone-len-reg", type=float, default=0.01)
    parser.add_argument("--temporal-reg", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-frames-per-sequence", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")
    device = resolve_device(args.device)
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_smpl_cache_dir = Path(args.input_smpl_cache_dir).resolve()

    frames_by_sequence = resolve_frames_by_sequence(args, dataset_root=dataset_root)
    views_by_sequence_frame = None
    if args.fit_space == "input_root":
        if not Path(args.manifest_path).exists():
            raise FileNotFoundError("--fit-space input_root requires an existing --manifest-path")
        if not input_smpl_cache_dir.exists():
            raise FileNotFoundError(
                f"--input-smpl-cache-dir does not exist: {input_smpl_cache_dir}"
            )
        views_by_sequence_frame, skipped_no_view = resolve_views_by_sequence_frame(args)
        frames_by_sequence = {
            sequence_id: set(frame_views)
            for sequence_id, frame_views in views_by_sequence_frame.items()
        }
    sequence_ids = sorted(frames_by_sequence)
    sequence_ids = select_sequence_shard(
        sequence_ids,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )

    print(f"Dataset root: {dataset_root}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")
    print(f"Fit space: {args.fit_space}")
    print(f"Fit joint source: {args.fit_joint_source}")
    if args.joint_regressor_path is not None:
        print(f"Joint regressor: {Path(args.joint_regressor_path).resolve()}")
    if args.fit_space == "input_root":
        print(f"Skipped manifest records without views: {skipped_no_view}")
    print(f"Shard: {args.shard_index}/{args.num_shards}")
    print(f"Sequences in shard: {len(sequence_ids)}")

    reports = []
    for sequence_id in sequence_ids:
        frame_ids = sorted(frames_by_sequence[sequence_id])
        if args.max_frames_per_sequence is not None:
            frame_ids = frame_ids[: args.max_frames_per_sequence]
        report = fit_sequence(
            args=args,
            dataset_root=dataset_root,
            output_dir=output_dir,
            sequence_id=sequence_id,
            frame_ids=frame_ids,
            device=device,
            views_by_frame=(
                views_by_sequence_frame.get(sequence_id, {})
                if views_by_sequence_frame is not None
                else None
            ),
            input_smpl_cache_dir=input_smpl_cache_dir,
        )
        reports.append(report)
        print(json.dumps(report, sort_keys=True))

    report_path = output_dir / f"fit_report_shard{args.shard_index:03d}_of_{args.num_shards:03d}.json"
    report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"Wrote report: {report_path}")


def resolve_device(device_arg: str) -> torch.device:
    requested = device_arg.strip()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def resolve_frames_by_sequence(
    args: argparse.Namespace,
    *,
    dataset_root: Path,
) -> dict[str, set[int]]:
    frames_by_sequence: dict[str, set[int]] = defaultdict(set)
    manifest_path = Path(args.manifest_path).resolve() if args.manifest_path else None
    allowed_splits = set(args.splits or [])
    allowed_sequences = set(args.sequences or [])
    if manifest_path is not None and manifest_path.exists():
        records = load_sample_records(manifest_path)
        for record in records:
            if allowed_splits and record.split not in allowed_splits:
                continue
            if allowed_sequences and record.sequence_id not in allowed_sequences:
                continue
            frames_by_sequence[record.sequence_id].add(int(record.frame_id))
        return frames_by_sequence

    for split_name, subjects, sampling in (
        ("train", MPII3D_TRAIN_SUBJECTS, MPII3D_HEATFORMER_TRAIN_SAMPLING),
        ("val", MPII3D_VAL_SUBJECTS, MPII3D_HEATFORMER_VAL_SAMPLING),
    ):
        if allowed_splits and split_name not in allowed_splits:
            continue
        for subject_id in subjects:
            for sequence_name in MPII3D_SEQUENCES:
                sequence_id = f"{subject_id}_{sequence_name}"
                if allowed_sequences and sequence_id not in allowed_sequences:
                    continue
                frame_count = annotation_frame_count(dataset_root, sequence_id=sequence_id)
                frames_by_sequence[sequence_id].update(range(0, frame_count, sampling))
    return frames_by_sequence


def resolve_views_by_sequence_frame(
    args: argparse.Namespace,
) -> tuple[dict[str, dict[int, tuple[ViewRecord, ...]]], int]:
    manifest_path = Path(args.manifest_path).resolve()
    records = load_sample_records(manifest_path)
    allowed_splits = set(args.splits or [])
    allowed_sequences = set(args.sequences or [])
    views_by_sequence_frame: dict[str, dict[int, tuple[ViewRecord, ...]]] = defaultdict(dict)
    skipped_no_view = 0
    for record in records:
        if allowed_splits and record.split not in allowed_splits:
            continue
        if allowed_sequences and record.sequence_id not in allowed_sequences:
            continue
        if not record.views:
            skipped_no_view += 1
            continue
        selected_views = select_manifest_views(record, num_views=args.num_views)
        views_by_sequence_frame[record.sequence_id][int(record.frame_id)] = tuple(selected_views)
    return views_by_sequence_frame, skipped_no_view


def select_manifest_views(record: SampleRecord, *, num_views: int) -> list[ViewRecord]:
    ordered_views = sorted(record.views, key=lambda item: item.camera_id)
    if not ordered_views:
        raise ValueError(f"Record '{record.sample_id}' has no valid views")
    return list(ordered_views[: min(len(ordered_views), num_views)])


def select_sequence_shard(
    sequence_ids: list[str],
    *,
    num_shards: int,
    shard_index: int,
) -> list[str]:
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    return [
        sequence_id
        for index, sequence_id in enumerate(sequence_ids)
        if index % num_shards == shard_index
    ]


def fit_sequence(
    *,
    args: argparse.Namespace,
    dataset_root: Path,
    output_dir: Path,
    sequence_id: str,
    frame_ids: list[int],
    device: torch.device,
    views_by_frame: dict[int, tuple[ViewRecord, ...]] | None,
    input_smpl_cache_dir: Path,
) -> dict[str, Any]:
    output_path = output_dir / f"{sequence_id}_smpl_params.npz"
    sequence_frame_count = annotation_frame_count(dataset_root, sequence_id=sequence_id)
    state = initialize_sequence_state(
        output_path=output_path,
        frame_count=sequence_frame_count,
        overwrite=args.overwrite,
    )
    requested = np.asarray(frame_ids, dtype=np.int64)
    if requested.size == 0:
        return {
            "sequence_id": sequence_id,
            "status": "empty",
            "requested_frames": 0,
            "fitted_frames": 0,
            "output_path": str(output_path),
        }
    valid_mask = state["valid_mask"]
    missing_frames = [
        int(frame_id)
        for frame_id in requested.tolist()
        if frame_id < 0 or frame_id >= sequence_frame_count or not valid_mask[frame_id]
    ]
    out_of_range = [frame_id for frame_id in missing_frames if frame_id < 0 or frame_id >= sequence_frame_count]
    if out_of_range:
        raise IndexError(
            f"{sequence_id} contains out-of-range frame ids, first={out_of_range[0]}, "
            f"frame_count={sequence_frame_count}"
        )
    if args.skip_existing and not missing_frames:
        return {
            "sequence_id": sequence_id,
            "status": "skipped",
            "requested_frames": int(requested.size),
            "fitted_frames": 0,
            "valid_frames": int(valid_mask.sum()),
            "output_path": str(output_path),
        }

    frames_to_fit = missing_frames if args.skip_existing else requested.tolist()
    if args.overwrite:
        frames_to_fit = requested.tolist()
    total_joint_loss = []
    total_final_loss = []
    total_frame_mpjpe = []
    total_joint_error = []
    for chunk_index, chunk_frame_ids in enumerate(chunked(frames_to_fit, args.batch_size), start=1):
        keypoints = load_sequence_keypoints(
            dataset_root=dataset_root,
            sequence_id=sequence_id,
            frame_ids=chunk_frame_ids,
            cameras=args.cameras,
            annotation_key=args.annotation_key,
            fit_space=args.fit_space,
            views_by_frame=views_by_frame,
            input_smpl_cache_dir=input_smpl_cache_dir,
        )
        fitted = fit_keypoint_batch(
            keypoints_3d=keypoints,
            smpl_model_path=args.smpl_model_path,
            device=device,
            num_iters=args.num_iters,
            lr=args.lr,
            fit_joint_source=args.fit_joint_source,
            joint_regressor_path=args.joint_regressor_path,
            headtop_topk=args.headtop_topk,
            headtop_axis=args.headtop_axis,
            joint_weight=args.joint_weight,
            pose_reg=args.pose_reg,
            shape_reg=args.shape_reg,
            bone_len_reg=args.bone_len_reg,
            temporal_reg=args.temporal_reg,
            grad_clip=args.grad_clip,
            progress_every=args.progress_every,
            sequence_id=sequence_id,
            chunk_index=chunk_index,
            optimize_global_orient=args.fit_space == "world",
        )
        write_fitted_chunk(
            state=state,
            frame_ids=chunk_frame_ids,
            keypoints=keypoints,
            fitted=fitted,
        )
        save_sequence_state(
            output_path=output_path,
            state=state,
            sequence_id=sequence_id,
            annotation_key=args.annotation_key,
            cameras=args.cameras,
            fit_space=args.fit_space,
            fit_joint_source=args.fit_joint_source,
            joint_regressor_path=args.joint_regressor_path,
            headtop_topk=args.headtop_topk,
            headtop_axis=args.headtop_axis,
        )
        total_joint_loss.append(float(fitted["joint_loss"]))
        total_final_loss.append(float(fitted["total_loss"]))
        total_frame_mpjpe.extend(np.asarray(fitted["frame_mpjpe"], dtype=np.float32).tolist())
        total_joint_error.append(np.asarray(fitted["joint_error"], dtype=np.float32))

    per_joint_mpjpe = (
        np.mean(np.concatenate(total_joint_error, axis=0), axis=0)
        if total_joint_error
        else None
    )
    return {
        "sequence_id": sequence_id,
        "status": "fit",
        "requested_frames": int(requested.size),
        "fitted_frames": int(len(frames_to_fit)),
        "valid_frames": int(state["valid_mask"].sum()),
        "fit_joint_source": args.fit_joint_source,
        "mean_mpjpe": float(np.mean(total_frame_mpjpe)) if total_frame_mpjpe else None,
        "median_mpjpe": float(np.median(total_frame_mpjpe)) if total_frame_mpjpe else None,
        "per_joint_mpjpe": per_joint_mpjpe.tolist() if per_joint_mpjpe is not None else None,
        "mean_joint_loss": float(np.mean(total_joint_loss)) if total_joint_loss else None,
        "mean_total_loss": float(np.mean(total_final_loss)) if total_final_loss else None,
        "output_path": str(output_path),
    }


def initialize_sequence_state(
    *,
    output_path: Path,
    frame_count: int,
    overwrite: bool,
) -> dict[str, np.ndarray]:
    if output_path.exists() and not overwrite:
        with np.load(output_path, allow_pickle=False) as payload:
            state = {
                key: payload[key]
                for key in (
                    "global_orient",
                    "body_pose",
                    "betas",
                    "transl",
                    "target_joints",
                    "frame_mpjpe",
                    "joint_error",
                    "joint_loss",
                    "total_loss",
                    "valid_mask",
                )
                if key in payload
            }
        if state.get("valid_mask", np.zeros(0, dtype=bool)).shape[0] == frame_count:
            state.setdefault("frame_mpjpe", np.full((frame_count,), np.nan, dtype=np.float32))
            state.setdefault(
                "joint_error",
                np.full(
                    (frame_count, len(MPII3D_HEATFORMER_EVAL_JOINTS)),
                    np.nan,
                    dtype=np.float32,
                ),
            )
            return state

    return {
        "global_orient": np.full((frame_count, 3), np.nan, dtype=np.float32),
        "body_pose": np.full((frame_count, 69), np.nan, dtype=np.float32),
        "betas": np.full((frame_count, 10), np.nan, dtype=np.float32),
        "transl": np.full((frame_count, 3), np.nan, dtype=np.float32),
        "target_joints": np.full(
            (frame_count, len(MPII3D_HEATFORMER_EVAL_JOINTS), 3),
            np.nan,
            dtype=np.float32,
        ),
        "frame_mpjpe": np.full((frame_count,), np.nan, dtype=np.float32),
        "joint_error": np.full(
            (frame_count, len(MPII3D_HEATFORMER_EVAL_JOINTS)),
            np.nan,
            dtype=np.float32,
        ),
        "joint_loss": np.full((frame_count,), np.nan, dtype=np.float32),
        "total_loss": np.full((frame_count,), np.nan, dtype=np.float32),
        "valid_mask": np.zeros((frame_count,), dtype=bool),
    }


def load_sequence_keypoints(
    *,
    dataset_root: Path,
    sequence_id: str,
    frame_ids: list[int],
    cameras: list[str],
    annotation_key: str,
    fit_space: str,
    views_by_frame: dict[int, tuple[ViewRecord, ...]] | None,
    input_smpl_cache_dir: Path,
) -> np.ndarray:
    payload = load_annotation_mat(str(sequence_dir(dataset_root, sequence_id)))
    if annotation_key not in payload:
        raise KeyError(f"{sequence_id} annot.mat is missing '{annotation_key}'")

    frame_array = np.asarray(frame_ids, dtype=np.int64)
    per_camera_world = []
    for camera_id in cameras:
        camera_index = camera_id_to_index(camera_id)
        camera_joints_mm = np.asarray(payload[annotation_key][camera_index, 0], dtype=np.float32)
        joints_camera = camera_joints_mm[frame_array].reshape(len(frame_ids), -1, 3) * 0.001
        camera = load_camera_parameters(
            dataset_root,
            sequence_id=sequence_id,
            camera_id=f"video_{camera_index}",
        )
        joints_world = (
            (joints_camera - camera.translation[None, None, :])
            @ np.asarray(camera.rotation, dtype=np.float32)
        )
        per_camera_world.append(joints_world)
    joints_world = np.mean(np.stack(per_camera_world, axis=0), axis=0)
    selected = joints_world[:, list(MPII3D_HEATFORMER_EVAL_JOINTS), :]
    if fit_space == "input_root":
        if views_by_frame is None:
            raise RuntimeError("views_by_frame must be provided for --fit-space input_root")
        selected = transform_world_keypoints_to_input_root(
            keypoints_world=selected,
            dataset_root=dataset_root,
            sequence_id=sequence_id,
            frame_ids=frame_ids,
            views_by_frame=views_by_frame,
            input_smpl_cache_dir=input_smpl_cache_dir,
        )
    return np.ascontiguousarray(selected.astype(np.float32, copy=False))


def transform_world_keypoints_to_input_root(
    *,
    keypoints_world: np.ndarray,
    dataset_root: Path,
    sequence_id: str,
    frame_ids: list[int],
    views_by_frame: dict[int, tuple[ViewRecord, ...]],
    input_smpl_cache_dir: Path,
) -> np.ndarray:
    transformed = []
    root_index = MPII3D_HEATFORMER_ROOT_INDEX
    for keypoints, frame_id in zip(keypoints_world, frame_ids, strict=True):
        views = views_by_frame.get(int(frame_id))
        if views is None:
            raise KeyError(f"Missing manifest views for {sequence_id} frame {frame_id}")
        per_view = []
        for view in views:
            camera = load_camera_parameters(
                dataset_root,
                sequence_id=sequence_id,
                camera_id=view.camera_id,
            )
            camera_rotation = np.asarray(camera.rotation, dtype=np.float32)
            camera_translation = np.asarray(camera.translation, dtype=np.float32)
            joints_camera = keypoints @ camera_rotation.T + camera_translation[None, :]
            joints_camera = joints_camera - joints_camera[root_index : root_index + 1]

            input_payload = load_cached_input_smpl(
                input_smpl_cache_dir=input_smpl_cache_dir,
                source_npz_path=view.npz_path,
            )
            input_global_orient = require_vector(
                input_payload,
                "global_orient",
                length=3,
                source_path=cache_path_for_source_npz(input_smpl_cache_dir, view.npz_path),
            )
            input_root_rotation = (
                axis_angle_to_matrix(torch.from_numpy(input_global_orient).reshape(1, 3))
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            per_view.append(joints_camera @ input_root_rotation)
        transformed.append(np.mean(np.stack(per_view, axis=0), axis=0))
    return np.ascontiguousarray(np.stack(transformed, axis=0).astype(np.float32, copy=False))


def load_cached_input_smpl(
    *,
    input_smpl_cache_dir: Path,
    source_npz_path: Path,
) -> dict[str, np.ndarray]:
    cache_path = cache_path_for_source_npz(input_smpl_cache_dir, source_npz_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cached fitted SMPL file does not exist: {cache_path}. "
            "Run scripts/precompute_input_smpl.py first."
        )
    with np.load(cache_path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def require_vector(
    payload: dict[str, np.ndarray],
    key: str,
    *,
    length: int,
    source_path: Path,
) -> np.ndarray:
    if key not in payload:
        raise KeyError(f"Missing required field '{key}' in {source_path}")
    value = np.asarray(payload[key], dtype=np.float32)
    if value.ndim == 2 and value.shape[0] == 1:
        value = value[0]
    if value.shape != (length,):
        raise ValueError(f"Expected field '{key}' in {source_path} to have shape ({length},), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"Field '{key}' in {source_path} contains non-finite values")
    return np.ascontiguousarray(value)


@dataclass(frozen=True)
class FitJointBuilder:
    """Build the fitted skeleton used by the MPI-INF-3DHP SMPL optimizer."""

    source: str
    smpl_indices: torch.Tensor
    joint_regressor: torch.Tensor | None = None
    joint_regressor_input: str | None = None
    headtop_vertex_ids: torch.Tensor | None = None

    def __call__(self, smpl_output) -> torch.Tensor:
        smpl_joints = smpl_output.joints[:, :24, :]
        if self.source in {"regressor", "regressor_headtop_proxy"}:
            if self.joint_regressor is None or self.joint_regressor_input is None:
                raise RuntimeError("Regressor joint source was requested without a loaded regressor")
            if self.joint_regressor_input == "vertices":
                fit_joints = torch.einsum("jv,bvc->bjc", self.joint_regressor, smpl_output.vertices)
            elif self.joint_regressor_input == "joints":
                fit_joints = torch.einsum("jk,bkc->bjc", self.joint_regressor, smpl_output.joints)
            else:
                raise ValueError(f"Unsupported joint_regressor_input={self.joint_regressor_input!r}")
            if self.source == "regressor_headtop_proxy":
                if self.headtop_vertex_ids is None:
                    raise RuntimeError("Regressor headtop proxy source was requested without vertex ids")
                headtop = smpl_output.vertices.index_select(1, self.headtop_vertex_ids).mean(dim=1)
                fit_joints = fit_joints.clone()
                fit_joints[:, 10, :] = headtop
            return fit_joints

        fit_joints = smpl_joints.index_select(1, self.smpl_indices)
        if self.source == "smpl24_headtop_proxy":
            if self.headtop_vertex_ids is None:
                raise RuntimeError("Headtop proxy source was requested without vertex ids")
            headtop = smpl_output.vertices.index_select(1, self.headtop_vertex_ids).mean(dim=1)
            fit_joints = fit_joints.clone()
            fit_joints[:, 10, :] = headtop
        return fit_joints


def build_fit_joint_builder(
    *,
    smpl,
    device: torch.device,
    fit_joint_source: str,
    joint_regressor_path: str | None,
    headtop_topk: int,
    headtop_axis: str,
) -> FitJointBuilder:
    smpl_indices = torch.as_tensor(
        MPII3D_HEATFORMER_TO_SMPL24,
        device=device,
        dtype=torch.long,
    )
    if fit_joint_source in {"regressor", "regressor_headtop_proxy"}:
        if joint_regressor_path is None:
            raise ValueError(f"--fit-joint-source {fit_joint_source} requires --joint-regressor-path")
        regressor, regressor_input = load_joint_regressor(
            joint_regressor_path,
            device=device,
            num_smpl_joints=int(getattr(smpl, "NUM_JOINTS", 23)) + 1,
        )
        headtop_vertex_ids = None
        if fit_joint_source == "regressor_headtop_proxy":
            headtop_vertex_ids = select_headtop_vertex_ids(
                smpl=smpl,
                device=device,
                topk=headtop_topk,
                axis=headtop_axis,
            )
        return FitJointBuilder(
            source=fit_joint_source,
            smpl_indices=smpl_indices,
            joint_regressor=regressor,
            joint_regressor_input=regressor_input,
            headtop_vertex_ids=headtop_vertex_ids,
        )

    headtop_vertex_ids = None
    if fit_joint_source == "smpl24_headtop_proxy":
        headtop_vertex_ids = select_headtop_vertex_ids(
            smpl=smpl,
            device=device,
            topk=headtop_topk,
            axis=headtop_axis,
        )
    return FitJointBuilder(
        source=fit_joint_source,
        smpl_indices=smpl_indices,
        headtop_vertex_ids=headtop_vertex_ids,
    )


def load_joint_regressor(
    path: str | Path,
    *,
    device: torch.device,
    num_smpl_joints: int,
) -> tuple[torch.Tensor, str]:
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Joint regressor does not exist: {resolved}")
    if resolved.suffix == ".npz":
        with np.load(resolved, allow_pickle=False) as payload:
            preferred_keys = ("J_regressor_h36m", "J_regressor", "regressor")
            key = next((item for item in preferred_keys if item in payload.files), payload.files[0])
            regressor = np.asarray(payload[key], dtype=np.float32)
    else:
        regressor = np.asarray(np.load(resolved, allow_pickle=False), dtype=np.float32)
    if regressor.ndim != 2:
        raise ValueError(f"Joint regressor must be a 2D array, got {regressor.shape}")
    if regressor.shape[0] != 17 and regressor.shape[1] == 17:
        regressor = regressor.T
    if regressor.shape[0] != 17:
        raise ValueError(f"Expected joint regressor to have 17 output joints, got {regressor.shape}")
    if regressor.shape[1] == 6890:
        regressor_input = "vertices"
    elif regressor.shape[1] <= num_smpl_joints:
        regressor_input = "joints"
    else:
        raise ValueError(
            f"Unsupported joint regressor shape {regressor.shape}. Expected [17, 6890] "
            f"or [17, <= {num_smpl_joints}]."
        )
    tensor = torch.from_numpy(np.ascontiguousarray(regressor)).to(device=device, dtype=torch.float32)
    return tensor, regressor_input


def select_headtop_vertex_ids(
    *,
    smpl,
    device: torch.device,
    topk: int,
    axis: str,
) -> torch.Tensor:
    if topk < 1:
        raise ValueError(f"--headtop-topk must be >= 1, got {topk}")
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    with torch.no_grad():
        betas = torch.zeros(1, 10, device=device)
        body_pose = torch.zeros(1, 69, device=device)
        global_orient = torch.zeros(1, 3, device=device)
        transl = torch.zeros(1, 3, device=device)
        output = smpl(
            body_pose=body_pose,
            betas=betas,
            global_orient=global_orient,
            transl=transl,
        )
        values = output.vertices[0, :, axis_index]
        k = min(int(topk), int(values.numel()))
        return torch.topk(values, k=k, largest=True).indices.to(device=device, dtype=torch.long)


def fit_keypoint_batch(
    *,
    keypoints_3d: np.ndarray,
    smpl_model_path: str,
    device: torch.device,
    num_iters: int,
    lr: float,
    fit_joint_source: str,
    joint_regressor_path: str | None,
    headtop_topk: int,
    headtop_axis: str,
    joint_weight: float,
    pose_reg: float,
    shape_reg: float,
    bone_len_reg: float,
    temporal_reg: float,
    grad_clip: float,
    progress_every: int,
    sequence_id: str,
    chunk_index: int,
    optimize_global_orient: bool,
) -> dict[str, np.ndarray | float]:
    if keypoints_3d.ndim != 3 or keypoints_3d.shape[1:] != (17, 3):
        raise ValueError(f"Expected keypoints [T, 17, 3], got {keypoints_3d.shape}")
    batch_size = int(keypoints_3d.shape[0])
    smpl = build_smpl_model(
        device=device,
        smpl_model_path=smpl_model_path,
        batch_size=batch_size,
    )
    smpl.eval()
    fit_joint_builder = build_fit_joint_builder(
        smpl=smpl,
        device=device,
        fit_joint_source=fit_joint_source,
        joint_regressor_path=joint_regressor_path,
        headtop_topk=headtop_topk,
        headtop_axis=headtop_axis,
    )

    keypoints = torch.from_numpy(keypoints_3d).to(device=device, dtype=torch.float32)
    root_index = MPII3D_HEATFORMER_ROOT_INDEX
    keypoints_centered = keypoints - keypoints[:, root_index : root_index + 1, :]
    gt_bone_len = normalized_bone_lengths(keypoints_centered)

    body_pose = torch.zeros(batch_size, 69, device=device, requires_grad=True)
    global_orient = torch.zeros(
        batch_size,
        3,
        device=device,
        requires_grad=optimize_global_orient,
    )
    betas = torch.zeros(1, 10, device=device, requires_grad=True)
    optimized_params = [body_pose, betas]
    if optimize_global_orient:
        optimized_params.append(global_orient)
    optimizer = torch.optim.Adam(optimized_params, lr=lr)

    final_losses: dict[str, torch.Tensor] = {}
    final_joint_error: torch.Tensor | None = None
    for iteration in range(num_iters):
        optimizer.zero_grad(set_to_none=True)
        betas_expanded = betas.expand(batch_size, -1)
        smpl_output = smpl(
            body_pose=body_pose,
            betas=betas_expanded,
            global_orient=global_orient,
            transl=torch.zeros(batch_size, 3, device=device),
        )
        smpl_joints = fit_joint_builder(smpl_output)
        smpl_joints_centered = smpl_joints - smpl_joints[:, root_index : root_index + 1, :]
        joint_error = torch.linalg.norm(smpl_joints_centered - keypoints_centered, dim=-1)
        per_joint_mse = joint_error.square()
        joint_loss = per_joint_mse.mean()
        pose_loss = body_pose.square().mean()
        shape_loss = betas.square().mean()
        bone_loss = (
            normalized_bone_lengths(smpl_joints_centered) - gt_bone_len
        ).square().mean()
        smooth_loss = body_pose.new_zeros(())
        if temporal_reg > 0.0 and batch_size > 1:
            smooth_loss = (body_pose[1:] - body_pose[:-1]).square().mean()
        total_loss = (
            joint_weight * joint_loss
            + pose_reg * pose_loss
            + shape_reg * shape_loss
            + bone_len_reg * bone_loss
            + temporal_reg * smooth_loss
        )
        if not torch.isfinite(total_loss):
            raise RuntimeError(
                f"Non-finite SMPL fitting loss at {sequence_id} chunk {chunk_index} "
                f"iteration {iteration}"
            )
        total_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(optimized_params, grad_clip)
        optimizer.step()
        final_losses = {
            "joint_loss": joint_loss.detach(),
            "total_loss": total_loss.detach(),
            "pose_loss": pose_loss.detach(),
            "shape_loss": shape_loss.detach(),
            "bone_loss": bone_loss.detach(),
            "smooth_loss": smooth_loss.detach(),
        }
        final_joint_error = joint_error.detach()
        if progress_every > 0 and (
            iteration == 0 or iteration == num_iters - 1 or (iteration + 1) % progress_every == 0
        ):
            print(
                f"[{sequence_id} chunk={chunk_index} iter={iteration + 1:04d}/{num_iters}] "
                f"loss={float(total_loss.detach().cpu()):.6f} "
                f"joint={float(joint_loss.detach().cpu()):.6f}",
                flush=True,
            )

    with torch.no_grad():
        if final_joint_error is None:
            raise RuntimeError("SMPL fitting did not run any optimization iterations")
        betas_expanded = betas.expand(batch_size, -1)
        final_smpl_output = smpl(
            body_pose=body_pose,
            betas=betas_expanded,
            global_orient=global_orient,
            transl=torch.zeros(batch_size, 3, device=device),
        )
        final_smpl_joints = fit_joint_builder(final_smpl_output)
        final_smpl_joints_centered = (
            final_smpl_joints - final_smpl_joints[:, root_index : root_index + 1, :]
        )
        final_joint_error = torch.linalg.norm(
            final_smpl_joints_centered - keypoints_centered,
            dim=-1,
        )
        final_joint_loss = final_joint_error.square().mean()
        transl = keypoints[:, root_index, :]
        return {
            "global_orient": global_orient.detach().cpu().numpy().astype(np.float32),
            "body_pose": body_pose.detach().cpu().numpy().astype(np.float32),
            "betas": betas_expanded.detach().cpu().numpy().astype(np.float32),
            "transl": transl.detach().cpu().numpy().astype(np.float32),
            "frame_mpjpe": final_joint_error.mean(dim=1).cpu().numpy().astype(np.float32),
            "joint_error": final_joint_error.cpu().numpy().astype(np.float32),
            "per_joint_mpjpe": final_joint_error.mean(dim=0).cpu().numpy().astype(np.float32),
            "joint_loss": float(final_joint_loss.cpu()),
            "total_loss": float(final_losses["total_loss"].cpu()),
        }


def normalized_bone_lengths(joints: torch.Tensor) -> torch.Tensor:
    lengths = []
    for start, end in MPII3D_HEATFORMER_BONE_PAIRS:
        lengths.append((joints[:, start] - joints[:, end]).norm(dim=-1))
    stacked = torch.stack(lengths, dim=-1)
    return stacked / stacked.mean(dim=-1, keepdim=True).clamp_min(1e-8)


def write_fitted_chunk(
    *,
    state: dict[str, np.ndarray],
    frame_ids: list[int],
    keypoints: np.ndarray,
    fitted: dict[str, np.ndarray | float],
) -> None:
    frame_array = np.asarray(frame_ids, dtype=np.int64)
    for key in ("global_orient", "body_pose", "betas", "transl"):
        state[key][frame_array] = np.asarray(fitted[key], dtype=np.float32)
    state["target_joints"][frame_array] = keypoints.astype(np.float32, copy=False)
    state["frame_mpjpe"][frame_array] = np.asarray(fitted["frame_mpjpe"], dtype=np.float32)
    state["joint_error"][frame_array] = np.asarray(fitted["joint_error"], dtype=np.float32)
    state["joint_loss"][frame_array] = np.float32(fitted["joint_loss"])
    state["total_loss"][frame_array] = np.float32(fitted["total_loss"])
    state["valid_mask"][frame_array] = True


def save_sequence_state(
    *,
    output_path: Path,
    state: dict[str, np.ndarray],
    sequence_id: str,
    annotation_key: str,
    cameras: list[str],
    fit_space: str,
    fit_joint_source: str,
    joint_regressor_path: str | None,
    headtop_topk: int,
    headtop_axis: str,
) -> None:
    frame_ids = np.flatnonzero(state["valid_mask"]).astype(np.int64)
    np.savez_compressed(
        output_path,
        **state,
        frame_ids=frame_ids,
        sequence_id=np.asarray(sequence_id),
        annotation_key=np.asarray(annotation_key),
        cameras=np.asarray(cameras),
        fit_space=np.asarray(fit_space),
        fit_joint_source=np.asarray(fit_joint_source),
        joint_regressor_path=np.asarray(joint_regressor_path or ""),
        headtop_topk=np.asarray(headtop_topk, dtype=np.int64),
        headtop_axis=np.asarray(headtop_axis),
        joint_names=np.asarray(
            [
                "pelvis",
                "left_hip",
                "left_knee",
                "left_ankle",
                "right_hip",
                "right_knee",
                "right_ankle",
                "spine",
                "neck",
                "head",
                "headtop",
                "left_shoulder",
                "left_elbow",
                "left_wrist",
                "right_shoulder",
                "right_elbow",
                "right_wrist",
            ]
        ),
        mpii3d_joint_indices=np.asarray(MPII3D_HEATFORMER_EVAL_JOINTS, dtype=np.int64),
        smpl_joint_indices=np.asarray(MPII3D_HEATFORMER_TO_SMPL24, dtype=np.int64),
        root_index=np.asarray(MPII3D_HEATFORMER_ROOT_INDEX, dtype=np.int64),
    )


def chunked(items: list[int], chunk_size: int) -> list[list[int]]:
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


if __name__ == "__main__":
    main()
