#!/usr/bin/env python
"""Visualize Panoptic Stage 1 input-view SMPL fits against target caches."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mvhpe3d.data.splits import (
    SampleRecord,
    ViewRecord,
    load_sample_records,
    resolve_split_records,
)
from mvhpe3d.metrics import batch_mpjpe, batch_pa_mpjpe
from mvhpe3d.utils import build_smpl_model, load_panoptic_camera_parameters
from mvhpe3d.utils.mhr_smpl_conversion import cache_path_for_source_npz

DEFAULT_DATASET_ROOT = Path("/opt/data/panoptic_kinoptic_single_actor_cropped")
_NPZ_CACHE: dict[Path, dict[str, np.ndarray]] = {}
_FRAME_INDEX_CACHE: dict[Path, dict[str, int]] = {}

SMPL_EDGES = (
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

PANOPTIC_EDGES = (
    (2, 0),
    (0, 1),
    (0, 3),
    (3, 4),
    (4, 5),
    (0, 9),
    (9, 10),
    (10, 11),
    (2, 6),
    (6, 7),
    (7, 8),
    (2, 12),
    (12, 13),
    (13, 14),
)

PANOPTIC_TO_SMPL24 = {
    2: 0,  # BodyCenter -> pelvis
    6: 1,  # lHip
    12: 2,  # rHip
    7: 4,  # lKnee
    13: 5,  # rKnee
    8: 7,  # lAnkle
    14: 8,  # rAnkle
    0: 12,  # Neck
    3: 16,  # lShoulder
    9: 17,  # rShoulder
    4: 18,  # lElbow
    10: 19,  # rElbow
    5: 20,  # lWrist
    11: 21,  # rWrist
}


@dataclass(frozen=True)
class TargetPayload:
    global_orient: np.ndarray
    body_pose: np.ndarray
    betas: np.ndarray
    transl: np.ndarray
    fitting_loss: float | None
    cache_path: Path


@dataclass(frozen=True)
class ViewDebug:
    view: ViewRecord
    input_smpl: dict[str, np.ndarray]
    source_payload: dict[str, np.ndarray]
    input_cache_path: Path
    mpjpe: float
    pa_mpjpe: float
    fitted_transl_reproj_error_px: float


@dataclass(frozen=True)
class SampleDebug:
    record: SampleRecord
    target: TargetPayload
    views: tuple[ViewDebug, ...]
    input_avg_mpjpe: float
    input_avg_pa_mpjpe: float
    fitted_transl_reproj_error_px: float
    target_fit_mpjpe: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create debug plots for Panoptic SAM3DBody input predictions, "
            "input-view fitted SMPL, Panoptic target SMPL, and raw Panoptic gt3d."
        )
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Defaults to <dataset-root>/panoptic_stage1_manifest.json.",
    )
    parser.add_argument(
        "--gt-smpl-dir",
        default=None,
        help="Defaults to <dataset-root>/smpl.",
    )
    parser.add_argument(
        "--input-smpl-cache-dir",
        default=None,
        help="Defaults to <dataset-root>/sam3dbody_fitted_smpl.",
    )
    parser.add_argument(
        "--split-config-path",
        default="configs/data/panoptic_stage1_splits.yaml",
    )
    parser.add_argument("--split-name", default="cross_camera_split")
    parser.add_argument("--stage", choices=("train", "val", "test"), default="test")
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=512,
        help="Number of split records to scan for highest-error examples.",
    )
    parser.add_argument(
        "--selection",
        choices=(
            "highest_input_mpjpe",
            "highest_fitted_reproj_error",
            "highest_target_fit_mpjpe",
            "first",
        ),
        default="highest_input_mpjpe",
    )
    parser.add_argument("--output-dir", default="outputs/debug/panoptic_inputs")
    parser.add_argument(
        "--device", default=None, help="Torch device, e.g. cuda:0 or cpu."
    )
    parser.add_argument("--smpl-model-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_views < 1:
        raise ValueError(f"--num-views must be >= 1, got {args.num_views}")
    if args.num_samples < 1:
        raise ValueError(f"--num-samples must be >= 1, got {args.num_samples}")
    if args.scan_limit < 1:
        raise ValueError(f"--scan-limit must be >= 1, got {args.scan_limit}")

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    manifest_path = (
        Path(args.manifest_path).expanduser().resolve()
        if args.manifest_path is not None
        else dataset_root / "panoptic_stage1_manifest.json"
    )
    gt_smpl_dir = (
        Path(args.gt_smpl_dir).expanduser().resolve()
        if args.gt_smpl_dir is not None
        else dataset_root / "smpl"
    )
    input_smpl_cache_dir = (
        Path(args.input_smpl_cache_dir).expanduser().resolve()
        if args.input_smpl_cache_dir is not None
        else dataset_root / "sam3dbody_fitted_smpl"
    )
    split_config_path = Path(args.split_config_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_sample_records(manifest_path)
    split_records = resolve_split_records(
        records,
        split_config_path=split_config_path,
        split_name=args.split_name,
        num_views=args.num_views,
    )
    candidates = split_records[args.stage]
    if not candidates:
        raise RuntimeError(f"No records found for stage '{args.stage}'")

    device = resolve_device(args.device)
    smpl_model = build_smpl_model(
        device=device,
        smpl_model_path=args.smpl_model_path,
        batch_size=1,
    )
    smpl_model.eval()

    scanned = scan_samples(
        records=candidates[: args.scan_limit],
        dataset_root=dataset_root,
        gt_smpl_dir=gt_smpl_dir,
        input_smpl_cache_dir=input_smpl_cache_dir,
        smpl_model=smpl_model,
        device=device,
        num_views=args.num_views,
        selection=args.selection,
        num_samples=args.num_samples,
    )
    if not scanned:
        raise RuntimeError(
            "No visualizable samples found. Check that input SMPL cache files exist under "
            f"{input_smpl_cache_dir}"
        )

    selected = select_samples(
        scanned, selection=args.selection, num_samples=args.num_samples
    )
    summary: list[dict[str, Any]] = []
    for index, sample in enumerate(selected):
        sample_summary = write_sample_visualization(
            sample=sample,
            index=index,
            output_dir=output_dir,
            dataset_root=dataset_root,
            smpl_model=smpl_model,
            device=device,
        )
        summary.append(sample_summary)

    summary_payload = {
        "dataset_root": str(dataset_root),
        "manifest_path": str(manifest_path),
        "gt_smpl_dir": str(gt_smpl_dir),
        "input_smpl_cache_dir": str(input_smpl_cache_dir),
        "split_config_path": str(split_config_path),
        "split_name": args.split_name,
        "stage": args.stage,
        "num_views": args.num_views,
        "scan_limit": args.scan_limit,
        "scanned_visualizable": len(scanned),
        "selection": args.selection,
        "samples": summary,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(selected)} Panoptic input debug samples to {output_dir}")
    print(f"Summary: {summary_path}")


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def scan_samples(
    *,
    records: list[SampleRecord],
    dataset_root: Path,
    gt_smpl_dir: Path,
    input_smpl_cache_dir: Path,
    smpl_model,
    device: torch.device,
    num_views: int,
    selection: str,
    num_samples: int,
) -> list[SampleDebug]:
    if selection == "highest_target_fit_mpjpe":
        return scan_high_target_fit_samples(
            records=records,
            dataset_root=dataset_root,
            gt_smpl_dir=gt_smpl_dir,
            input_smpl_cache_dir=input_smpl_cache_dir,
            smpl_model=smpl_model,
            device=device,
            num_views=num_views,
            num_samples=num_samples,
        )

    samples: list[SampleDebug] = []
    for record in records:
        try:
            sample = build_sample_debug(
                record=record,
                dataset_root=dataset_root,
                gt_smpl_dir=gt_smpl_dir,
                input_smpl_cache_dir=input_smpl_cache_dir,
                smpl_model=smpl_model,
                device=device,
                num_views=num_views,
            )
        except (FileNotFoundError, KeyError, ValueError):
            continue
        samples.append(sample)
        if selection == "first" and len(samples) >= 1:
            continue
    return samples


def scan_high_target_fit_samples(
    *,
    records: list[SampleRecord],
    dataset_root: Path,
    gt_smpl_dir: Path,
    input_smpl_cache_dir: Path,
    smpl_model,
    device: torch.device,
    num_views: int,
    num_samples: int,
) -> list[SampleDebug]:
    prefilter_count = max(num_samples * 100, 256)
    scored_by_cached_loss: list[tuple[float, SampleRecord]] = []
    missing_cached_loss: list[SampleRecord] = []
    for record in records:
        try:
            target = load_target_payload(
                gt_smpl_dir=gt_smpl_dir,
                sequence_id=record.sequence_id,
                frame_id=record.frame_id,
            )
        except (FileNotFoundError, KeyError, ValueError):
            continue
        if target.fitting_loss is None or not np.isfinite(target.fitting_loss):
            missing_cached_loss.append(record)
        else:
            scored_by_cached_loss.append((target.fitting_loss, record))

    prefiltered_records = [
        record
        for _, record in sorted(
            scored_by_cached_loss,
            key=lambda item: item[0],
            reverse=True,
        )[:prefilter_count]
    ]
    if not prefiltered_records:
        prefiltered_records = missing_cached_loss[:prefilter_count]

    scored_records: list[tuple[float, SampleRecord]] = []
    for record in prefiltered_records:
        try:
            target_fit_mpjpe = compute_target_fit_mpjpe_for_record(
                record=record,
                dataset_root=dataset_root,
                gt_smpl_dir=gt_smpl_dir,
                smpl_model=smpl_model,
                device=device,
            )
        except (FileNotFoundError, KeyError, ValueError):
            continue
        if np.isfinite(target_fit_mpjpe):
            scored_records.append((target_fit_mpjpe, record))

    samples: list[SampleDebug] = []
    for _, record in sorted(scored_records, key=lambda item: item[0], reverse=True):
        try:
            samples.append(
                build_sample_debug(
                    record=record,
                    dataset_root=dataset_root,
                    gt_smpl_dir=gt_smpl_dir,
                    input_smpl_cache_dir=input_smpl_cache_dir,
                    smpl_model=smpl_model,
                    device=device,
                    num_views=num_views,
                )
            )
        except (FileNotFoundError, KeyError, ValueError):
            continue
        if len(samples) >= num_samples:
            break
    return samples


def compute_target_fit_mpjpe_for_record(
    *,
    record: SampleRecord,
    dataset_root: Path,
    gt_smpl_dir: Path,
    smpl_model,
    device: torch.device,
) -> float:
    target = load_target_payload(
        gt_smpl_dir=gt_smpl_dir,
        sequence_id=record.sequence_id,
        frame_id=record.frame_id,
    )
    target_world_joints = build_smpl_joints(
        smpl_model=smpl_model,
        device=device,
        body_pose=target.body_pose,
        betas=target.betas,
        global_orient=target.global_orient,
        transl=target.transl,
    )
    panoptic_world_joints = load_panoptic_gt3d(
        dataset_root=dataset_root,
        sequence_id=record.sequence_id,
        frame_id=record.frame_id,
    )
    panoptic_confidence = load_panoptic_gt3d_confidence(
        dataset_root=dataset_root,
        sequence_id=record.sequence_id,
        frame_id=record.frame_id,
    )
    return compute_target_fit_mpjpe(
        target_world_joints=target_world_joints,
        panoptic_world_joints=panoptic_world_joints,
        panoptic_confidence=panoptic_confidence,
    )


def select_samples(
    samples: list[SampleDebug],
    *,
    selection: str,
    num_samples: int,
) -> list[SampleDebug]:
    if selection == "highest_input_mpjpe":
        ordered = sorted(
            samples, key=lambda sample: sample.input_avg_mpjpe, reverse=True
        )
    elif selection == "highest_fitted_reproj_error":
        ordered = sorted(
            samples,
            key=lambda sample: sample.fitted_transl_reproj_error_px,
            reverse=True,
        )
    elif selection == "highest_target_fit_mpjpe":
        ordered = sorted(
            samples,
            key=lambda sample: (
                sample.target_fit_mpjpe
                if np.isfinite(sample.target_fit_mpjpe)
                else float("-inf")
            ),
            reverse=True,
        )
    else:
        ordered = samples
    return ordered[:num_samples]


def build_sample_debug(
    *,
    record: SampleRecord,
    dataset_root: Path,
    gt_smpl_dir: Path,
    input_smpl_cache_dir: Path,
    smpl_model,
    device: torch.device,
    num_views: int,
) -> SampleDebug:
    selected_views = select_views(record, num_views=num_views)
    target = load_target_payload(
        gt_smpl_dir=gt_smpl_dir,
        sequence_id=record.sequence_id,
        frame_id=record.frame_id,
    )
    target_joints = build_smpl_joints(
        smpl_model=smpl_model,
        device=device,
        body_pose=target.body_pose,
        betas=target.betas,
        global_orient=np.zeros(3, dtype=np.float32),
        transl=np.zeros(3, dtype=np.float32),
    )
    target_world_joints = build_smpl_joints(
        smpl_model=smpl_model,
        device=device,
        body_pose=target.body_pose,
        betas=target.betas,
        global_orient=target.global_orient,
        transl=target.transl,
    )
    panoptic_world_joints = load_panoptic_gt3d(
        dataset_root=dataset_root,
        sequence_id=record.sequence_id,
        frame_id=record.frame_id,
    )
    panoptic_confidence = load_panoptic_gt3d_confidence(
        dataset_root=dataset_root,
        sequence_id=record.sequence_id,
        frame_id=record.frame_id,
    )
    target_fit_mpjpe = compute_target_fit_mpjpe(
        target_world_joints=target_world_joints,
        panoptic_world_joints=panoptic_world_joints,
        panoptic_confidence=panoptic_confidence,
    )

    views: list[ViewDebug] = []
    for view in selected_views:
        source_payload = load_npz(view.npz_path)
        input_cache_path = cache_path_for_source_npz(
            input_smpl_cache_dir, view.npz_path
        )
        if not input_cache_path.exists():
            raise FileNotFoundError(
                f"Input SMPL cache does not exist: {input_cache_path}"
            )
        input_smpl = load_npz(input_cache_path)
        input_joints = build_smpl_joints(
            smpl_model=smpl_model,
            device=device,
            body_pose=as_vector(input_smpl["body_pose"], 69),
            betas=as_vector(input_smpl["betas"], 10),
            global_orient=np.zeros(3, dtype=np.float32),
            transl=np.zeros(3, dtype=np.float32),
        )
        mpjpe = float(
            batch_mpjpe(to_joint_tensor(input_joints), to_joint_tensor(target_joints))
        )
        pa_mpjpe = float(
            batch_pa_mpjpe(
                to_joint_tensor(input_joints), to_joint_tensor(target_joints)
            )
        )
        fitted_transl_reproj_error_px = compute_fitted_transl_reprojection_error(
            record=record,
            view=view,
            dataset_root=dataset_root,
            source_payload=source_payload,
            input_smpl=input_smpl,
            smpl_model=smpl_model,
            device=device,
        )
        views.append(
            ViewDebug(
                view=view,
                input_smpl=input_smpl,
                source_payload=source_payload,
                input_cache_path=input_cache_path,
                mpjpe=mpjpe,
                pa_mpjpe=pa_mpjpe,
                fitted_transl_reproj_error_px=fitted_transl_reproj_error_px,
            )
        )

    return SampleDebug(
        record=record,
        target=target,
        views=tuple(views),
        input_avg_mpjpe=float(np.mean([view.mpjpe for view in views])),
        input_avg_pa_mpjpe=float(np.mean([view.pa_mpjpe for view in views])),
        fitted_transl_reproj_error_px=float(
            np.mean([view.fitted_transl_reproj_error_px for view in views])
        ),
        target_fit_mpjpe=target_fit_mpjpe,
    )


def compute_target_fit_mpjpe(
    *,
    target_world_joints: np.ndarray,
    panoptic_world_joints: np.ndarray,
    panoptic_confidence: np.ndarray,
) -> float:
    errors = []
    for panoptic_index, smpl_index in sorted(PANOPTIC_TO_SMPL24.items()):
        if float(panoptic_confidence[panoptic_index]) <= 0.05:
            continue
        errors.append(
            float(
                np.linalg.norm(
                    target_world_joints[smpl_index]
                    - panoptic_world_joints[panoptic_index]
                )
            )
        )
    if not errors:
        return float("nan")
    return float(np.mean(errors))


def compute_fitted_transl_reprojection_error(
    *,
    record: SampleRecord,
    view: ViewRecord,
    dataset_root: Path,
    source_payload: dict[str, np.ndarray],
    input_smpl: dict[str, np.ndarray],
    smpl_model,
    device: torch.device,
) -> float:
    input_camera_joints = build_smpl_joints(
        smpl_model=smpl_model,
        device=device,
        body_pose=as_vector(input_smpl["body_pose"], 69),
        betas=as_vector(input_smpl["betas"], 10),
        global_orient=as_vector(input_smpl["global_orient"], 3),
        transl=as_vector(input_smpl["transl"], 3),
    )
    panoptic_world_joints = load_panoptic_gt3d(
        dataset_root=dataset_root,
        sequence_id=record.sequence_id,
        frame_id=record.frame_id,
    )
    camera = load_panoptic_camera_parameters(
        dataset_root,
        sequence_id=record.sequence_id,
        camera_id=view.camera_id,
    )
    panoptic_camera_joints = transform_points_world_to_camera(
        panoptic_world_joints,
        rotation=camera.rotation,
        translation=camera.translation,
    )
    input_projected = project_points(
        input_camera_joints[:24],
        intrinsics=np.asarray(source_payload["cam_int"], dtype=np.float32),
    )
    gt_projected = project_points(panoptic_camera_joints, intrinsics=camera.intrinsics)

    errors: list[float] = []
    for panoptic_index, smpl_index in sorted(PANOPTIC_TO_SMPL24.items()):
        input_point = input_projected[smpl_index]
        gt_point = gt_projected[panoptic_index]
        if input_point is None or gt_point is None:
            continue
        errors.append(
            float(
                np.linalg.norm(
                    np.asarray(input_point, dtype=np.float32)
                    - np.asarray(gt_point, dtype=np.float32)
                )
            )
        )
    if not errors:
        return float("inf")
    return float(np.mean(errors))


def select_views(record: SampleRecord, *, num_views: int) -> tuple[ViewRecord, ...]:
    ordered = tuple(sorted(record.views, key=lambda view: view.camera_id))
    if len(ordered) < num_views:
        raise ValueError(
            f"Record {record.sample_id} has {len(ordered)} views, need {num_views}"
        )
    return ordered[:num_views]


def load_target_payload(
    *,
    gt_smpl_dir: Path,
    sequence_id: str,
    frame_id: str,
) -> TargetPayload:
    cache_path = gt_smpl_dir / f"{sequence_id}_smpl_params.npz"
    payload = load_npz(cache_path)
    frame_index = resolve_frame_index(
        payload,
        frame_id=frame_id,
        sequence_id=sequence_id,
        cache_path=cache_path.resolve(),
    )
    fitting_loss = None
    if "fitting_loss" in payload:
        fitting_loss = float(
            np.asarray(payload["fitting_loss"]).reshape(-1)[frame_index]
        )
    return TargetPayload(
        global_orient=select_frame_vector(payload, "global_orient", frame_index, 3),
        body_pose=select_frame_vector(payload, "body_pose", frame_index, 69),
        betas=select_frame_vector(payload, "betas", frame_index, 10),
        transl=select_frame_vector(payload, "transl", frame_index, 3),
        fitting_loss=fitting_loss,
        cache_path=cache_path,
    )


def resolve_frame_index(
    payload: dict[str, np.ndarray],
    *,
    frame_id: str,
    sequence_id: str,
    cache_path: Path,
) -> int:
    if "frame_ids" not in payload:
        index = int(frame_id) - 1
        if index < 0:
            raise ValueError(
                f"Invalid 1-based frame_id for sequence {sequence_id}: {frame_id}"
            )
        return index
    frame_index = _FRAME_INDEX_CACHE.get(cache_path)
    if frame_index is None:
        frame_index = {
            normalize_frame_id(value): index
            for index, value in enumerate(np.asarray(payload["frame_ids"]).reshape(-1))
        }
        _FRAME_INDEX_CACHE[cache_path] = frame_index
    normalized = normalize_frame_id(frame_id)
    if normalized not in frame_index:
        raise KeyError(
            f"Frame {frame_id} was not found in target cache for {sequence_id}"
        )
    return frame_index[normalized]


def select_frame_vector(
    payload: dict[str, np.ndarray],
    key: str,
    frame_index: int,
    expected_dim: int,
) -> np.ndarray:
    value = np.asarray(payload[key], dtype=np.float32)
    if value.ndim == 1:
        return as_vector(value, expected_dim)
    return as_vector(value[frame_index], expected_dim)


def write_sample_visualization(
    *,
    sample: SampleDebug,
    index: int,
    output_dir: Path,
    dataset_root: Path,
    smpl_model,
    device: torch.device,
) -> dict[str, Any]:
    record = sample.record
    sample_dir = (
        output_dir
        / f"sample_{index:03d}_{record.sequence_id}_{normalize_frame_id(record.frame_id)}"
    )
    sample_dir.mkdir(parents=True, exist_ok=True)

    target_world_joints = build_smpl_joints(
        smpl_model=smpl_model,
        device=device,
        body_pose=sample.target.body_pose,
        betas=sample.target.betas,
        global_orient=sample.target.global_orient,
        transl=sample.target.transl,
    )
    panoptic_world_joints = load_panoptic_gt3d(
        dataset_root=dataset_root,
        sequence_id=record.sequence_id,
        frame_id=record.frame_id,
    )

    view_panels: list[np.ndarray] = []
    view_summaries: list[dict[str, Any]] = []
    for view_index, view_debug in enumerate(sample.views):
        panel, view_summary, separate_overlays = build_view_panel(
            sample=sample,
            view_debug=view_debug,
            view_index=view_index,
            dataset_root=dataset_root,
            target_world_joints=target_world_joints,
            panoptic_world_joints=panoptic_world_joints,
            smpl_model=smpl_model,
            device=device,
        )
        view_path = (
            sample_dir / f"view_{view_index + 1}_{view_debug.view.camera_id}.png"
        )
        cv2.imwrite(str(view_path), panel)
        view_panels.append(panel)
        view_summary["image_path"] = str(view_path)
        separate_paths = {}
        for overlay_name, overlay_image in separate_overlays.items():
            overlay_path = (
                sample_dir
                / f"view_{view_index + 1}_{view_debug.view.camera_id}_{overlay_name}.png"
            )
            cv2.imwrite(str(overlay_path), overlay_image)
            separate_paths[overlay_name] = str(overlay_path)
        view_summary["separate_images"] = separate_paths
        view_summaries.append(view_summary)

    canonical_path = sample_dir / "canonical_skeletons.png"
    write_canonical_plot(
        sample=sample,
        output_path=canonical_path,
        smpl_model=smpl_model,
        device=device,
    )
    summary_image = build_summary_image(view_panels, canonical_path)
    summary_image_path = sample_dir / "summary.png"
    cv2.imwrite(str(summary_image_path), summary_image)

    sample_summary = {
        "sample_dir": str(sample_dir),
        "summary_image": str(summary_image_path),
        "canonical_image": str(canonical_path),
        "sample_id": record.sample_id,
        "sequence_id": record.sequence_id,
        "frame_id": record.frame_id,
        "input_avg_mpjpe_m": sample.input_avg_mpjpe,
        "input_avg_pa_mpjpe_m": sample.input_avg_pa_mpjpe,
        "fitted_transl_reproj_error_px": sample.fitted_transl_reproj_error_px,
        "target_fit_mpjpe_m": sample.target_fit_mpjpe,
        "target_fitting_loss": sample.target.fitting_loss,
        "target_cache_path": str(sample.target.cache_path),
        "views": view_summaries,
    }
    (sample_dir / "metrics.json").write_text(
        json.dumps(sample_summary, indent=2), encoding="utf-8"
    )
    return sample_summary


def build_view_panel(
    *,
    sample: SampleDebug,
    view_debug: ViewDebug,
    view_index: int,
    dataset_root: Path,
    target_world_joints: np.ndarray,
    panoptic_world_joints: np.ndarray,
    smpl_model,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    record = sample.record
    image_path = resolve_panoptic_rgb_path(
        dataset_root=dataset_root,
        sequence_id=record.sequence_id,
        camera_id=view_debug.view.camera_id,
        frame_id=record.frame_id,
    )
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    camera = load_panoptic_camera_parameters(
        dataset_root,
        sequence_id=record.sequence_id,
        camera_id=view_debug.view.camera_id,
    )
    target_camera_joints = transform_points_world_to_camera(
        target_world_joints,
        rotation=camera.rotation,
        translation=camera.translation,
    )
    panoptic_camera_joints = transform_points_world_to_camera(
        panoptic_world_joints,
        rotation=camera.rotation,
        translation=camera.translation,
    )

    pred_cam_t = as_vector(view_debug.source_payload["pred_cam_t"], 3)

    def build_input_camera_joints(input_transl: np.ndarray) -> np.ndarray:
        return build_smpl_joints(
            smpl_model=smpl_model,
            device=device,
            body_pose=as_vector(view_debug.input_smpl["body_pose"], 69),
            betas=as_vector(view_debug.input_smpl["betas"], 10),
            global_orient=as_vector(view_debug.input_smpl["global_orient"], 3),
            transl=input_transl,
        )

    def make_overlay(input_transl: np.ndarray) -> np.ndarray:
        input_camera_joints = build_input_camera_joints(input_transl)
        overlay = image_bgr.copy()
        draw_skeleton(
            overlay,
            panoptic_camera_joints,
            edges=PANOPTIC_EDGES,
            intrinsics=camera.intrinsics,
            color_bgr=(0, 220, 255),
            radius=3,
            thickness=2,
        )
        draw_skeleton(
            overlay,
            target_camera_joints[:24],
            edges=SMPL_EDGES,
            intrinsics=camera.intrinsics,
            color_bgr=(65, 210, 80),
            radius=3,
            thickness=2,
        )
        draw_skeleton(
            overlay,
            input_camera_joints[:24],
            edges=SMPL_EDGES,
            intrinsics=np.asarray(
                view_debug.source_payload["cam_int"], dtype=np.float32
            ),
            color_bgr=(235, 150, 35),
            radius=3,
            thickness=2,
        )
        return draw_legend(
            overlay,
            lines=[
                ("input SMPL", (235, 150, 35)),
                ("target SMPL", (65, 210, 80)),
                ("raw gt3d", (0, 220, 255)),
            ],
        )

    input_fit_transl = as_vector(view_debug.input_smpl["transl"], 3)
    input_fit_camera_joints = build_input_camera_joints(input_fit_transl)
    raw_gt3d_overlay = make_single_skeleton_overlay(
        image_bgr=image_bgr,
        title="raw gt3d",
        joints=panoptic_camera_joints,
        edges=PANOPTIC_EDGES,
        intrinsics=camera.intrinsics,
        color_bgr=(0, 220, 255),
        label="raw gt3d",
    )
    target_overlay = make_single_skeleton_overlay(
        image_bgr=image_bgr,
        title=f"target SMPL  fit {sample.target_fit_mpjpe * 1000:.0f}mm",
        joints=target_camera_joints[:24],
        edges=SMPL_EDGES,
        intrinsics=camera.intrinsics,
        color_bgr=(65, 210, 80),
        label="target SMPL",
    )
    input_overlay = make_single_skeleton_overlay(
        image_bgr=image_bgr,
        title="input SMPL fitted transl",
        joints=input_fit_camera_joints[:24],
        edges=SMPL_EDGES,
        intrinsics=np.asarray(view_debug.source_payload["cam_int"], dtype=np.float32),
        color_bgr=(235, 150, 35),
        label="input SMPL",
    )
    rgb_panel = add_title(image_bgr, "RGB crop")
    raw_panel = add_title(image_bgr, "RGB crop")
    metric_title = (
        f"{view_debug.view.camera_id}  MPJPE {view_debug.mpjpe * 1000:.0f}mm  "
        f"PA {view_debug.pa_mpjpe * 1000:.0f}mm"
    )
    metric_overlay_panel = add_title(make_overlay(pred_cam_t), metric_title)
    fitted_title = (
        f"fitted transl  2D err {view_debug.fitted_transl_reproj_error_px:.0f}px"
    )
    fitted_overlay_panel = add_title(make_overlay(input_fit_transl), fitted_title)
    panel = hstack_with_padding(
        [raw_panel, metric_overlay_panel, fitted_overlay_panel], pad=12
    )

    separate_overlays = {
        "rgb": rgb_panel,
        "raw_gt3d": raw_gt3d_overlay,
        "target_smpl": target_overlay,
        "input_smpl": input_overlay,
    }

    return (
        panel,
        {
            "camera_id": view_debug.view.camera_id,
            "source_npz_path": str(view_debug.view.npz_path),
            "input_cache_path": str(view_debug.input_cache_path),
            "rgb_path": str(image_path),
            "input_mpjpe_m": view_debug.mpjpe,
            "input_pa_mpjpe_m": view_debug.pa_mpjpe,
            "fitted_transl_reproj_error_px": view_debug.fitted_transl_reproj_error_px,
            "pred_cam_t": pred_cam_t.tolist(),
            "input_fit_transl": input_fit_transl.tolist(),
            "input_fit_transl_minus_pred_cam_t_norm_m": float(
                np.linalg.norm(input_fit_transl - pred_cam_t)
            ),
        },
        separate_overlays,
    )


def make_single_skeleton_overlay(
    *,
    image_bgr: np.ndarray,
    title: str,
    joints: np.ndarray,
    edges: tuple[tuple[int, int], ...],
    intrinsics: np.ndarray,
    color_bgr: tuple[int, int, int],
    label: str,
) -> np.ndarray:
    overlay = image_bgr.copy()
    draw_skeleton(
        overlay,
        joints,
        edges=edges,
        intrinsics=intrinsics,
        color_bgr=color_bgr,
        radius=3,
        thickness=2,
    )
    overlay = draw_legend(overlay, lines=[(label, color_bgr)])
    return add_title(overlay, title)


def write_canonical_plot(
    *,
    sample: SampleDebug,
    output_path: Path,
    smpl_model,
    device: torch.device,
) -> None:
    target_joints = build_smpl_joints(
        smpl_model=smpl_model,
        device=device,
        body_pose=sample.target.body_pose,
        betas=sample.target.betas,
        global_orient=np.zeros(3, dtype=np.float32),
        transl=np.zeros(3, dtype=np.float32),
    )[:24]
    plotted = [("target SMPL", target_joints, "#41b95a")]
    for index, view_debug in enumerate(sample.views):
        input_joints = build_smpl_joints(
            smpl_model=smpl_model,
            device=device,
            body_pose=as_vector(view_debug.input_smpl["body_pose"], 69),
            betas=as_vector(view_debug.input_smpl["betas"], 10),
            global_orient=np.zeros(3, dtype=np.float32),
            transl=np.zeros(3, dtype=np.float32),
        )[:24]
        plotted.append(
            (f"input {index + 1}: {view_debug.view.camera_id}", input_joints, "#eb9623")
        )

    fig = plt.figure(figsize=(8, 7), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    for label, joints, color in plotted:
        plot_smpl_skeleton(ax, joints, label=label, color=color)

    all_points = np.concatenate([joints for _, joints, _ in plotted], axis=0)
    set_equal_3d_axes(ax, all_points)
    ax.view_init(elev=12, azim=-68)
    title = (
        f"{sample.record.sequence_id} frame {sample.record.frame_id}\n"
        f"target fit {sample.target_fit_mpjpe * 1000:.0f}mm, "
        f"input avg MPJPE {sample.input_avg_mpjpe * 1000:.0f}mm, "
        f"PA {sample.input_avg_pa_mpjpe * 1000:.0f}mm, "
        f"fit 2D {sample.fitted_transl_reproj_error_px:.0f}px"
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("-y")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_smpl_skeleton(ax, joints: np.ndarray, *, label: str, color: str) -> None:
    coords = canonical_plot_coords(joints)
    for start, end in SMPL_EDGES:
        segment = coords[[start, end]]
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color=color,
            linewidth=1.8,
            alpha=0.9,
        )
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], color=color, s=12, label=label)


def canonical_plot_coords(joints: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float32)
    root_centered = joints - joints[:1]
    return np.stack(
        [root_centered[:, 0], root_centered[:, 2], -root_centered[:, 1]], axis=1
    )


def set_equal_3d_axes(ax, points: np.ndarray) -> None:
    coords = canonical_plot_coords(points)
    center = 0.5 * (coords.min(axis=0) + coords.max(axis=0))
    radius = max(float(np.max(coords.max(axis=0) - coords.min(axis=0))) * 0.55, 0.5)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def build_summary_image(
    view_panels: list[np.ndarray], canonical_path: Path
) -> np.ndarray:
    canonical = cv2.imread(str(canonical_path), cv2.IMREAD_COLOR)
    if canonical is None:
        raise FileNotFoundError(f"Failed to read canonical plot: {canonical_path}")
    target_width = max(panel.shape[1] for panel in view_panels)
    resized_panels = [resize_to_width(panel, target_width) for panel in view_panels]
    canonical = resize_to_width(canonical, target_width)
    return vstack_with_padding([*resized_panels, canonical], pad=14)


def load_panoptic_gt3d(
    *, dataset_root: Path, sequence_id: str, frame_id: str
) -> np.ndarray:
    array = load_panoptic_gt3d_array(
        dataset_root=dataset_root,
        sequence_id=sequence_id,
        frame_id=frame_id,
    )
    return np.ascontiguousarray(array[:, :3] * 0.01)


def load_panoptic_gt3d_confidence(
    *,
    dataset_root: Path,
    sequence_id: str,
    frame_id: str,
) -> np.ndarray:
    array = load_panoptic_gt3d_array(
        dataset_root=dataset_root,
        sequence_id=sequence_id,
        frame_id=frame_id,
    )
    return np.ascontiguousarray(array[:, 3])


def load_panoptic_gt3d_array(
    *,
    dataset_root: Path,
    sequence_id: str,
    frame_id: str,
) -> np.ndarray:
    path = (
        dataset_root
        / sequence_id
        / "gt3d"
        / f"{int(normalize_frame_id(frame_id)):08d}.npy"
    )
    payload = np.load(path, allow_pickle=True)
    array = np.asarray(payload, dtype=np.float32)
    if array.shape != (19, 4):
        raise ValueError(f"Expected gt3d shape (19, 4), got {array.shape}: {path}")
    return np.ascontiguousarray(array)


def resolve_panoptic_rgb_path(
    *,
    dataset_root: Path,
    sequence_id: str,
    camera_id: str,
    frame_id: str,
) -> Path:
    camera_dir = panoptic_camera_dir_name(camera_id)
    path = (
        dataset_root
        / sequence_id
        / "rgb"
        / camera_dir
        / f"{int(normalize_frame_id(frame_id)):08d}.jpg"
    )
    if not path.exists():
        raise FileNotFoundError(f"Panoptic RGB image does not exist: {path}")
    return path


def panoptic_camera_dir_name(camera_id: str) -> str:
    if not camera_id.startswith("kinect_"):
        raise KeyError(f"Unsupported Panoptic camera id: {camera_id}")
    return f"kinect_{int(camera_id.split('_', maxsplit=1)[1])}"


def build_smpl_joints(
    *,
    smpl_model,
    device: torch.device,
    body_pose: np.ndarray,
    betas: np.ndarray,
    global_orient: np.ndarray,
    transl: np.ndarray,
) -> np.ndarray:
    with torch.no_grad():
        output = smpl_model(
            body_pose=torch.as_tensor(
                body_pose, dtype=torch.float32, device=device
            ).view(1, -1),
            betas=torch.as_tensor(betas, dtype=torch.float32, device=device).view(
                1, -1
            ),
            global_orient=torch.as_tensor(
                global_orient, dtype=torch.float32, device=device
            ).view(1, -1),
            transl=torch.as_tensor(transl, dtype=torch.float32, device=device).view(
                1, -1
            ),
        )
    return output.joints[0].detach().cpu().numpy().astype(np.float32, copy=False)


def to_joint_tensor(joints: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(joints[:24], dtype=np.float32)).unsqueeze(0)


def transform_points_world_to_camera(
    points_world: np.ndarray,
    *,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    points_world = np.asarray(points_world, dtype=np.float32)
    rotation = np.asarray(rotation, dtype=np.float32)
    translation = np.asarray(translation, dtype=np.float32).reshape(1, 3)
    return np.ascontiguousarray((rotation @ points_world.T).T + translation)


def draw_skeleton(
    image_bgr: np.ndarray,
    joints_camera: np.ndarray,
    *,
    edges: tuple[tuple[int, int], ...],
    intrinsics: np.ndarray,
    color_bgr: tuple[int, int, int],
    radius: int,
    thickness: int,
) -> None:
    projected = project_points(joints_camera, intrinsics=intrinsics)
    for start, end in edges:
        if start >= len(projected) or end >= len(projected):
            continue
        p0 = projected[start]
        p1 = projected[end]
        if p0 is None or p1 is None:
            continue
        cv2.line(image_bgr, p0, p1, color_bgr, thickness, cv2.LINE_AA)
    for point in projected:
        if point is None:
            continue
        cv2.circle(image_bgr, point, radius + 1, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image_bgr, point, radius, color_bgr, -1, cv2.LINE_AA)


def project_points(
    joints_camera: np.ndarray,
    *,
    intrinsics: np.ndarray,
) -> list[tuple[int, int] | None]:
    joints_camera = np.asarray(joints_camera, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    projected: list[tuple[int, int] | None] = []
    for joint in joints_camera:
        z = float(joint[2])
        if z <= 1e-5:
            projected.append(None)
            continue
        x = float(intrinsics[0, 0] * joint[0] / z + intrinsics[0, 2])
        y = float(intrinsics[1, 1] * joint[1] / z + intrinsics[1, 2])
        projected.append((int(round(x)), int(round(y))))
    return projected


def draw_legend(
    image_bgr: np.ndarray, *, lines: list[tuple[str, tuple[int, int, int]]]
) -> np.ndarray:
    output = image_bgr.copy()
    x0, y0 = 8, 8
    row_height = 20
    width = 138
    height = 10 + row_height * len(lines)
    cv2.rectangle(output, (x0, y0), (x0 + width, y0 + height), (0, 0, 0), -1)
    cv2.addWeighted(output, 0.72, image_bgr, 0.28, 0, output)
    for index, (label, color) in enumerate(lines):
        y = y0 + 19 + index * row_height
        cv2.circle(output, (x0 + 12, y - 4), 5, color, -1, cv2.LINE_AA)
        cv2.putText(
            output,
            label,
            (x0 + 24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return output


def add_title(image_bgr: np.ndarray, title: str) -> np.ndarray:
    title_height = 30
    output = cv2.copyMakeBorder(
        image_bgr,
        title_height,
        0,
        0,
        0,
        borderType=cv2.BORDER_CONSTANT,
        value=(250, 250, 250),
    )
    cv2.putText(
        output,
        title,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (25, 25, 25),
        1,
        cv2.LINE_AA,
    )
    return output


def hstack_with_padding(images: list[np.ndarray], *, pad: int) -> np.ndarray:
    max_height = max(image.shape[0] for image in images)
    padded = [pad_to_height(image, max_height) for image in images]
    spacer = np.full((max_height, pad, 3), 255, dtype=np.uint8)
    parts: list[np.ndarray] = []
    for index, image in enumerate(padded):
        if index:
            parts.append(spacer)
        parts.append(image)
    return np.hstack(parts)


def vstack_with_padding(images: list[np.ndarray], *, pad: int) -> np.ndarray:
    max_width = max(image.shape[1] for image in images)
    padded = [pad_to_width(image, max_width) for image in images]
    spacer = np.full((pad, max_width, 3), 255, dtype=np.uint8)
    parts: list[np.ndarray] = []
    for index, image in enumerate(padded):
        if index:
            parts.append(spacer)
        parts.append(image)
    return np.vstack(parts)


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    if image.shape[1] == width:
        return image
    scale = width / float(image.shape[1])
    height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def pad_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    bottom = height - image.shape[0]
    return cv2.copyMakeBorder(
        image,
        0,
        bottom,
        0,
        0,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )


def pad_to_width(image: np.ndarray, width: int) -> np.ndarray:
    if image.shape[1] == width:
        return image
    right = width - image.shape[1]
    return cv2.copyMakeBorder(
        image,
        0,
        0,
        0,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    resolved = Path(path).resolve()
    cached = _NPZ_CACHE.get(resolved)
    if cached is not None:
        return cached
    with np.load(resolved, allow_pickle=False) as payload:
        cached = {key: np.asarray(payload[key]) for key in payload.files}
    _NPZ_CACHE[resolved] = cached
    return cached


def as_vector(value: np.ndarray, expected_dim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    array = array.reshape(-1)
    if array.shape[0] != expected_dim:
        raise ValueError(f"Expected vector dim {expected_dim}, got {array.shape}")
    return np.ascontiguousarray(array.astype(np.float32, copy=False))


def normalize_frame_id(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.generic):
        value = value.item()
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit():
        return str(int(text))
    return text


if __name__ == "__main__":
    main()
