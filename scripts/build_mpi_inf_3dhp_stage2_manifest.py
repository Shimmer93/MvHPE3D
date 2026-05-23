#!/usr/bin/env python
"""Build a Stage 2 manifest for MPI-INF-3DHP SAM3DBody exports."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mvhpe3d.data.mpi_inf_3dhp import (
    MPII3D_HEATFORMER_CAMERA_IDS,
    MPII3D_HEATFORMER_TRAIN_SAMPLING,
    MPII3D_HEATFORMER_VAL_SAMPLING,
    MPII3D_SEQUENCES,
    MPII3D_TRAIN_SUBJECTS,
    MPII3D_VAL_SUBJECTS,
    annotation_frame_count,
    camera_id_to_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="/dysData/shimmer/datasets/mpi_inf_3dhp")
    parser.add_argument("--sam3dbody-root", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--train-subjects", nargs="*", default=list(MPII3D_TRAIN_SUBJECTS))
    parser.add_argument("--val-subjects", nargs="*", default=list(MPII3D_VAL_SUBJECTS))
    parser.add_argument("--sequences", nargs="*", default=list(MPII3D_SEQUENCES))
    parser.add_argument("--cameras", nargs="*", default=list(MPII3D_HEATFORMER_CAMERA_IDS))
    parser.add_argument("--train-sampling", type=int, default=MPII3D_HEATFORMER_TRAIN_SAMPLING)
    parser.add_argument("--val-sampling", type=int, default=MPII3D_HEATFORMER_VAL_SAMPLING)
    parser.add_argument(
        "--min-views",
        type=int,
        default=0,
        help=(
            "Minimum valid SAM3DBody views to keep in the manifest. "
            "Default 0 preserves the full HeatFormer sampling grid; training "
            "filters by data.num_views at dataset-loading time."
        ),
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--absolute-paths", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    sam3dbody_root = Path(args.sam3dbody_root).resolve()
    output_path = Path(args.output_path).resolve()
    samples = []
    report = {}
    for split_name, subjects, sampling in (
        ("train", args.train_subjects, args.train_sampling),
        ("val", args.val_subjects, args.val_sampling),
    ):
        for subject_id in subjects:
            for sequence_name in args.sequences:
                sequence_id = f"{subject_id}_{sequence_name}"
                sequence_samples, sequence_report = build_sequence_samples(
                    dataset_root=dataset_root,
                    sam3dbody_root=sam3dbody_root,
                    manifest_dir=output_path.parent,
                    sequence_id=sequence_id,
                    split_name=split_name,
                    cameras=args.cameras,
                    sampling=sampling,
                    min_views=args.min_views,
                    max_frames=args.max_frames,
                    use_absolute_paths=args.absolute_paths,
                )
                samples.extend(sequence_samples)
                report[sequence_id] = sequence_report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"samples": samples}, indent=2), encoding="utf-8")
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {output_path}")
    print(f"Wrote report: {report_path}")
    print(f"Samples: {len(samples)}")


def build_sequence_samples(
    *,
    dataset_root: Path,
    sam3dbody_root: Path,
    manifest_dir: Path,
    sequence_id: str,
    split_name: str,
    cameras: list[str],
    sampling: int,
    min_views: int,
    max_frames: int | None,
    use_absolute_paths: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if min_views < 0:
        raise ValueError(f"min_views must be >= 0, got {min_views}")
    frame_count = annotation_frame_count(dataset_root, sequence_id=sequence_id)
    frame_ids = list(range(0, frame_count, sampling))
    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]
    kept = []
    skipped_below_min_views = 0
    skipped_invalid = 0
    available_view_counts: dict[int, int] = {}
    for frame_id in frame_ids:
        views = []
        for camera_id in cameras:
            npz_path = resolve_prediction_path(
                sam3dbody_root=sam3dbody_root,
                sequence_id=sequence_id,
                camera_id=camera_id,
                frame_id=frame_id,
            )
            if npz_path is None:
                continue
            if not is_valid_single_person_prediction(npz_path):
                skipped_invalid += 1
                continue
            views.append(
                {
                    "camera_id": f"video_{camera_id_to_index(camera_id)}",
                    "npz_path": stringify_path(
                        npz_path,
                        manifest_dir=manifest_dir,
                        use_absolute_paths=use_absolute_paths,
                    ),
                }
            )
        available_view_counts[len(views)] = available_view_counts.get(len(views), 0) + 1
        if len(views) < min_views:
            skipped_below_min_views += 1
            continue
        kept.append(
            {
                "sample_id": f"{sequence_id}_frame{frame_id:08d}",
                "sequence_id": sequence_id,
                "frame_id": f"{frame_id:08d}",
                "split": split_name,
                "subject_id": sequence_id.split("_", maxsplit=1)[0],
                "action_id": sequence_id.split("_", maxsplit=1)[1],
                "views": sorted(views, key=lambda item: item["camera_id"]),
            }
        )
    report = {
        "split": split_name,
        "sampling": sampling,
        "frames": len(frame_ids),
        "samples": len(kept),
        "min_views": min_views,
        "skipped_frames_below_min_views": skipped_below_min_views,
        "skipped_frames_missing_views": skipped_below_min_views,
        "skipped_invalid_view_files": skipped_invalid,
        "available_view_count_distribution": {
            str(view_count): available_view_counts.get(view_count, 0)
            for view_count in range(0, len(cameras) + 1)
        },
        "cameras": [f"video_{camera_id_to_index(camera_id)}" for camera_id in cameras],
    }
    return kept, report


def resolve_prediction_path(
    *,
    sam3dbody_root: Path,
    sequence_id: str,
    camera_id: str,
    frame_id: int,
) -> Path | None:
    normalized_camera = f"video_{camera_id_to_index(camera_id)}"
    candidates = (
        sam3dbody_root / sequence_id / normalized_camera / f"frame_{frame_id:08d}.npz",
        sam3dbody_root / sequence_id / normalized_camera / f"{frame_id:08d}.npz",
        sam3dbody_root / sequence_id / normalized_camera / f"frame_{frame_id:06d}.npz",
    )
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def is_valid_single_person_prediction(npz_path: Path) -> bool:
    with np.load(npz_path, allow_pickle=False) as payload:
        for key, expected_dim in (
            ("mhr_model_params", 204),
            ("shape_params", 45),
            ("pred_cam_t", 3),
        ):
            if key not in payload:
                return False
            value = np.asarray(payload[key])
            if value.ndim == 1 and value.shape[0] == expected_dim:
                continue
            if value.ndim == 2 and value.shape == (1, expected_dim):
                continue
            return False
        if "cam_int" not in payload or np.asarray(payload["cam_int"]).shape != (3, 3):
            return False
        if "image_size" not in payload or np.asarray(payload["image_size"]).shape != (2,):
            return False
    return True


def stringify_path(path: Path, *, manifest_dir: Path, use_absolute_paths: bool) -> str:
    resolved = path.resolve()
    if use_absolute_paths:
        return str(resolved)
    return os.path.relpath(resolved, manifest_dir)


if __name__ == "__main__":
    main()
