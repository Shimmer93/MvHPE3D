#!/usr/bin/env python
"""Extract MPI-INF-3DHP frames on the HeatFormer sampling grid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

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
    parser.add_argument("--output-root", default="data/mpi_inf_3dhp/frames")
    parser.add_argument("--train-subjects", nargs="*", default=list(MPII3D_TRAIN_SUBJECTS))
    parser.add_argument("--val-subjects", nargs="*", default=list(MPII3D_VAL_SUBJECTS))
    parser.add_argument("--sequences", nargs="*", default=list(MPII3D_SEQUENCES))
    parser.add_argument("--cameras", nargs="*", default=list(MPII3D_HEATFORMER_CAMERA_IDS))
    parser.add_argument("--train-sampling", type=int, default=MPII3D_HEATFORMER_TRAIN_SAMPLING)
    parser.add_argument("--val-sampling", type=int, default=MPII3D_HEATFORMER_VAL_SAMPLING)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--image-ext", default=".jpg")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report = {}
    for split_name, subjects, sampling in (
        ("train", args.train_subjects, args.train_sampling),
        ("val", args.val_subjects, args.val_sampling),
    ):
        for subject_id in subjects:
            for sequence_name in args.sequences:
                sequence_id = f"{subject_id}_{sequence_name}"
                count = extract_sequence(
                    dataset_root=dataset_root,
                    output_root=output_root,
                    sequence_id=sequence_id,
                    cameras=args.cameras,
                    sampling=sampling,
                    max_frames=args.max_frames,
                    image_ext=args.image_ext,
                    overwrite=args.overwrite,
                )
                report[sequence_id] = {
                    "split": split_name,
                    "sampling": sampling,
                    "frames_per_camera": count,
                    "cameras": list(args.cameras),
                }
                print(f"{sequence_id}: extracted {count} frames per camera")
    report_path = output_root / "extract_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote report: {report_path}")


def extract_sequence(
    *,
    dataset_root: Path,
    output_root: Path,
    sequence_id: str,
    cameras: list[str],
    sampling: int,
    max_frames: int | None,
    image_ext: str,
    overwrite: bool,
) -> int:
    if sampling < 1:
        raise ValueError(f"sampling must be >= 1, got {sampling}")
    subject_id, sequence_name = sequence_id.split("_", maxsplit=1)
    sequence_dir = dataset_root / subject_id / sequence_name
    frame_count = annotation_frame_count(dataset_root, sequence_id=sequence_id)
    selected_frame_ids = list(range(0, frame_count, sampling))
    if max_frames is not None:
        selected_frame_ids = selected_frame_ids[:max_frames]
    for camera_id in cameras:
        video_path = sequence_dir / "imageSequence" / f"video_{camera_id_to_index(camera_id)}.avi"
        extract_camera_frames(
            video_path=video_path,
            output_dir=output_root / sequence_id / f"video_{camera_id_to_index(camera_id)}",
            frame_ids=selected_frame_ids,
            image_ext=image_ext,
            overwrite=overwrite,
        )
    return len(selected_frame_ids)


def extract_camera_frames(
    *,
    video_path: Path,
    output_dir: Path,
    frame_ids: list[int],
    image_ext: str,
    overwrite: bool,
) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        for frame_id in frame_ids:
            output_path = output_dir / f"frame_{frame_id:08d}{image_ext}"
            if output_path.exists() and not overwrite:
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Failed to read frame {frame_id} from {video_path}")
            if not cv2.imwrite(str(output_path), frame):
                raise RuntimeError(f"Failed to write frame image: {output_path}")
    finally:
        capture.release()


if __name__ == "__main__":
    main()
