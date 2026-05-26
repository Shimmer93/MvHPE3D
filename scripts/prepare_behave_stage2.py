#!/usr/bin/env python
"""Prepare BEHAVE data for Stage 2 training.

This script does not modify the raw BEHAVE tree. It creates a reproducible
workspace containing image links, exported GT SMPL targets, and a Stage 2
manifest that can be rebuilt after SAM3DBody compact outputs are generated.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np


DEFAULT_CAMERAS = ("k0", "k1", "k2", "k3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="/dysData/shimmer/datasets/behave")
    parser.add_argument("--output-root", default="data/behave")
    parser.add_argument("--manifest-path", default="data/behave/behave_stage2_manifest.json")
    parser.add_argument("--sam3dbody-root", default="data/behave/sam3dbody")
    parser.add_argument(
        "--heatformer-db-root",
        default="/dysData/shimmer/datasets/behave/preprocessed_data_z",
        help="Directory containing BEHAVE_train_db.pt and BEHAVE_valid_db.pt.",
    )
    parser.add_argument(
        "--protocol",
        choices=("heatformer", "folder"),
        default="heatformer",
        help="Use HeatFormer DB samples or scan BEHAVE folders directly.",
    )
    parser.add_argument("--cameras", nargs="*", default=list(DEFAULT_CAMERAS))
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help="Optional subset of BEHAVE sequence directory names.",
    )
    parser.add_argument(
        "--frame-source",
        choices=("images", "smpl"),
        default="images",
        help="Use sparse RGB timestamp folders or dense smpl_fit_all frame_times.",
    )
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames-per-sequence", type=int, default=None)
    parser.add_argument("--min-views", type=int, default=0)
    parser.add_argument("--val-subjects", nargs="*", default=["Sub08"])
    parser.add_argument("--test-subjects", nargs="*", default=["Sub08"])
    parser.add_argument(
        "--split-json",
        default=None,
        help=(
            "Optional exact split file. Supported keys: train_sequences, "
            "val_sequences, test_sequences, train_samples, val_samples, test_samples."
        ),
    )
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--skip-frame-links", action="store_true")
    parser.add_argument("--skip-gt-smpl", action="store_true")
    parser.add_argument("--absolute-paths", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_step < 1:
        raise ValueError(f"--frame-step must be >= 1, got {args.frame_step}")
    if args.min_views < 0:
        raise ValueError(f"--min-views must be >= 0, got {args.min_views}")

    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    manifest_path = Path(args.manifest_path).resolve()
    sam3dbody_root = Path(args.sam3dbody_root).resolve()
    heatformer_db_root = Path(args.heatformer_db_root).resolve()
    frames_root = output_root / "frames"
    gt_smpl_root = output_root / "gt_smpl"
    split_policy = load_split_policy(args.split_json)

    samples: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "sam3dbody_root": str(sam3dbody_root),
        "heatformer_db_root": str(heatformer_db_root),
        "protocol": args.protocol,
        "cameras": list(args.cameras),
        "frame_source": args.frame_source,
        "frame_step": args.frame_step,
        "min_views": args.min_views,
        "split_source": (
            "split_json"
            if args.split_json
            else ("heatformer_db" if args.protocol == "heatformer" else "subject_holdout_fallback")
        ),
        "sequences": {},
    }

    if args.protocol == "heatformer":
        samples, sequence_report = prepare_heatformer_protocol(
            dataset_root=dataset_root,
            heatformer_db_root=heatformer_db_root,
            frames_root=frames_root,
            gt_smpl_root=gt_smpl_root,
            sam3dbody_root=sam3dbody_root,
            manifest_dir=manifest_path.parent,
            cameras=list(args.cameras),
            sequences=args.sequences,
            max_frames_per_split=args.max_frames_per_sequence,
            min_views=args.min_views,
            make_frame_links=not args.skip_frame_links,
            export_gt_smpl=not args.skip_gt_smpl,
            copy_images=args.copy_images,
            use_absolute_paths=args.absolute_paths,
        )
        report["sequences"] = sequence_report
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({"samples": samples}, indent=2), encoding="utf-8")
        report_path = manifest_path.with_suffix(".report.json")
        finalize_report(report=report, samples=samples)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote manifest: {manifest_path}")
        print(f"Wrote report: {report_path}")
        print(f"Samples: {len(samples)}")
        return

    sequence_dirs = [
        path
        for path in sorted(dataset_root.iterdir())
        if path.is_dir() and path.name.startswith("Date")
    ]
    if args.sequences is not None:
        requested_sequences = {str(sequence) for sequence in args.sequences}
        sequence_dirs = [path for path in sequence_dirs if path.name in requested_sequences]
        missing_sequences = sorted(requested_sequences - {path.name for path in sequence_dirs})
        if missing_sequences:
            raise FileNotFoundError(
                "Requested BEHAVE sequences were not found: "
                + ", ".join(missing_sequences)
            )
    for sequence_dir in sequence_dirs:
        sequence_id = sequence_dir.name
        sequence_meta = parse_sequence_id(sequence_id)
        split = assign_split(
            sequence_id=sequence_id,
            subject_id=sequence_meta["subject_id"],
            split_policy=split_policy,
            val_subjects=set(args.val_subjects),
            test_subjects=set(args.test_subjects),
        )
        if split is None:
            continue

        sequence_report = prepare_sequence(
            sequence_dir=sequence_dir,
            sequence_meta=sequence_meta,
            split=split,
            cameras=list(args.cameras),
            frames_root=frames_root,
            gt_smpl_root=gt_smpl_root,
            sam3dbody_root=sam3dbody_root,
            manifest_dir=manifest_path.parent,
            frame_step=args.frame_step,
            frame_source=args.frame_source,
            max_frames=args.max_frames_per_sequence,
            min_views=args.min_views,
            make_frame_links=not args.skip_frame_links,
            export_gt_smpl=not args.skip_gt_smpl,
            copy_images=args.copy_images,
            use_absolute_paths=args.absolute_paths,
            split_policy=split_policy,
        )
        samples.extend(sequence_report.pop("samples"))
        report["sequences"][sequence_id] = sequence_report

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"samples": samples}, indent=2), encoding="utf-8")
    report_path = manifest_path.with_suffix(".report.json")
    finalize_report(report=report, samples=samples)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote report: {report_path}")
    print(f"Samples: {len(samples)}")


def parse_sequence_id(sequence_id: str) -> dict[str, str]:
    parts = sequence_id.split("_")
    if len(parts) < 3:
        raise ValueError(
            f"Expected BEHAVE sequence id like Date01_Sub01_action, got {sequence_id!r}"
        )
    return {
        "date_id": parts[0],
        "subject_id": parts[1],
        "action_id": "_".join(parts[2:]),
    }


def prepare_heatformer_protocol(
    *,
    dataset_root: Path,
    heatformer_db_root: Path,
    frames_root: Path,
    gt_smpl_root: Path,
    sam3dbody_root: Path,
    manifest_dir: Path,
    cameras: list[str],
    sequences: list[str] | None,
    max_frames_per_split: int | None,
    min_views: int,
    make_frame_links: bool,
    export_gt_smpl: bool,
    copy_images: bool,
    use_absolute_paths: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requested_sequences = set(sequences) if sequences is not None else None
    all_samples: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    db_records: list[dict[str, Any]] = []

    for db_split, manifest_split, db_filename in (
        ("train", "train", "BEHAVE_train_db.pt"),
        ("valid", "val", "BEHAVE_valid_db.pt"),
    ):
        db_path = heatformer_db_root / db_filename
        if not db_path.exists():
            raise FileNotFoundError(f"HeatFormer BEHAVE DB does not exist: {db_path}")
        db = joblib.load(db_path)
        split_kept = 0
        for db_index, item in enumerate(db):
            sequence_id, frame_time = parse_heatformer_img_name(str(item["img_name"]))
            if requested_sequences is not None and sequence_id not in requested_sequences:
                continue
            if max_frames_per_split is not None and split_kept >= max_frames_per_split:
                continue
            sequence_dir = dataset_root / sequence_id
            if not sequence_dir.exists():
                raise FileNotFoundError(f"HeatFormer DB sequence does not exist: {sequence_dir}")
            sequence_meta = parse_sequence_id(sequence_id)
            db_records.append(
                {
                    "db_split": db_split,
                    "manifest_split": manifest_split,
                    "db_index": db_index,
                    "item": item,
                    "sequence_id": sequence_id,
                    "frame_time": frame_time,
                    "sequence_dir": sequence_dir,
                    "sequence_meta": sequence_meta,
                }
            )
            split_kept += 1

    frame_index_by_key = build_sparse_frame_index(db_records)
    if export_gt_smpl:
        export_heatformer_gt_smpl(
            records=db_records,
            frame_index_by_key=frame_index_by_key,
            output_dir=gt_smpl_root,
        )

    for record in db_records:
        sequence_id = record["sequence_id"]
        frame_time = record["frame_time"]
        sequence_dir = record["sequence_dir"]
        sequence_meta = record["sequence_meta"]
        manifest_split = record["manifest_split"]
        db_split = record["db_split"]
        db_index = record["db_index"]
        frame_index = frame_index_by_key[(sequence_id, frame_time)]

        frame_report = report.setdefault(
            sequence_id,
                {
                    "sample_split_counts": {},
                    "view_count_distribution": {},
                    "missing_images": 0,
                    "linked_images": 0,
                    "copied_images": 0,
                    "missing_sam_views": 0,
                    "invalid_sam_views": 0,
                    "skipped_below_min_views": 0,
                    "samples_kept": 0,
                },
            )
        views = []
        for camera_id in cameras:
            source_image = sequence_dir / frame_time / f"{camera_id}.color.jpg"
            linked_image = frames_root / sequence_id / camera_id / f"frame_{frame_index:08d}.jpg"
            if not source_image.exists():
                frame_report["missing_images"] += 1
                continue
            if make_frame_links:
                if copy_images:
                    frame_report["copied_images"] += copy_file(source_image, linked_image)
                else:
                    frame_report["linked_images"] += link_file(source_image, linked_image)
            sam_npz = resolve_sam3dbody_npz(
                sam3dbody_root=sam3dbody_root,
                sequence_id=sequence_id,
                camera_id=camera_id,
                frame_index=frame_index,
                frame_time=frame_time,
            )
            if sam_npz is None:
                frame_report["missing_sam_views"] += 1
                continue
            if not is_valid_single_person_prediction(sam_npz):
                frame_report["invalid_sam_views"] += 1
                continue
            views.append(
                {
                    "camera_id": camera_id,
                    "npz_path": stringify_path(
                        sam_npz,
                        manifest_dir=manifest_dir,
                        use_absolute_paths=use_absolute_paths,
                    ),
                }
            )

        view_count = len(views)
        view_distribution = frame_report["view_count_distribution"]
        view_distribution[str(view_count)] = view_distribution.get(str(view_count), 0) + 1
        if view_count < min_views:
            frame_report["skipped_below_min_views"] += 1
            continue

        sample_id = f"{sequence_id}_frame{frame_index:08d}"
        frame_report["samples_kept"] += 1
        split_counts = frame_report["sample_split_counts"]
        split_counts[manifest_split] = split_counts.get(manifest_split, 0) + 1
        all_samples.append(
            {
                "sample_id": sample_id,
                "sequence_id": sequence_id,
                "frame_id": f"{frame_index:08d}",
                "frame_time": frame_time,
                "split": manifest_split,
                "subject_id": sequence_meta["subject_id"],
                "action_id": sequence_meta["action_id"],
                "target_path": stringify_path(
                    gt_smpl_root / f"{sequence_id}_smpl_params.npz",
                    manifest_dir=manifest_dir,
                    use_absolute_paths=use_absolute_paths,
                ),
                "heatformer_db_split": db_split,
                "heatformer_db_index": db_index,
                "views": sorted(views, key=lambda view: view["camera_id"]),
            }
        )

    return all_samples, report


def build_sparse_frame_index(records: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    """Assign compact per-sequence frame indices in HeatFormer DB order."""
    frame_times_by_sequence: dict[str, dict[str, int]] = {}
    for record in records:
        sequence_id = str(record["sequence_id"])
        frame_time = str(record["frame_time"])
        sequence_frames = frame_times_by_sequence.setdefault(sequence_id, {})
        if frame_time not in sequence_frames:
            sequence_frames[frame_time] = len(sequence_frames)
    return {
        (sequence_id, frame_time): frame_index
        for sequence_id, sequence_frames in frame_times_by_sequence.items()
        for frame_time, frame_index in sequence_frames.items()
    }


def export_heatformer_gt_smpl(
    *,
    records: list[dict[str, Any]],
    frame_index_by_key: dict[tuple[str, str], int],
    output_dir: Path,
) -> None:
    """Export sparse sequence-level SMPL arrays from HeatFormer DB entries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records_by_sequence: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        records_by_sequence.setdefault(str(record["sequence_id"]), []).append(record)

    for sequence_id, sequence_records in records_by_sequence.items():
        num_frames = (
            max(
                frame_index_by_key[(sequence_id, str(record["frame_time"]))]
                for record in sequence_records
            )
            + 1
        )
        global_orient = np.zeros((num_frames, 3), dtype=np.float32)
        body_pose = np.zeros((num_frames, 69), dtype=np.float32)
        betas = np.zeros((num_frames, 10), dtype=np.float32)
        transl = np.zeros((num_frames, 3), dtype=np.float32)
        valid_mask = np.zeros((num_frames,), dtype=bool)
        frame_times = np.empty((num_frames,), dtype="<U32")
        seen_indices: set[int] = set()

        for record in sequence_records:
            frame_time = str(record["frame_time"])
            frame_index = frame_index_by_key[(sequence_id, frame_time)]
            if frame_index in seen_indices:
                continue
            item = record["item"]
            pose = to_numpy(item["pose"], dtype=np.float32).reshape(-1)
            if pose.shape[0] < 72:
                raise ValueError(
                    f"Expected HeatFormer pose with at least 72 values for "
                    f"{sequence_id}/{frame_time}, got shape {pose.shape}"
                )
            beta = to_numpy(item["betas"], dtype=np.float32).reshape(-1)
            if beta.shape[0] < 10:
                raise ValueError(
                    f"Expected HeatFormer betas with at least 10 values for "
                    f"{sequence_id}/{frame_time}, got shape {beta.shape}"
                )
            trans = to_numpy(item["trans"], dtype=np.float32).reshape(-1)
            if trans.shape[0] < 3:
                raise ValueError(
                    f"Expected HeatFormer trans with at least 3 values for "
                    f"{sequence_id}/{frame_time}, got shape {trans.shape}"
                )
            global_orient[frame_index] = pose[:3]
            body_pose[frame_index] = pose[3:72]
            betas[frame_index] = beta[:10]
            transl[frame_index] = trans[:3]
            valid_mask[frame_index] = True
            frame_times[frame_index] = frame_time
            seen_indices.add(frame_index)

        np.savez_compressed(
            output_dir / f"{sequence_id}_smpl_params.npz",
            global_orient=np.ascontiguousarray(global_orient),
            body_pose=np.ascontiguousarray(body_pose),
            betas=np.ascontiguousarray(betas),
            transl=np.ascontiguousarray(transl),
            valid_mask=valid_mask,
            frame_times=frame_times,
            gender=np.asarray("neutral"),
        )


def to_numpy(value: Any, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def parse_heatformer_img_name(img_name: str) -> tuple[str, str]:
    parts = Path(img_name).parts
    try:
        sequences_index = parts.index("sequences")
    except ValueError as exc:
        raise ValueError(f"Unexpected HeatFormer BEHAVE img_name: {img_name!r}") from exc
    try:
        return parts[sequences_index + 1], parts[sequences_index + 2]
    except IndexError as exc:
        raise ValueError(f"Unexpected HeatFormer BEHAVE img_name: {img_name!r}") from exc


def dense_frame_index(sequence_dir: Path, frame_time: str) -> int:
    smpl_path = sequence_dir / "smpl_fit_all.npz"
    if not smpl_path.exists():
        raise FileNotFoundError(f"Missing BEHAVE SMPL file: {smpl_path}")
    with np.load(smpl_path, allow_pickle=False) as payload:
        frame_times = [str(item) for item in np.asarray(payload["frame_times"])]
    try:
        return frame_times.index(frame_time)
    except ValueError as exc:
        raise KeyError(
            f"Frame time {frame_time!r} was not found in {smpl_path}"
        ) from exc


def load_split_policy(path: str | None) -> dict[str, set[str]]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    policy: dict[str, set[str]] = {}
    for key in (
        "train_sequences",
        "val_sequences",
        "test_sequences",
        "train_samples",
        "val_samples",
        "test_samples",
    ):
        policy[key] = {str(item) for item in payload.get(key, [])}
    return policy


def assign_split(
    *,
    sequence_id: str,
    subject_id: str,
    split_policy: dict[str, set[str]],
    val_subjects: set[str],
    test_subjects: set[str],
) -> str | None:
    if split_policy:
        if sequence_id in split_policy.get("train_sequences", set()):
            return "train"
        if sequence_id in split_policy.get("val_sequences", set()):
            return "val"
        if sequence_id in split_policy.get("test_sequences", set()):
            return "test"
        has_sequence_lists = any(
            split_policy.get(key)
            for key in ("train_sequences", "val_sequences", "test_sequences")
        )
        if has_sequence_lists:
            return None

    if subject_id in test_subjects:
        return "test"
    if subject_id in val_subjects:
        return "val"
    return "train"


def prepare_sequence(
    *,
    sequence_dir: Path,
    sequence_meta: dict[str, str],
    split: str,
    cameras: list[str],
    frames_root: Path,
    gt_smpl_root: Path,
    sam3dbody_root: Path,
    manifest_dir: Path,
    frame_step: int,
    frame_source: str,
    max_frames: int | None,
    min_views: int,
    make_frame_links: bool,
    export_gt_smpl: bool,
    copy_images: bool,
    use_absolute_paths: bool,
    split_policy: dict[str, set[str]],
) -> dict[str, Any]:
    sequence_id = sequence_dir.name
    smpl_path = sequence_dir / "smpl_fit_all.npz"
    if not smpl_path.exists():
        raise FileNotFoundError(f"Missing BEHAVE SMPL file: {smpl_path}")

    smpl_payload = np.load(smpl_path, allow_pickle=False)
    frame_times = [str(item) for item in np.asarray(smpl_payload["frame_times"])]
    selected_indices = select_frame_indices(
        sequence_dir=sequence_dir,
        frame_times=frame_times,
        cameras=cameras,
        frame_source=frame_source,
        frame_step=frame_step,
    )
    if max_frames is not None:
        selected_indices = selected_indices[:max_frames]

    if export_gt_smpl:
        export_sequence_gt_smpl(
            smpl_payload=smpl_payload,
            output_path=gt_smpl_root / f"{sequence_id}_smpl_params.npz",
        )

    samples: list[dict[str, Any]] = []
    view_count_distribution: Counter[int] = Counter()
    missing_images = 0
    linked_images = 0
    copied_images = 0
    missing_sam_views = 0
    invalid_sam_views = 0
    skipped_below_min_views = 0
    sample_split_counts: Counter[str] = Counter()

    for frame_index in selected_indices:
        frame_time = frame_times[frame_index]
        frame_dir = sequence_dir / frame_time
        sample_id = f"{sequence_id}_frame{frame_index:08d}"
        sample_split = split_for_sample(sample_id, split, split_policy)
        if sample_split is None:
            continue

        views = []
        for camera_id in cameras:
            source_image = frame_dir / f"{camera_id}.color.jpg"
            linked_image = frames_root / sequence_id / camera_id / f"frame_{frame_index:08d}.jpg"
            if not source_image.exists():
                missing_images += 1
                continue
            if make_frame_links:
                if copy_images:
                    copied_images += copy_file(source_image, linked_image)
                else:
                    linked_images += link_file(source_image, linked_image)

            sam_npz = resolve_sam3dbody_npz(
                sam3dbody_root=sam3dbody_root,
                sequence_id=sequence_id,
                camera_id=camera_id,
                frame_index=frame_index,
                frame_time=frame_time,
            )
            if sam_npz is None:
                missing_sam_views += 1
                continue
            if not is_valid_single_person_prediction(sam_npz):
                invalid_sam_views += 1
                continue
            views.append(
                {
                    "camera_id": camera_id,
                    "npz_path": stringify_path(
                        sam_npz,
                        manifest_dir=manifest_dir,
                        use_absolute_paths=use_absolute_paths,
                    ),
                }
            )

        view_count_distribution[len(views)] += 1
        if len(views) < min_views:
            skipped_below_min_views += 1
            continue

        sample_split_counts[sample_split] += 1
        samples.append(
            {
                "sample_id": sample_id,
                "sequence_id": sequence_id,
                "frame_id": f"{frame_index:08d}",
                "frame_time": frame_time,
                "split": sample_split,
                "subject_id": sequence_meta["subject_id"],
                "action_id": sequence_meta["action_id"],
                "target_path": stringify_path(
                    gt_smpl_root / f"{sequence_id}_smpl_params.npz",
                    manifest_dir=manifest_dir,
                    use_absolute_paths=use_absolute_paths,
                ),
                "views": sorted(views, key=lambda item: item["camera_id"]),
            }
        )

    return {
        "samples": samples,
        "split": split,
        "frame_source": frame_source,
        "frames_total": len(frame_times),
        "frames_selected": len(selected_indices),
        "samples_kept": len(samples),
        "sample_split_counts": dict(sample_split_counts),
        "view_count_distribution": {
            str(key): value for key, value in sorted(view_count_distribution.items())
        },
        "missing_images": missing_images,
        "linked_images": linked_images,
        "copied_images": copied_images,
        "missing_sam_views": missing_sam_views,
        "invalid_sam_views": invalid_sam_views,
        "skipped_below_min_views": skipped_below_min_views,
    }


def select_frame_indices(
    *,
    sequence_dir: Path,
    frame_times: list[str],
    cameras: list[str],
    frame_source: str,
    frame_step: int,
) -> list[int]:
    if frame_source == "smpl":
        return list(range(0, len(frame_times), frame_step))
    if frame_source != "images":
        raise ValueError(f"Unsupported frame_source: {frame_source}")

    index_by_time = {frame_time: index for index, frame_time in enumerate(frame_times)}
    image_frame_times = []
    for frame_dir in sorted(sequence_dir.iterdir()):
        if not frame_dir.is_dir() or frame_dir.name not in index_by_time:
            continue
        if any((frame_dir / f"{camera_id}.color.jpg").exists() for camera_id in cameras):
            image_frame_times.append(frame_dir.name)
    return [index_by_time[frame_time] for frame_time in image_frame_times[::frame_step]]


def split_for_sample(
    sample_id: str,
    default_split: str,
    split_policy: dict[str, set[str]],
) -> str | None:
    if not split_policy:
        return default_split
    if sample_id in split_policy.get("train_samples", set()):
        return "train"
    if sample_id in split_policy.get("val_samples", set()):
        return "val"
    if sample_id in split_policy.get("test_samples", set()):
        return "test"
    has_sample_lists = any(
        split_policy.get(key)
        for key in ("train_samples", "val_samples", "test_samples")
    )
    return None if has_sample_lists else default_split


def export_sequence_gt_smpl(*, smpl_payload: np.lib.npyio.NpzFile, output_path: Path) -> None:
    poses = np.asarray(smpl_payload["poses"], dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] < 72:
        raise ValueError(f"Expected poses with shape [T, >=72], got {poses.shape}")
    betas = np.asarray(smpl_payload["betas"], dtype=np.float32)
    transl = np.asarray(smpl_payload["trans"], dtype=np.float32)
    frame_times = np.asarray(smpl_payload["frame_times"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        global_orient=np.ascontiguousarray(poses[:, :3]),
        body_pose=np.ascontiguousarray(poses[:, 3:72]),
        betas=np.ascontiguousarray(betas),
        transl=np.ascontiguousarray(transl),
        valid_mask=np.ones((poses.shape[0],), dtype=bool),
        frame_times=frame_times,
        gender=np.asarray(smpl_payload["gender"]) if "gender" in smpl_payload else np.asarray("neutral"),
    )


def resolve_sam3dbody_npz(
    *,
    sam3dbody_root: Path,
    sequence_id: str,
    camera_id: str,
    frame_index: int,
    frame_time: str,
) -> Path | None:
    folder = sam3dbody_root / sequence_id / camera_id
    candidates = (
        folder / f"frame_{frame_index:08d}.npz",
        folder / f"{frame_time}.npz",
        folder / f"{camera_id}.color.npz",
    )
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def is_valid_single_person_prediction(npz_path: Path) -> bool:
    try:
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
    except Exception:
        return False
    return True


def link_file(source: Path, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    source = source.resolve()
    if target.is_symlink():
        current_target = Path(os.readlink(target))
        if not current_target.is_absolute():
            current_target = (target.parent / current_target).resolve()
        if current_target == source:
            return 0
        target.unlink()
    elif target.exists():
        return 0
    target.symlink_to(source)
    return 1


def copy_file(source: Path, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return 0
    shutil.copy2(source, target)
    return 1


def stringify_path(path: Path, *, manifest_dir: Path, use_absolute_paths: bool) -> str:
    resolved = path.resolve()
    if use_absolute_paths:
        return str(resolved)
    try:
        return str(resolved.relative_to(manifest_dir.resolve()))
    except ValueError:
        return os.path.relpath(resolved, manifest_dir.resolve())


def finalize_report(*, report: dict[str, Any], samples: list[dict[str, Any]]) -> None:
    split_counts: Counter[str] = Counter(str(sample.get("split")) for sample in samples)
    view_counts: Counter[int] = Counter(len(sample.get("views", [])) for sample in samples)
    subject_counts: Counter[str] = Counter(str(sample.get("subject_id")) for sample in samples)
    report["summary"] = {
        "samples": len(samples),
        "split_counts": dict(sorted(split_counts.items())),
        "view_count_distribution": {
            str(key): value for key, value in sorted(view_counts.items())
        },
        "subject_counts": dict(sorted(subject_counts.items())),
    }


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
