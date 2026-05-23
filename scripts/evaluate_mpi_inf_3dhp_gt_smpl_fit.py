#!/usr/bin/env python
"""Summarize MPI-INF-3DHP pseudo-SMPL fitting quality reports."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mvhpe3d.data.mpi_inf_3dhp import MPII3D_HEATFORMER_EVAL_JOINTS
from mvhpe3d.data.splits import load_sample_records


JOINT_NAMES = (
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", default="data/mpi_inf_3dhp/mpi_inf_3dhp_stage2_manifest.json")
    parser.add_argument("--gt-smpl-dir", default="data/mpi_inf_3dhp/gt_smpl_fit_input_root")
    parser.add_argument("--splits", nargs="*", default=("train", "val"))
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--output-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_path).resolve()
    gt_smpl_dir = Path(args.gt_smpl_dir).resolve()
    output_path = (
        Path(args.output_path).resolve()
        if args.output_path is not None
        else gt_smpl_dir / "fit_quality_report.json"
    )
    split_filter = set(args.splits or [])
    sequence_filter = set(args.sequences or [])

    frames_by_sequence: dict[str, set[int]] = defaultdict(set)
    for record in load_sample_records(manifest_path):
        if split_filter and record.split not in split_filter:
            continue
        if sequence_filter and record.sequence_id not in sequence_filter:
            continue
        if not record.views:
            continue
        frames_by_sequence[record.sequence_id].add(int(record.frame_id))

    sequence_reports: dict[str, dict[str, Any]] = {}
    all_frame_errors = []
    all_joint_errors = []
    missing = []
    for sequence_id, frame_ids in sorted(frames_by_sequence.items()):
        sequence_path = gt_smpl_dir / f"{sequence_id}_smpl_params.npz"
        if not sequence_path.exists():
            missing.append(str(sequence_path))
            continue
        with np.load(sequence_path, allow_pickle=False) as payload:
            valid_mask = np.asarray(payload.get("valid_mask", []), dtype=bool)
            if "frame_mpjpe" not in payload:
                raise KeyError(
                    f"{sequence_path} does not contain 'frame_mpjpe'. "
                    "Regenerate fits with the updated fitter."
                )
            frame_mpjpe = np.asarray(payload["frame_mpjpe"], dtype=np.float32)
            joint_error = (
                np.asarray(payload["joint_error"], dtype=np.float32)
                if "joint_error" in payload
                else None
            )

        selected_frames = [
            frame_id
            for frame_id in sorted(frame_ids)
            if 0 <= frame_id < frame_mpjpe.shape[0]
            and frame_id < valid_mask.shape[0]
            and bool(valid_mask[frame_id])
            and np.isfinite(frame_mpjpe[frame_id])
        ]
        if not selected_frames:
            sequence_reports[sequence_id] = {"samples": 0, "status": "empty"}
            continue

        frame_values = frame_mpjpe[selected_frames]
        all_frame_errors.append(frame_values)
        report = {
            "samples": int(frame_values.shape[0]),
            "mean_mpjpe": float(frame_values.mean()),
            "median_mpjpe": float(np.median(frame_values)),
        }
        if joint_error is not None:
            joint_values = joint_error[selected_frames]
            all_joint_errors.append(joint_values)
            report["per_joint_mpjpe"] = per_joint_report(joint_values)
        sequence_reports[sequence_id] = report

    if not all_frame_errors:
        raise RuntimeError(f"No fitted frames were found under {gt_smpl_dir}")

    frame_values = np.concatenate(all_frame_errors, axis=0)
    summary: dict[str, Any] = {
        "manifest_path": str(manifest_path),
        "gt_smpl_dir": str(gt_smpl_dir),
        "splits": sorted(split_filter),
        "sequences": sequence_reports,
        "missing_files": missing,
        "overall": {
            "samples": int(frame_values.shape[0]),
            "mean_mpjpe": float(frame_values.mean()),
            "median_mpjpe": float(np.median(frame_values)),
        },
    }
    if all_joint_errors:
        summary["overall"]["per_joint_mpjpe"] = per_joint_report(
            np.concatenate(all_joint_errors, axis=0)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2))
    print(f"Wrote report: {output_path}")


def per_joint_report(joint_error: np.ndarray) -> dict[str, float]:
    if joint_error.ndim != 2 or joint_error.shape[1] != len(MPII3D_HEATFORMER_EVAL_JOINTS):
        raise ValueError(f"Expected joint_error [N, 17], got {joint_error.shape}")
    means = joint_error.mean(axis=0)
    return {
        joint_name: float(value)
        for joint_name, value in zip(JOINT_NAMES, means.tolist(), strict=True)
    }


if __name__ == "__main__":
    main()
