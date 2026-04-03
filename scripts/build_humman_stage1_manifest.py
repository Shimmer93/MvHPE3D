#!/usr/bin/env python
"""Build a Stage 1 HuMMan manifest from SAM3DBody compact-parameter exports."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FRAME_STEM_RE = re.compile(
    r"^(?P<sequence_id>.+)_(?P<camera_id>iphone|kinect_\d{3})_(?P<frame_id>\d+)$"
)
SEQUENCE_RE = re.compile(r"^(?P<subject_id>p[^_]+)_(?P<action_id>a[^_]+)$")


@dataclass(frozen=True)
class ParsedPrediction:
    sequence_id: str
    camera_id: str
    frame_id: str
    npz_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a split-agnostic HuMMan Stage 1 manifest from exported SAM3DBody npz files."
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        required=True,
        help="Directory containing exported SAM3DBody compact-parameter .npz files.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to the output manifest JSON.",
    )
    parser.add_argument(
        "--min-views",
        type=int,
        default=2,
        help="Minimum number of valid views required to keep a sample.",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute paths into the manifest instead of paths relative to the manifest file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_dir = Path(args.predictions_dir).resolve()
    output_path = Path(args.output_path).resolve()

    if not predictions_dir.exists():
        raise FileNotFoundError(f"Predictions directory does not exist: {predictions_dir}")
    if args.min_views < 1:
        raise ValueError(f"--min-views must be >= 1, got {args.min_views}")

    parsed_predictions, skipped_files = collect_predictions(predictions_dir)
    samples = build_manifest_samples(
        parsed_predictions,
        min_views=args.min_views,
        manifest_dir=output_path.parent,
        use_absolute_paths=args.absolute_paths,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"samples": samples}, indent=2), encoding="utf-8")

    kept_views = sum(len(sample["views"]) for sample in samples)
    print(f"Wrote manifest: {output_path}")
    print(f"Valid exported views: {len(parsed_predictions)}")
    print(f"Skipped exported files: {len(skipped_files)}")
    print(f"Kept samples: {len(samples)}")
    print(f"Kept views in manifest: {kept_views}")
    if skipped_files:
        print("Skipped files:")
        for path, reason in skipped_files[:20]:
            print(f"  - {path}: {reason}")
        if len(skipped_files) > 20:
            print(f"  ... and {len(skipped_files) - 20} more")


def collect_predictions(predictions_dir: Path) -> tuple[list[ParsedPrediction], list[tuple[Path, str]]]:
    parsed_predictions: list[ParsedPrediction] = []
    skipped_files: list[tuple[Path, str]] = []

    for npz_path in sorted(predictions_dir.rglob("*.npz")):
        try:
            parsed_predictions.append(parse_prediction_file(npz_path))
        except ValueError as exc:
            skipped_files.append((npz_path, str(exc)))

    return parsed_predictions, skipped_files


def parse_prediction_file(npz_path: Path) -> ParsedPrediction:
    match = FRAME_STEM_RE.match(npz_path.stem)
    if match is None:
        raise ValueError(
            "filename does not match expected pattern '<sequence_id>_<camera_id>_<frame_id>.npz'"
        )

    with np.load(npz_path, allow_pickle=False) as payload:
        mhr_model_params = _read_array(payload, "mhr_model_params")
        shape_params = _read_array(payload, "shape_params")
        pred_cam_t = _read_array(payload, "pred_cam_t")
        cam_int = _read_array(payload, "cam_int")
        image_size = _read_array(payload, "image_size")

    _require_single_person_vector(mhr_model_params, "mhr_model_params", expected_dim=204)
    _require_single_person_vector(shape_params, "shape_params", expected_dim=45)
    _require_single_person_vector(pred_cam_t, "pred_cam_t", expected_dim=3)

    if cam_int.shape != (3, 3):
        raise ValueError(f"cam_int must have shape (3, 3), got {cam_int.shape}")
    if image_size.shape != (2,):
        raise ValueError(f"image_size must have shape (2,), got {image_size.shape}")

    return ParsedPrediction(
        sequence_id=match.group("sequence_id"),
        camera_id=match.group("camera_id"),
        frame_id=match.group("frame_id"),
        npz_path=npz_path.resolve(),
    )


def build_manifest_samples(
    parsed_predictions: list[ParsedPrediction],
    *,
    min_views: int,
    manifest_dir: Path,
    use_absolute_paths: bool,
) -> list[dict[str, Any]]:
    grouped_predictions: dict[tuple[str, str], list[ParsedPrediction]] = defaultdict(list)
    for prediction in parsed_predictions:
        grouped_predictions[(prediction.sequence_id, prediction.frame_id)].append(prediction)

    samples: list[dict[str, Any]] = []
    for (sequence_id, frame_id), grouped_views in sorted(grouped_predictions.items()):
        grouped_views = sorted(grouped_views, key=lambda item: item.camera_id)
        if len(grouped_views) < min_views:
            continue

        subject_id, action_id = parse_sequence_metadata(sequence_id)
        sample: dict[str, Any] = {
            "sample_id": f"{sequence_id}_frame{frame_id}",
            "sequence_id": sequence_id,
            "frame_id": frame_id,
            "views": [
                {
                    "camera_id": view.camera_id,
                    "npz_path": stringify_path(
                        view.npz_path,
                        manifest_dir=manifest_dir,
                        use_absolute_paths=use_absolute_paths,
                    ),
                }
                for view in grouped_views
            ],
        }
        if subject_id is not None:
            sample["subject_id"] = subject_id
        if action_id is not None:
            sample["action_id"] = action_id

        samples.append(sample)

    return samples


def parse_sequence_metadata(sequence_id: str) -> tuple[str | None, str | None]:
    match = SEQUENCE_RE.match(sequence_id)
    if match is None:
        return None, None
    return match.group("subject_id"), match.group("action_id")


def stringify_path(path: Path, *, manifest_dir: Path, use_absolute_paths: bool) -> str:
    resolved = path.resolve()
    if use_absolute_paths:
        return str(resolved)
    return os.path.relpath(resolved, manifest_dir)


def _read_array(payload: Any, key: str) -> np.ndarray:
    if key not in payload:
        raise ValueError(f"missing required field '{key}'")
    return np.asarray(payload[key])


def _require_single_person_vector(array: np.ndarray, key: str, *, expected_dim: int) -> None:
    if array.ndim == 1 and array.shape[0] == expected_dim:
        return
    if array.ndim == 2 and array.shape == (1, expected_dim):
        return
    if array.ndim == 2 and array.shape[0] == 0:
        raise ValueError(f"{key} has no detected person")
    if array.ndim == 2 and array.shape[0] > 1:
        raise ValueError(f"{key} has multiple detected people: shape {array.shape}")
    raise ValueError(f"{key} has invalid shape {array.shape}, expected (1, {expected_dim})")


if __name__ == "__main__":
    main()
