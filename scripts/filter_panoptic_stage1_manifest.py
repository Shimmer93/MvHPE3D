#!/usr/bin/env python
"""Filter Panoptic Stage 1 manifest views by projected 3D-joint visibility."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mvhpe3d.utils import load_panoptic_camera_parameters


DEFAULT_DATASET_ROOT = Path("/opt/data/panoptic_kinoptic_single_actor_cropped")


@dataclass(frozen=True)
class ViewVisibility:
    keep: bool
    visible_joint_count: int
    total_confident_joints: int
    root_visible: bool
    reason: str


@dataclass
class FilterStats:
    input_samples: int = 0
    output_samples: int = 0
    input_views: int = 0
    output_views: int = 0
    missing_gt3d_views: int = 0
    missing_prediction_views: int = 0
    malformed_prediction_views: int = 0
    low_visibility_views: int = 0
    root_not_visible_views: int = 0
    dropped_samples: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a Panoptic Stage 1 manifest by projecting native gt3d joints "
            "into each cropped RGB view and keeping only views with enough visible joints."
        )
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Input manifest. Defaults to <dataset-root>/panoptic_stage1_manifest.json.",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Panoptic cropped dataset root containing <sequence>/gt3d and camera metadata.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help=(
            "Filtered manifest path. Defaults to "
            "<dataset-root>/panoptic_stage1_manifest_visible.json."
        ),
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional JSON report path with aggregate filtering statistics.",
    )
    parser.add_argument(
        "--min-views",
        type=int,
        default=2,
        help="Minimum valid views required to keep a sample.",
    )
    parser.add_argument(
        "--min-visible-joints",
        type=int,
        default=8,
        help="Minimum number of confident Panoptic joints projected inside the crop.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.05,
        help="Minimum Panoptic joint confidence used for visibility counting.",
    )
    parser.add_argument(
        "--visibility-margin-px",
        type=float,
        default=8.0,
        help="Pixel margin around the crop when deciding whether a joint is visible.",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=1.0e-4,
        help="Minimum positive camera-space depth in meters.",
    )
    parser.add_argument(
        "--require-root-visible",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require Panoptic BodyCenter joint 2 to project inside the crop.",
    )
    parser.add_argument(
        "--absolute-paths",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Rewrite view npz paths as absolute or relative to output manifest. "
            "Default preserves the input path style."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional debugging cap on input samples.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10000,
        help="Print progress every N samples. Use 0 to disable progress logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_views < 1:
        raise ValueError(f"--min-views must be >= 1, got {args.min_views}")
    if args.min_visible_joints < 1:
        raise ValueError(f"--min-visible-joints must be >= 1, got {args.min_visible_joints}")

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    manifest_path = (
        Path(args.manifest_path).expanduser().resolve()
        if args.manifest_path is not None
        else dataset_root / "panoptic_stage1_manifest.json"
    )
    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path is not None
        else dataset_root / "panoptic_stage1_manifest_visible.json"
    )
    report_path = (
        Path(args.report_path).expanduser().resolve()
        if args.report_path is not None
        else output_path.with_suffix(".report.json")
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_samples = payload["samples"] if isinstance(payload, dict) else payload
    if args.max_samples is not None:
        raw_samples = raw_samples[: args.max_samples]

    filtered_samples, stats = filter_samples(
        raw_samples,
        dataset_root=dataset_root,
        input_manifest_dir=manifest_path.parent,
        output_manifest_dir=output_path.parent,
        min_views=args.min_views,
        min_visible_joints=args.min_visible_joints,
        confidence_threshold=args.confidence_threshold,
        visibility_margin_px=args.visibility_margin_px,
        min_depth=args.min_depth,
        require_root_visible=args.require_root_visible,
        absolute_paths=args.absolute_paths,
        log_every=args.log_every,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"samples": filtered_samples}, indent=2), encoding="utf-8")
    report_payload = {
        "input_manifest_path": str(manifest_path),
        "output_manifest_path": str(output_path),
        "dataset_root": str(dataset_root),
        "min_views": args.min_views,
        "min_visible_joints": args.min_visible_joints,
        "confidence_threshold": args.confidence_threshold,
        "visibility_margin_px": args.visibility_margin_px,
        "min_depth": args.min_depth,
        "require_root_visible": args.require_root_visible,
        "stats": stats.__dict__,
    }
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    print(f"Wrote filtered manifest: {output_path}")
    print(f"Wrote report: {report_path}")
    print(f"Samples: {stats.input_samples} -> {stats.output_samples}")
    print(f"Views: {stats.input_views} -> {stats.output_views}")
    print(f"Dropped samples: {stats.dropped_samples}")
    print(f"Low-visibility views: {stats.low_visibility_views}")
    print(f"Root-not-visible views: {stats.root_not_visible_views}")


def filter_samples(
    raw_samples: list[dict[str, Any]],
    *,
    dataset_root: Path,
    input_manifest_dir: Path,
    output_manifest_dir: Path,
    min_views: int,
    min_visible_joints: int,
    confidence_threshold: float,
    visibility_margin_px: float,
    min_depth: float,
    require_root_visible: bool,
    absolute_paths: bool | None,
    log_every: int = 0,
) -> tuple[list[dict[str, Any]], FilterStats]:
    stats = FilterStats(input_samples=len(raw_samples))
    filtered_samples: list[dict[str, Any]] = []
    camera_cache: dict[tuple[str, str], Any] = {}
    gt3d_cache: dict[tuple[str, str], np.ndarray] = {}

    for sample_index, raw_sample in enumerate(raw_samples, start=1):
        views = raw_sample.get("views", [])
        stats.input_views += len(views)
        kept_views = []
        for raw_view in views:
            view_visibility = evaluate_view_visibility(
                raw_sample,
                raw_view,
                dataset_root=dataset_root,
                input_manifest_dir=input_manifest_dir,
                camera_cache=camera_cache,
                gt3d_cache=gt3d_cache,
                min_visible_joints=min_visible_joints,
                confidence_threshold=confidence_threshold,
                visibility_margin_px=visibility_margin_px,
                min_depth=min_depth,
                require_root_visible=require_root_visible,
            )
            update_stats_for_view(stats, view_visibility)
            if not view_visibility.keep:
                continue
            kept_views.append(
                rewrite_view_path(
                    raw_view,
                    input_manifest_dir=input_manifest_dir,
                    output_manifest_dir=output_manifest_dir,
                    absolute_paths=absolute_paths,
                )
            )

        if len(kept_views) < min_views:
            stats.dropped_samples += 1
            continue

        filtered_sample = dict(raw_sample)
        filtered_sample["views"] = kept_views
        filtered_samples.append(filtered_sample)
        stats.output_views += len(kept_views)
        if log_every > 0 and sample_index % log_every == 0:
            print(
                f"Processed {sample_index}/{len(raw_samples)} samples; "
                f"kept {len(filtered_samples)} samples, {stats.output_views} views",
                flush=True,
            )

    stats.output_samples = len(filtered_samples)
    return filtered_samples, stats


def evaluate_view_visibility(
    raw_sample: dict[str, Any],
    raw_view: dict[str, Any],
    *,
    dataset_root: Path,
    input_manifest_dir: Path,
    camera_cache: dict[tuple[str, str], Any],
    gt3d_cache: dict[tuple[str, str], np.ndarray],
    min_visible_joints: int,
    confidence_threshold: float,
    visibility_margin_px: float,
    min_depth: float,
    require_root_visible: bool,
) -> ViewVisibility:
    sequence_id = str(raw_sample["sequence_id"])
    frame_id = normalize_frame_id(raw_sample["frame_id"])
    camera_id = str(raw_view["camera_id"])

    npz_path = resolve_manifest_path(raw_view["npz_path"], input_manifest_dir)
    if not npz_path.exists():
        return ViewVisibility(False, 0, 0, False, "missing_prediction")
    try:
        image_size = load_prediction_image_size(npz_path)
    except (KeyError, ValueError):
        return ViewVisibility(False, 0, 0, False, "malformed_prediction")

    try:
        gt3d = load_gt3d(
            dataset_root=dataset_root,
            sequence_id=sequence_id,
            frame_id=frame_id,
            cache=gt3d_cache,
        )
    except FileNotFoundError:
        return ViewVisibility(False, 0, 0, False, "missing_gt3d")

    camera_key = (sequence_id, camera_id)
    camera = camera_cache.get(camera_key)
    if camera is None:
        camera = load_panoptic_camera_parameters(
            dataset_root,
            sequence_id=sequence_id,
            camera_id=camera_id,
        )
        camera_cache[camera_key] = camera

    visible_mask, confident_mask = project_visibility_mask(
        gt3d,
        image_size=image_size,
        camera=camera,
        confidence_threshold=confidence_threshold,
        visibility_margin_px=visibility_margin_px,
        min_depth=min_depth,
    )
    visible_count = int(visible_mask.sum())
    root_visible = bool(visible_mask[2])
    if require_root_visible and not root_visible:
        return ViewVisibility(False, visible_count, int(confident_mask.sum()), root_visible, "root_not_visible")
    if visible_count < min_visible_joints:
        return ViewVisibility(False, visible_count, int(confident_mask.sum()), root_visible, "low_visibility")
    return ViewVisibility(True, visible_count, int(confident_mask.sum()), root_visible, "keep")


def update_stats_for_view(stats: FilterStats, visibility: ViewVisibility) -> None:
    if visibility.keep:
        return
    if visibility.reason == "missing_prediction":
        stats.missing_prediction_views += 1
    elif visibility.reason == "malformed_prediction":
        stats.malformed_prediction_views += 1
    elif visibility.reason == "missing_gt3d":
        stats.missing_gt3d_views += 1
    elif visibility.reason == "root_not_visible":
        stats.root_not_visible_views += 1
    else:
        stats.low_visibility_views += 1


def project_visibility_mask(
    gt3d: np.ndarray,
    *,
    image_size: np.ndarray,
    camera,
    confidence_threshold: float,
    visibility_margin_px: float,
    min_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = float(image_size[0]), float(image_size[1])
    points_world_m = np.asarray(gt3d[:, :3], dtype=np.float32) * 0.01
    points_camera = (camera.rotation @ points_world_m.T).T + camera.translation.reshape(1, 3)
    z = points_camera[:, 2]
    projected = (camera.intrinsics @ points_camera.T).T
    xy = projected[:, :2] / np.clip(projected[:, 2:3], 1.0e-6, None)
    confident = np.asarray(gt3d[:, 3], dtype=np.float32) >= float(confidence_threshold)
    visible = (
        confident
        & (z > float(min_depth))
        & (xy[:, 0] >= -visibility_margin_px)
        & (xy[:, 0] < width + visibility_margin_px)
        & (xy[:, 1] >= -visibility_margin_px)
        & (xy[:, 1] < height + visibility_margin_px)
    )
    return visible, confident


def load_prediction_image_size(npz_path: Path) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as payload:
        image_size = np.asarray(payload["image_size"], dtype=np.float32)
    if image_size.shape != (2,):
        raise ValueError(f"image_size must have shape (2,), got {image_size.shape}: {npz_path}")
    return image_size


def load_gt3d(
    *,
    dataset_root: Path,
    sequence_id: str,
    frame_id: str,
    cache: dict[tuple[str, str], np.ndarray],
) -> np.ndarray:
    key = (sequence_id, frame_id)
    cached = cache.get(key)
    if cached is not None:
        return cached
    path = dataset_root / sequence_id / "gt3d" / f"{int(frame_id):08d}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Panoptic gt3d file does not exist: {path}")
    payload = np.asarray(np.load(path, allow_pickle=True), dtype=np.float32)
    if payload.shape != (19, 4):
        raise ValueError(f"Expected gt3d shape (19, 4), got {payload.shape}: {path}")
    cache[key] = payload
    return payload


def rewrite_view_path(
    raw_view: dict[str, Any],
    *,
    input_manifest_dir: Path,
    output_manifest_dir: Path,
    absolute_paths: bool | None,
) -> dict[str, Any]:
    if absolute_paths is None:
        return dict(raw_view)
    rewritten = dict(raw_view)
    path = resolve_manifest_path(raw_view["npz_path"], input_manifest_dir)
    if absolute_paths:
        rewritten["npz_path"] = str(path)
    else:
        rewritten["npz_path"] = os.path.relpath(path, output_manifest_dir)
    return rewritten


def resolve_manifest_path(value: str, manifest_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


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
