#!/usr/bin/env python
"""Export SAM3DBody compact predictions for cropped Panoptic/Kinoptic data.

The expected input layout is:

    <dataset-root>/<sequence>/rgb/<camera>/<frame>.jpg

By default, outputs are flattened as:

    <output-root>/<sequence>_<camera-zero-padded>_<frame>.npz

For example:

    170307_dance5_kinect_001_00001011.npz

That filename convention keeps the exported predictions compatible with the
existing Stage 1 manifest parser, which expects `<sequence>_<camera>_<frame>`.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORTER_SCRIPT = REPO_ROOT / "external" / "sam-3d-body" / "demo_save_compact_params.py"
DEFAULT_DATASET_ROOT = Path("/opt/data/panoptic_kinoptic_single_actor_cropped")
DEFAULT_CHECKPOINT_PATH = Path("/opt/data/SAM_3dbody_checkpoints/model.ckpt")
DEFAULT_MHR_PATH = Path("/opt/data/SAM_3dbody_checkpoints/assets/mhr_model.pt")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}
KINECT_RE = re.compile(r"^kinect_(?P<index>\d+)$")


@dataclass(frozen=True)
class PanopticCameraTask:
    sequence_id: str
    camera_id: str
    image_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SAM3DBody compact-parameter export over a cropped "
            "Panoptic/Kinoptic single-actor dataset."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Panoptic cropped dataset root containing <sequence>/rgb/<camera> directories.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Prediction output root. Defaults to <dataset-root>/sam3dbody. "
            "Flat layout writes .npz files directly here."
        ),
    )
    parser.add_argument(
        "--layout",
        choices=("flat", "nested"),
        default="flat",
        help=(
            "flat: <output-root>/<sequence>_<camera>_<frame>.npz; "
            "nested: <output-root>/<sequence>/<camera>/<frame>.npz"
        ),
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Optional sequence ids to export, e.g. 170307_dance5 170407_office2.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help=(
            "Optional camera ids to export. Both raw ids like kinect_1 and "
            "zero-padded ids like kinect_001 are accepted."
        ),
    )
    parser.add_argument(
        "--camera-id-width",
        type=int,
        default=3,
        help="Zero-padding width for Kinect camera ids in output names. Use 0 to preserve raw ids.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="SAM3DBody checkpoint path.",
    )
    parser.add_argument(
        "--mhr-path",
        default=str(DEFAULT_MHR_PATH),
        help="MHR model path passed to SAM3DBody.",
    )
    parser.add_argument(
        "--detector-name",
        default="",
        help=(
            "Detector name passed to SAM3DBody. Default is empty because the "
            "Panoptic input is already single-actor cropped."
        ),
    )
    parser.add_argument("--detector-path", default=None)
    parser.add_argument("--segmentor-name", default=None)
    parser.add_argument("--segmentor-path", default=None)
    parser.add_argument(
        "--fov-name",
        default="",
        help=(
            "FOV estimator name. Default is empty, so the exporter uses its "
            "deterministic image-size intrinsics fallback."
        ),
    )
    parser.add_argument("--fov-path", default=None)
    parser.add_argument(
        "--inference-type",
        choices=("body", "full"),
        default="body",
        help="SAM3DBody inference type.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Images per SAM3DBody forward pass. Values >1 use body-only full-image crops.",
    )
    parser.add_argument("--bbox-thresh", type=float, default=0.8)
    parser.add_argument("--use-mask", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run cameras even when all expected final .npz files already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned exporter commands without running them.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used to run the SAM3DBody exporter.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value for exporter subprocesses.",
    )
    parser.add_argument(
        "--momentum-enabled",
        default="0",
        help="MOMENTUM_ENABLED value for exporter subprocesses. Default matches the HuMMan launcher.",
    )
    parser.add_argument(
        "--work-root",
        default=None,
        help="Temporary per-camera exporter output root. Defaults to <output-root>/.panoptic_sam3dbody_work.",
    )
    add_optional_smpl_export_args(parser)
    return parser.parse_args()


def add_optional_smpl_export_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--export-smpl-params", action="store_true")
    parser.add_argument("--smpl-model-path", default=None)
    parser.add_argument("--smpl-gender", default=None)
    parser.add_argument("--export-smplx-params", action="store_true")
    parser.add_argument("--smplx-model-path", default=None)
    parser.add_argument("--smplx-gender", default=None)
    parser.add_argument("--smpl-conversion-method", choices=("pytorch", "pymomentum"), default=None)
    parser.add_argument("--smpl-conversion-batch-size", type=int, default=None)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else dataset_root / "sam3dbody"
    )
    work_root = (
        Path(args.work_root).expanduser().resolve()
        if args.work_root
        else output_root / ".panoptic_sam3dbody_work"
    )

    validate_args(args, dataset_root)
    tasks = discover_camera_tasks(
        dataset_root=dataset_root,
        sequences=args.sequences,
        cameras=args.cameras,
        camera_id_width=args.camera_id_width,
    )
    if not tasks:
        raise RuntimeError(f"No Panoptic RGB camera folders found under {dataset_root}")

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        work_root.mkdir(parents=True, exist_ok=True)

    print(f"Dataset root: {dataset_root}")
    print(f"Output root: {output_root}")
    print(f"Camera tasks: {len(tasks)}")
    for task in tasks:
        process_camera_task(
            task=task,
            args=args,
            output_root=output_root,
            work_root=work_root,
        )


def validate_args(args: argparse.Namespace, dataset_root: Path) -> None:
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")
    if args.camera_id_width < 0:
        raise ValueError(f"--camera-id-width must be >= 0, got {args.camera_id_width}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not EXPORTER_SCRIPT.exists():
        raise FileNotFoundError(f"SAM3DBody exporter not found: {EXPORTER_SCRIPT}")
    if not args.dry_run:
        require_existing_path(args.checkpoint_path, "--checkpoint-path")
        require_existing_path(args.mhr_path, "--mhr-path")


def require_existing_path(value: str | None, arg_name: str) -> None:
    if not value:
        raise ValueError(f"{arg_name} is required")
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{arg_name} does not exist: {path}")


def discover_camera_tasks(
    *,
    dataset_root: Path,
    sequences: list[str] | None,
    cameras: list[str] | None,
    camera_id_width: int,
) -> list[PanopticCameraTask]:
    requested_sequences = set(sequences or [])
    requested_cameras = set(cameras or [])
    tasks: list[PanopticCameraTask] = []

    for sequence_dir in sorted(dataset_root.iterdir()):
        if not sequence_dir.is_dir():
            continue
        if requested_sequences and sequence_dir.name not in requested_sequences:
            continue
        rgb_dir = sequence_dir / "rgb"
        if not rgb_dir.is_dir():
            continue
        for camera_dir in sorted(rgb_dir.iterdir(), key=lambda path: camera_sort_key(path.name)):
            if not camera_dir.is_dir():
                continue
            if requested_cameras and not camera_id_matches(
                camera_dir.name,
                requested_cameras=requested_cameras,
                camera_id_width=camera_id_width,
            ):
                continue
            if not list_image_paths(camera_dir):
                continue
            tasks.append(
                PanopticCameraTask(
                    sequence_id=sequence_dir.name,
                    camera_id=camera_dir.name,
                    image_dir=camera_dir,
                )
            )

    return tasks


def camera_sort_key(camera_id: str) -> tuple[int, str]:
    match = KINECT_RE.match(camera_id)
    if match is None:
        return (10**9, camera_id)
    return (int(match.group("index")), camera_id)


def camera_id_matches(
    camera_id: str,
    *,
    requested_cameras: set[str],
    camera_id_width: int,
) -> bool:
    normalized = normalize_camera_id(camera_id, width=camera_id_width)
    return camera_id in requested_cameras or normalized in requested_cameras


def normalize_camera_id(camera_id: str, *, width: int) -> str:
    if width == 0:
        return camera_id
    match = KINECT_RE.match(camera_id)
    if match is None:
        return camera_id
    return f"kinect_{int(match.group('index')):0{width}d}"


def list_image_paths(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def process_camera_task(
    *,
    task: PanopticCameraTask,
    args: argparse.Namespace,
    output_root: Path,
    work_root: Path,
) -> None:
    image_paths = list_image_paths(task.image_dir)
    final_paths = [
        final_output_path(
            output_root=output_root,
            layout=args.layout,
            sequence_id=task.sequence_id,
            camera_id=task.camera_id,
            frame_stem=image_path.stem,
            camera_id_width=args.camera_id_width,
        )
        for image_path in image_paths
    ]

    camera_label = f"{task.sequence_id}/{task.camera_id}"
    if not args.overwrite and all(path.exists() for path in final_paths):
        print(f"[skip] {camera_label}: {len(final_paths)} predictions already exist")
        return

    staging_dir = work_root / task.sequence_id / task.camera_id
    command = build_export_command(
        args=args,
        image_dir=task.image_dir,
        output_dir=staging_dir,
    )
    print(f"[run] {camera_label}: {len(image_paths)} images")
    if args.dry_run:
        print(" ".join(quote_command_part(part) for part in command))
        return

    staging_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True, env=build_subprocess_env(args))
    move_staged_outputs(
        image_paths=image_paths,
        staging_dir=staging_dir,
        final_paths=final_paths,
        overwrite=args.overwrite,
    )


def final_output_path(
    *,
    output_root: Path,
    layout: str,
    sequence_id: str,
    camera_id: str,
    frame_stem: str,
    camera_id_width: int,
) -> Path:
    normalized_camera_id = normalize_camera_id(camera_id, width=camera_id_width)
    if layout == "nested":
        return output_root / sequence_id / normalized_camera_id / f"{frame_stem}.npz"
    if layout == "flat":
        return output_root / f"{sequence_id}_{normalized_camera_id}_{frame_stem}.npz"
    raise ValueError(f"Unknown output layout: {layout}")


def build_export_command(
    *,
    args: argparse.Namespace,
    image_dir: Path,
    output_dir: Path,
) -> list[str]:
    command = [
        args.python_executable,
        str(EXPORTER_SCRIPT),
        "--image_folder",
        str(image_dir),
        "--output_folder",
        str(output_dir),
        "--checkpoint_path",
        str(args.checkpoint_path),
        "--mhr_path",
        str(args.mhr_path),
        "--detector_name",
        str(args.detector_name),
        "--inference_type",
        str(args.inference_type),
        "--batch_size",
        str(args.batch_size),
        "--bbox_thresh",
        str(args.bbox_thresh),
    ]
    append_optional_arg(command, "--detector_path", args.detector_path)
    append_optional_arg(command, "--segmentor_name", args.segmentor_name)
    append_optional_arg(command, "--segmentor_path", args.segmentor_path)
    append_optional_arg(command, "--fov_name", args.fov_name)
    append_optional_arg(command, "--fov_path", args.fov_path)
    if args.use_mask:
        command.append("--use_mask")
    append_optional_smpl_export_args(command, args)
    return command


def append_optional_smpl_export_args(command: list[str], args: argparse.Namespace) -> None:
    if args.export_smpl_params:
        command.append("--export_smpl_params")
    append_optional_arg(command, "--smpl_model_path", args.smpl_model_path)
    append_optional_arg(command, "--smpl_gender", args.smpl_gender)
    if args.export_smplx_params:
        command.append("--export_smplx_params")
    append_optional_arg(command, "--smplx_model_path", args.smplx_model_path)
    append_optional_arg(command, "--smplx_gender", args.smplx_gender)
    append_optional_arg(command, "--smpl_conversion_method", args.smpl_conversion_method)
    append_optional_arg(
        command,
        "--smpl_conversion_batch_size",
        None if args.smpl_conversion_batch_size is None else str(args.smpl_conversion_batch_size),
    )


def append_optional_arg(command: list[str], name: str, value: str | None) -> None:
    if value is not None:
        command.extend([name, str(value)])


def build_subprocess_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    if args.momentum_enabled is not None:
        env["MOMENTUM_ENABLED"] = str(args.momentum_enabled)
    return env


def move_staged_outputs(
    *,
    image_paths: list[Path],
    staging_dir: Path,
    final_paths: list[Path],
    overwrite: bool,
) -> None:
    missing: list[Path] = []
    for image_path, final_path in zip(image_paths, final_paths, strict=True):
        staged_path = staging_dir / f"{image_path.stem}.npz"
        if not staged_path.exists():
            missing.append(staged_path)
            continue
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if final_path.exists() and not overwrite:
            continue
        if final_path.exists():
            final_path.unlink()
        shutil.move(str(staged_path), str(final_path))

    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:10])
        suffix = "" if len(missing) <= 10 else f"\n  ... and {len(missing) - 10} more"
        raise RuntimeError(f"Exporter did not produce expected files:\n{preview}{suffix}")


def quote_command_part(value: str) -> str:
    if not value or any(char.isspace() for char in value):
        return repr(value)
    return value


if __name__ == "__main__":
    main()
