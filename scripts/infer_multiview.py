#!/usr/bin/env python
"""Run Stage 2/3 inference on multi-view image or video inputs."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mvhpe3d.lightning import Stage2FusionLightningModule, Stage3TemporalLightningModule
from mvhpe3d.models import (
    Stage2JointGraphRefinerConfig,
    Stage2JointResidualConfig,
    Stage2ParamRefineConfig,
    Stage3TemporalRefineConfig,
)
from mvhpe3d.utils import MHRToSMPLConverter, axis_angle_to_rotation_6d, load_experiment_config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
NPZ_SUFFIX = ".npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run calibration-free MvHPE3D inference from multi-view compact NPZs, "
            "images, or videos. Images use Stage 2; videos support Stage 2 or Stage 3."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("image", "video"),
        required=True,
        help="Input semantics. image runs frame-independent Stage 2; video can run Stage 2 or Stage 3.",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        required=True,
        help=(
            "One path per camera/view. Each path may be an NPZ file, an NPZ directory, "
            "an image file, an image directory, or a video file."
        ),
    )
    parser.add_argument("--checkpoint-path", required=True, help="Stage 2 or Stage 3 checkpoint")
    parser.add_argument(
        "--config",
        required=True,
        help="Stage 2 or Stage 3 experiment config matching --checkpoint-path",
    )
    parser.add_argument(
        "--stage2-checkpoint-path",
        default=None,
        help="Stage 2 checkpoint used by a Stage 3 checkpoint when not stored in hparams",
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "npz", "media"),
        default="auto",
        help="Use npz to skip SAM3DBody export, media for raw images/videos, or auto-detect.",
    )
    parser.add_argument(
        "--sam3dbody-checkpoint-path",
        default=None,
        help="SAM3DBody checkpoint used when --input-format resolves to media",
    )
    parser.add_argument("--sam3dbody-detector-path", default=None)
    parser.add_argument("--sam3dbody-segmentor-path", default=None)
    parser.add_argument("--sam3dbody-fov-path", default=None)
    parser.add_argument("--sam3dbody-mhr-path", default=None)
    parser.add_argument(
        "--mhr-assets-dir",
        default=None,
        help="MHR asset directory for compact MHR -> fitted SMPL conversion",
    )
    parser.add_argument("--smpl-model-path", default=None, help="Neutral SMPL model path")
    parser.add_argument("--output-dir", default="outputs/inference", help="Output directory")
    parser.add_argument("--work-dir", default=None, help="Optional media preprocessing directory")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--person-index", type=int, default=0, help="Person row to use from each view NPZ")
    parser.add_argument("--batch-size", type=int, default=256, help="MHR->SMPL conversion batch size")
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Stage 3 temporal window length. Defaults to data.window_size from the config, then 5.",
    )
    parser.add_argument("--frame-step", type=int, default=1, help="Use every Nth frame for video extraction")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on inferred frames")
    parser.add_argument(
        "--overwrite-media-cache",
        action="store_true",
        help="Recreate extracted frames and exported compact NPZs under --work-dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.person_index < 0:
        raise ValueError(f"--person-index must be >= 0, got {args.person_index}")
    if args.frame_step < 1:
        raise ValueError(f"--frame-step must be >= 1, got {args.frame_step}")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError(f"--max-frames must be >= 1, got {args.max_frames}")

    experiment = load_experiment_config(args.config)
    model_config = build_model_config(experiment["model"], checkpoint_path=args.checkpoint_path)
    model_stage = infer_model_stage(model_config)
    if args.mode == "image" and model_stage != "stage2":
        raise ValueError("Image inference must use a Stage 2 model/config. Use --mode video for Stage 3.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir).resolve() if args.work_dir else output_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    frame_view_npz_paths = prepare_frame_view_npz_paths(args=args, work_dir=work_dir)
    if args.max_frames is not None:
        frame_view_npz_paths = frame_view_npz_paths[: args.max_frames]
    if not frame_view_npz_paths:
        raise RuntimeError("No input frames were resolved for inference")

    num_views = len(frame_view_npz_paths[0])
    if any(len(frame_paths) != num_views for frame_paths in frame_view_npz_paths):
        raise RuntimeError("Resolved frame list has inconsistent view counts")

    stage2_input_map = build_stage2_input_map(
        frame_view_npz_paths=frame_view_npz_paths,
        args=args,
        device=device,
    )
    module = load_inference_module(
        checkpoint_path=args.checkpoint_path,
        model_config=model_config,
        args=args,
    )
    module.eval()
    module.to(device)

    with torch.no_grad():
        outputs = run_inference(
            module=module,
            model_stage=model_stage,
            model_config=model_config,
            frame_view_npz_paths=frame_view_npz_paths,
            stage2_input_map=stage2_input_map,
            experiment=experiment,
            args=args,
            device=device,
        )

    prediction_path = output_dir / "predictions.npz"
    save_prediction_npz(prediction_path, outputs)
    summary_path = output_dir / "summary.json"
    summary = build_summary(
        args=args,
        model_stage=model_stage,
        model_config=model_config,
        frame_view_npz_paths=frame_view_npz_paths,
        prediction_path=prediction_path,
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved predictions to {prediction_path}")
    print(f"Saved summary to {summary_path}")


def build_model_config(config: dict[str, Any], *, checkpoint_path: str):
    model_kwargs = dict(config)
    model_name = str(model_kwargs.pop("name", "stage2_param_refine"))
    model_kwargs.pop("_config_path", None)
    requested_causal = model_kwargs.get("causal")
    model_kwargs.update(load_checkpoint_model_overrides(checkpoint_path))
    if requested_causal is not None:
        model_kwargs["causal"] = requested_causal

    if model_name == "stage3_temporal_refine":
        return Stage3TemporalRefineConfig(**model_kwargs)
    if model_name == "stage2_joint_graph_refiner":
        return Stage2JointGraphRefinerConfig(**model_kwargs)
    if model_name == "stage2_joint_residual":
        return Stage2JointResidualConfig(**model_kwargs)
    return Stage2ParamRefineConfig(**model_kwargs)


def load_checkpoint_model_overrides(checkpoint_path: str) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint.get("hyper_parameters", {}).get("model_config", {})
    return dict(model_config) if isinstance(model_config, dict) else {}


def infer_model_stage(model_config) -> str:
    if isinstance(model_config, Stage3TemporalRefineConfig):
        return "stage3"
    return "stage2"


def load_inference_module(*, checkpoint_path: str, model_config, args: argparse.Namespace):
    if isinstance(model_config, Stage3TemporalRefineConfig):
        return Stage3TemporalLightningModule.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
            model_config=model_config,
            smpl_model_path=args.smpl_model_path,
            stage2_checkpoint_path=args.stage2_checkpoint_path,
            strict=False,
        )
    return Stage2FusionLightningModule.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
        model_config=model_config,
        smpl_model_path=args.smpl_model_path,
        strict=False,
    )


def prepare_frame_view_npz_paths(*, args: argparse.Namespace, work_dir: Path) -> list[list[Path]]:
    view_paths = [Path(path).resolve() for path in args.views]
    if len(view_paths) < 1:
        raise ValueError("At least one --views path is required")
    input_format = resolve_input_format(view_paths, requested=args.input_format)
    if input_format == "npz":
        return collect_npz_frame_views(view_paths, mode=args.mode)
    return prepare_media_frame_views(view_paths=view_paths, args=args, work_dir=work_dir)


def resolve_input_format(view_paths: list[Path], *, requested: str) -> str:
    if requested != "auto":
        return requested
    if all(is_npz_path(path) or is_npz_dir(path) for path in view_paths):
        return "npz"
    return "media"


def prepare_media_frame_views(
    *,
    view_paths: list[Path],
    args: argparse.Namespace,
    work_dir: Path,
) -> list[list[Path]]:
    if args.sam3dbody_checkpoint_path is None:
        raise ValueError(
            "Raw image/video inference requires --sam3dbody-checkpoint-path. "
            "Use --input-format npz to run from pre-exported compact NPZs."
        )
    media_root = work_dir / "media"
    compact_root = work_dir / "compact_npz"
    if args.overwrite_media_cache:
        if media_root.exists():
            shutil.rmtree(media_root)
        if compact_root.exists():
            shutil.rmtree(compact_root)
    media_root.mkdir(parents=True, exist_ok=True)
    compact_root.mkdir(parents=True, exist_ok=True)

    if args.mode == "video" and all(path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES for path in view_paths):
        image_dirs = []
        for view_index, video_path in enumerate(view_paths):
            image_dir = media_root / f"view_{view_index:03d}"
            if args.overwrite_media_cache or not image_dir.exists():
                extract_video_frames(
                    video_path=video_path,
                    output_dir=image_dir,
                    frame_step=args.frame_step,
                    max_frames=args.max_frames,
                )
            image_dirs.append(image_dir)
        compact_dirs = export_compact_npz_dirs(image_dirs=image_dirs, args=args, compact_root=compact_root)
        return collect_npz_frame_views(compact_dirs, mode="video")

    if all(path.is_dir() for path in view_paths):
        compact_dirs = export_compact_npz_dirs(image_dirs=view_paths, args=args, compact_root=compact_root)
        return collect_npz_frame_views(compact_dirs, mode=args.mode)

    if args.mode == "image" and all(path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES for path in view_paths):
        image_dirs = []
        expected_npzs = []
        for view_index, image_path in enumerate(view_paths):
            image_dir = media_root / f"view_{view_index:03d}"
            image_dir.mkdir(parents=True, exist_ok=True)
            copied_path = image_dir / image_path.name
            if args.overwrite_media_cache or not copied_path.exists():
                shutil.copy2(image_path, copied_path)
            image_dirs.append(image_dir)
            expected_npzs.append(compact_root / f"view_{view_index:03d}" / f"{image_path.stem}.npz")
        export_compact_npz_dirs(image_dirs=image_dirs, args=args, compact_root=compact_root)
        return [[path.resolve() for path in expected_npzs]]

    raise ValueError(
        "Unsupported media layout. For image mode, pass image files or image directories. "
        "For video mode, pass video files or pre-extracted image directories."
    )


def export_compact_npz_dirs(
    *,
    image_dirs: list[Path],
    args: argparse.Namespace,
    compact_root: Path,
) -> list[Path]:
    output_dirs = []
    exporter_script = REPO_ROOT / "external" / "sam-3d-body" / "demo_save_compact_params.py"
    for view_index, image_dir in enumerate(image_dirs):
        output_dir = compact_root / f"view_{view_index:03d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        if not args.overwrite_media_cache and compact_outputs_complete(
            image_dir=image_dir,
            output_dir=output_dir,
        ):
            output_dirs.append(output_dir)
            continue
        command = [
            sys.executable,
            str(exporter_script),
            "--image_folder",
            str(image_dir),
            "--output_folder",
            str(output_dir),
            "--checkpoint_path",
            str(args.sam3dbody_checkpoint_path),
        ]
        append_optional_arg(command, "--detector_path", args.sam3dbody_detector_path)
        append_optional_arg(command, "--segmentor_path", args.sam3dbody_segmentor_path)
        append_optional_arg(command, "--fov_path", args.sam3dbody_fov_path)
        append_optional_arg(command, "--mhr_path", args.sam3dbody_mhr_path)
        subprocess.run(command, check=True)
        output_dirs.append(output_dir)
    return output_dirs


def compact_outputs_complete(*, image_dir: Path, output_dir: Path) -> bool:
    image_stems = {
        path.stem
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    if not image_stems:
        return False
    return all((output_dir / f"{stem}.npz").exists() for stem in image_stems)


def append_optional_arg(command: list[str], name: str, value: str | None) -> None:
    if value:
        command.extend([name, value])


def extract_video_frames(
    *,
    video_path: Path,
    output_dir: Path,
    frame_step: int,
    max_frames: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")
    frame_index = 0
    written = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_step == 0:
                output_path = output_dir / f"frame_{written:06d}.jpg"
                if not cv2.imwrite(str(output_path), frame):
                    raise RuntimeError(f"Failed to write extracted frame: {output_path}")
                written += 1
                if max_frames is not None and written >= max_frames:
                    break
            frame_index += 1
    finally:
        capture.release()
    if written == 0:
        raise RuntimeError(f"No frames were extracted from {video_path}")


def collect_npz_frame_views(view_paths: list[Path], *, mode: str) -> list[list[Path]]:
    if all(path.is_file() and path.suffix.lower() == NPZ_SUFFIX for path in view_paths):
        return [[path.resolve() for path in view_paths]]
    if not all(path.is_dir() for path in view_paths):
        raise ValueError("NPZ inputs must be all .npz files or all directories containing .npz files")

    per_view = [collect_npz_by_stem(path) for path in view_paths]
    common_stems = set(per_view[0])
    for mapping in per_view[1:]:
        common_stems &= set(mapping)
    if not common_stems:
        raise RuntimeError("No common NPZ frame stems were found across all views")
    ordered_stems = sorted(common_stems, key=natural_key)
    if mode == "image" and len(ordered_stems) == 0:
        raise RuntimeError("No image NPZ inputs were found")
    return [[mapping[stem] for mapping in per_view] for stem in ordered_stems]


def collect_npz_by_stem(directory: Path) -> dict[str, Path]:
    mapping = {path.stem: path.resolve() for path in directory.glob("*.npz")}
    if not mapping:
        raise FileNotFoundError(f"No .npz files found under {directory}")
    return mapping


def is_npz_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == NPZ_SUFFIX


def is_npz_dir(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.npz"))


def build_stage2_input_map(
    *,
    frame_view_npz_paths: list[list[Path]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, np.ndarray]:
    unique_paths = sorted({path.resolve() for frame in frame_view_npz_paths for path in frame})
    input_map: dict[str, np.ndarray] = {}
    pending_paths = []
    pending_mhr = []
    pending_shape = []
    pending_cam_t = []

    for npz_path in unique_paths:
        payload = load_npz_payload(npz_path)
        direct_smpl = read_direct_smpl_parameters(
            payload,
            person_index=args.person_index,
            npz_path=npz_path,
        )
        if direct_smpl is not None:
            input_map[str(npz_path)] = build_stage2_input(
                body_pose=direct_smpl["body_pose"],
                betas=direct_smpl["betas"],
            )
            continue
        compact = read_compact_person_parameters(
            payload,
            person_index=args.person_index,
            npz_path=npz_path,
        )
        pending_paths.append(npz_path)
        pending_mhr.append(compact["mhr_model_params"])
        pending_shape.append(compact["shape_params"])
        pending_cam_t.append(compact["pred_cam_t"])

    if pending_paths:
        converter = MHRToSMPLConverter(
            smpl_model_path=args.smpl_model_path,
            mhr_assets_dir=args.mhr_assets_dir,
            cache_dir=str(Path(args.output_dir).resolve() / "_smpl_cache"),
            batch_size=args.batch_size,
        )
        for start in range(0, len(pending_paths), args.batch_size):
            end = start + args.batch_size
            batch_paths = pending_paths[start:end]
            mhr = torch.from_numpy(np.stack(pending_mhr[start:end], axis=0)).to(device)
            shape = torch.from_numpy(np.stack(pending_shape[start:end], axis=0)).to(device)
            cam_t = torch.from_numpy(np.stack(pending_cam_t[start:end], axis=0)).to(device)
            source_paths = [str(path) for path in batch_paths] if args.person_index == 0 else None
            converted = converter.convert(
                mhr_model_params=mhr.float(),
                shape_params=shape.float(),
                pred_cam_t=cam_t.float(),
                source_npz_paths=source_paths,
            )
            for offset, npz_path in enumerate(batch_paths):
                input_map[str(npz_path.resolve())] = build_stage2_input(
                    body_pose=converted["body_pose"][offset].detach().cpu().numpy(),
                    betas=converted["betas"][offset].detach().cpu().numpy(),
                )

    return input_map


def read_direct_smpl_parameters(
    payload: dict[str, np.ndarray],
    *,
    person_index: int,
    npz_path: Path,
) -> dict[str, np.ndarray] | None:
    for body_key, betas_key in (("body_pose", "betas"), ("smpl_body_pose", "smpl_betas")):
        if body_key in payload and betas_key in payload:
            return {
                "body_pose": select_person_vector(
                    payload[body_key],
                    expected_dim=69,
                    person_index=person_index,
                    key=body_key,
                    npz_path=npz_path,
                ),
                "betas": select_person_vector(
                    payload[betas_key],
                    expected_dim=10,
                    person_index=person_index,
                    key=betas_key,
                    npz_path=npz_path,
                ),
            }
    return None


def read_compact_person_parameters(
    payload: dict[str, np.ndarray],
    *,
    person_index: int,
    npz_path: Path,
) -> dict[str, np.ndarray]:
    return {
        "mhr_model_params": select_person_vector(
            payload["mhr_model_params"],
            expected_dim=204,
            person_index=person_index,
            key="mhr_model_params",
            npz_path=npz_path,
        ),
        "shape_params": select_person_vector(
            payload["shape_params"],
            expected_dim=45,
            person_index=person_index,
            key="shape_params",
            npz_path=npz_path,
        ),
        "pred_cam_t": select_person_vector(
            payload.get("pred_cam_t", np.zeros((1, 3), dtype=np.float32)),
            expected_dim=3,
            person_index=person_index,
            key="pred_cam_t",
            npz_path=npz_path,
        ),
    }


def select_person_vector(
    value: np.ndarray,
    *,
    expected_dim: int,
    person_index: int,
    key: str,
    npz_path: Path,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        if person_index != 0:
            raise IndexError(f"{npz_path} field '{key}' has one person, requested {person_index}")
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != expected_dim:
        raise ValueError(
            f"{npz_path} field '{key}' has shape {array.shape}, expected [N, {expected_dim}]"
        )
    if person_index >= array.shape[0]:
        raise IndexError(
            f"{npz_path} field '{key}' has {array.shape[0]} people, requested {person_index}"
        )
    return np.ascontiguousarray(array[person_index].astype(np.float32, copy=False))


def build_stage2_input(*, body_pose: np.ndarray, betas: np.ndarray) -> np.ndarray:
    body_pose_6d = axis_angle_to_rotation_6d(
        torch.from_numpy(np.asarray(body_pose, dtype=np.float32).reshape(-1, 3))
    ).reshape(-1)
    return np.concatenate(
        [
            body_pose_6d.detach().cpu().numpy().astype(np.float32, copy=False),
            np.asarray(betas, dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def run_inference(
    *,
    module,
    model_stage: str,
    model_config,
    frame_view_npz_paths: list[list[Path]],
    stage2_input_map: dict[str, np.ndarray],
    experiment: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    tensor_outputs: dict[str, list[np.ndarray]] = {}
    frame_ids = [frame_view_npz_paths[index][0].stem for index in range(len(frame_view_npz_paths))]

    if model_stage == "stage2":
        for frame_paths in frame_view_npz_paths:
            views_input = build_views_input_tensor(frame_paths, stage2_input_map, device=device)
            predictions = module(views_input[None, ...])
            append_prediction_outputs(tensor_outputs, predictions)
    else:
        window_size = resolve_stage3_window_size(args=args, experiment=experiment)
        for frame_index in range(len(frame_view_npz_paths)):
            window_indices = temporal_window_indices(
                frame_index=frame_index,
                num_frames=len(frame_view_npz_paths),
                window_size=window_size,
                causal=bool(model_config.causal),
            )
            window_inputs = [
                build_views_input_tensor(frame_view_npz_paths[index], stage2_input_map, device=device)
                for index in window_indices
            ]
            views_input = torch.stack(window_inputs, dim=0)[None, ...]
            predictions = module(views_input)
            append_prediction_outputs(tensor_outputs, predictions)

    return {
        "frame_ids": np.asarray(frame_ids),
        "view_npz_paths": np.asarray(
            [[str(path) for path in frame_paths] for frame_paths in frame_view_npz_paths]
        ),
        "tensors": {
            key: np.concatenate(values, axis=0)
            for key, values in tensor_outputs.items()
        },
    }


def build_views_input_tensor(
    frame_paths: list[Path],
    stage2_input_map: dict[str, np.ndarray],
    *,
    device: torch.device,
) -> torch.Tensor:
    values = [stage2_input_map[str(path.resolve())] for path in frame_paths]
    return torch.from_numpy(np.stack(values, axis=0)).to(device=device, dtype=torch.float32)


def append_prediction_outputs(
    tensor_outputs: dict[str, list[np.ndarray]],
    predictions: dict[str, torch.Tensor],
) -> None:
    keys = (
        "pred_body_pose",
        "pred_betas",
        "pred_pose_6d",
        "base_body_pose",
        "base_betas",
        "stage2_pred_pose_6d",
        "stage2_pred_betas",
        "pose_residual_6d",
        "betas_residual",
        "target_frame_index",
    )
    for key in keys:
        value = predictions.get(key)
        if value is None or not torch.is_tensor(value):
            continue
        tensor_outputs.setdefault(key, []).append(value.detach().cpu().numpy())


def resolve_stage3_window_size(*, args: argparse.Namespace, experiment: dict[str, Any]) -> int:
    if args.window_size is not None:
        return args.window_size
    data_window_size = experiment.get("data", {}).get("window_size")
    if data_window_size is not None:
        return int(data_window_size)
    return 5


def temporal_window_indices(
    *,
    frame_index: int,
    num_frames: int,
    window_size: int,
    causal: bool,
) -> list[int]:
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    if causal:
        return [min(max(frame_index + offset, 0), num_frames - 1) for offset in range(1 - window_size, 1)]
    if window_size % 2 == 0:
        raise ValueError(f"Centered Stage 3 window_size must be odd, got {window_size}")
    radius = window_size // 2
    return [min(max(frame_index + offset, 0), num_frames - 1) for offset in range(-radius, radius + 1)]


def save_prediction_npz(path: Path, outputs: dict[str, Any]) -> None:
    payload = {
        "frame_ids": outputs["frame_ids"],
        "view_npz_paths": outputs["view_npz_paths"],
    }
    payload.update(outputs["tensors"])
    np.savez_compressed(path, **payload)


def build_summary(
    *,
    args: argparse.Namespace,
    model_stage: str,
    model_config,
    frame_view_npz_paths: list[list[Path]],
    prediction_path: Path,
) -> dict[str, Any]:
    return {
        "mode": args.mode,
        "model_stage": model_stage,
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "config": str(Path(args.config).resolve()),
        "prediction_path": str(prediction_path),
        "num_frames": len(frame_view_npz_paths),
        "num_views": len(frame_view_npz_paths[0]) if frame_view_npz_paths else 0,
        "causal": bool(getattr(model_config, "causal", False)),
        "window_size": args.window_size,
        "views": [str(Path(path).resolve()) for path in args.views],
    }


def load_npz_payload(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def natural_key(value: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


if __name__ == "__main__":
    main()
