#!/usr/bin/env python
"""Render per-sequence MP4 videos with input, fused prediction, and GT SMPL overlays."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import trimesh

if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import pyrender

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mvhpe3d.data import (
    Stage1DataConfig,
    Stage1HuMManDataModule,
    Stage2DataConfig,
    Stage2HuMManDataModule,
    multiview_collate,
)
from mvhpe3d.lightning import Stage1FusionLightningModule, Stage2FusionLightningModule
from mvhpe3d.metrics import batch_mpjpe, batch_pa_mpjpe
from mvhpe3d.models import (
    Stage1MLPFusionConfig,
    Stage1ResidualFusionConfig,
    Stage2ParamRefineConfig,
)
from mvhpe3d.utils import (
    build_smpl_model,
    cache_path_for_source_npz,
    load_experiment_config,
    resolve_smpl_model_path as resolve_smpl_model_path_impl,
    validate_mhr_asset_folder,
)
from mvhpe3d.visualization import (
    load_camera_parameters,
    resolve_rgb_image_path,
)

SMPL_SKELETON_EDGES = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save per-sequence MP4 videos with input, predicted, and GT SMPL overlays"
    )
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment/stage1_cross_camera.yaml",
        help="Path to the experiment YAML file",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Optional override for the data manifest path",
    )
    parser.add_argument(
        "--gt-smpl-dir",
        type=str,
        default=None,
        help="Optional override for the HuMMan GT SMPL directory",
    )
    parser.add_argument(
        "--split-config-path",
        type=str,
        default=None,
        help="Optional override for the split policy YAML path",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default=None,
        help="Optional override for the named split policy in the split config YAML",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=("train", "val", "test"),
        default="test",
        help="Which split to visualize",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/stage1_videos",
        help="Directory to save the MP4 videos",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for inference, e.g. cuda:0 or cpu",
    )
    parser.add_argument(
        "--rgb-dir",
        type=str,
        default=None,
        help="Optional override for the HuMMan cropped RGB directory",
    )
    parser.add_argument(
        "--cameras-dir",
        type=str,
        default=None,
        help="Optional override for the HuMMan camera JSON directory used by the dataset loader",
    )
    parser.add_argument(
        "--mhr-assets-dir",
        type=str,
        default=None,
        help="Optional override for the MHR asset directory used for input-view conversion",
    )
    parser.add_argument(
        "--input-smpl-cache-dir",
        type=str,
        default=None,
        help="Optional directory for caching fitted SMPL parameters converted from input views",
    )
    parser.add_argument(
        "--smpl-model-path",
        type=str,
        default=None,
        help="Path to the neutral SMPL model .pkl file",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.55,
        help="Alpha used for semi-transparent mesh overlays",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Frames per second for the output MP4",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="FourCC codec for cv2.VideoWriter, e.g. mp4v or avc1",
    )
    parser.add_argument(
        "--sequence-id",
        type=str,
        default=None,
        help="Optional single sequence_id to visualize",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Optional cap on the number of output videos",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on the number of frames per sequence",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Use every Nth frame in each sequence",
    )
    parser.add_argument(
        "--pred-camera-mode",
        type=str,
        choices=("input", "gt"),
        default="input",
        help="Camera/root placement used for prediction overlays",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        choices=("first", "best_improvement"),
        default="first",
        help="How to choose which sequences to visualize",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        choices=("mpjpe", "pa_mpjpe"),
        default="mpjpe",
        help="Metric used when --selection-mode=best_improvement",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=None,
        help="Optional minimum average input-minus-pred improvement required for selection",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=1,
        help="Centered temporal smoothing window for predicted SMPL params; 1 disables smoothing",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional override for experiment seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_step < 1:
        raise ValueError(f"--frame-step must be >= 1, got {args.frame_step}")
    if args.fps <= 0:
        raise ValueError(f"--fps must be > 0, got {args.fps}")
    if args.smoothing_window < 1:
        raise ValueError(f"--smoothing-window must be >= 1, got {args.smoothing_window}")

    experiment = load_experiment_config(args.config)
    if not is_stage2_experiment(experiment):
        validate_mhr_asset_folder(args.mhr_assets_dir or "/opt/data/assets")

    data_config = build_data_config(experiment["data"], args)
    model_config = build_model_config(
        experiment["model"],
        checkpoint_path=args.checkpoint_path,
    )
    data_config.batch_size = 1
    data_config.drop_last_train = False
    if args.seed is not None:
        data_config.seed = args.seed

    datamodule = build_datamodule(data_config)
    datamodule.setup(None)
    dataset = select_dataset(datamodule, args.stage)

    device = resolve_device(args.device)
    module = load_visualization_module(
        checkpoint_path=args.checkpoint_path,
        model_config=model_config,
        data_config=data_config,
        args=args,
    )
    module.eval()
    module.to(device)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(data_config.manifest_path).resolve().parent
    rgb_dir = resolve_required_dir(args.rgb_dir, fallback=data_root / "rgb", name="rgb")
    cameras_dir = Path(data_config.cameras_dir).resolve()
    smpl_model_path = resolve_smpl_model_path(args.smpl_model_path)
    smpl_model = build_smpl_model(
        device=device,
        smpl_model_path=str(smpl_model_path),
        batch_size=1,
    )
    faces = np.asarray(smpl_model.faces, dtype=np.int32)
    overlay_renderer = CameraMeshOverlayRenderer()

    try:
        grouped_indices = build_sequence_index(dataset)
        selected_sequence_ids = sorted(grouped_indices)
        if args.sequence_id is not None:
            if args.sequence_id not in grouped_indices:
                raise KeyError(
                    f"sequence_id '{args.sequence_id}' was not found in the selected {args.stage} split"
                )
            selected_sequence_ids = [args.sequence_id]
        else:
            selected_sequence_ids = select_sequence_ids(
                dataset=dataset,
                grouped_indices=grouped_indices,
                selected_sequence_ids=selected_sequence_ids,
                module=module,
                device=device,
                selection_mode=args.selection_mode,
                selection_metric=args.selection_metric,
                min_improvement=args.min_improvement,
                frame_step=args.frame_step,
                max_frames=args.max_frames,
                max_sequences=args.max_sequences,
                input_smpl_cache_dir=Path(resolve_input_smpl_cache_dir(args, data_config)),
            )

        summaries = []
        for sequence_id in selected_sequence_ids:
            frame_indices = grouped_indices[sequence_id][:: args.frame_step]
            if args.max_frames is not None:
                frame_indices = frame_indices[: args.max_frames]
            if not frame_indices:
                continue

            reference_batch = multiview_collate([dataset[frame_indices[0]]])
            reference_meta = reference_batch["meta"][0]
            reference_image_path = resolve_rgb_image_path(
                rgb_dir,
                sequence_id=sequence_id,
                camera_id=str(reference_meta["camera_ids"][0]),
                frame_id=str(reference_meta["frame_id"]),
            )
            reference_image = cv2.imread(str(reference_image_path), cv2.IMREAD_COLOR)
            if reference_image is None:
                raise FileNotFoundError(f"Failed to read RGB image: {reference_image_path}")
            fixed_view_height, fixed_view_width = reference_image.shape[:2]
            sequence_fixed_view_scene = build_sequence_fixed_view_scene(
                dataset=dataset,
                sequence_frame_indices=frame_indices,
                module=module,
                smpl_model=smpl_model,
                device=device,
                cameras_dir=cameras_dir,
                width=fixed_view_width,
                height=fixed_view_height,
                smoothing_window=args.smoothing_window,
                input_smpl_cache_dir=Path(resolve_input_smpl_cache_dir(args, data_config)),
            )

            video_path = output_dir / f"{sequence_id}.mp4"
            writer: cv2.VideoWriter | None = None
            frame_width: int | None = None
            frame_height: int | None = None
            written_frames = 0

            for dataset_index in frame_indices:
                batch = multiview_collate([dataset[dataset_index]])
                frame_image, frame_summary = build_video_frame(
                    batch=batch,
                    module=module,
                    smpl_model=smpl_model,
                    faces=faces,
                    device=device,
                    rgb_dir=rgb_dir,
                    cameras_dir=cameras_dir,
                    overlay_alpha=args.overlay_alpha,
                    pred_camera_mode=args.pred_camera_mode,
                    overlay_renderer=overlay_renderer,
                    dataset=dataset,
                    sequence_frame_indices=frame_indices,
                    dataset_index=dataset_index,
                    smoothing_window=args.smoothing_window,
                    input_smpl_cache_dir=Path(resolve_input_smpl_cache_dir(args, data_config)),
                    fixed_view_scene_spec=sequence_fixed_view_scene,
                )
                if writer is None:
                    frame_height, frame_width = frame_image.shape[:2]
                    writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*args.codec),
                        args.fps,
                        (frame_width, frame_height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Failed to open video writer for {video_path} with codec '{args.codec}'"
                        )
                else:
                    assert frame_width is not None and frame_height is not None
                    if frame_image.shape[1] != frame_width or frame_image.shape[0] != frame_height:
                        frame_image = cv2.resize(
                            frame_image,
                            (frame_width, frame_height),
                            interpolation=cv2.INTER_LINEAR,
                        )
                writer.write(frame_image)
                written_frames += 1

            if writer is not None:
                writer.release()

            summaries.append(
                {
                    "sequence_id": sequence_id,
                    "video_path": str(video_path),
                    "num_frames": written_frames,
                    "stage": args.stage,
                    "fps": args.fps,
                }
            )
            print(f"Saved {written_frames} frames to {video_path}")
    finally:
        overlay_renderer.close()

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({"videos": summaries}, indent=2), encoding="utf-8")
    print(f"Saved summary to {summary_path}")


def build_data_config(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> Stage1DataConfig | Stage2DataConfig:
    data_kwargs = dict(config)
    data_name = str(data_kwargs.pop("name", "humman_stage1"))
    data_kwargs.pop("_config_path", None)

    if args.manifest_path is not None:
        data_kwargs["manifest_path"] = args.manifest_path
    if args.gt_smpl_dir is not None:
        data_kwargs["gt_smpl_dir"] = args.gt_smpl_dir
    if args.cameras_dir is not None:
        data_kwargs["cameras_dir"] = args.cameras_dir
    if args.split_config_path is not None:
        data_kwargs["split_config_path"] = args.split_config_path
    if args.split_name is not None:
        data_kwargs["split_name"] = args.split_name
    if args.seed is not None:
        data_kwargs["seed"] = args.seed
    if data_name == "humman_stage2":
        if args.input_smpl_cache_dir is not None:
            data_kwargs["input_smpl_cache_dir"] = args.input_smpl_cache_dir
        return Stage2DataConfig(**data_kwargs)

    return Stage1DataConfig(**data_kwargs)


def build_model_config(
    config: dict[str, Any],
    *,
    checkpoint_path: str,
) -> Stage1MLPFusionConfig | Stage1ResidualFusionConfig | Stage2ParamRefineConfig:
    model_kwargs = dict(config)
    model_name = str(model_kwargs.pop("name", "stage1_mlp_fusion"))
    model_kwargs.pop("_config_path", None)
    model_kwargs.update(load_checkpoint_model_overrides(checkpoint_path))
    if model_name == "stage2_param_refine":
        return Stage2ParamRefineConfig(**model_kwargs)
    if model_name == "stage1_residual_fusion":
        return Stage1ResidualFusionConfig(**model_kwargs)
    return Stage1MLPFusionConfig(**model_kwargs)


def load_checkpoint_model_overrides(checkpoint_path: str) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint.get("hyper_parameters", {}).get("model_config", {})
    if not isinstance(model_config, dict):
        return {}
    return dict(model_config)


def select_dataset(datamodule, stage: str):
    if stage == "train":
        if datamodule.train_dataset is None:
            raise RuntimeError("train_dataset was not initialized")
        return datamodule.train_dataset
    if stage == "val":
        if datamodule.val_dataset is None:
            raise RuntimeError("val_dataset was not initialized")
        return datamodule.val_dataset
    if datamodule.test_dataset is None:
        raise RuntimeError("test_dataset was not initialized")
    return datamodule.test_dataset


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_required_dir(path_arg: str | None, *, fallback: Path, name: str) -> Path:
    resolved = Path(path_arg).resolve() if path_arg is not None else fallback.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{name} directory does not exist: {resolved}")
    return resolved


def resolve_smpl_model_path(path_arg: str | None) -> Path:
    return resolve_smpl_model_path_impl(path_arg)


def resolve_input_smpl_cache_dir(
    args: argparse.Namespace,
    data_config: Stage1DataConfig | Stage2DataConfig,
) -> str:
    if args.input_smpl_cache_dir is not None:
        return str(Path(args.input_smpl_cache_dir).resolve())
    manifest_parent = Path(data_config.manifest_path).resolve().parent
    return str((manifest_parent / "sam3dbody_fitted_smpl").resolve())


def resolve_input_view_smpl_params(
    *,
    module,
    views_input: torch.Tensor,
    pred_cam_t: torch.Tensor,
    batch_meta: list[dict[str, Any]],
    input_smpl_cache_dir: Path,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if isinstance(module, Stage1FusionLightningModule):
        return module.convert_input_views_to_smpl(
            views_input=views_input,
            pred_cam_t=pred_cam_t,
            batch_meta=batch_meta,
        )
    return load_cached_input_view_smpl(
        batch_meta=batch_meta,
        input_smpl_cache_dir=input_smpl_cache_dir,
        device=device,
    )


def load_cached_input_view_smpl(
    *,
    batch_meta: list[dict[str, Any]],
    input_smpl_cache_dir: Path,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    stacked: dict[str, list[torch.Tensor]] = {
        "body_pose": [],
        "betas": [],
        "global_orient": [],
        "transl": [],
    }
    for sample_meta in batch_meta:
        sample_paths = sample_meta.get("view_npz_paths")
        if not isinstance(sample_paths, list):
            raise KeyError("Expected batch meta to contain 'view_npz_paths'")
        for sample_path in sample_paths:
            cache_path = cache_path_for_source_npz(input_smpl_cache_dir, str(sample_path))
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"Cached fitted SMPL file does not exist: {cache_path}. "
                    "Run scripts/precompute_input_smpl.py first."
                )
            with np.load(cache_path, allow_pickle=False) as payload:
                for key in stacked:
                    stacked[key].append(
                        torch.from_numpy(np.asarray(payload[key], dtype=np.float32))
                    )
    return {
        key: torch.stack(values, dim=0).to(device=device, dtype=torch.float32)
        for key, values in stacked.items()
    }


def expand_per_view_tensor(tensor: torch.Tensor, *, num_views: int) -> torch.Tensor:
    return (
        tensor[:, None, :]
        .expand(tensor.shape[0], num_views, tensor.shape[-1])
        .reshape(tensor.shape[0] * num_views, tensor.shape[-1])
    )


def compute_prediction_joint_metrics(
    *,
    module,
    pred_body_pose: torch.Tensor,
    pred_betas: torch.Tensor,
    target_body_pose: torch.Tensor,
    target_betas: torch.Tensor,
    pred_cam_t: torch.Tensor,
    input_view_smpl_params: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    batch_size = pred_body_pose.shape[0]
    num_views = pred_cam_t.shape[1]
    repeated_pred_body_pose = expand_per_view_tensor(pred_body_pose, num_views=num_views)
    repeated_pred_betas = expand_per_view_tensor(pred_betas, num_views=num_views)
    repeated_target_body_pose = expand_per_view_tensor(target_body_pose, num_views=num_views)
    repeated_target_betas = expand_per_view_tensor(target_betas, num_views=num_views)
    input_global_orient = input_view_smpl_params["global_orient"].to(pred_body_pose.device)
    input_transl = pred_cam_t.reshape(batch_size * num_views, 3).to(pred_body_pose.device)
    pred_joints = module._build_smpl_joints(
        body_pose=repeated_pred_body_pose,
        betas=repeated_pred_betas,
        global_orient=input_global_orient,
        transl=input_transl,
    )
    target_joints = module._build_smpl_joints(
        body_pose=repeated_target_body_pose,
        betas=repeated_target_betas,
        global_orient=input_global_orient,
        transl=input_transl,
    )
    return {
        "mpjpe": batch_mpjpe(pred_joints, target_joints),
        "pa_mpjpe": batch_pa_mpjpe(pred_joints, target_joints),
    }


def compute_input_joint_metrics(
    *,
    module,
    input_view_smpl_params: dict[str, torch.Tensor],
    target_body_pose: torch.Tensor,
    target_betas: torch.Tensor,
    pred_cam_t: torch.Tensor,
) -> dict[str, torch.Tensor]:
    batch_size = target_body_pose.shape[0]
    num_views = pred_cam_t.shape[1]
    repeated_target_body_pose = expand_per_view_tensor(target_body_pose, num_views=num_views)
    repeated_target_betas = expand_per_view_tensor(target_betas, num_views=num_views)
    input_global_orient = input_view_smpl_params["global_orient"].to(target_body_pose.device)
    input_transl = pred_cam_t.reshape(batch_size * num_views, 3).to(target_body_pose.device)
    pred_joints = module._build_smpl_joints(
        body_pose=input_view_smpl_params["body_pose"].to(target_body_pose.device),
        betas=input_view_smpl_params["betas"].to(target_betas.device),
        global_orient=input_global_orient,
        transl=input_transl,
    )
    target_joints = module._build_smpl_joints(
        body_pose=repeated_target_body_pose,
        betas=repeated_target_betas,
        global_orient=input_global_orient,
        transl=input_transl,
    )
    return {
        "mpjpe": batch_mpjpe(pred_joints, target_joints),
        "pa_mpjpe": batch_pa_mpjpe(pred_joints, target_joints),
    }


def is_stage2_experiment(experiment: dict[str, Any]) -> bool:
    model_name = str(experiment.get("model", {}).get("name", ""))
    data_name = str(experiment.get("data", {}).get("name", ""))
    return model_name == "stage2_param_refine" or data_name == "humman_stage2"


def build_datamodule(data_config: Stage1DataConfig | Stage2DataConfig):
    if isinstance(data_config, Stage2DataConfig):
        return Stage2HuMManDataModule(data_config)
    return Stage1HuMManDataModule(data_config)


def load_visualization_module(
    *,
    checkpoint_path: str,
    model_config: Stage1MLPFusionConfig | Stage1ResidualFusionConfig | Stage2ParamRefineConfig,
    data_config: Stage1DataConfig | Stage2DataConfig,
    args: argparse.Namespace,
):
    if isinstance(model_config, Stage2ParamRefineConfig):
        return Stage2FusionLightningModule.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
            model_config=model_config,
            smpl_model_path=args.smpl_model_path,
            strict=False,
        )
    return Stage1FusionLightningModule.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
        model_config=model_config,
        mhr_assets_dir=args.mhr_assets_dir,
        smpl_model_path=args.smpl_model_path,
        input_smpl_cache_dir=resolve_input_smpl_cache_dir(args, data_config),
        strict=False,
    )


def build_sequence_index(dataset) -> dict[str, list[int]]:
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for index, record in enumerate(dataset.records):
        grouped[record.sequence_id].append((int(record.frame_id), index))
    return {
        sequence_id: [index for _, index in sorted(items)]
        for sequence_id, items in grouped.items()
    }


def select_sequence_ids(
    *,
    dataset,
    grouped_indices: dict[str, list[int]],
    selected_sequence_ids: list[str],
    module,
    device: torch.device,
    selection_mode: str,
    selection_metric: str,
    min_improvement: float | None,
    frame_step: int,
    max_frames: int | None,
    max_sequences: int | None,
    input_smpl_cache_dir: Path,
) -> list[str]:
    if selection_mode == "first":
        if max_sequences is not None:
            return selected_sequence_ids[:max_sequences]
        return selected_sequence_ids

    ranked_sequences = []
    with torch.no_grad():
        for sequence_id in selected_sequence_ids:
            frame_indices = grouped_indices[sequence_id][::frame_step]
            if max_frames is not None:
                frame_indices = frame_indices[:max_frames]
            if not frame_indices:
                continue
            improvements = []
            pred_values = []
            input_values = []
            for dataset_index in frame_indices:
                batch = multiview_collate([dataset[dataset_index]])
                score = compute_selection_score(
                    batch=batch,
                    module=module,
                    device=device,
                    metric_name=selection_metric,
                    input_smpl_cache_dir=input_smpl_cache_dir,
                )
                improvements.append(score["improvement"])
                pred_values.append(score["pred_metric"])
                input_values.append(score["input_metric"])
            mean_improvement = float(np.mean(improvements))
            if min_improvement is not None and mean_improvement < min_improvement:
                continue
            ranked_sequences.append(
                {
                    "sequence_id": sequence_id,
                    "improvement": mean_improvement,
                    "pred_metric": float(np.mean(pred_values)),
                    "input_metric": float(np.mean(input_values)),
                }
            )

    ranked_sequences.sort(key=lambda item: item["improvement"], reverse=True)
    if max_sequences is not None:
        ranked_sequences = ranked_sequences[:max_sequences]
    return [item["sequence_id"] for item in ranked_sequences]


def compute_selection_score(
    *,
    batch: dict[str, Any],
    module,
    device: torch.device,
    metric_name: str,
    input_smpl_cache_dir: Path,
) -> dict[str, float]:
    views_input = batch["views_input"].to(device)
    pred_cam_t = batch["view_aux"]["pred_cam_t"].to(device)
    predictions = module(views_input)
    input_view_smpl_params = resolve_input_view_smpl_params(
        module=module,
        views_input=views_input,
        pred_cam_t=pred_cam_t,
        batch_meta=batch["meta"],
        input_smpl_cache_dir=input_smpl_cache_dir,
        device=device,
    )
    pred_metrics = compute_prediction_joint_metrics(
        module=module,
        pred_body_pose=predictions["pred_body_pose"],
        pred_betas=predictions["pred_betas"],
        target_body_pose=batch["target_body_pose"].to(device),
        target_betas=batch["target_betas"].to(device),
        pred_cam_t=pred_cam_t,
        input_view_smpl_params=input_view_smpl_params,
    )
    input_metrics = compute_input_joint_metrics(
        module=module,
        input_view_smpl_params=input_view_smpl_params,
        target_body_pose=batch["target_body_pose"].to(device),
        target_betas=batch["target_betas"].to(device),
        pred_cam_t=pred_cam_t,
    )
    pred_value = float(pred_metrics[metric_name].detach().cpu().item())
    input_value = float(input_metrics[metric_name].detach().cpu().item())
    return {
        "pred_metric": pred_value,
        "input_metric": input_value,
        "improvement": input_value - pred_value,
    }


def build_video_frame(
    *,
    batch: dict[str, Any],
    module,
    smpl_model,
    faces: np.ndarray,
    device: torch.device,
    rgb_dir: Path,
    cameras_dir: Path,
    overlay_alpha: float,
    pred_camera_mode: str,
    overlay_renderer: "CameraMeshOverlayRenderer",
    dataset,
    sequence_frame_indices: list[int],
    dataset_index: int,
    smoothing_window: int,
    input_smpl_cache_dir: Path,
    fixed_view_scene_spec: dict[str, np.ndarray | float],
) -> tuple[np.ndarray, dict[str, Any]]:
    meta = batch["meta"][0]
    sample_id = str(meta["sample_id"])
    sequence_id = str(meta["sequence_id"])
    frame_id = str(meta["frame_id"])

    views_input = batch["views_input"].to(device)
    pred_cam_t = batch["view_aux"]["pred_cam_t"].to(device)
    with torch.no_grad():
        predictions = module(views_input)
    input_view_smpl = resolve_input_view_smpl_params(
        module=module,
        views_input=views_input,
        pred_cam_t=pred_cam_t,
        batch_meta=batch["meta"],
        input_smpl_cache_dir=input_smpl_cache_dir,
        device=device,
    )

    pred_body_pose, pred_betas = get_visualization_prediction(
        batch=batch,
        predictions=predictions,
        module=module,
        dataset=dataset,
        sequence_frame_indices=sequence_frame_indices,
        dataset_index=dataset_index,
        device=device,
        smoothing_window=smoothing_window,
    )
    gt_body_pose = batch["target_body_pose"][0].detach().cpu().numpy()
    gt_betas = batch["target_betas"][0].detach().cpu().numpy()
    gt_world_global_orient = batch["target_aux"]["global_orient"][0].detach().cpu().numpy()
    gt_world_transl = batch["target_aux"]["transl"][0].detach().cpu().numpy()

    input_body_pose = input_view_smpl["body_pose"].detach().cpu().numpy()
    input_betas = input_view_smpl["betas"].detach().cpu().numpy()
    input_global_orient = input_view_smpl["global_orient"].detach().cpu().numpy()
    input_transl = input_view_smpl["transl"].detach().cpu().numpy()
    gt_camera_global_orient = batch["target_aux"]["camera_global_orient"][0].detach().cpu().numpy()
    gt_camera_transl = batch["target_aux"]["camera_transl"][0].detach().cpu().numpy()

    gt_world_vertices, gt_world_joints = build_smpl_outputs(
        smpl_model=smpl_model,
        device=device,
        body_pose=gt_body_pose,
        betas=gt_betas,
        global_orient=gt_world_global_orient,
        transl=gt_world_transl,
    )
    pred_world_vertices, pred_world_joints = build_smpl_outputs(
        smpl_model=smpl_model,
        device=device,
        body_pose=pred_body_pose,
        betas=pred_betas,
        global_orient=gt_world_global_orient,
        transl=gt_world_transl,
    )
    view_panels: list[np.ndarray] = []
    per_view_summary: list[dict[str, Any]] = []
    reference_image_path = resolve_rgb_image_path(
        rgb_dir,
        sequence_id=sequence_id,
        camera_id=str(meta["camera_ids"][0]),
        frame_id=frame_id,
    )
    reference_image = cv2.imread(str(reference_image_path), cv2.IMREAD_COLOR)
    if reference_image is None:
        raise FileNotFoundError(f"Failed to read RGB image: {reference_image_path}")
    fixed_view_height, fixed_view_width = reference_image.shape[:2]
    input_fixed_view_colors = (
        (0, 150, 225),
        (45, 180, 240),
        (0, 130, 200),
        (75, 195, 245),
    )
    fixed_view_meshes: list[tuple[str, np.ndarray, np.ndarray, tuple[int, int, int]]] = []
    for view_index, camera_id in enumerate(meta["camera_ids"]):
        input_vertices_camera, input_joints_camera = build_smpl_outputs(
            smpl_model=smpl_model,
            device=device,
            body_pose=input_body_pose[view_index],
            betas=input_betas[view_index],
            global_orient=input_global_orient[view_index],
            transl=input_transl[view_index],
        )
        camera = load_camera_parameters(
            cameras_dir,
            sequence_id=sequence_id,
            camera_id=str(camera_id),
        )
        input_vertices_world = transform_camera_vertices_to_world(
            vertices_camera=input_vertices_camera,
            rotation=camera.rotation,
            translation=camera.translation,
        )
        input_joints_world = transform_camera_vertices_to_world(
            vertices_camera=input_joints_camera,
            rotation=camera.rotation,
            translation=camera.translation,
        )
        fixed_view_meshes.append(
            (
                f"View {view_index + 1} Input",
                input_vertices_world,
                input_joints_world,
                input_fixed_view_colors[view_index % len(input_fixed_view_colors)],
            )
        )
    fixed_view_meshes.extend(
        [
            ("Unified Prediction", pred_world_vertices, pred_world_joints, (55, 85, 235)),
            ("Unified GT", gt_world_vertices, gt_world_joints, (85, 200, 95)),
        ]
    )
    view_panels.append(
        build_view_panel(
            [
                (
                    title,
                    overlay_renderer.render_fixed_view(
                        vertices=vertices,
                        joints=joints,
                        faces=faces,
                        color_bgr=color_bgr,
                        width=fixed_view_width,
                        height=fixed_view_height,
                        scene_spec=fixed_view_scene_spec,
                    ),
                )
                for title, vertices, joints, color_bgr in fixed_view_meshes
            ],
            footer="",
        )
    )

    for view_index, camera_id in enumerate(meta["camera_ids"]):
        input_color_bgr = input_fixed_view_colors[view_index % len(input_fixed_view_colors)]
        image_path = resolve_rgb_image_path(
            rgb_dir,
            sequence_id=sequence_id,
            camera_id=str(camera_id),
            frame_id=frame_id,
        )
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read RGB image: {image_path}")

        pred_cam_int = batch["view_aux"]["cam_int"][0, view_index].detach().cpu().numpy()
        gt_camera = load_camera_parameters(
            cameras_dir,
            sequence_id=sequence_id,
            camera_id=str(camera_id),
        )

        input_vertices_camera, input_joints_camera = build_smpl_outputs(
            smpl_model=smpl_model,
            device=device,
            body_pose=input_body_pose[view_index],
            betas=input_betas[view_index],
            global_orient=input_global_orient[view_index],
            transl=input_transl[view_index],
        )
        gt_vertices_camera = transform_world_vertices_to_camera(
            vertices_world=gt_world_vertices,
            rotation=gt_camera.rotation,
            translation=gt_camera.translation,
        )
        if pred_camera_mode == "gt":
            pred_vertices_camera = transform_world_vertices_to_camera(
                vertices_world=pred_world_vertices,
                rotation=gt_camera.rotation,
                translation=gt_camera.translation,
            )
            pred_overlay_intrinsics = gt_camera.intrinsics
            pred_translation_offset = np.zeros(3, dtype=np.float32)
        else:
            pred_vertices_oriented, pred_joints_oriented = build_smpl_outputs(
                smpl_model=smpl_model,
                device=device,
                body_pose=pred_body_pose,
                betas=pred_betas,
                global_orient=input_global_orient[view_index],
                transl=np.zeros(3, dtype=np.float32),
            )
            pred_translation_offset = input_joints_camera[0] - pred_joints_oriented[0]
            pred_vertices_camera = np.ascontiguousarray(
                pred_vertices_oriented + pred_translation_offset.reshape(1, 3)
            )
            pred_overlay_intrinsics = pred_cam_int

        input_overlay, input_mask = overlay_renderer.render_overlay(
            image_bgr=image_bgr,
            vertices_camera=input_vertices_camera,
            faces=faces,
            intrinsics=pred_cam_int,
            color_bgr=input_color_bgr,
            alpha=overlay_alpha,
        )
        pred_overlay, pred_mask = overlay_renderer.render_overlay(
            image_bgr=image_bgr,
            vertices_camera=pred_vertices_camera,
            faces=faces,
            intrinsics=pred_overlay_intrinsics,
            color_bgr=(40, 70, 230),
            alpha=overlay_alpha,
        )
        gt_overlay, gt_mask = overlay_renderer.render_overlay(
            image_bgr=image_bgr,
            vertices_camera=gt_vertices_camera,
            faces=faces,
            intrinsics=gt_camera.intrinsics,
            color_bgr=(70, 210, 70),
            alpha=overlay_alpha,
        )
        view_panels.append(
            build_view_panel(
                [
                    (f"View {view_index + 1} RGB", image_bgr),
                    (f"View {view_index + 1} Input", input_overlay),
                    ("Unified Prediction", pred_overlay),
                    ("Unified GT", gt_overlay),
                ],
                footer="",
            )
        )
        per_view_summary.append(
            {
                "camera_id": str(camera_id),
                "image_path": str(image_path),
                "input_visible": bool(input_mask.any()),
                "pred_visible": bool(pred_mask.any()),
                "gt_visible": bool(gt_mask.any()),
                "pred_camera_mode": pred_camera_mode,
                "pred_translation_offset": pred_translation_offset.astype(np.float32).tolist(),
            }
        )

    frame_image = build_contact_sheet(
        title_lines=[
            f"sequence_id: {sequence_id}",
            f"smoothing_window: {smoothing_window}",
        ],
        view_panels=view_panels,
    )
    return frame_image, {
        "sample_id": sample_id,
        "sequence_id": sequence_id,
        "frame_id": frame_id,
        "pred_camera_mode": pred_camera_mode,
        "smoothing_window": smoothing_window,
        "views": per_view_summary,
    }


def get_visualization_prediction(
    *,
    batch: dict[str, Any],
    predictions: dict[str, torch.Tensor],
    module,
    dataset,
    sequence_frame_indices: list[int],
    dataset_index: int,
    device: torch.device,
    smoothing_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    if smoothing_window <= 1:
        return (
            predictions["pred_body_pose"][0].detach().cpu().numpy(),
            predictions["pred_betas"][0].detach().cpu().numpy(),
        )

    center_position = sequence_frame_indices.index(dataset_index)
    window_indices = select_centered_window_indices(
        sequence_indices=sequence_frame_indices,
        center_position=center_position,
        window_size=smoothing_window,
    )

    pred_body_pose_values = []
    pred_betas_values = []
    with torch.no_grad():
        for neighbor_index in window_indices:
            neighbor_batch = multiview_collate([dataset[neighbor_index]])
            neighbor_predictions = module(neighbor_batch["views_input"].to(device))
            pred_body_pose_values.append(neighbor_predictions["pred_body_pose"][0].detach().cpu().numpy())
            pred_betas_values.append(neighbor_predictions["pred_betas"][0].detach().cpu().numpy())

    return (
        np.mean(np.stack(pred_body_pose_values, axis=0), axis=0).astype(np.float32, copy=False),
        np.mean(np.stack(pred_betas_values, axis=0), axis=0).astype(np.float32, copy=False),
    )


def build_sequence_fixed_view_scene(
    *,
    dataset,
    sequence_frame_indices: list[int],
    module,
    smpl_model,
    device: torch.device,
    width: int,
    height: int,
    cameras_dir: Path,
    smoothing_window: int,
    input_smpl_cache_dir: Path,
) -> dict[str, np.ndarray | float]:
    sequence_vertices: list[np.ndarray] = []
    with torch.no_grad():
        for dataset_index in sequence_frame_indices:
            batch = multiview_collate([dataset[dataset_index]])
            views_input = batch["views_input"].to(device)
            pred_cam_t = batch["view_aux"]["pred_cam_t"].to(device)
            predictions = module(views_input)
            input_view_smpl = resolve_input_view_smpl_params(
                module=module,
                views_input=views_input,
                pred_cam_t=pred_cam_t,
                batch_meta=batch["meta"],
                input_smpl_cache_dir=input_smpl_cache_dir,
                device=device,
            )
            pred_body_pose, pred_betas = get_visualization_prediction(
                batch=batch,
                predictions=predictions,
                module=module,
                dataset=dataset,
                sequence_frame_indices=sequence_frame_indices,
                dataset_index=dataset_index,
                device=device,
                smoothing_window=smoothing_window,
            )
            gt_body_pose = batch["target_body_pose"][0].detach().cpu().numpy()
            gt_betas = batch["target_betas"][0].detach().cpu().numpy()
            input_body_pose = input_view_smpl["body_pose"].detach().cpu().numpy()
            input_betas = input_view_smpl["betas"].detach().cpu().numpy()
            sequence_id = str(batch["meta"][0]["sequence_id"])
            camera_ids = [str(camera_id) for camera_id in batch["meta"][0]["camera_ids"]]

            for view_index in range(input_body_pose.shape[0]):
                input_vertices_camera, input_joints_camera = build_smpl_outputs(
                    smpl_model=smpl_model,
                    device=device,
                    body_pose=input_body_pose[view_index],
                    betas=input_betas[view_index],
                    global_orient=input_view_smpl["global_orient"][view_index].detach().cpu().numpy(),
                    transl=input_view_smpl["transl"][view_index].detach().cpu().numpy(),
                )
                camera = load_camera_parameters(
                    cameras_dir,
                    sequence_id=sequence_id,
                    camera_id=camera_ids[view_index],
                )
                input_vertices_world = transform_camera_vertices_to_world(
                    vertices_camera=input_vertices_camera,
                    rotation=camera.rotation,
                    translation=camera.translation,
                )
                sequence_vertices.append(input_vertices_world)

            pred_world_vertices, _ = build_smpl_outputs(
                smpl_model=smpl_model,
                device=device,
                body_pose=pred_body_pose,
                betas=pred_betas,
                global_orient=batch["target_aux"]["global_orient"][0].detach().cpu().numpy(),
                transl=batch["target_aux"]["transl"][0].detach().cpu().numpy(),
            )
            gt_world_vertices, _ = build_smpl_outputs(
                smpl_model=smpl_model,
                device=device,
                body_pose=gt_body_pose,
                betas=gt_betas,
                global_orient=batch["target_aux"]["global_orient"][0].detach().cpu().numpy(),
                transl=batch["target_aux"]["transl"][0].detach().cpu().numpy(),
            )
            sequence_vertices.extend((pred_world_vertices, gt_world_vertices))

    return build_fixed_view_scene(
        sequence_vertices,
        width=width,
        height=height,
    )


def select_centered_window_indices(
    *,
    sequence_indices: list[int],
    center_position: int,
    window_size: int,
) -> list[int]:
    half = window_size // 2
    start = max(0, center_position - half)
    end = min(len(sequence_indices), start + window_size)
    start = max(0, end - window_size)
    return sequence_indices[start:end]


def build_smpl_vertices(
    *,
    smpl_model,
    device: torch.device,
    body_pose: np.ndarray,
    betas: np.ndarray,
    global_orient: np.ndarray,
    transl: np.ndarray,
) -> np.ndarray:
    body_pose_tensor = torch.as_tensor(body_pose, dtype=torch.float32, device=device).view(1, -1)
    betas_tensor = torch.as_tensor(betas, dtype=torch.float32, device=device).view(1, -1)
    global_orient_tensor = torch.as_tensor(global_orient, dtype=torch.float32, device=device).view(
        1, -1
    )
    transl_tensor = torch.as_tensor(transl, dtype=torch.float32, device=device).view(1, -1)
    output = smpl_model(
        body_pose=body_pose_tensor,
        betas=betas_tensor,
        global_orient=global_orient_tensor,
        transl=transl_tensor,
    )
    return output.vertices[0].detach().cpu().numpy()


def build_smpl_outputs(
    *,
    smpl_model,
    device: torch.device,
    body_pose: np.ndarray,
    betas: np.ndarray,
    global_orient: np.ndarray,
    transl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    body_pose_tensor = torch.as_tensor(body_pose, dtype=torch.float32, device=device).view(1, -1)
    betas_tensor = torch.as_tensor(betas, dtype=torch.float32, device=device).view(1, -1)
    global_orient_tensor = torch.as_tensor(global_orient, dtype=torch.float32, device=device).view(
        1, -1
    )
    transl_tensor = torch.as_tensor(transl, dtype=torch.float32, device=device).view(1, -1)
    output = smpl_model(
        body_pose=body_pose_tensor,
        betas=betas_tensor,
        global_orient=global_orient_tensor,
        transl=transl_tensor,
    )
    return (
        output.vertices[0].detach().cpu().numpy(),
        output.joints[0].detach().cpu().numpy(),
    )

def transform_world_vertices_to_camera(
    *,
    vertices_world: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    vertices_world = np.asarray(vertices_world, dtype=np.float32)
    rotation = np.asarray(rotation, dtype=np.float32)
    translation = np.asarray(translation, dtype=np.float32).reshape(1, 3)
    return np.ascontiguousarray((rotation @ vertices_world.T).T + translation)


def transform_camera_vertices_to_world(
    *,
    vertices_camera: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    vertices_camera = np.asarray(vertices_camera, dtype=np.float32)
    rotation = np.asarray(rotation, dtype=np.float32)
    translation = np.asarray(translation, dtype=np.float32).reshape(1, 3)
    return np.ascontiguousarray(((rotation.T @ (vertices_camera - translation).T).T))


class CameraMeshOverlayRenderer:
    """Offscreen renderer for shaded camera-space mesh overlays."""

    def __init__(self) -> None:
        self._renderers: dict[tuple[int, int], pyrender.OffscreenRenderer] = {}

    def close(self) -> None:
        for renderer in self._renderers.values():
            try:
                renderer.delete()
            except Exception:
                pass
        self._renderers.clear()

    def render_overlay(
        self,
        *,
        image_bgr: np.ndarray,
        vertices_camera: np.ndarray,
        faces: np.ndarray,
        intrinsics: np.ndarray,
        color_bgr: tuple[int, int, int],
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = image_bgr.shape[:2]
        renderer = self._get_renderer(width=width, height=height)
        mesh = trimesh.Trimesh(
            np.asarray(vertices_camera, dtype=np.float32).copy(),
            np.asarray(faces, dtype=np.int32),
            process=False,
        )
        mesh.apply_transform(opencv_to_opengl_transform())
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.1,
            roughnessFactor=0.75,
            baseColorFactor=bgr_to_rgba(color_bgr),
            alphaMode="OPAQUE",
        )
        render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)

        scene = pyrender.Scene(
            bg_color=np.array([0, 0, 0, 0], dtype=np.uint8),
            ambient_light=np.array([0.35, 0.35, 0.35], dtype=np.float32),
        )
        scene.add(render_mesh)
        add_camera_lights(scene)
        camera = pyrender.IntrinsicsCamera(
            fx=float(intrinsics[0, 0]),
            fy=float(intrinsics[1, 1]),
            cx=float(intrinsics[0, 2]),
            cy=float(intrinsics[1, 2]),
        )
        scene.add(camera, pose=np.eye(4, dtype=np.float32))
        color_rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        return composite_rgba_on_image(image_bgr, color_rgba, alpha=alpha)

    def render_fixed_view(
        self,
        *,
        vertices: np.ndarray,
        joints: np.ndarray,
        faces: np.ndarray,
        color_bgr: tuple[int, int, int],
        width: int = 320,
        height: int = 320,
        scene_spec: dict[str, np.ndarray | float] | None = None,
    ) -> np.ndarray:
        renderer = self._get_renderer(width=width, height=height)
        posed_vertices, posed_joints, intrinsics = prepare_fixed_view_geometry(
            vertices=vertices,
            joints=joints,
            width=width,
            height=height,
            scene_spec=scene_spec,
        )
        mesh = trimesh.Trimesh(
            np.asarray(posed_vertices, dtype=np.float32).copy(),
            np.asarray(faces, dtype=np.int32),
            process=False,
        )
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.1,
            roughnessFactor=0.75,
            baseColorFactor=bgr_to_rgba(color_bgr),
            alphaMode="OPAQUE",
        )
        render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
        scene = pyrender.Scene(
            bg_color=np.array([255, 255, 255, 255], dtype=np.uint8),
            ambient_light=np.array([0.45, 0.45, 0.45], dtype=np.float32),
        )
        scene.add(render_mesh)
        add_camera_lights(scene)
        camera = pyrender.IntrinsicsCamera(
            fx=float(intrinsics[0, 0]),
            fy=float(intrinsics[1, 1]),
            cx=float(intrinsics[0, 2]),
            cy=float(intrinsics[1, 2]),
        )
        scene.add(camera, pose=np.eye(4, dtype=np.float32))
        color_rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color_rgb = color_rgba[..., :3].astype(np.uint8)
        color_bgr_image = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
        return draw_projected_skeleton(
            image_bgr=color_bgr_image,
            joints_camera=posed_joints,
            intrinsics=intrinsics,
            color_bgr=blend_color(color_bgr, target=(255, 255, 255), weight=0.35),
        )

    def _get_renderer(self, *, width: int, height: int) -> pyrender.OffscreenRenderer:
        key = (width, height)
        renderer = self._renderers.get(key)
        if renderer is None:
            renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
            self._renderers[key] = renderer
        return renderer


def opencv_to_opengl_transform() -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[1, 1] = -1.0
    transform[2, 2] = -1.0
    return transform

def add_camera_lights(scene: pyrender.Scene) -> None:
    light = pyrender.DirectionalLight(color=np.ones(3, dtype=np.float32), intensity=2.6)
    light_poses = (
        np.eye(4, dtype=np.float32),
        np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.8660254, -0.5, 0.0],
                [0.0, 0.5, 0.8660254, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        np.asarray(
            [
                [0.8660254, 0.0, 0.5, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-0.5, 0.0, 0.8660254, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    for pose in light_poses:
        scene.add(light, pose=pose)


def prepare_fixed_view_geometry(
    *,
    vertices: np.ndarray,
    joints: np.ndarray,
    width: int,
    height: int,
    scene_spec: dict[str, np.ndarray | float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float32)
    joints = np.asarray(joints, dtype=np.float32)
    if scene_spec is None:
        scene_spec = build_fixed_view_scene([vertices], width=width, height=height)
    center = np.asarray(scene_spec["center"], dtype=np.float32)
    rotation = np.asarray(scene_spec["rotation"], dtype=np.float32)
    intrinsics = np.asarray(scene_spec["intrinsics"], dtype=np.float32)
    depth = float(scene_spec["depth"])
    centered_vertices = vertices - center.reshape(1, 3)
    centered_joints = joints - center.reshape(1, 3)
    rotated_vertices = (rotation @ centered_vertices.T).T
    rotated_joints = (rotation @ centered_joints.T).T
    translated_vertices = rotated_vertices + np.array([0.0, 0.0, -depth], dtype=np.float32)
    translated_joints = rotated_joints + np.array([0.0, 0.0, -depth], dtype=np.float32)
    return translated_vertices, translated_joints, intrinsics


def build_fixed_view_scene(
    vertices_list: list[np.ndarray],
    *,
    width: int,
    height: int,
) -> dict[str, np.ndarray | float]:
    stacked_vertices = np.concatenate(
        [np.asarray(vertices, dtype=np.float32) for vertices in vertices_list],
        axis=0,
    )
    center = 0.5 * (np.min(stacked_vertices, axis=0) + np.max(stacked_vertices, axis=0))
    rotation = fixed_view_rotation_matrix()
    centered_vertices = stacked_vertices - center.reshape(1, 3)
    rotated_vertices = (rotation @ centered_vertices.T).T
    extent = np.max(rotated_vertices, axis=0) - np.min(rotated_vertices, axis=0)
    scale = max(float(np.max(extent)), 1e-3)
    depth = 1.75 * scale
    focal = 1.34 * min(width, height)
    intrinsics = np.array(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return {
        "center": center.astype(np.float32, copy=False),
        "rotation": rotation,
        "depth": depth,
        "intrinsics": intrinsics,
    }


def fixed_view_rotation_matrix() -> np.ndarray:
    # HuMMan world coordinates use a downward Y direction for body-up, so flip
    # the world frame first to keep world-space bodies upright in the fixed view.
    world_alignment = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )
    yaw = np.deg2rad(-28.0)
    pitch = np.deg2rad(12.0)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    rotation_y = np.array(
        [
            [cos_yaw, 0.0, sin_yaw],
            [0.0, 1.0, 0.0],
            [-sin_yaw, 0.0, cos_yaw],
        ],
        dtype=np.float32,
    )
    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_pitch, -sin_pitch],
            [0.0, sin_pitch, cos_pitch],
        ],
        dtype=np.float32,
    )
    return rotation_x @ rotation_y @ world_alignment


def draw_projected_skeleton(
    *,
    image_bgr: np.ndarray,
    joints_camera: np.ndarray,
    intrinsics: np.ndarray,
    color_bgr: tuple[int, int, int],
) -> np.ndarray:
    output = image_bgr.copy()
    projected = project_points_to_image(joints_camera=joints_camera, intrinsics=intrinsics)
    for joint_start, joint_end in SMPL_SKELETON_EDGES:
        start = projected[joint_start]
        end = projected[joint_end]
        if start is None or end is None:
            continue
        cv2.line(output, start, end, color_bgr, 2, cv2.LINE_AA)
    for point in projected:
        if point is None:
            continue
        cv2.circle(output, point, 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(output, point, 2, color_bgr, -1, cv2.LINE_AA)
    return output


def project_points_to_image(
    *,
    joints_camera: np.ndarray,
    intrinsics: np.ndarray,
) -> list[tuple[int, int] | None]:
    joints_camera = np.asarray(joints_camera, dtype=np.float32)[:24]
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    projected: list[tuple[int, int] | None] = []
    for joint in joints_camera:
        depth = float(-joint[2])
        if depth <= 1e-4:
            projected.append(None)
            continue
        x = float(joint[0] / -joint[2])
        y = float(joint[1] / -joint[2])
        pixel_x = intrinsics[0, 0] * x + intrinsics[0, 2]
        pixel_y = intrinsics[1, 2] - intrinsics[1, 1] * y
        projected.append((int(round(pixel_x)), int(round(pixel_y))))
    return projected


def blend_color(
    color_bgr: tuple[int, int, int],
    *,
    target: tuple[int, int, int],
    weight: float,
) -> tuple[int, int, int]:
    source = np.asarray(color_bgr, dtype=np.float32)
    target_array = np.asarray(target, dtype=np.float32)
    blended = source * (1.0 - weight) + target_array * weight
    return tuple(int(round(value)) for value in blended.tolist())


def composite_rgba_on_image(
    image_bgr: np.ndarray,
    color_rgba: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    render_rgb = color_rgba[..., :3].astype(np.float32)
    render_bgr = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR)
    render_alpha = (color_rgba[..., 3].astype(np.float32) / 255.0) * float(alpha)
    render_alpha = np.clip(render_alpha, 0.0, 1.0)
    output = image_bgr.astype(np.float32).copy()
    output *= 1.0 - render_alpha[..., None]
    output += render_bgr * render_alpha[..., None]
    return np.clip(output, 0.0, 255.0).astype(np.uint8), (render_alpha > 1e-4).astype(np.uint8)


def bgr_to_rgba(color_bgr: tuple[int, int, int]) -> tuple[float, float, float, float]:
    b, g, r = color_bgr
    return (r / 255.0, g / 255.0, b / 255.0, 1.0)


def build_view_panel(items: list[tuple[str, np.ndarray]], *, footer: str) -> np.ndarray:
    tile_height, tile_width = items[0][1].shape[:2]
    header_height = 34
    footer_height = 28 if footer else 0
    tile_gap = 12
    panel_width = tile_width * len(items) + tile_gap * max(len(items) - 1, 0)
    panel = np.full(
        (tile_height + header_height + footer_height, panel_width, 3),
        255,
        dtype=np.uint8,
    )

    for item_index, (title, image) in enumerate(items):
        x0 = item_index * (tile_width + tile_gap)
        panel[header_height : header_height + tile_height, x0 : x0 + tile_width] = image
        cv2.rectangle(
            panel,
            (x0, header_height),
            (x0 + tile_width - 1, header_height + tile_height - 1),
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            title,
            (x0 + 12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    if footer:
        cv2.putText(
            panel,
            footer,
            (12, header_height + tile_height + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (50, 50, 50),
            1,
            cv2.LINE_AA,
        )
    return panel


def build_contact_sheet(*, title_lines: list[str], view_panels: list[np.ndarray]) -> np.ndarray:
    margin = 24
    text_height = 28
    panel_gap = 18
    width = max(panel.shape[1] for panel in view_panels) + 2 * margin
    title_height = margin + len(title_lines) * text_height + 8
    total_height = title_height + sum(panel.shape[0] for panel in view_panels)
    total_height += panel_gap * max(len(view_panels) - 1, 0) + margin

    canvas = np.full((total_height, width, 3), 255, dtype=np.uint8)
    y = margin + 4
    for line in title_lines:
        cv2.putText(
            canvas,
            line,
            (margin, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        y += text_height

    y += 8
    for panel in view_panels:
        x = max((width - panel.shape[1]) // 2, 0)
        canvas[y : y + panel.shape[0], x : x + panel.shape[1]] = panel
        y += panel.shape[0] + panel_gap
    return canvas


if __name__ == "__main__":
    main()
