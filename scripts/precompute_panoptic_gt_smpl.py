#!/usr/bin/env python
"""Fit SMPL target caches from cropped Panoptic/Kinoptic 3D joints.

Panoptic `gt3d/*.npy` files store coco19 3D joints, not SMPL parameters. This
script creates the SMPL-compatible supervision cache expected by the Stage 1
Panoptic dataset:

    <output-dir>/<sequence_id>_smpl_params.npz

Each cache contains `frame_ids`, `global_orient`, `body_pose`, `betas`, and
`transl`. The fitted targets are an approximate pseudo-GT bridge for running the
current SMPL-parameter fusion baseline on Panoptic.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mvhpe3d.utils import PANOPTIC_TO_SMPL24, build_smpl_model

DEFAULT_DATASET_ROOT = Path("/opt/data/panoptic_kinoptic_single_actor_cropped")


@dataclass(frozen=True)
class PanopticFrame:
    frame_id: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit approximate SMPL supervision targets from Panoptic gt3d/*.npy "
            "joint annotations."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Panoptic cropped dataset root containing <sequence>/gt3d/*.npy.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output SMPL target cache directory. Defaults to <dataset-root>/smpl.",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Optional sequence ids to process.",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional manifest; when set, fit only frames referenced by that manifest.",
    )
    parser.add_argument(
        "--smpl-model-path", default=None, help="Neutral SMPL model path."
    )
    parser.add_argument(
        "--device", default=None, help="Torch device, e.g. cuda:0 or cpu."
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-iters", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--gt-unit", choices=("cm", "m"), default="cm")
    parser.add_argument("--confidence-threshold", type=float, default=0.05)
    parser.add_argument("--pose-prior-weight", type=float, default=1.0e-4)
    parser.add_argument("--betas-prior-weight", type=float, default=1.0e-3)
    parser.add_argument(
        "--joint-loss",
        choices=("mse", "huber"),
        default="mse",
        help="Data term for fitting Panoptic joints. Huber is more robust to bad/noisy joints.",
    )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=0.05,
        help="Huber threshold in meters when --joint-loss huber is used.",
    )
    parser.add_argument(
        "--shared-betas",
        action="store_true",
        help=(
            "Estimate one sequence-level beta vector, then refit all frames with "
            "that fixed shape. Recommended for Panoptic pseudo-GT."
        ),
    )
    parser.add_argument(
        "--shared-betas-max-frames",
        type=int,
        default=512,
        help="Maximum frames sampled per sequence for estimating shared betas.",
    )
    parser.add_argument(
        "--shared-betas-num-iters",
        type=int,
        default=None,
        help="Optimizer iterations for the shared-betas estimation pass. Defaults to --num-iters.",
    )
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help="Initialize each chunk from the previous fitted chunk's last frame.",
    )
    parser.add_argument(
        "--temporal-smooth-weight",
        type=float,
        default=0.0,
        help="Optional smoothness weight on consecutive body poses/global roots.",
    )
    parser.add_argument(
        "--temporal-transl-smooth-weight",
        type=float,
        default=0.0,
        help="Optional smoothness weight on consecutive translations.",
    )
    parser.add_argument(
        "--fit-betas",
        action="store_true",
        help="Fit per-frame betas from joints. Default keeps betas at zero.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing <sequence>_smpl_params.npz files.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional debugging cap per sequence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")
    if args.num_iters < 1:
        raise ValueError(f"--num-iters must be >= 1, got {args.num_iters}")
    if args.shared_betas and args.fit_betas:
        raise ValueError("--shared-betas and --fit-betas are mutually exclusive")
    if args.shared_betas_max_frames < 1:
        raise ValueError("--shared-betas-max-frames must be >= 1")
    if args.huber_delta <= 0:
        raise ValueError("--huber-delta must be > 0")
    if args.temporal_smooth_weight < 0 or args.temporal_transl_smooth_weight < 0:
        raise ValueError("Temporal smoothness weights must be >= 0")

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else dataset_root / "smpl"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    requested_frames = load_requested_frames(args.manifest_path)

    sequence_dirs = discover_sequence_dirs(
        dataset_root,
        sequences=args.sequences,
        requested_frames=requested_frames,
    )
    if not sequence_dirs:
        raise RuntimeError(
            f"No Panoptic sequences with gt3d/*.npy found under {dataset_root}"
        )

    smpl_model = build_smpl_model(
        device=device,
        smpl_model_path=args.smpl_model_path,
        batch_size=args.batch_size,
    )
    smpl_model.eval()

    for sequence_dir in sequence_dirs:
        output_path = output_dir / f"{sequence_dir.name}_smpl_params.npz"
        if output_path.exists() and not args.overwrite:
            print(f"[skip] {output_path} exists")
            continue
        frames = collect_sequence_frames(
            sequence_dir,
            requested_frame_ids=requested_frames.get(sequence_dir.name),
            max_frames=args.max_frames,
        )
        if not frames:
            print(f"[skip] {sequence_dir.name}: no matching gt3d frames")
            continue
        fit_sequence(
            frames=frames,
            output_path=output_path,
            smpl_model=smpl_model,
            args=args,
            device=device,
        )


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_requested_frames(manifest_path: str | None) -> dict[str, set[str]]:
    if manifest_path is None:
        return {}
    import json

    path = Path(manifest_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_samples = payload["samples"] if isinstance(payload, dict) else payload
    requested: dict[str, set[str]] = {}
    for sample in raw_samples:
        sequence_id = str(sample["sequence_id"])
        frame_id = normalize_frame_id(sample["frame_id"])
        requested.setdefault(sequence_id, set()).add(frame_id)
    return requested


def discover_sequence_dirs(
    dataset_root: Path,
    *,
    sequences: list[str] | None,
    requested_frames: dict[str, set[str]],
) -> list[Path]:
    requested_sequences = set(sequences or [])
    if requested_frames:
        requested_sequences &= (
            set(requested_frames) if requested_sequences else set(requested_frames)
        )
    sequence_dirs = []
    for sequence_dir in sorted(dataset_root.iterdir()):
        if not sequence_dir.is_dir():
            continue
        if requested_sequences and sequence_dir.name not in requested_sequences:
            continue
        if (sequence_dir / "gt3d").is_dir():
            sequence_dirs.append(sequence_dir)
    return sequence_dirs


def collect_sequence_frames(
    sequence_dir: Path,
    *,
    requested_frame_ids: set[str] | None,
    max_frames: int | None,
) -> list[PanopticFrame]:
    frames = []
    for path in sorted((sequence_dir / "gt3d").glob("*.npy")):
        frame_id = normalize_frame_id(path.stem)
        if requested_frame_ids is not None and frame_id not in requested_frame_ids:
            continue
        frames.append(PanopticFrame(frame_id=frame_id, path=path))
        if max_frames is not None and len(frames) >= max_frames:
            break
    return frames


def fit_sequence(
    *,
    frames: list[PanopticFrame],
    output_path: Path,
    smpl_model,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    frame_ids: list[int] = []
    global_orients: list[np.ndarray] = []
    body_poses: list[np.ndarray] = []
    betas_values: list[np.ndarray] = []
    transls: list[np.ndarray] = []
    final_losses: list[float] = []

    unit_scale = 0.01 if args.gt_unit == "cm" else 1.0
    shared_betas = None
    if args.shared_betas:
        shared_betas = estimate_shared_betas(
            frames=frames,
            smpl_model=smpl_model,
            args=args,
            device=device,
            unit_scale=unit_scale,
        )

    previous_result: dict[str, np.ndarray] | None = None
    iterator = range(0, len(frames), args.batch_size)
    for start in tqdm(
        iterator,
        desc=f"Fitting {output_path.stem}",
        total=ceil_div(len(frames), args.batch_size),
    ):
        chunk = frames[start : start + args.batch_size]
        joints, confidences = load_panoptic_joint_batch(chunk, unit_scale=unit_scale)
        result = fit_smpl_batch(
            target_joints=torch.from_numpy(joints).to(
                device=device, dtype=torch.float32
            ),
            confidences=torch.from_numpy(confidences).to(
                device=device, dtype=torch.float32
            ),
            smpl_model=smpl_model,
            num_iters=args.num_iters,
            learning_rate=args.learning_rate,
            confidence_threshold=args.confidence_threshold,
            pose_prior_weight=args.pose_prior_weight,
            betas_prior_weight=args.betas_prior_weight,
            fit_betas=args.fit_betas,
            fixed_betas=shared_betas,
            init_result=previous_result if args.warm_start else None,
            temporal_smooth_weight=args.temporal_smooth_weight,
            temporal_transl_smooth_weight=args.temporal_transl_smooth_weight,
            joint_loss=args.joint_loss,
            huber_delta=args.huber_delta,
        )
        previous_result = result
        frame_ids.extend(int(frame.frame_id) for frame in chunk)
        global_orients.append(result["global_orient"])
        body_poses.append(result["body_pose"])
        betas_values.append(result["betas"])
        transls.append(result["transl"])
        final_losses.extend(result["loss"].tolist())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        frame_ids=np.asarray(frame_ids, dtype=np.int32),
        global_orient=np.concatenate(global_orients, axis=0).astype(
            np.float32, copy=False
        ),
        body_pose=np.concatenate(body_poses, axis=0).astype(np.float32, copy=False),
        betas=np.concatenate(betas_values, axis=0).astype(np.float32, copy=False),
        transl=np.concatenate(transls, axis=0).astype(np.float32, copy=False),
        fitting_loss=np.asarray(final_losses, dtype=np.float32),
        panoptic_joint_indices=np.asarray(sorted(PANOPTIC_TO_SMPL24), dtype=np.int32),
        smpl_joint_indices=np.asarray(
            [PANOPTIC_TO_SMPL24[index] for index in sorted(PANOPTIC_TO_SMPL24)],
            dtype=np.int32,
        ),
    )
    print(f"[write] {output_path} ({len(frame_ids)} frames)")


def estimate_shared_betas(
    *,
    frames: list[PanopticFrame],
    smpl_model,
    args: argparse.Namespace,
    device: torch.device,
    unit_scale: float,
) -> torch.Tensor:
    sampled_frames = sample_frames_uniformly(
        frames, max_frames=args.shared_betas_max_frames
    )
    beta_chunks = []
    iterator = range(0, len(sampled_frames), args.batch_size)
    for start in tqdm(
        iterator,
        desc="Estimating shared betas",
        total=ceil_div(len(sampled_frames), args.batch_size),
        leave=False,
    ):
        chunk = sampled_frames[start : start + args.batch_size]
        joints, confidences = load_panoptic_joint_batch(chunk, unit_scale=unit_scale)
        result = fit_smpl_batch(
            target_joints=torch.from_numpy(joints).to(
                device=device, dtype=torch.float32
            ),
            confidences=torch.from_numpy(confidences).to(
                device=device, dtype=torch.float32
            ),
            smpl_model=smpl_model,
            num_iters=args.shared_betas_num_iters or args.num_iters,
            learning_rate=args.learning_rate,
            confidence_threshold=args.confidence_threshold,
            pose_prior_weight=args.pose_prior_weight,
            betas_prior_weight=args.betas_prior_weight,
            fit_betas=True,
            fixed_betas=None,
            init_result=None,
            temporal_smooth_weight=0.0,
            temporal_transl_smooth_weight=0.0,
            joint_loss=args.joint_loss,
            huber_delta=args.huber_delta,
        )
        beta_chunks.append(result["betas"])
    betas = np.concatenate(beta_chunks, axis=0)
    shared_betas_np = np.median(betas, axis=0).astype(np.float32, copy=False)
    return torch.from_numpy(shared_betas_np).to(device=device, dtype=torch.float32)


def sample_frames_uniformly(
    frames: list[PanopticFrame],
    *,
    max_frames: int,
) -> list[PanopticFrame]:
    if len(frames) <= max_frames:
        return frames
    indices = np.linspace(0, len(frames) - 1, num=max_frames, dtype=np.int64)
    return [frames[int(index)] for index in indices]


def load_panoptic_joint_batch(
    frames: list[PanopticFrame],
    *,
    unit_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    joints = []
    confidences = []
    for frame in frames:
        payload = load_panoptic_gt3d_file(frame.path)
        joints.append(payload[:, :3] * unit_scale)
        confidences.append(payload[:, 3])
    return (
        np.stack(joints, axis=0).astype(np.float32, copy=False),
        np.stack(confidences, axis=0).astype(np.float32, copy=False),
    )


def load_panoptic_gt3d_file(path: Path) -> np.ndarray:
    payload = np.load(path, allow_pickle=True)
    array = np.asarray(payload, dtype=np.float32)
    if array.shape != (19, 4):
        raise ValueError(
            f"Expected Panoptic gt3d file with shape (19, 4), got {array.shape}: {path}"
        )
    return array


def fit_smpl_batch(
    *,
    target_joints: torch.Tensor,
    confidences: torch.Tensor,
    smpl_model,
    num_iters: int,
    learning_rate: float,
    confidence_threshold: float,
    pose_prior_weight: float,
    betas_prior_weight: float,
    fit_betas: bool,
    fixed_betas: torch.Tensor | None = None,
    init_result: dict[str, np.ndarray] | None = None,
    temporal_smooth_weight: float = 0.0,
    temporal_transl_smooth_weight: float = 0.0,
    joint_loss: str = "mse",
    huber_delta: float = 0.05,
) -> dict[str, np.ndarray]:
    batch_size = target_joints.shape[0]
    panoptic_indices = torch.tensor(
        sorted(PANOPTIC_TO_SMPL24),
        dtype=torch.long,
        device=target_joints.device,
    )
    smpl_indices = torch.tensor(
        [PANOPTIC_TO_SMPL24[index] for index in sorted(PANOPTIC_TO_SMPL24)],
        dtype=torch.long,
        device=target_joints.device,
    )
    selected_targets = target_joints.index_select(1, panoptic_indices)
    selected_confidences = confidences.index_select(1, panoptic_indices)
    weights = torch.clamp(selected_confidences, min=0.0)
    weights = torch.where(
        weights >= confidence_threshold,
        weights,
        torch.zeros_like(weights),
    )
    # Always keep the pelvis/body center active so translation has an anchor.
    body_center_column = int((panoptic_indices == 2).nonzero(as_tuple=False).item())
    weights[:, body_center_column] = torch.clamp(
        selected_confidences[:, body_center_column],
        min=1.0,
    )
    weights = weights / torch.clamp(weights.mean(dim=1, keepdim=True), min=1.0e-6)

    global_orient_init = initialize_parameter_from_previous_result(
        init_result,
        key="global_orient",
        shape=(batch_size, 3),
        device=target_joints.device,
    )
    body_pose_init = initialize_parameter_from_previous_result(
        init_result,
        key="body_pose",
        shape=(batch_size, 69),
        device=target_joints.device,
    )
    transl_init = target_joints[:, 2, :].detach().clone()
    if init_result is not None and "transl" in init_result:
        previous_transl = torch.as_tensor(
            init_result["transl"][-1],
            dtype=torch.float32,
            device=target_joints.device,
        )
        transl_init = transl_init + (previous_transl - transl_init[:1])

    global_orient = global_orient_init.requires_grad_(True)
    body_pose = body_pose_init.requires_grad_(True)
    transl = transl_init.requires_grad_(True)
    if fixed_betas is not None:
        betas = fixed_betas.to(device=target_joints.device, dtype=torch.float32)
        if betas.ndim == 1:
            betas = betas.unsqueeze(0).expand(batch_size, -1).clone()
        elif betas.shape[0] != batch_size:
            raise ValueError(
                f"Expected fixed_betas with batch size {batch_size} or shape [10], "
                f"got {tuple(betas.shape)}"
            )
        betas = betas.detach()
    else:
        betas = initialize_parameter_from_previous_result(
            init_result,
            key="betas",
            shape=(batch_size, 10),
            device=target_joints.device,
        ).requires_grad_(fit_betas)

    parameters: list[torch.Tensor] = [global_orient, body_pose, transl]
    if fit_betas and fixed_betas is None:
        parameters.append(betas)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    previous_global_orient, previous_body_pose, previous_transl = (
        previous_temporal_state(
            init_result=init_result,
            device=target_joints.device,
        )
    )
    for _ in range(num_iters):
        optimizer.zero_grad(set_to_none=True)
        output = smpl_model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl=transl,
        )
        smpl_joints = output.joints[:, :24, :].index_select(1, smpl_indices)
        joint_error = joint_data_term(
            smpl_joints,
            selected_targets,
            loss_type=joint_loss,
            huber_delta=huber_delta,
        )
        data_loss = (joint_error * weights).sum(dim=1) / torch.clamp(
            weights.sum(dim=1), min=1.0e-6
        )
        prior_loss = pose_prior_weight * body_pose.pow(2).mean(dim=1)
        if fit_betas and fixed_betas is None:
            prior_loss = prior_loss + betas_prior_weight * betas.pow(2).mean(dim=1)
        smooth_loss = temporal_smooth_loss(
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            previous_global_orient=previous_global_orient,
            previous_body_pose=previous_body_pose,
            previous_transl=previous_transl,
            pose_weight=temporal_smooth_weight,
            transl_weight=temporal_transl_smooth_weight,
        )
        loss = (data_loss + prior_loss).mean()
        loss = loss + smooth_loss
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        output = smpl_model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl=transl,
        )
        smpl_joints = output.joints[:, :24, :].index_select(1, smpl_indices)
        final_error = ((smpl_joints - selected_targets) ** 2).sum(dim=-1)
        final_loss = (final_error * weights).sum(dim=1) / torch.clamp(
            weights.sum(dim=1), min=1.0e-6
        )

    return {
        "global_orient": global_orient.detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False),
        "body_pose": body_pose.detach().cpu().numpy().astype(np.float32, copy=False),
        "betas": betas.detach().cpu().numpy().astype(np.float32, copy=False),
        "transl": transl.detach().cpu().numpy().astype(np.float32, copy=False),
        "loss": final_loss.detach().cpu().numpy().astype(np.float32, copy=False),
    }


def initialize_parameter_from_previous_result(
    init_result: dict[str, np.ndarray] | None,
    *,
    key: str,
    shape: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    if init_result is None or key not in init_result:
        return torch.zeros(shape, dtype=torch.float32, device=device)
    previous_value = torch.as_tensor(
        init_result[key][-1],
        dtype=torch.float32,
        device=device,
    )
    return previous_value.reshape(1, -1).expand(shape[0], shape[1]).clone()


def previous_temporal_state(
    *,
    init_result: dict[str, np.ndarray] | None,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if init_result is None:
        return None, None, None
    return (
        last_result_tensor(init_result, "global_orient", device=device),
        last_result_tensor(init_result, "body_pose", device=device),
        last_result_tensor(init_result, "transl", device=device),
    )


def last_result_tensor(
    result: dict[str, np.ndarray],
    key: str,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    if key not in result:
        return None
    return torch.as_tensor(result[key][-1], dtype=torch.float32, device=device).reshape(
        1, -1
    )


def joint_data_term(
    smpl_joints: torch.Tensor,
    target_joints: torch.Tensor,
    *,
    loss_type: str,
    huber_delta: float,
) -> torch.Tensor:
    squared_distance = ((smpl_joints - target_joints) ** 2).sum(dim=-1)
    if loss_type == "mse":
        return squared_distance
    if loss_type != "huber":
        raise ValueError(f"Unsupported joint loss: {loss_type}")
    distance = torch.sqrt(squared_distance + 1.0e-8)
    return torch.where(
        distance <= huber_delta,
        0.5 * squared_distance,
        huber_delta * (distance - 0.5 * huber_delta),
    )


def temporal_smooth_loss(
    *,
    global_orient: torch.Tensor,
    body_pose: torch.Tensor,
    transl: torch.Tensor,
    previous_global_orient: torch.Tensor | None,
    previous_body_pose: torch.Tensor | None,
    previous_transl: torch.Tensor | None,
    pose_weight: float,
    transl_weight: float,
) -> torch.Tensor:
    loss = global_orient.new_zeros(())
    if pose_weight > 0.0:
        global_sequence = prepend_previous(global_orient, previous_global_orient)
        body_sequence = prepend_previous(body_pose, previous_body_pose)
        loss = loss + pose_weight * (
            diff_mse(global_sequence) + diff_mse(body_sequence)
        )
    if transl_weight > 0.0:
        transl_sequence = prepend_previous(transl, previous_transl)
        loss = loss + transl_weight * diff_mse(transl_sequence)
    return loss


def prepend_previous(
    values: torch.Tensor,
    previous: torch.Tensor | None,
) -> torch.Tensor:
    if previous is None:
        return values
    return torch.cat(
        [previous.to(device=values.device, dtype=values.dtype), values], dim=0
    )


def diff_mse(values: torch.Tensor) -> torch.Tensor:
    if values.shape[0] < 2:
        return values.new_zeros(())
    return (values[1:] - values[:-1]).pow(2).mean()


def normalize_frame_id(value) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit():
        return str(int(text))
    return text


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


if __name__ == "__main__":
    main()
