"""MPI-INF-3DHP helpers following the HeatFormer multiview protocol."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.io as sio

from mvhpe3d.utils import load_camera_parameters


MPII3D_TRAIN_SUBJECTS: tuple[str, ...] = tuple(f"S{index}" for index in range(1, 8))
MPII3D_VAL_SUBJECTS: tuple[str, ...] = ("S8",)
MPII3D_SEQUENCES: tuple[str, ...] = ("Seq1", "Seq2")
MPII3D_HEATFORMER_CAMERA_IDS: tuple[str, ...] = ("video_0", "video_2", "video_7", "video_8")
MPII3D_HEATFORMER_TRAIN_SAMPLING = 10
MPII3D_HEATFORMER_VAL_SAMPLING = 5

MPII3D_JOINT_NAMES: tuple[str, ...] = (
    "spine3",
    "spine4",
    "spine2",
    "spine",
    "pelvis",
    "neck",
    "head",
    "headtop",
    "left_clavicle",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hand",
    "right_clavicle",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hand",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot",
    "left_toe",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_foot",
    "right_toe",
)

# HeatFormer evaluates MPI-INF-3DHP in Human3.6M 17-joint order.
MPII3D_HEATFORMER_EVAL_JOINTS: tuple[int, ...] = (
    4,   # hip / pelvis
    18,  # left hip
    19,  # left knee
    20,  # left ankle
    23,  # right hip
    24,  # right knee
    25,  # right ankle
    3,   # Spine (H36M)
    5,   # neck
    6,   # Head (H36M)
    7,   # headtop
    9,   # left shoulder
    10,  # left elbow
    11,  # left wrist
    14,  # right shoulder
    15,  # right elbow
    16,  # right wrist
)

MPII3D_HEATFORMER_TO_SMPL24: tuple[int, ...] = (
    0,   # hip / pelvis
    1,   # left hip
    4,   # left knee
    7,   # left ankle
    2,   # right hip
    5,   # right knee
    8,   # right ankle
    6,   # Spine (H36M)
    12,  # neck
    15,  # Head (H36M)
    15,  # headtop -> SMPL head proxy
    16,  # left shoulder
    18,  # left elbow
    20,  # left wrist
    17,  # right shoulder
    19,  # right elbow
    21,  # right wrist
)

MPII3D_HEATFORMER_ROOT_INDEX = 0
MPII3D_HEATFORMER_LEG_SWAP_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 4),
    (2, 5),
    (3, 6),
)


def parse_sequence_id(sequence_id: str) -> tuple[str, str]:
    """Parse manifest sequence ids such as ``S1_Seq1``."""
    parts = sequence_id.split("_", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"MPI-INF-3DHP sequence_id must look like 'S1_Seq1', got {sequence_id!r}")
    subject_id, sequence_name = parts
    return subject_id, sequence_name


def camera_id_to_index(camera_id: str | int) -> int:
    """Return the integer video index for camera ids like ``video_7`` or ``7``."""
    if isinstance(camera_id, int):
        return camera_id
    text = str(camera_id)
    if text.startswith("video_"):
        text = text.split("_", maxsplit=1)[1]
    return int(text)


def sequence_dir(root: str | Path, sequence_id: str) -> Path:
    subject_id, sequence_name = parse_sequence_id(sequence_id)
    return Path(root).resolve() / subject_id / sequence_name


@lru_cache(maxsize=32)
def load_annotation_mat(sequence_path: str) -> dict:
    """Load one MPI-INF-3DHP ``annot.mat`` file."""
    path = Path(sequence_path).resolve() / "annot.mat"
    if not path.exists():
        raise FileNotFoundError(f"MPI-INF-3DHP annotation file does not exist: {path}")
    return sio.loadmat(path)


def load_mpii3d_joint_target(
    root: str | Path,
    *,
    sequence_id: str,
    frame_id: str,
    camera_ids: tuple[str, ...] | list[str] = MPII3D_HEATFORMER_CAMERA_IDS,
    universal: bool = True,
) -> dict[str, np.ndarray]:
    """Load HeatFormer-order 3DHP joints in world meters.

    Raw MPI-INF-3DHP annotations are stored per camera in millimeters. This
    helper follows HeatFormer by back-projecting camera joints through each
    selected camera's extrinsics and averaging the resulting world joints.
    """
    seq_dir = sequence_dir(root, sequence_id)
    payload = load_annotation_mat(str(seq_dir))
    key = "univ_annot3" if universal else "annot3"
    if key not in payload:
        raise KeyError(f"Annotation payload {seq_dir / 'annot.mat'} is missing '{key}'")

    frame_index = int(frame_id)
    per_view_world_joints = []
    per_view_camera_joints = []
    for camera_id in camera_ids:
        camera_index = camera_id_to_index(camera_id)
        camera_joints_mm = np.asarray(payload[key][camera_index, 0], dtype=np.float32)
        if frame_index < 0 or frame_index >= camera_joints_mm.shape[0]:
            raise IndexError(
                f"Frame {frame_index} is out of range for {sequence_id} camera {camera_id} "
                f"with {camera_joints_mm.shape[0]} annotated frames"
            )
        joints_camera = camera_joints_mm[frame_index].reshape(-1, 3) * 0.001
        per_view_camera_joints.append(
            _apply_heatformer_leg_swap(
                joints_camera[list(MPII3D_HEATFORMER_EVAL_JOINTS)]
            )
        )
        camera = load_camera_parameters(root, sequence_id=sequence_id, camera_id=str(camera_id))
        joints_world = (
            (joints_camera - camera.translation[None, :])
            @ np.asarray(camera.rotation, dtype=np.float32)
        )
        per_view_world_joints.append(joints_world)

    joints_world = np.mean(np.stack(per_view_world_joints, axis=0), axis=0)
    selected = _apply_heatformer_leg_swap(
        joints_world[list(MPII3D_HEATFORMER_EVAL_JOINTS)]
    )
    camera_selected = np.stack(per_view_camera_joints, axis=0)
    return {
        "joints": np.ascontiguousarray(selected.astype(np.float32, copy=False)),
        "confidence": np.ones((len(MPII3D_HEATFORMER_EVAL_JOINTS),), dtype=np.float32),
        "camera_joints": np.ascontiguousarray(
            camera_selected.astype(np.float32, copy=False)
        ),
        "camera_confidence": np.ones(
            (len(camera_ids), len(MPII3D_HEATFORMER_EVAL_JOINTS)),
            dtype=np.float32,
        ),
        "smpl_indices": np.asarray(MPII3D_HEATFORMER_TO_SMPL24, dtype=np.int64),
        "root_index": np.asarray(MPII3D_HEATFORMER_ROOT_INDEX, dtype=np.int64),
    }


def load_mpii3d_joint_2d_target(
    root: str | Path,
    *,
    sequence_id: str,
    frame_id: str,
    camera_ids: tuple[str, ...] | list[str] = MPII3D_HEATFORMER_CAMERA_IDS,
) -> dict[str, np.ndarray]:
    """Load HeatFormer-order MPI-INF-3DHP 2D joints for selected cameras.

    The annotations are in image pixels and are returned per view in the same
    camera order requested by the Stage 2 sample.
    """
    seq_dir = sequence_dir(root, sequence_id)
    payload = load_annotation_mat(str(seq_dir))
    if "annot2" not in payload:
        raise KeyError(f"Annotation payload {seq_dir / 'annot.mat'} is missing 'annot2'")

    frame_index = int(frame_id)
    per_view_joints = []
    per_view_confidence = []
    for camera_id in camera_ids:
        camera_index = camera_id_to_index(camera_id)
        camera_joints = np.asarray(payload["annot2"][camera_index, 0], dtype=np.float32)
        if frame_index < 0 or frame_index >= camera_joints.shape[0]:
            raise IndexError(
                f"Frame {frame_index} is out of range for {sequence_id} camera {camera_id} "
                f"with {camera_joints.shape[0]} annotated frames"
            )
        joints_2d = camera_joints[frame_index].reshape(-1, 2)
        selected = _apply_heatformer_leg_swap(
            joints_2d[list(MPII3D_HEATFORMER_EVAL_JOINTS)]
        )
        confidence = np.isfinite(selected).all(axis=-1).astype(np.float32)
        confidence = _apply_heatformer_leg_swap(confidence)
        per_view_joints.append(selected.astype(np.float32, copy=False))
        per_view_confidence.append(confidence)

    return {
        "joints_2d": np.ascontiguousarray(np.stack(per_view_joints, axis=0)),
        "confidence": np.ascontiguousarray(np.stack(per_view_confidence, axis=0)),
    }


def annotation_frame_count(root: str | Path, *, sequence_id: str) -> int:
    """Return the number of annotated frames for a sequence."""
    payload = load_annotation_mat(str(sequence_dir(root, sequence_id)))
    annot = np.asarray(payload["annot3"][0, 0])
    return int(annot.shape[0])


def _apply_heatformer_leg_swap(values: np.ndarray) -> np.ndarray:
    """Apply HeatFormer's MPI-INF-3DHP lower-body left/right correction."""
    swapped = np.asarray(values).copy()
    joint_axis = -2 if swapped.ndim >= 2 else -1
    for left, right in MPII3D_HEATFORMER_LEG_SWAP_PAIRS:
        left_index = [slice(None)] * swapped.ndim
        right_index = [slice(None)] * swapped.ndim
        left_index[joint_axis] = left
        right_index[joint_axis] = right
        left_value = swapped[tuple(left_index)].copy()
        swapped[tuple(left_index)] = swapped[tuple(right_index)]
        swapped[tuple(right_index)] = left_value
    return swapped
