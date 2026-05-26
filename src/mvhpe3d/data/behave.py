"""BEHAVE helpers following the HeatFormer protocol."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np


BEHAVE_HEATFORMER_CAMERA_IDS: tuple[str, ...] = ("k0", "k1", "k2", "k3")
BEHAVE_HEATFORMER_JOINT_COUNT = 15
BEHAVE_HEATFORMER_ROOT_INDEX = 8
BEHAVE_HEATFORMER_SCORE_THRESHOLD = 0.3


def behave_camera_id_to_index(camera_id: str | int) -> int:
    if isinstance(camera_id, int):
        return camera_id
    text = str(camera_id)
    if text.startswith("k"):
        text = text[1:]
    return int(text)


def behave_db_path(root: str | Path, split: str) -> Path:
    split_name = "valid" if split in {"val", "valid", "test"} else "train"
    filename = f"BEHAVE_{split_name}_db.pt"
    return Path(root).resolve() / "preprocessed_data_z" / filename


@lru_cache(maxsize=4)
def load_behave_db(path: str) -> list[dict[str, Any]]:
    db_path = Path(path).resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"BEHAVE HeatFormer DB does not exist: {db_path}")
    return joblib.load(db_path)


def load_behave_db_entry(
    root: str | Path,
    *,
    split: str,
    index: int,
) -> dict[str, Any]:
    path = behave_db_path(root, split)
    db = load_behave_db(str(path))
    if index < 0 or index >= len(db):
        raise IndexError(f"BEHAVE DB index {index} is out of range for {path}")
    return db[index]


def load_behave_joint_target(
    root: str | Path,
    *,
    split: str,
    index: int,
    camera_ids: tuple[str, ...] | list[str] = BEHAVE_HEATFORMER_CAMERA_IDS,
    score_threshold: float = BEHAVE_HEATFORMER_SCORE_THRESHOLD,
) -> dict[str, np.ndarray]:
    item = load_behave_db_entry(root, split=split, index=index)
    joints_openpose = np.asarray(item["w_3d_openpose"], dtype=np.float32)
    joints = joints_openpose[:BEHAVE_HEATFORMER_JOINT_COUNT, :3]
    confidence = (
        joints_openpose[:BEHAVE_HEATFORMER_JOINT_COUNT, 3] >= score_threshold
    ).astype(np.float32)

    camera_indices = [behave_camera_id_to_index(camera_id) for camera_id in camera_ids]
    rotation = np.asarray(item["R"], dtype=np.float32)[camera_indices]
    translation = np.asarray(item["t"], dtype=np.float32)[camera_indices]
    camera_joints = (
        np.einsum("vij,kj->vki", rotation, joints) + translation[:, None, :]
    )
    camera_confidence = np.repeat(confidence[None], len(camera_indices), axis=0)
    return {
        "joints": np.ascontiguousarray(joints),
        "confidence": np.ascontiguousarray(confidence),
        "camera_joints": np.ascontiguousarray(camera_joints.astype(np.float32, copy=False)),
        "camera_confidence": np.ascontiguousarray(camera_confidence),
        "smpl_indices": np.arange(BEHAVE_HEATFORMER_JOINT_COUNT, dtype=np.int64),
        "root_index": np.asarray(BEHAVE_HEATFORMER_ROOT_INDEX, dtype=np.int64),
    }


def load_behave_joint_2d_target(
    root: str | Path,
    *,
    split: str,
    index: int,
    camera_ids: tuple[str, ...] | list[str] = BEHAVE_HEATFORMER_CAMERA_IDS,
    score_threshold: float = BEHAVE_HEATFORMER_SCORE_THRESHOLD,
) -> dict[str, np.ndarray]:
    item = load_behave_db_entry(root, split=split, index=index)
    camera_indices = [behave_camera_id_to_index(camera_id) for camera_id in camera_ids]
    openpose_2d = np.asarray(item["c_2d_openpose"], dtype=np.float32)[camera_indices]
    joints_2d = openpose_2d[:, :BEHAVE_HEATFORMER_JOINT_COUNT, :2]
    confidence = (
        openpose_2d[:, :BEHAVE_HEATFORMER_JOINT_COUNT, 2] >= score_threshold
    ).astype(np.float32)
    return {
        "joints_2d": np.ascontiguousarray(joints_2d),
        "confidence": np.ascontiguousarray(confidence),
    }


def load_behave_camera(
    root: str | Path,
    *,
    split: str,
    index: int,
    camera_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    item = load_behave_db_entry(root, split=split, index=index)
    camera_index = behave_camera_id_to_index(camera_id)
    rotation = np.asarray(item["R"], dtype=np.float32)[camera_index]
    translation = np.asarray(item["t"], dtype=np.float32)[camera_index]
    intrinsics = np.asarray(item["K"], dtype=np.float32)[camera_index]
    return (
        np.ascontiguousarray(rotation),
        np.ascontiguousarray(translation),
        np.ascontiguousarray(intrinsics),
    )

