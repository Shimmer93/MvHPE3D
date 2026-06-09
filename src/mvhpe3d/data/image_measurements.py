"""Helpers for cached per-joint image measurement features."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def resolve_image_measurement_cache_path(
    cache_dir: str | Path,
    *,
    sequence_id: str,
    camera_id: str,
    frame_id: str,
) -> Path:
    """Return the cached per-view joint image measurement path."""
    return Path(cache_dir).resolve() / f"{sequence_id}_{camera_id}_{frame_id}.npz"


def load_image_measurement_payload(cache_path: str | Path) -> dict[str, np.ndarray]:
    """Load cached per-joint image features and auxiliary validity masks."""
    path = Path(cache_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image measurement cache file does not exist: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if "joint_features" not in payload:
            raise KeyError(f"Missing required field 'joint_features' in {path}")
        joint_features = np.asarray(payload["joint_features"], dtype=np.float32)
        mask_joint_features = (
            np.asarray(payload["mask_joint_features"], dtype=np.float32)
            if "mask_joint_features" in payload
            else None
        )
        if mask_joint_features is None and "mask_feature_dim" in payload:
            mask_feature_dim = int(np.asarray(payload["mask_feature_dim"]).item())
            if mask_feature_dim > 0:
                flat_joint_features = joint_features.reshape(joint_features.shape[0], -1)
                if flat_joint_features.shape[-1] <= mask_feature_dim:
                    raise ValueError(
                        f"Cannot split {mask_feature_dim} mask features from "
                        f"'joint_features' in {path} with shape {joint_features.shape}"
                    )
                mask_joint_features = flat_joint_features[:, -mask_feature_dim:]
                joint_features = flat_joint_features[:, :-mask_feature_dim]
        projected_uv = np.asarray(
            payload["projected_uv"] if "projected_uv" in payload else np.zeros((17, 2)),
            dtype=np.float32,
        )
        measured_uv = np.asarray(
            payload["measured_uv"] if "measured_uv" in payload else projected_uv,
            dtype=np.float32,
        )
        valid = np.asarray(
            payload["valid"] if "valid" in payload else np.ones(joint_features.shape[0]),
            dtype=bool,
        )
        joint_confidence = np.asarray(
            payload["joint_confidence"]
            if "joint_confidence" in payload
            else valid.astype(np.float32),
            dtype=np.float32,
        )

    if joint_features.ndim < 2:
        raise ValueError(
            f"Expected 'joint_features' in {path} to have shape [J, ...], "
            f"got {joint_features.shape}"
        )
    if joint_features.ndim > 2:
        joint_features = joint_features.reshape(joint_features.shape[0], -1)
    joint_count = int(joint_features.shape[0])
    if projected_uv.shape != (joint_count, 2):
        raise ValueError(
            f"Expected 'projected_uv' in {path} to have shape {(joint_count, 2)}, "
            f"got {projected_uv.shape}"
        )
    if measured_uv.shape != (joint_count, 2):
        raise ValueError(
            f"Expected 'measured_uv' in {path} to have shape {(joint_count, 2)}, "
            f"got {measured_uv.shape}"
        )
    if valid.shape != (joint_count,):
        raise ValueError(
            f"Expected 'valid' in {path} to have shape {(joint_count,)}, got {valid.shape}"
        )
    if joint_confidence.shape != (joint_count,):
        raise ValueError(
            "Expected 'joint_confidence' in "
            f"{path} to have shape {(joint_count,)}, got {joint_confidence.shape}"
        )
    result = {
        "joint_features": np.ascontiguousarray(joint_features),
        "projected_uv": np.ascontiguousarray(projected_uv),
        "measured_uv": np.ascontiguousarray(measured_uv),
        "valid": np.ascontiguousarray(valid),
        "joint_confidence": np.ascontiguousarray(joint_confidence),
    }
    if mask_joint_features is not None:
        if mask_joint_features.ndim > 2:
            mask_joint_features = mask_joint_features.reshape(mask_joint_features.shape[0], -1)
        if mask_joint_features.ndim != 2 or mask_joint_features.shape[0] != joint_count:
            raise ValueError(
                f"Expected mask_joint_features in {path} to have shape "
                f"[{joint_count}, D], got {mask_joint_features.shape}"
            )
        result["mask_joint_features"] = np.ascontiguousarray(mask_joint_features)
    return result
