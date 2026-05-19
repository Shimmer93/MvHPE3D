"""Helpers for cached frozen RGB feature inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def resolve_rgb_feature_cache_path(
    cache_dir: str | Path,
    *,
    sequence_id: str,
    camera_id: str,
    frame_id: str,
) -> Path:
    """Return the cached frozen RGB feature path for one manifest view."""
    return Path(cache_dir).resolve() / f"{sequence_id}_{camera_id}_{frame_id}.npz"


def load_rgb_feature_payload(cache_path: str | Path) -> np.ndarray:
    """Load a 1D frozen RGB feature vector from an `.npz` cache file."""
    path = Path(cache_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"RGB feature cache file does not exist: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if "rgb_feature" not in payload:
            raise KeyError(f"Missing required field 'rgb_feature' in {path}")
        feature = np.asarray(payload["rgb_feature"], dtype=np.float32)

    if feature.ndim == 2 and feature.shape[0] == 1:
        feature = feature[0]
    if feature.ndim != 1:
        raise ValueError(
            f"Expected 'rgb_feature' in {path} to be 1D, got shape {feature.shape}"
        )
    return np.ascontiguousarray(feature)
