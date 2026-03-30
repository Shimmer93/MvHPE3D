"""Batch collation for fixed-view multiview samples."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data._utils.collate import default_collate


def multiview_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate multiview samples while keeping metadata as a python list."""
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    collated: dict[str, Any] = {}
    keys = batch[0].keys()
    for key in keys:
        values = [item[key] for item in batch]
        first = values[0]

        if key == "meta":
            collated[key] = values
        elif isinstance(first, Mapping):
            collated[key] = {
                sub_key: default_collate([value[sub_key] for value in values])
                for sub_key in first
            }
        else:
            collated[key] = default_collate(values)

    return collated


def _ensure_tensor_dict(value: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Convert every mapping value to a tensor.

    This helper is currently unused outside tests, but it is useful to keep the
    expected tensor-oriented schema explicit in one place.
    """
    return {key: torch.as_tensor(item) for key, item in value.items()}
