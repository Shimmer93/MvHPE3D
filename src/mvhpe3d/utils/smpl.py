"""Shared helpers for loading and using SMPL models."""

from __future__ import annotations

from pathlib import Path

import smplx
import torch


def resolve_smpl_model_path(path_arg: str | None = None) -> Path:
    """Resolve the neutral SMPL model path from an explicit argument or known defaults."""
    if path_arg is not None:
        resolved = Path(path_arg).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"SMPL model does not exist: {resolved}")
        return resolved

    candidates = [
        Path("/opt/data/weights/smpl/SMPL_NEUTRAL.pkl"),
        Path("/opt/data/weights/SMPL_NEUTRAL.pkl"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not resolve an SMPL model. Pass --smpl-model-path explicitly."
    )


def build_smpl_model(
    *,
    device: torch.device,
    smpl_model_path: str | None = None,
    batch_size: int = 1,
):
    """Instantiate the neutral SMPL body model on the requested device."""
    resolved_path = resolve_smpl_model_path(smpl_model_path)
    return smplx.SMPL(
        model_path=str(resolved_path),
        gender="neutral",
        batch_size=batch_size,
    ).to(device)
