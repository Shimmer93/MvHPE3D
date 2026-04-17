"""Loss definitions for MvHPE3D."""

from .smpl_loss import (
    Stage1Loss,
    Stage1LossConfig,
    Stage2Loss,
    Stage2LossConfig,
    Stage3Loss,
    Stage3LossConfig,
)

__all__ = [
    "Stage1LossConfig",
    "Stage1Loss",
    "Stage2LossConfig",
    "Stage2Loss",
    "Stage3LossConfig",
    "Stage3Loss",
]
