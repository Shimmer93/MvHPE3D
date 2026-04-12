"""Stage-specific model definitions."""

from .mlp_fusion import Stage1MLPFusionConfig, Stage1MLPFusionModel
from .residual_fusion import (
    Stage1ResidualFusionConfig,
    Stage1ResidualFusionModel,
)

__all__ = [
    "Stage1MLPFusionConfig",
    "Stage1MLPFusionModel",
    "Stage1ResidualFusionConfig",
    "Stage1ResidualFusionModel",
]
