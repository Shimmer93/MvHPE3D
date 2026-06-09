"""Model definitions for MvHPE3D."""

from .stage1 import (
    Stage1MLPFusionConfig,
    Stage1MLPFusionModel,
    Stage1ResidualFusionConfig,
    Stage1ResidualFusionModel,
)
from .stage2 import (
    Stage21RootCorrectionAdapterConfig,
    Stage21RootCorrectionAdapterModel,
    Stage22GatedBodyAdapterConfig,
    Stage22GatedBodyAdapterModel,
    Stage23RootBodyAdapterConfig,
    Stage23RootBodyAdapterModel,
    Stage2JointGraphRefinerConfig,
    Stage2JointGraphRefinerModel,
    Stage2JointResidualConfig,
    Stage2JointResidualModel,
    Stage2ParamRefineConfig,
    Stage2ParamRefineModel,
    Stage2RRGBGuidedResidualRefinerConfig,
    Stage2RRGBGuidedResidualRefinerModel,
)
from .stage3 import (
    Stage3TemporalRefineConfig,
    Stage3TemporalRefineModel,
    Stage3ViewTimeTokenConfig,
    Stage3ViewTimeTokenModel,
)

__all__ = [
    "Stage1MLPFusionConfig",
    "Stage1MLPFusionModel",
    "Stage1ResidualFusionConfig",
    "Stage1ResidualFusionModel",
    "Stage21RootCorrectionAdapterConfig",
    "Stage21RootCorrectionAdapterModel",
    "Stage22GatedBodyAdapterConfig",
    "Stage22GatedBodyAdapterModel",
    "Stage23RootBodyAdapterConfig",
    "Stage23RootBodyAdapterModel",
    "Stage2JointGraphRefinerConfig",
    "Stage2JointGraphRefinerModel",
    "Stage2JointResidualConfig",
    "Stage2JointResidualModel",
    "Stage2ParamRefineConfig",
    "Stage2ParamRefineModel",
    "Stage2RRGBGuidedResidualRefinerConfig",
    "Stage2RRGBGuidedResidualRefinerModel",
    "Stage3TemporalRefineConfig",
    "Stage3TemporalRefineModel",
    "Stage3ViewTimeTokenConfig",
    "Stage3ViewTimeTokenModel",
]
