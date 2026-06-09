"""Stage 2 parameter-space fusion and refinement models."""

from .gated_body_adapter import Stage22GatedBodyAdapterConfig, Stage22GatedBodyAdapterModel
from .joint_graph_refiner import Stage2JointGraphRefinerConfig, Stage2JointGraphRefinerModel
from .joint_residual import Stage2JointResidualConfig, Stage2JointResidualModel
from .param_refine import Stage2ParamRefineConfig, Stage2ParamRefineModel
from .root_body_adapter import Stage23RootBodyAdapterConfig, Stage23RootBodyAdapterModel
from .root_correction_adapter import Stage21RootCorrectionAdapterConfig, Stage21RootCorrectionAdapterModel
from .stage2r_rgb_guided_residual_refiner import (
    Stage2RRGBGuidedResidualRefinerConfig,
    Stage2RRGBGuidedResidualRefinerModel,
)

__all__ = [
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
]
