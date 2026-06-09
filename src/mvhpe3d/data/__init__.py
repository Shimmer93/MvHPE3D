"""Data loading utilities for MvHPE3D."""

from .canonicalization import canonicalize_stage1_target, canonicalize_stage2_target
from .collate import multiview_collate
from .datamodule import (
    Stage1DataConfig,
    Stage1HuMManDataModule,
    Stage2DataConfig,
    Stage2HuMManDataModule,
    Stage3DataConfig,
    Stage3HuMManDataModule,
)
from .mpi_inf_3dhp import (
    MPII3D_HEATFORMER_CAMERA_IDS,
    MPII3D_HEATFORMER_TRAIN_SAMPLING,
    MPII3D_HEATFORMER_VAL_SAMPLING,
    MPII3D_SEQUENCES,
    MPII3D_TRAIN_SUBJECTS,
    MPII3D_VAL_SUBJECTS,
)
from .behave import (
    BEHAVE_HEATFORMER_CAMERA_IDS,
    BEHAVE_HEATFORMER_JOINT_COUNT,
    BEHAVE_HEATFORMER_ROOT_INDEX,
    BEHAVE_HEATFORMER_SCORE_THRESHOLD,
)
from .h36m import (
    H36M_HEATFORMER_CAMERA_IDS,
    H36M_HEATFORMER_JOINT_COUNT,
    H36M_HEATFORMER_ROOT_INDEX,
    H36M_HEATFORMER_TRAIN_SAMPLING,
    H36M_HEATFORMER_TRAIN_SUBJECTS,
    H36M_HEATFORMER_VAL_SAMPLING,
    H36M_HEATFORMER_VAL_SUBJECTS,
)
from .rgb_features import load_rgb_feature_payload, resolve_rgb_feature_cache_path

__all__ = [
    "MPII3D_HEATFORMER_CAMERA_IDS",
    "MPII3D_HEATFORMER_TRAIN_SAMPLING",
    "MPII3D_HEATFORMER_VAL_SAMPLING",
    "MPII3D_SEQUENCES",
    "MPII3D_TRAIN_SUBJECTS",
    "MPII3D_VAL_SUBJECTS",
    "BEHAVE_HEATFORMER_CAMERA_IDS",
    "BEHAVE_HEATFORMER_JOINT_COUNT",
    "BEHAVE_HEATFORMER_ROOT_INDEX",
    "BEHAVE_HEATFORMER_SCORE_THRESHOLD",
    "H36M_HEATFORMER_CAMERA_IDS",
    "H36M_HEATFORMER_JOINT_COUNT",
    "H36M_HEATFORMER_ROOT_INDEX",
    "H36M_HEATFORMER_TRAIN_SAMPLING",
    "H36M_HEATFORMER_TRAIN_SUBJECTS",
    "H36M_HEATFORMER_VAL_SAMPLING",
    "H36M_HEATFORMER_VAL_SUBJECTS",
    "Stage1DataConfig",
    "Stage1HuMManDataModule",
    "Stage2DataConfig",
    "Stage2HuMManDataModule",
    "Stage3DataConfig",
    "Stage3HuMManDataModule",
    "canonicalize_stage1_target",
    "canonicalize_stage2_target",
    "load_rgb_feature_payload",
    "multiview_collate",
    "resolve_rgb_feature_cache_path",
]
