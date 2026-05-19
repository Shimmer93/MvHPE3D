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
from .rgb_features import load_rgb_feature_payload, resolve_rgb_feature_cache_path

__all__ = [
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
