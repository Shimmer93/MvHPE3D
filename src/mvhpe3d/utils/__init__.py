"""Utility helpers for MvHPE3D."""

from .camera import (
    CameraParameters,
    camera_id_to_camera_key,
    load_camera_parameters,
    resolve_camera_json_path,
    transform_smpl_world_to_camera,
)
from .config import load_experiment_config
from .mhr_smpl_conversion import (
    MHRToSMPLConverter,
    cache_path_for_source_npz,
    resolve_mhr_asset_folder,
    validate_mhr_asset_folder,
)
from .rotation import (
    axis_angle_to_matrix,
    axis_angle_to_rotation_6d,
    matrix_to_axis_angle,
    rotation_6d_to_axis_angle,
    rotation_6d_to_matrix,
)
from .smpl import build_smpl_model, resolve_smpl_model_path

__all__ = [
    "CameraParameters",
    "MHRToSMPLConverter",
    "axis_angle_to_matrix",
    "axis_angle_to_rotation_6d",
    "build_smpl_model",
    "cache_path_for_source_npz",
    "camera_id_to_camera_key",
    "load_camera_parameters",
    "load_experiment_config",
    "resolve_camera_json_path",
    "resolve_mhr_asset_folder",
    "matrix_to_axis_angle",
    "rotation_6d_to_axis_angle",
    "rotation_6d_to_matrix",
    "resolve_smpl_model_path",
    "transform_smpl_world_to_camera",
    "validate_mhr_asset_folder",
]
