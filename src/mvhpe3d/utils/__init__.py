"""Utility helpers for MvHPE3D."""

from .camera import (
    CameraParameters,
    camera_id_to_camera_key,
    load_camera_parameters,
    load_panoptic_camera_parameters,
    panoptic_camera_id_to_camera_key,
    resolve_camera_json_path,
    resolve_panoptic_camera_json_path,
    transform_smpl_world_to_camera,
)
from .config import load_experiment_config
from .mhr_smpl_conversion import (
    MHRToSMPLConverter,
    cache_path_for_source_npz,
    resolve_mhr_asset_folder,
    validate_mhr_asset_folder,
)
from .panoptic import (
    PANOPTIC_BODY_CENTER_INDEX,
    PANOPTIC_EVAL_JOINT_INDICES,
    PANOPTIC_EVAL_ROOT_COLUMN,
    PANOPTIC_EVAL_SMPL24_INDICES,
    PANOPTIC_GT_UNIT_SCALE,
    PANOPTIC_TO_SMPL24,
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
    "load_panoptic_camera_parameters",
    "load_experiment_config",
    "panoptic_camera_id_to_camera_key",
    "PANOPTIC_BODY_CENTER_INDEX",
    "PANOPTIC_EVAL_JOINT_INDICES",
    "PANOPTIC_EVAL_ROOT_COLUMN",
    "PANOPTIC_EVAL_SMPL24_INDICES",
    "PANOPTIC_GT_UNIT_SCALE",
    "PANOPTIC_TO_SMPL24",
    "resolve_camera_json_path",
    "resolve_panoptic_camera_json_path",
    "resolve_mhr_asset_folder",
    "matrix_to_axis_angle",
    "rotation_6d_to_axis_angle",
    "rotation_6d_to_matrix",
    "resolve_smpl_model_path",
    "transform_smpl_world_to_camera",
    "validate_mhr_asset_folder",
]
