"""Visualization helpers for MvHPE3D."""

from .smpl_overlay import (
    CameraParameters,
    camera_id_to_camera_key,
    load_camera_parameters,
    overlay_mask_on_image,
    project_vertices_camera_to_image,
    render_projected_mesh_mask,
    render_projected_mesh_mask_camera,
    resolve_camera_json_path,
    resolve_rgb_image_path,
)

__all__ = [
    "CameraParameters",
    "camera_id_to_camera_key",
    "load_camera_parameters",
    "overlay_mask_on_image",
    "project_vertices_camera_to_image",
    "render_projected_mesh_mask",
    "render_projected_mesh_mask_camera",
    "resolve_camera_json_path",
    "resolve_rgb_image_path",
]
