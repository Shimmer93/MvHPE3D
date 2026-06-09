"""HuMMan multiview Stage 2 dataset built from cached per-view fitted SMPL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from mvhpe3d.data.image_measurements import (
    load_image_measurement_payload,
    resolve_image_measurement_cache_path,
)
from mvhpe3d.data.rgb_features import (
    load_rgb_feature_payload,
    resolve_rgb_feature_cache_path,
)
from mvhpe3d.data.mpi_inf_3dhp import (
    load_mpii3d_joint_2d_target,
    load_mpii3d_joint_target,
)
from mvhpe3d.data.behave import (
    load_behave_camera,
    load_behave_joint_2d_target,
    load_behave_joint_target,
)
from mvhpe3d.data.h36m import (
    load_h36m_camera,
    load_h36m_joint_2d_target,
    load_h36m_joint_target,
)
from mvhpe3d.utils import (
    CameraParameters,
    axis_angle_to_rotation_6d,
    cache_path_for_source_npz,
    load_camera_parameters,
    transform_smpl_world_to_camera,
)

from ..canonicalization import canonicalize_stage2_target
from .humman_multiview import HuMManStage1Dataset


class HuMManStage2Dataset(HuMManStage1Dataset):
    """Dataset for Stage 2 parameter-space fusion and refinement.

    Each returned sample follows the schema:
    - ``views_input``: tensor of shape ``[N, 148]`` from cached per-view
      canonical ``body_pose_6d + betas``
    - ``target_body_pose``: canonical target SMPL body pose in axis-angle
    - ``target_body_pose_6d``: canonical target pose in 6D rotation form
    - ``target_betas``: canonical target SMPL shape coefficients
    - ``view_aux`` / ``target_aux`` / ``meta``: same auxiliary structure as Stage 1
    """

    def __init__(
        self,
        records,
        *,
        num_views: int,
        train: bool,
        gt_smpl_dir: str | Path,
        cameras_dir: str | Path,
        input_smpl_cache_dir: str | Path,
        rgb_feature_cache_dir: str | Path | None = None,
        image_measurement_cache_dir: str | Path | None = None,
        segmentation_mask_cache_dir: str | Path | None = None,
        pose_pca_coeff_target_path: str | Path | None = None,
        joint_target_dataset: str | None = None,
        joint_target_root: str | Path | None = None,
        joint_target_use_smpl_targets: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__(
            records,
            num_views=num_views,
            train=train,
            gt_smpl_dir=gt_smpl_dir,
            cameras_dir=cameras_dir,
            seed=seed,
        )
        self.input_smpl_cache_dir = Path(input_smpl_cache_dir).resolve()
        self.rgb_feature_cache_dir = (
            Path(rgb_feature_cache_dir).resolve()
            if rgb_feature_cache_dir is not None
            else None
        )
        self.image_measurement_cache_dir = (
            Path(image_measurement_cache_dir).resolve()
            if image_measurement_cache_dir is not None
            else None
        )
        self.segmentation_mask_cache_dir = (
            Path(segmentation_mask_cache_dir).resolve()
            if segmentation_mask_cache_dir is not None
            else None
        )
        self.pose_pca_coeff_target_path = (
            Path(pose_pca_coeff_target_path).resolve()
            if pose_pca_coeff_target_path is not None
            else None
        )
        self.pose_pca_coeff_targets, self.pose_pca_coeff_dim = self._load_pose_pca_coeff_targets(
            self.pose_pca_coeff_target_path
        )
        self.joint_target_dataset = joint_target_dataset
        self.joint_target_root = Path(joint_target_root).resolve() if joint_target_root is not None else None
        self.joint_target_use_smpl_targets = bool(joint_target_use_smpl_targets)
        self._input_smpl_cache: dict[str, dict[str, np.ndarray]] = {}
        self._stage2_input_cache: dict[str, np.ndarray] = {}
        self._canonical_target_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        self._rgb_feature_cache: dict[str, np.ndarray] = {}
        self._image_measurement_cache: dict[str, dict[str, np.ndarray]] = {}
        self._segmentation_mask_cache: dict[str, dict[str, np.ndarray]] = {}
        self._camera_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        self._joint_target_cache: dict[
            tuple[str, str, tuple[str, ...]],
            dict[str, np.ndarray],
        ] = {}
        self._joint_2d_target_cache: dict[
            tuple[str, str, tuple[str, ...]],
            dict[str, np.ndarray],
        ] = {}

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        selected_views = self._select_views(record, index=index)

        view_inputs = []
        view_aux = {
            "pred_cam_t": [],
            "cam_int": [],
            "image_size": [],
            "input_global_orient": [],
            "input_transl": [],
        }
        view_rgb_features: list[np.ndarray] = []
        view_image_joint_features: list[np.ndarray] = []
        view_image_joint_valid: list[np.ndarray] = []
        view_image_joint_confidence: list[np.ndarray] = []
        view_image_joint_uv: list[np.ndarray] = []
        view_image_joint_projected_uv: list[np.ndarray] = []
        view_image_mask_features: list[np.ndarray] = []
        view_segmentation_masks: list[np.ndarray] = []
        view_segmentation_distances: list[np.ndarray] = []
        view_segmentation_bboxes: list[np.ndarray] = []
        view_segmentation_valid: list[np.ndarray] = []
        camera_ids: list[str] = []

        for view in selected_views:
            payload = self._load_npz(view.npz_path)
            fitted_payload = self._load_cached_input_smpl(view.npz_path)
            view_inputs.append(self._build_stage2_input(fitted_payload))
            view_aux["pred_cam_t"].append(
                self._require_field(payload, "pred_cam_t", expected_last_dim=3)
            )
            if self.joint_target_dataset in {"behave", "h36m"}:
                _, _, intrinsics = self._load_joint_dataset_view_camera(
                    record=record,
                    camera_id=view.camera_id,
                )
                view_aux["cam_int"].append(intrinsics)
            else:
                view_aux["cam_int"].append(self._require_matrix(payload, "cam_int", shape=(3, 3)))
            view_aux["image_size"].append(
                self._require_field(payload, "image_size", expected_last_dim=2)
            )
            view_aux["input_global_orient"].append(
                self._require_field(fitted_payload, "global_orient", expected_last_dim=3)
            )
            view_aux["input_transl"].append(
                self._require_field(fitted_payload, "transl", expected_last_dim=3)
            )
            if self.uses_joint_targets:
                if self.joint_target_dataset in {"behave", "h36m"}:
                    camera_rotation, camera_translation, _ = self._load_joint_dataset_view_camera(
                        record=record,
                        camera_id=view.camera_id,
                    )
                else:
                    camera_rotation, camera_translation = self._load_view_camera(
                        sequence_id=record.sequence_id,
                        camera_id=view.camera_id,
                    )
                view_aux.setdefault("camera_rotation", []).append(camera_rotation)
                view_aux.setdefault("camera_translation", []).append(camera_translation)
            if self.rgb_feature_cache_dir is not None:
                view_rgb_features.append(
                    self._load_rgb_feature(
                        sequence_id=record.sequence_id,
                        camera_id=view.camera_id,
                        frame_id=record.frame_id,
                    )
                )
            if self.image_measurement_cache_dir is not None:
                image_measurement = self._load_image_measurement(
                    sequence_id=record.sequence_id,
                    camera_id=view.camera_id,
                    frame_id=record.frame_id,
                )
                view_image_joint_features.append(image_measurement["joint_features"])
                view_image_joint_valid.append(image_measurement["valid"])
                view_image_joint_confidence.append(
                    image_measurement["joint_confidence"]
                )
                view_image_joint_uv.append(image_measurement["measured_uv"])
                view_image_joint_projected_uv.append(
                    image_measurement["projected_uv"]
                )
                if "mask_joint_features" in image_measurement:
                    view_image_mask_features.append(image_measurement["mask_joint_features"])
            if self.segmentation_mask_cache_dir is not None:
                mask_payload = self._load_segmentation_mask_supervision(
                    sequence_id=record.sequence_id,
                    camera_id=view.camera_id,
                    frame_id=record.frame_id,
                )
                view_segmentation_masks.append(mask_payload["mask"])
                view_segmentation_distances.append(mask_payload["outside_distance"])
                view_segmentation_bboxes.append(mask_payload["bbox"])
                view_segmentation_valid.append(mask_payload["valid"])
            camera_ids.append(view.camera_id)

        if self.uses_joint_targets:
            if self.joint_target_use_smpl_targets:
                target_payload = self._load_gt_target(record)
                canonical_target = self._load_canonical_target(record)
                camera_target_aux = self._build_camera_frame_target_aux(
                    record=record,
                    camera_ids=camera_ids,
                    target_payload=target_payload,
                )
            else:
                target_payload = self._zero_smpl_target()
                canonical_target = self._zero_canonical_target()
                camera_target_aux = self._zero_camera_target_aux(num_views=self.num_views)
            joint_target = self._load_joint_target(record, camera_ids=camera_ids)
            joint_2d_target = self._load_joint_2d_target(record, camera_ids=camera_ids)
        else:
            target_payload = self._load_gt_target(record)
            canonical_target = self._load_canonical_target(record)
            camera_target_aux = self._build_camera_frame_target_aux(
                record=record,
                camera_ids=camera_ids,
                target_payload=target_payload,
            )
            joint_target = None
            joint_2d_target = None

        sample = {
            "views_input": torch.from_numpy(np.stack(view_inputs, axis=0)),
            "target_body_pose": torch.from_numpy(canonical_target["target_body_pose"]),
            "target_body_pose_6d": torch.from_numpy(canonical_target["target_body_pose_6d"]),
            "target_betas": torch.from_numpy(canonical_target["target_betas"]),
            "view_aux": {
                key: torch.from_numpy(np.stack(values, axis=0))
                for key, values in view_aux.items()
            },
            "target_aux": {
                "global_orient": torch.from_numpy(
                    self._require_field(target_payload, "global_orient", expected_last_dim=3)
                ),
                "transl": torch.from_numpy(
                    self._require_field(target_payload, "transl", expected_last_dim=3)
                ),
                "camera_rotation": torch.from_numpy(camera_target_aux["camera_rotation"]),
                "camera_translation": torch.from_numpy(camera_target_aux["camera_translation"]),
                "camera_global_orient": torch.from_numpy(camera_target_aux["camera_global_orient"]),
                "camera_transl": torch.from_numpy(camera_target_aux["camera_transl"]),
            },
            "meta": {
                "sample_id": record.sample_id,
                "sequence_id": record.sequence_id,
                "frame_id": record.frame_id,
                "camera_ids": camera_ids,
                "view_npz_paths": [str(view.npz_path) for view in selected_views],
            },
        }
        if self.rgb_feature_cache_dir is not None:
            sample["view_rgb_feature"] = torch.from_numpy(
                np.stack(view_rgb_features, axis=0)
            )
        if self.image_measurement_cache_dir is not None:
            sample["view_image_joint_feature"] = torch.from_numpy(
                np.stack(view_image_joint_features, axis=0)
            )
            sample["view_image_joint_valid"] = torch.from_numpy(
                np.stack(view_image_joint_valid, axis=0)
            )
            sample["view_image_joint_confidence"] = torch.from_numpy(
                np.stack(view_image_joint_confidence, axis=0)
            )
            sample["view_image_joint_uv"] = torch.from_numpy(
                np.stack(view_image_joint_uv, axis=0)
            )
            sample["view_image_joint_projected_uv"] = torch.from_numpy(
                np.stack(view_image_joint_projected_uv, axis=0)
            )
            if view_image_mask_features:
                if len(view_image_mask_features) != len(selected_views):
                    raise ValueError(
                        "Mask image features must be present for every selected view "
                        f"of sample {record.sample_id}"
                    )
                sample["view_image_mask_feature"] = torch.from_numpy(
                    np.stack(view_image_mask_features, axis=0)
                )
        if self.segmentation_mask_cache_dir is not None:
            if len(view_segmentation_masks) != len(selected_views):
                raise ValueError(
                    "Segmentation mask supervision must be present for every selected "
                    f"view of sample {record.sample_id}"
                )
            sample["view_segmentation_mask"] = torch.from_numpy(
                np.stack(view_segmentation_masks, axis=0)
            )
            sample["view_segmentation_distance"] = torch.from_numpy(
                np.stack(view_segmentation_distances, axis=0)
            )
            sample["view_segmentation_bbox"] = torch.from_numpy(
                np.stack(view_segmentation_bboxes, axis=0)
            )
            sample["view_segmentation_valid"] = torch.from_numpy(
                np.stack(view_segmentation_valid, axis=0)
            )
        if joint_target is not None:
            sample["target_joints"] = torch.from_numpy(joint_target["joints"])
            sample["target_joint_confidence"] = torch.from_numpy(joint_target["confidence"])
            if "camera_joints" in joint_target:
                sample["target_camera_joints"] = torch.from_numpy(
                    joint_target["camera_joints"]
                )
            if "camera_confidence" in joint_target:
                sample["target_camera_joint_confidence"] = torch.from_numpy(
                    joint_target["camera_confidence"]
                )
            sample["target_joint_smpl_indices"] = torch.from_numpy(joint_target["smpl_indices"])
            sample["target_joint_root_index"] = torch.as_tensor(
                int(joint_target["root_index"]),
                dtype=torch.long,
            )
            sample["target_smpl_valid"] = torch.as_tensor(
                self.joint_target_use_smpl_targets,
                dtype=torch.bool,
            )
        if joint_2d_target is not None:
            sample["target_joints_2d"] = torch.from_numpy(joint_2d_target["joints_2d"])
            sample["target_joints_2d_confidence"] = torch.from_numpy(
                joint_2d_target["confidence"]
            )
        if self.pose_pca_coeff_targets is not None:
            target_pose_pca_coeff = self.pose_pca_coeff_targets.get(record.sample_id)
            target_pose_pca_coeff_valid = target_pose_pca_coeff is not None
            if target_pose_pca_coeff is None:
                target_pose_pca_coeff = np.zeros(
                    (self.pose_pca_coeff_dim,),
                    dtype=np.float32,
                )
            sample["target_pose_pca_coeff"] = torch.from_numpy(target_pose_pca_coeff)
            sample["target_pose_pca_coeff_valid"] = torch.as_tensor(
                target_pose_pca_coeff_valid,
                dtype=torch.bool,
            )
        return sample

    @staticmethod
    def _load_pose_pca_coeff_targets(
        path: Path | None,
    ) -> tuple[dict[str, np.ndarray] | None, int]:
        if path is None:
            return None, 0
        with np.load(path) as data:
            if "sample_ids" not in data or "coeff" not in data:
                raise KeyError(
                    f"Pose PCA coefficient target file {path} must contain sample_ids and coeff"
                )
            sample_ids = [str(item) for item in data["sample_ids"].tolist()]
            coeff = np.asarray(data["coeff"], dtype=np.float32)
        if coeff.ndim != 2:
            raise ValueError(f"Expected coeff shape [N, K], got {coeff.shape}")
        if len(sample_ids) != coeff.shape[0]:
            raise ValueError(
                f"Expected one sample id per coeff row, got {len(sample_ids)} and {coeff.shape[0]}"
            )
        return {sample_id: np.ascontiguousarray(row) for sample_id, row in zip(sample_ids, coeff)}, int(coeff.shape[1])

    @property
    def uses_joint_targets(self) -> bool:
        return self.joint_target_dataset is not None

    def _load_gt_target(self, record) -> dict[str, np.ndarray]:
        if not (self.uses_joint_targets and self.joint_target_use_smpl_targets):
            return super()._load_gt_target(record)

        sequence_payload = self._load_gt_sequence(record.sequence_id)
        frame_index = int(record.frame_id)
        valid_mask = sequence_payload.get("valid_mask")
        if valid_mask is not None:
            valid_mask = np.asarray(valid_mask, dtype=bool)
            if frame_index < 0 or frame_index >= valid_mask.shape[0] or not bool(valid_mask[frame_index]):
                raise FileNotFoundError(
                    f"Pseudo-SMPL target for {record.sequence_id} frame {record.frame_id} "
                    f"is missing in {self.gt_smpl_dir / f'{record.sequence_id}_smpl_params.npz'}. "
                    "Run scripts/fit_mpi_inf_3dhp_gt_smpl.py first."
                )

        target_payload: dict[str, np.ndarray] = {}
        for key in ("global_orient", "body_pose", "betas", "transl"):
            if key not in sequence_payload:
                raise KeyError(
                    f"Missing required pseudo-SMPL field '{key}' for sequence "
                    f"'{record.sequence_id}'"
                )
            target_payload[key] = self._select_dense_frame(
                sequence_payload[key],
                frame_index,
                key=key,
                sequence_id=record.sequence_id,
            )
        return target_payload

    def _build_camera_frame_target_aux(
        self,
        *,
        record,
        camera_ids: list[str],
        target_payload: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        if self.joint_target_dataset not in {"behave", "h36m"}:
            return super()._build_camera_frame_target_aux(
                record=record,
                camera_ids=camera_ids,
                target_payload=target_payload,
            )

        world_global_orient = self._require_field(
            target_payload,
            "global_orient",
            expected_last_dim=3,
        )
        world_transl = self._require_field(target_payload, "transl", expected_last_dim=3)
        camera_global_orients = []
        camera_translations = []
        camera_rotations = []
        camera_translation_vectors = []
        for camera_id in camera_ids:
            rotation, translation, intrinsics = self._load_joint_dataset_view_camera(
                record=record,
                camera_id=camera_id,
            )
            camera = CameraParameters(
                intrinsics=intrinsics,
                rotation=rotation,
                translation=translation,
            )
            camera_global_orient, camera_transl = transform_smpl_world_to_camera(
                global_orient=world_global_orient,
                transl=world_transl,
                camera=camera,
            )
            camera_global_orients.append(camera_global_orient)
            camera_translations.append(camera_transl)
            camera_rotations.append(rotation)
            camera_translation_vectors.append(translation)

        return {
            "camera_rotation": np.ascontiguousarray(
                np.stack(camera_rotations, axis=0).astype(np.float32, copy=False)
            ),
            "camera_translation": np.ascontiguousarray(
                np.stack(camera_translation_vectors, axis=0).astype(np.float32, copy=False)
            ),
            "camera_global_orient": np.ascontiguousarray(
                np.stack(camera_global_orients, axis=0).astype(np.float32, copy=False)
            ),
            "camera_transl": np.ascontiguousarray(
                np.stack(camera_translations, axis=0).astype(np.float32, copy=False)
            ),
        }

    @staticmethod
    def _select_dense_frame(
        array: np.ndarray,
        frame_index: int,
        *,
        key: str,
        sequence_id: str,
    ) -> np.ndarray:
        value = np.asarray(array, dtype=np.float32)
        if value.ndim == 1:
            return np.ascontiguousarray(value)
        if value.ndim != 2:
            raise ValueError(
                f"Pseudo-SMPL field '{key}' for sequence '{sequence_id}' must have "
                f"shape [T, D] or [D], got {value.shape}"
            )
        if not 0 <= frame_index < value.shape[0]:
            raise IndexError(
                f"Frame index {frame_index} is out of range for pseudo-SMPL field "
                f"'{key}' in sequence '{sequence_id}' with length {value.shape[0]}"
            )
        selected = value[frame_index]
        if not np.isfinite(selected).all():
            raise ValueError(
                f"Pseudo-SMPL field '{key}' for sequence '{sequence_id}' frame "
                f"{frame_index} contains non-finite values"
            )
        return np.ascontiguousarray(selected)

    def _load_cached_input_smpl(self, source_npz_path: Path) -> dict[str, np.ndarray]:
        cache_key = str(source_npz_path.resolve())
        cached = self._input_smpl_cache.get(cache_key)
        if cached is not None:
            return cached

        cache_path = cache_path_for_source_npz(self.input_smpl_cache_dir, source_npz_path)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cached fitted SMPL file does not exist: {cache_path}. "
                "Run scripts/precompute_input_smpl.py first."
            )
        cached = self._load_npz(cache_path)
        self._input_smpl_cache[cache_key] = cached
        return cached

    def _load_view_camera(
        self,
        *,
        sequence_id: str,
        camera_id: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        cache_key = (sequence_id, camera_id)
        cached = self._camera_cache.get(cache_key)
        if cached is not None:
            return cached
        camera = load_camera_parameters(
            self.cameras_dir,
            sequence_id=sequence_id,
            camera_id=camera_id,
        )
        cached = (
            np.ascontiguousarray(np.asarray(camera.rotation, dtype=np.float32)),
            np.ascontiguousarray(np.asarray(camera.translation, dtype=np.float32)),
        )
        self._camera_cache[cache_key] = cached
        return cached

    def _load_behave_view_camera(self, *, record, camera_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        split, index = self._behave_db_locator(record)
        cache_key = (record.sequence_id, record.frame_id, camera_id, split, str(index))
        cached = self._camera_cache.get(cache_key)
        if cached is not None:
            rotation, translation = cached
            _, _, intrinsics = load_behave_camera(
                self.joint_target_root,
                split=split,
                index=index,
                camera_id=camera_id,
            )
            return rotation, translation, intrinsics
        rotation, translation, intrinsics = load_behave_camera(
            self.joint_target_root,
            split=split,
            index=index,
            camera_id=camera_id,
        )
        self._camera_cache[cache_key] = (rotation, translation)
        return rotation, translation, intrinsics

    def _load_h36m_view_camera(self, *, record, camera_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        subject_id, _, _, _ = self._h36m_db_locator(record)
        cache_key = (record.sequence_id, record.frame_id, camera_id, "h36m", subject_id)
        cached = self._camera_cache.get(cache_key)
        if cached is not None:
            rotation, translation = cached
            _, _, intrinsics = load_h36m_camera(
                self.joint_target_root,
                subject_id=subject_id,
                camera_id=camera_id,
            )
            return rotation, translation, intrinsics
        rotation, translation, intrinsics = load_h36m_camera(
            self.joint_target_root,
            subject_id=subject_id,
            camera_id=camera_id,
        )
        self._camera_cache[cache_key] = (rotation, translation)
        return rotation, translation, intrinsics

    def _load_joint_dataset_view_camera(
        self,
        *,
        record,
        camera_id: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.joint_target_dataset == "behave":
            return self._load_behave_view_camera(record=record, camera_id=camera_id)
        if self.joint_target_dataset == "h36m":
            return self._load_h36m_view_camera(record=record, camera_id=camera_id)
        raise ValueError(f"Unsupported joint_target_dataset={self.joint_target_dataset!r}")

    def _build_stage2_input(self, payload: dict[str, np.ndarray]) -> np.ndarray:
        cached_input = payload.get("_stage2_input_cached")
        if cached_input is not None:
            return np.asarray(cached_input, dtype=np.float32)
        body_pose = self._require_field(payload, "body_pose", expected_last_dim=69)
        betas = self._require_field(payload, "betas", expected_last_dim=10)
        body_pose_6d = payload.get("body_pose_6d")
        if body_pose_6d is None:
            body_pose_6d = axis_angle_to_rotation_6d(
                torch.from_numpy(body_pose.reshape(-1, 3))
            ).reshape(-1).cpu().numpy()
            payload["body_pose_6d"] = np.ascontiguousarray(body_pose_6d.astype(np.float32, copy=False))
        built_input = np.concatenate(
            [
                np.asarray(body_pose_6d, dtype=np.float32),
                betas,
            ],
            axis=-1,
        ).astype(np.float32, copy=False)
        payload["_stage2_input_cached"] = built_input
        return built_input

    def _load_canonical_target(self, record) -> dict[str, np.ndarray]:
        cache_key = (record.sequence_id, record.frame_id)
        cached = self._canonical_target_cache.get(cache_key)
        if cached is not None:
            return cached

        target_payload = self._load_gt_target(record)
        cached = canonicalize_stage2_target(
            smpl_body_pose=self._require_field(target_payload, "body_pose", expected_last_dim=69),
            smpl_betas=self._require_field(target_payload, "betas", expected_last_dim=10),
        )
        self._canonical_target_cache[cache_key] = cached
        return cached

    def _load_rgb_feature(
        self,
        *,
        sequence_id: str,
        camera_id: str,
        frame_id: str,
    ) -> np.ndarray:
        if self.rgb_feature_cache_dir is None:
            raise RuntimeError("rgb_feature_cache_dir is not configured")
        cache_path = resolve_rgb_feature_cache_path(
            self.rgb_feature_cache_dir,
            sequence_id=sequence_id,
            camera_id=camera_id,
            frame_id=frame_id,
        )
        cache_key = str(cache_path)
        cached = self._rgb_feature_cache.get(cache_key)
        if cached is not None:
            return cached
        feature = load_rgb_feature_payload(cache_path)
        self._rgb_feature_cache[cache_key] = feature
        return feature

    def _load_image_measurement(
        self,
        *,
        sequence_id: str,
        camera_id: str,
        frame_id: str,
    ) -> dict[str, np.ndarray]:
        if self.image_measurement_cache_dir is None:
            raise RuntimeError("image_measurement_cache_dir is not configured")
        cache_path = resolve_image_measurement_cache_path(
            self.image_measurement_cache_dir,
            sequence_id=sequence_id,
            camera_id=camera_id,
            frame_id=frame_id,
        )
        cache_key = str(cache_path)
        cached = self._image_measurement_cache.get(cache_key)
        if cached is not None:
            return cached
        payload = load_image_measurement_payload(cache_path)
        self._image_measurement_cache[cache_key] = payload
        return payload

    def _load_segmentation_mask_supervision(
        self,
        *,
        sequence_id: str,
        camera_id: str,
        frame_id: str,
    ) -> dict[str, np.ndarray]:
        if self.segmentation_mask_cache_dir is None:
            raise RuntimeError("segmentation_mask_cache_dir is not configured")
        cache_path = (
            self.segmentation_mask_cache_dir / f"{sequence_id}_{camera_id}_{frame_id}.npz"
        )
        cache_key = str(cache_path)
        cached = self._segmentation_mask_cache.get(cache_key)
        if cached is not None:
            return cached
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Segmentation mask supervision cache file does not exist: {cache_path}"
            )
        with np.load(cache_path, allow_pickle=False) as payload:
            required = {"mask", "outside_distance", "bbox", "valid"}
            missing = sorted(required.difference(payload.files))
            if missing:
                raise KeyError(
                    f"Segmentation mask cache {cache_path} is missing keys: {missing}"
                )
            mask = np.asarray(payload["mask"], dtype=np.float32)
            outside_distance = np.asarray(payload["outside_distance"], dtype=np.float32)
            bbox = np.asarray(payload["bbox"], dtype=np.float32)
            valid = np.asarray(payload["valid"], dtype=bool)
        if mask.ndim != 2:
            raise ValueError(
                f"Expected mask in {cache_path} to have shape [H, W], got {mask.shape}"
            )
        if outside_distance.shape != mask.shape:
            raise ValueError(
                f"Expected outside_distance in {cache_path} to match mask shape "
                f"{mask.shape}, got {outside_distance.shape}"
            )
        if bbox.shape != (4,):
            raise ValueError(f"Expected bbox in {cache_path} to have shape [4], got {bbox.shape}")
        if valid.shape not in {(), (1,)}:
            raise ValueError(f"Expected valid in {cache_path} to be scalar, got {valid.shape}")
        payload = {
            "mask": np.ascontiguousarray(mask),
            "outside_distance": np.ascontiguousarray(outside_distance),
            "bbox": np.ascontiguousarray(bbox),
            "valid": np.asarray(bool(valid.reshape(-1)[0]), dtype=bool),
        }
        self._segmentation_mask_cache[cache_key] = payload
        return payload

    def _load_joint_target(self, record, *, camera_ids: list[str]) -> dict[str, np.ndarray]:
        camera_key = tuple(camera_ids)
        cache_key = (record.sequence_id, record.frame_id, camera_key)
        cached = self._joint_target_cache.get(cache_key)
        if cached is not None:
            return cached
        if self.joint_target_root is None:
            raise RuntimeError(
                f"joint_target_root must be configured for {self.joint_target_dataset} joint targets"
            )
        if self.joint_target_dataset == "mpi_inf_3dhp":
            cached = load_mpii3d_joint_target(
                self.joint_target_root,
                sequence_id=record.sequence_id,
                frame_id=record.frame_id,
                camera_ids=camera_key,
            )
        elif self.joint_target_dataset == "behave":
            split, index = self._behave_db_locator(record)
            cached = load_behave_joint_target(
                self.joint_target_root,
                split=split,
                index=index,
                camera_ids=camera_key,
            )
        elif self.joint_target_dataset == "h36m":
            subject_id, action_id, subaction_id, frame_index = self._h36m_db_locator(record)
            cached = load_h36m_joint_target(
                self.joint_target_root,
                subject_id=subject_id,
                action_id=action_id,
                subaction_id=subaction_id,
                frame_index=frame_index,
                camera_ids=camera_key,
            )
        else:
            raise ValueError(f"Unsupported joint_target_dataset={self.joint_target_dataset!r}")
        self._joint_target_cache[cache_key] = cached
        return cached

    def _load_joint_2d_target(self, record, *, camera_ids: list[str]) -> dict[str, np.ndarray]:
        camera_key = tuple(camera_ids)
        cache_key = (record.sequence_id, record.frame_id, camera_key)
        cached = self._joint_2d_target_cache.get(cache_key)
        if cached is not None:
            return cached
        if self.joint_target_root is None:
            raise RuntimeError(
                f"joint_target_root must be configured for {self.joint_target_dataset} joint targets"
            )
        if self.joint_target_dataset == "mpi_inf_3dhp":
            cached = load_mpii3d_joint_2d_target(
                self.joint_target_root,
                sequence_id=record.sequence_id,
                frame_id=record.frame_id,
                camera_ids=camera_key,
            )
        elif self.joint_target_dataset == "behave":
            split, index = self._behave_db_locator(record)
            cached = load_behave_joint_2d_target(
                self.joint_target_root,
                split=split,
                index=index,
                camera_ids=camera_key,
            )
        elif self.joint_target_dataset == "h36m":
            subject_id, action_id, subaction_id, frame_index = self._h36m_db_locator(record)
            cached = load_h36m_joint_2d_target(
                self.joint_target_root,
                subject_id=subject_id,
                action_id=action_id,
                subaction_id=subaction_id,
                frame_index=frame_index,
                camera_ids=camera_key,
            )
        else:
            raise ValueError(f"Unsupported joint_target_dataset={self.joint_target_dataset!r}")
        self._joint_2d_target_cache[cache_key] = cached
        return cached

    @staticmethod
    def _behave_db_locator(record) -> tuple[str, int]:
        metadata = record.metadata or {}
        split = str(metadata.get("heatformer_db_split", record.split or "valid"))
        if "heatformer_db_index" not in metadata:
            raise KeyError(
                f"Record {record.sample_id!r} is missing heatformer_db_index metadata"
            )
        return split, int(metadata["heatformer_db_index"])

    @staticmethod
    def _h36m_db_locator(record) -> tuple[str, str, str, str]:
        metadata = record.metadata or {}
        missing_keys = [
            key
            for key in (
                "h36m_subject_id",
                "h36m_action_id",
                "h36m_subaction_id",
                "h36m_frame_index",
            )
            if key not in metadata
        ]
        if missing_keys:
            raise KeyError(
                f"Record {record.sample_id!r} is missing H36M metadata keys: "
                + ", ".join(missing_keys)
            )
        return (
            str(metadata["h36m_subject_id"]),
            str(metadata["h36m_action_id"]),
            str(metadata["h36m_subaction_id"]),
            str(metadata["h36m_frame_index"]),
        )

    @staticmethod
    def _zero_smpl_target() -> dict[str, np.ndarray]:
        return {
            "global_orient": np.zeros(3, dtype=np.float32),
            "transl": np.zeros(3, dtype=np.float32),
            "body_pose": np.zeros(69, dtype=np.float32),
            "betas": np.zeros(10, dtype=np.float32),
        }

    @staticmethod
    def _zero_canonical_target() -> dict[str, np.ndarray]:
        return {
            "target_body_pose": np.zeros(69, dtype=np.float32),
            "target_body_pose_6d": np.zeros((23, 6), dtype=np.float32),
            "target_betas": np.zeros(10, dtype=np.float32),
        }

    @staticmethod
    def _zero_camera_target_aux(*, num_views: int) -> dict[str, np.ndarray]:
        return {
            "camera_rotation": np.repeat(np.eye(3, dtype=np.float32)[None], num_views, axis=0),
            "camera_translation": np.zeros((num_views, 3), dtype=np.float32),
            "camera_global_orient": np.zeros((num_views, 3), dtype=np.float32),
            "camera_transl": np.zeros((num_views, 3), dtype=np.float32),
        }
