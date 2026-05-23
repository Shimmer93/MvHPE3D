"""HuMMan multiview Stage 2 dataset built from cached per-view fitted SMPL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from mvhpe3d.data.rgb_features import (
    load_rgb_feature_payload,
    resolve_rgb_feature_cache_path,
)
from mvhpe3d.data.mpi_inf_3dhp import load_mpii3d_joint_target
from mvhpe3d.utils import (
    axis_angle_to_rotation_6d,
    cache_path_for_source_npz,
    load_camera_parameters,
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
        self.joint_target_dataset = joint_target_dataset
        self.joint_target_root = Path(joint_target_root).resolve() if joint_target_root is not None else None
        self.joint_target_use_smpl_targets = bool(joint_target_use_smpl_targets)
        self._input_smpl_cache: dict[str, dict[str, np.ndarray]] = {}
        self._stage2_input_cache: dict[str, np.ndarray] = {}
        self._canonical_target_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        self._rgb_feature_cache: dict[str, np.ndarray] = {}
        self._camera_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        self._joint_target_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}

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
        camera_ids: list[str] = []

        for view in selected_views:
            payload = self._load_npz(view.npz_path)
            fitted_payload = self._load_cached_input_smpl(view.npz_path)
            view_inputs.append(self._build_stage2_input(fitted_payload))
            view_aux["pred_cam_t"].append(
                self._require_field(payload, "pred_cam_t", expected_last_dim=3)
            )
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
        else:
            target_payload = self._load_gt_target(record)
            canonical_target = self._load_canonical_target(record)
            camera_target_aux = self._build_camera_frame_target_aux(
                record=record,
                camera_ids=camera_ids,
                target_payload=target_payload,
            )
            joint_target = None

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
        if joint_target is not None:
            sample["target_joints"] = torch.from_numpy(joint_target["joints"])
            sample["target_joint_confidence"] = torch.from_numpy(joint_target["confidence"])
            sample["target_joint_smpl_indices"] = torch.from_numpy(joint_target["smpl_indices"])
            sample["target_joint_root_index"] = torch.as_tensor(
                int(joint_target["root_index"]),
                dtype=torch.long,
            )
            sample["target_smpl_valid"] = torch.as_tensor(
                self.joint_target_use_smpl_targets,
                dtype=torch.bool,
            )
        return sample

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

    def _load_joint_target(self, record, *, camera_ids: list[str]) -> dict[str, np.ndarray]:
        cache_key = (record.sequence_id, record.frame_id)
        cached = self._joint_target_cache.get(cache_key)
        if cached is not None:
            return cached
        if self.joint_target_dataset != "mpi_inf_3dhp":
            raise ValueError(f"Unsupported joint_target_dataset={self.joint_target_dataset!r}")
        if self.joint_target_root is None:
            raise RuntimeError("joint_target_root must be configured for MPI-INF-3DHP joint targets")
        cached = load_mpii3d_joint_target(
            self.joint_target_root,
            sequence_id=record.sequence_id,
            frame_id=record.frame_id,
            camera_ids=tuple(camera_ids),
        )
        self._joint_target_cache[cache_key] = cached
        return cached

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
