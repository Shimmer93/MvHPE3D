"""HuMMan multiview Stage 2 dataset built from cached per-view fitted SMPL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from mvhpe3d.utils import axis_angle_to_rotation_6d, cache_path_for_source_npz

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
        self._input_smpl_cache: dict[str, dict[str, np.ndarray]] = {}
        self._stage2_input_cache: dict[str, np.ndarray] = {}
        self._canonical_target_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        selected_views = self._select_views(record, index=index)

        view_inputs = []
        view_aux = {
            "pred_cam_t": [],
            "cam_int": [],
            "image_size": [],
        }
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
            camera_ids.append(view.camera_id)

        target_payload = self._load_gt_target(record)
        canonical_target = self._load_canonical_target(record)
        camera_target_aux = self._build_camera_frame_target_aux(
            record=record,
            camera_ids=camera_ids,
            target_payload=target_payload,
        )

        return {
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
