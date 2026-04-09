"""HuMMan multiview Stage 1 dataset."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..canonicalization import canonicalize_stage1_target
from ..splits import SampleRecord
from mvhpe3d.utils import load_camera_parameters, transform_smpl_world_to_camera


class HuMManStage1Dataset(Dataset[dict[str, Any]]):
    """Dataset for Stage 1 canonical body fusion.

    Each returned sample follows the schema:
    - ``views_input``: tensor of shape ``[N, D]`` from per-view
      ``mhr_model_params + shape_params``
    - ``target_body_pose``: canonical target SMPL body pose
    - ``target_betas``: canonical target SMPL shape coefficients
    - ``view_aux``: per-view auxiliary visualization fields
    - ``target_aux``: GT root orientation / translation for later rendering
    - ``meta``: identifiers and camera names kept as Python data
    """

    def __init__(
        self,
        records: list[SampleRecord],
        *,
        num_views: int,
        train: bool,
        gt_smpl_dir: str | Path,
        cameras_dir: str | Path,
        seed: int = 0,
    ) -> None:
        if num_views < 1:
            raise ValueError(f"num_views must be >= 1, got {num_views}")
        self.records = records
        self.num_views = num_views
        self.train = train
        self.gt_smpl_dir = Path(gt_smpl_dir).resolve()
        self.cameras_dir = Path(cameras_dir).resolve()
        self.seed = seed
        self._gt_sequence_cache: dict[str, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.records)

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
            view_inputs.append(self._build_stage1_input(payload))
            view_aux["pred_cam_t"].append(
                self._require_field(payload, "pred_cam_t", expected_last_dim=3)
            )
            view_aux["cam_int"].append(self._require_matrix(payload, "cam_int", shape=(3, 3)))
            view_aux["image_size"].append(
                self._require_field(payload, "image_size", expected_last_dim=2)
            )
            camera_ids.append(view.camera_id)

        target_payload = self._load_gt_target(record)
        canonical_target = canonicalize_stage1_target(
            smpl_body_pose=self._require_field(target_payload, "body_pose", expected_last_dim=69),
            smpl_betas=self._require_field(target_payload, "betas", expected_last_dim=10),
        )
        camera_target_aux = self._build_camera_frame_target_aux(
            record=record,
            camera_ids=camera_ids,
            target_payload=target_payload,
        )

        return {
            "views_input": torch.from_numpy(np.stack(view_inputs, axis=0)),
            "target_body_pose": torch.from_numpy(canonical_target["target_body_pose"]),
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

    def _select_views(self, record: SampleRecord, *, index: int) -> list[Any]:
        if len(record.views) < self.num_views:
            raise ValueError(
                f"Record '{record.sample_id}' has {len(record.views)} views, "
                f"but num_views={self.num_views}"
            )

        ordered_views = sorted(record.views, key=lambda item: item.camera_id)
        if not self.train:
            return list(ordered_views[: self.num_views])

        rng = random.Random(self.seed + index)
        return rng.sample(list(ordered_views), k=self.num_views)

    def _load_gt_target(self, record: SampleRecord) -> dict[str, np.ndarray]:
        sequence_payload = self._load_gt_sequence(record.sequence_id)
        frame_index = self._frame_id_to_index(record.frame_id)

        target_payload: dict[str, np.ndarray] = {}
        for key in ("global_orient", "body_pose", "betas", "transl"):
            if key not in sequence_payload:
                raise KeyError(
                    f"Missing required GT SMPL field '{key}' for sequence '{record.sequence_id}'"
                )
            target_payload[key] = self._select_frame(sequence_payload[key], frame_index, key=key)
        return target_payload

    def _load_gt_sequence(self, sequence_id: str) -> dict[str, np.ndarray]:
        cached = self._gt_sequence_cache.get(sequence_id)
        if cached is not None:
            return cached

        sequence_path = self.gt_smpl_dir / f"{sequence_id}_smpl_params.npz"
        if not sequence_path.exists():
            raise FileNotFoundError(
                f"GT SMPL sequence file does not exist: {sequence_path}"
            )
        cached = self._load_npz(sequence_path)
        self._gt_sequence_cache[sequence_id] = cached
        return cached

    @staticmethod
    def _load_npz(npz_path: Path) -> dict[str, np.ndarray]:
        with np.load(npz_path, allow_pickle=False) as payload:
            return {key: payload[key] for key in payload.files}

    @staticmethod
    def _require_field(
        payload: dict[str, np.ndarray],
        key: str,
        *,
        expected_last_dim: int | None = None,
    ) -> np.ndarray:
        if key not in payload:
            raise KeyError(f"Missing required field '{key}'")
        value = np.asarray(payload[key], dtype=np.float32)
        if value.ndim == 0:
            raise ValueError(f"Field '{key}' must not be scalar")

        # Optional exporter output may include a leading person dimension.
        if value.ndim == 2 and value.shape[0] == 1:
            value = value[0]

        if expected_last_dim is not None and value.shape[-1] != expected_last_dim:
            raise ValueError(
                f"Field '{key}' has shape {value.shape}, expected trailing dim "
                f"{expected_last_dim}"
        )
        return np.ascontiguousarray(value)

    @staticmethod
    def _require_matrix(
        payload: dict[str, np.ndarray], key: str, *, shape: tuple[int, ...]
    ) -> np.ndarray:
        if key not in payload:
            raise KeyError(f"Missing required field '{key}'")
        value = np.asarray(payload[key], dtype=np.float32)
        if value.shape != shape:
            raise ValueError(f"Field '{key}' has shape {value.shape}, expected {shape}")
        return np.ascontiguousarray(value)

    def _build_stage1_input(self, payload: dict[str, np.ndarray]) -> np.ndarray:
        mhr_params = self._require_field(payload, "mhr_model_params", expected_last_dim=204)
        shape_params = self._require_field(payload, "shape_params", expected_last_dim=45)
        return np.concatenate([mhr_params, shape_params], axis=-1).astype(np.float32, copy=False)

    def _build_camera_frame_target_aux(
        self,
        *,
        record: SampleRecord,
        camera_ids: list[str],
        target_payload: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        world_global_orient = self._require_field(
            target_payload,
            "global_orient",
            expected_last_dim=3,
        )
        world_transl = self._require_field(target_payload, "transl", expected_last_dim=3)
        camera_global_orients = []
        camera_translations = []
        for camera_id in camera_ids:
            camera = load_camera_parameters(
                self.cameras_dir,
                sequence_id=record.sequence_id,
                camera_id=camera_id,
            )
            camera_global_orient, camera_transl = transform_smpl_world_to_camera(
                global_orient=world_global_orient,
                transl=world_transl,
                camera=camera,
            )
            camera_global_orients.append(camera_global_orient)
            camera_translations.append(camera_transl)

        return {
            "camera_global_orient": np.ascontiguousarray(
                np.stack(camera_global_orients, axis=0).astype(np.float32, copy=False)
            ),
            "camera_transl": np.ascontiguousarray(
                np.stack(camera_translations, axis=0).astype(np.float32, copy=False)
            ),
        }

    @staticmethod
    def _frame_id_to_index(frame_id: str) -> int:
        frame_number = int(frame_id)
        if frame_number < 1:
            raise ValueError(f"Expected 1-based positive frame_id, got '{frame_id}'")
        return frame_number - 1

    @staticmethod
    def _select_frame(array: np.ndarray, frame_index: int, *, key: str) -> np.ndarray:
        value = np.asarray(array, dtype=np.float32)
        if value.ndim == 1:
            return np.ascontiguousarray(value)
        if value.ndim != 2:
            raise ValueError(
                f"GT field '{key}' must have shape [T, D] or [D], got {value.shape}"
            )
        if not 0 <= frame_index < value.shape[0]:
            raise IndexError(
                f"Frame index {frame_index} is out of range for GT field '{key}' "
                f"with length {value.shape[0]}"
            )
        return np.ascontiguousarray(value[frame_index])
