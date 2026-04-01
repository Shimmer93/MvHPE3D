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


class HuMManStage1Dataset(Dataset[dict[str, Any]]):
    """Dataset for Stage 1 canonical body fusion.

    Each returned sample follows the schema:
    - ``views_input``: tensor of shape ``[N, D]`` from per-view mhr_model_params + shape_params
    - ``target_mhr_params``: canonical target MHR model parameters
    - ``target_shape_params``: canonical target shape parameters
    - ``view_aux``: per-view visualization fields
    - ``meta``: identifiers and camera names kept as Python data
    """

    def __init__(
        self,
        records: list[SampleRecord],
        *,
        num_views: int,
        train: bool,
        seed: int = 0,
    ) -> None:
        if num_views < 1:
            raise ValueError(f"num_views must be >= 1, got {num_views}")
        self.records = records
        self.num_views = num_views
        self.train = train
        self.seed = seed

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

        target_payload = self._load_target_payload(record)
        canonical_target = canonicalize_stage1_target(
            mhr_model_params=self._require_field(target_payload, "mhr_model_params"),
            shape_params=self._require_field(
                target_payload, "shape_params", expected_last_dim=45
            ),
        )

        return {
            "views_input": torch.from_numpy(np.stack(view_inputs, axis=0)),
            "target_mhr_params": torch.from_numpy(canonical_target["target_mhr_params"]),
            "target_shape_params": torch.from_numpy(canonical_target["target_shape_params"]),
            "view_aux": {
                key: torch.from_numpy(np.stack(values, axis=0))
                for key, values in view_aux.items()
            },
            "meta": {
                "sample_id": record.sample_id,
                "sequence_id": record.sequence_id,
                "frame_id": record.frame_id,
                "camera_ids": camera_ids,
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

    def _load_target_payload(self, record: SampleRecord) -> dict[str, np.ndarray]:
        if record.target_path is None:
            first_view = sorted(record.views, key=lambda item: item.camera_id)[0]
            return self._load_npz(first_view.npz_path)
        return self._load_npz(record.target_path)

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
        mhr_params = self._require_field(payload, "mhr_model_params")
        shape_params = self._require_field(payload, "shape_params", expected_last_dim=45)
        return np.concatenate([mhr_params, shape_params], axis=-1).astype(np.float32, copy=False)
