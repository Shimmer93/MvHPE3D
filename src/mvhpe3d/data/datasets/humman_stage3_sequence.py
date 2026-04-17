"""HuMMan temporal Stage 3 dataset built from Stage 2 per-view fitted SMPL."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from ..splits import SampleRecord
from .humman_stage2_multiview import HuMManStage2Dataset


class HuMManStage3Dataset(HuMManStage2Dataset):
    """Dataset for temporal refinement over Stage 2 per-frame multiview inputs.

    Each returned sample follows the schema:
    - ``views_input``: tensor of shape ``[T, V, 148]``
    - ``target_body_pose``: center-frame canonical target SMPL body pose
    - ``target_body_pose_6d``: center-frame target pose in 6D rotation form
    - ``target_betas``: center-frame target SMPL shape coefficients
    - ``meta``: center-frame ids plus the full temporal window frame ids
    """

    def __init__(
        self,
        records: list[SampleRecord],
        *,
        num_views: int,
        train: bool,
        gt_smpl_dir: str | Path,
        cameras_dir: str | Path,
        input_smpl_cache_dir: str | Path,
        window_size: int,
        seed: int = 0,
    ) -> None:
        super().__init__(
            records,
            num_views=num_views,
            train=train,
            gt_smpl_dir=gt_smpl_dir,
            cameras_dir=cameras_dir,
            input_smpl_cache_dir=input_smpl_cache_dir,
            seed=seed,
        )
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError(f"window_size must be a positive odd integer, got {window_size}")
        self.window_size = window_size
        self.window_radius = window_size // 2
        self.sequence_records = self._group_records_by_sequence(records)
        self.window_index: list[tuple[str, int, tuple[str, ...]]] = []
        for sequence_id, sequence_records in self.sequence_records.items():
            for center_index in range(len(sequence_records)):
                common_camera_ids = self._resolve_window_camera_ids(
                    sequence_records,
                    center_index=center_index,
                )
                if len(common_camera_ids) >= self.num_views:
                    self.window_index.append((sequence_id, center_index, common_camera_ids))

    def __len__(self) -> int:
        return len(self.window_index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sequence_id, center_index, common_camera_ids = self.window_index[index]
        sequence_records = self.sequence_records[sequence_id]
        center_record = sequence_records[center_index]
        selected_camera_ids = self._select_temporal_camera_ids(
            common_camera_ids,
            index=index,
        )

        temporal_inputs: list[np.ndarray] = []
        window_frame_ids: list[str] = []
        window_view_npz_paths: list[list[str]] = []
        for offset in range(-self.window_radius, self.window_radius + 1):
            frame_index = min(max(center_index + offset, 0), len(sequence_records) - 1)
            frame_record = sequence_records[frame_index]
            frame_views = self._select_views_by_camera_ids(frame_record, selected_camera_ids)
            frame_inputs = []
            frame_npz_paths: list[str] = []
            for view in frame_views:
                fitted_payload = self._load_cached_input_smpl(view.npz_path)
                frame_inputs.append(self._build_stage2_input(fitted_payload))
                frame_npz_paths.append(str(view.npz_path))
            temporal_inputs.append(np.stack(frame_inputs, axis=0))
            window_frame_ids.append(frame_record.frame_id)
            window_view_npz_paths.append(frame_npz_paths)

        canonical_target = self._load_canonical_target(center_record)

        return {
            "views_input": torch.from_numpy(np.stack(temporal_inputs, axis=0)),
            "target_body_pose": torch.from_numpy(canonical_target["target_body_pose"]),
            "target_body_pose_6d": torch.from_numpy(canonical_target["target_body_pose_6d"]),
            "target_betas": torch.from_numpy(canonical_target["target_betas"]),
            "meta": {
                "sample_id": center_record.sample_id,
                "sequence_id": center_record.sequence_id,
                "frame_id": center_record.frame_id,
                "window_frame_ids": window_frame_ids,
                "camera_ids": selected_camera_ids,
                "window_view_npz_paths": window_view_npz_paths,
            },
        }

    @staticmethod
    def _group_records_by_sequence(records: list[SampleRecord]) -> dict[str, list[SampleRecord]]:
        grouped: dict[str, list[SampleRecord]] = defaultdict(list)
        for record in records:
            grouped[record.sequence_id].append(record)
        return {
            sequence_id: sorted(items, key=lambda record: int(record.frame_id))
            for sequence_id, items in grouped.items()
        }

    @staticmethod
    def _select_views_by_camera_ids(
        record: SampleRecord, camera_ids: list[str]
    ) -> list[Any]:
        view_map = {view.camera_id: view for view in record.views}
        missing = [camera_id for camera_id in camera_ids if camera_id not in view_map]
        if missing:
            raise KeyError(
                f"Record '{record.sample_id}' is missing required temporal view(s): {missing}"
            )
        return [view_map[camera_id] for camera_id in camera_ids]

    def _resolve_window_camera_ids(
        self,
        sequence_records: list[SampleRecord],
        *,
        center_index: int,
    ) -> tuple[str, ...]:
        common_camera_ids: set[str] | None = None
        for offset in range(-self.window_radius, self.window_radius + 1):
            frame_index = min(max(center_index + offset, 0), len(sequence_records) - 1)
            frame_record = sequence_records[frame_index]
            frame_camera_ids = {view.camera_id for view in frame_record.views}
            if common_camera_ids is None:
                common_camera_ids = frame_camera_ids
            else:
                common_camera_ids &= frame_camera_ids
            if common_camera_ids is not None and len(common_camera_ids) < self.num_views:
                return ()
        assert common_camera_ids is not None
        return tuple(sorted(common_camera_ids))

    def _select_temporal_camera_ids(
        self,
        common_camera_ids: tuple[str, ...],
        *,
        index: int,
    ) -> list[str]:
        if len(common_camera_ids) < self.num_views:
            raise ValueError(
                f"Temporal window only has {len(common_camera_ids)} shared views, "
                f"but num_views={self.num_views}"
            )
        if not self.train:
            return list(common_camera_ids[: self.num_views])
        rng = random.Random(self.seed + index)
        return rng.sample(list(common_camera_ids), k=self.num_views)
