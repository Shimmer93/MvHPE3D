"""HuMMan Stage 3 dataset that exposes sparse view-time SMPL tokens."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from ..splits import SampleRecord
from .humman_stage2_multiview import HuMManStage2Dataset


class HuMManStage3TokenDataset(HuMManStage2Dataset):
    """Dataset for target-frame SMPL refinement from view-time tokens.

    The target frame is still represented by ``views_input`` with shape
    ``[V, 148]`` so a Stage 2 backbone can produce the base prediction.
    Temporal context is represented by fixed-size token tensors:
    - ``view_time_tokens``: ``[window_size * V, 148]``
    - ``token_time_offsets``: ``[window_size * V]``
    - ``token_camera_indices``: ``[window_size * V]``
    - ``token_valid_mask``: ``[window_size * V]``
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
        causal: bool = False,
        token_sampling: str = "dense_sync",
        token_drop_prob: float = 0.5,
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
        if window_size < 1:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if not causal and window_size % 2 == 0:
            raise ValueError(
                f"Centered Stage 3 token window_size must be odd, got {window_size}"
            )
        if not 0.0 <= token_drop_prob < 1.0:
            raise ValueError(f"token_drop_prob must be in [0, 1), got {token_drop_prob}")
        valid_sampling = {
            "dense_sync",
            "sparse_interleaved",
            "random_view_time",
            "causal_sparse",
        }
        if token_sampling not in valid_sampling:
            raise ValueError(
                f"Unsupported token_sampling '{token_sampling}'. "
                f"Expected one of {sorted(valid_sampling)}."
            )
        self.causal = causal
        self.window_size = window_size
        self.window_radius = window_size // 2
        self.target_window_index = window_size - 1 if causal else self.window_radius
        self.token_sampling = token_sampling
        self.token_drop_prob = token_drop_prob
        self.sequence_records = self._group_records_by_sequence(records)
        self.window_index: list[tuple[str, int]] = []
        for sequence_id, sequence_records in self.sequence_records.items():
            for center_index, center_record in enumerate(sequence_records):
                if len(center_record.views) >= self.num_views:
                    self.window_index.append((sequence_id, center_index))

    def __len__(self) -> int:
        return len(self.window_index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sequence_id, center_index = self.window_index[index]
        sequence_records = self.sequence_records[sequence_id]
        center_record = sequence_records[center_index]
        selected_camera_ids = self._select_target_camera_ids(center_record, index=index)
        center_views = self._select_views_by_camera_ids(center_record, selected_camera_ids)

        target_view_inputs = []
        view_aux = {
            "input_global_orient": [],
            "input_transl": [],
        }
        for view in center_views:
            fitted_payload = self._load_cached_input_smpl(view.npz_path)
            target_view_inputs.append(self._build_stage2_input(fitted_payload))
            view_aux["input_global_orient"].append(
                self._require_field(fitted_payload, "global_orient", expected_last_dim=3)
            )
            view_aux["input_transl"].append(
                self._require_field(fitted_payload, "transl", expected_last_dim=3)
            )

        token_rng = random.Random(self.seed + 104729 * (index + 1))
        token_values: list[np.ndarray] = []
        token_time_offsets: list[int] = []
        token_camera_indices: list[int] = []
        token_valid_mask: list[bool] = []
        token_camera_ids: list[str] = []
        token_frame_ids: list[str] = []

        zero_token = np.zeros(138 + 10, dtype=np.float32)
        window_frame_ids: list[str] = []
        for time_position, offset in enumerate(self._window_offsets()):
            frame_index = min(max(center_index + offset, 0), len(sequence_records) - 1)
            frame_record = sequence_records[frame_index]
            frame_view_map = {view.camera_id: view for view in frame_record.views}
            window_frame_ids.append(frame_record.frame_id)
            for view_position, camera_id in enumerate(selected_camera_ids):
                include_token = self._include_token(
                    time_position=time_position,
                    view_position=view_position,
                    rng=token_rng,
                )
                view = frame_view_map.get(camera_id)
                if include_token and view is not None:
                    fitted_payload = self._load_cached_input_smpl(view.npz_path)
                    token_values.append(self._build_stage2_input(fitted_payload))
                    token_valid_mask.append(True)
                else:
                    token_values.append(zero_token)
                    token_valid_mask.append(False)
                token_time_offsets.append(offset)
                token_camera_indices.append(self._camera_id_to_index(camera_id))
                token_camera_ids.append(camera_id)
                token_frame_ids.append(frame_record.frame_id)

        if not any(token_valid_mask):
            self._restore_target_frame_tokens(
                token_values=token_values,
                token_valid_mask=token_valid_mask,
                target_view_inputs=target_view_inputs,
            )

        canonical_target = self._load_canonical_target(center_record)

        return {
            "views_input": torch.from_numpy(np.stack(target_view_inputs, axis=0)),
            "view_time_tokens": torch.from_numpy(np.stack(token_values, axis=0)),
            "token_time_offsets": torch.as_tensor(token_time_offsets, dtype=torch.long),
            "token_camera_indices": torch.as_tensor(token_camera_indices, dtype=torch.long),
            "token_valid_mask": torch.as_tensor(token_valid_mask, dtype=torch.bool),
            "target_body_pose": torch.from_numpy(canonical_target["target_body_pose"]),
            "target_body_pose_6d": torch.from_numpy(canonical_target["target_body_pose_6d"]),
            "target_betas": torch.from_numpy(canonical_target["target_betas"]),
            "view_aux": {
                key: torch.from_numpy(np.stack(values, axis=0))
                for key, values in view_aux.items()
            },
            "meta": {
                "sample_id": center_record.sample_id,
                "sequence_id": center_record.sequence_id,
                "frame_id": center_record.frame_id,
                "window_frame_ids": window_frame_ids,
                "target_window_index": self.target_window_index,
                "causal": self.causal,
                "camera_ids": selected_camera_ids,
                "token_sampling": self.token_sampling,
                "token_camera_ids": token_camera_ids,
                "token_frame_ids": token_frame_ids,
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
                f"Record '{record.sample_id}' is missing required target view(s): {missing}"
            )
        return [view_map[camera_id] for camera_id in camera_ids]

    def _window_offsets(self) -> range:
        if self.causal:
            return range(1 - self.window_size, 1)
        return range(-self.window_radius, self.window_radius + 1)

    def _select_target_camera_ids(
        self,
        record: SampleRecord,
        *,
        index: int,
    ) -> list[str]:
        camera_ids = sorted(view.camera_id for view in record.views)
        if len(camera_ids) < self.num_views:
            raise ValueError(
                f"Target frame only has {len(camera_ids)} views, but num_views={self.num_views}"
            )
        if not self.train:
            return camera_ids[: self.num_views]
        rng = random.Random(self.seed + index)
        return rng.sample(camera_ids, k=self.num_views)

    def _include_token(
        self,
        *,
        time_position: int,
        view_position: int,
        rng: random.Random,
    ) -> bool:
        if self.token_sampling == "dense_sync":
            return True
        if self.token_sampling in {"sparse_interleaved", "causal_sparse"}:
            return view_position == time_position % self.num_views
        if self.token_sampling == "random_view_time":
            return rng.random() >= self.token_drop_prob
        raise RuntimeError(f"Unhandled token_sampling: {self.token_sampling}")

    def _restore_target_frame_tokens(
        self,
        *,
        token_values: list[np.ndarray],
        token_valid_mask: list[bool],
        target_view_inputs: list[np.ndarray],
    ) -> None:
        target_start = self.target_window_index * self.num_views
        for view_position, target_input in enumerate(target_view_inputs):
            token_index = target_start + view_position
            token_values[token_index] = target_input
            token_valid_mask[token_index] = True

    @staticmethod
    def _camera_id_to_index(camera_id: str) -> int:
        digits = "".join(character for character in camera_id if character.isdigit())
        if digits:
            return int(digits)
        return sum(camera_id.encode("utf-8"))
