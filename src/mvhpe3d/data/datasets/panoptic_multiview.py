"""Panoptic/Kinoptic multiview Stage 1 dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..splits import SampleRecord
from .humman_multiview import HuMManStage1Dataset
from mvhpe3d.utils import PANOPTIC_GT_UNIT_SCALE, load_panoptic_camera_parameters


class PanopticStage1Dataset(HuMManStage1Dataset):
    """Stage 1 dataset adapter for cropped Panoptic/Kinoptic inputs.

    The per-view inputs use the same SAM3DBody compact `.npz` schema as HuMMan.
    The supervision target is still SMPL-compatible `body_pose + betas`; raw
    Panoptic 19-joint `gt3d/*.npy` files are not used directly by this model.

    Supported target layouts:
    - per-frame manifest `target_path` pointing at an `.npz` with
      `global_orient`, `body_pose`, `betas`, and `transl`
    - sequence cache `<gt_smpl_dir>/<sequence_id>_smpl_params.npz`, optionally
      with a `frame_ids` array for sparse/non-contiguous Panoptic frame ids
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
        super().__init__(
            records,
            num_views=num_views,
            train=train,
            gt_smpl_dir=gt_smpl_dir,
            cameras_dir=cameras_dir,
            seed=seed,
        )
        self._frame_index_cache: dict[str, dict[str, int]] = {}

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        sample = super().__getitem__(index)
        native_gt = self._load_native_panoptic_gt3d(record)
        if native_gt is not None:
            sample["target_aux"]["panoptic_joints_world"] = torch.from_numpy(
                np.ascontiguousarray(
                    native_gt[:, :3].astype(np.float32, copy=False)
                    * PANOPTIC_GT_UNIT_SCALE
                )
            )
            sample["target_aux"]["panoptic_confidence"] = torch.from_numpy(
                np.ascontiguousarray(native_gt[:, 3].astype(np.float32, copy=False))
            )
        return sample

    def _load_gt_target(self, record: SampleRecord) -> dict[str, np.ndarray]:
        if record.target_path is not None:
            if not record.target_path.exists():
                raise FileNotFoundError(
                    f"Panoptic target SMPL file does not exist: {record.target_path}"
                )
            return self._normalize_target_payload(
                self._load_npz(record.target_path),
                source=str(record.target_path),
            )

        sequence_payload = self._load_gt_sequence(record.sequence_id)
        frame_index = self._resolve_sequence_frame_index(
            sequence_id=record.sequence_id,
            sequence_payload=sequence_payload,
            frame_id=record.frame_id,
        )

        target_payload: dict[str, np.ndarray] = {}
        for key in ("global_orient", "body_pose", "betas", "transl"):
            if key not in sequence_payload:
                raise KeyError(
                    f"Missing required GT SMPL field '{key}' for sequence '{record.sequence_id}'"
                )
            target_payload[key] = self._select_frame(
                sequence_payload[key], frame_index, key=key
            )
        return target_payload

    def _load_camera_parameters(self, *, record: SampleRecord, camera_id: str):
        return load_panoptic_camera_parameters(
            self.cameras_dir,
            sequence_id=record.sequence_id,
            camera_id=camera_id,
        )

    def _resolve_sequence_frame_index(
        self,
        *,
        sequence_id: str,
        sequence_payload: dict[str, np.ndarray],
        frame_id: str,
    ) -> int:
        if "frame_ids" not in sequence_payload:
            return self._frame_id_to_index(frame_id)

        frame_index = self._frame_index_cache.get(sequence_id)
        if frame_index is None:
            raw_frame_ids = np.asarray(sequence_payload["frame_ids"]).reshape(-1)
            frame_index = {
                self._normalize_frame_id(raw_frame_id): index
                for index, raw_frame_id in enumerate(raw_frame_ids)
            }
            self._frame_index_cache[sequence_id] = frame_index

        normalized_frame_id = self._normalize_frame_id(frame_id)
        if normalized_frame_id not in frame_index:
            raise KeyError(
                f"Frame id '{frame_id}' was not found in SMPL target frame_ids "
                f"for sequence '{sequence_id}'"
            )
        return frame_index[normalized_frame_id]

    def _normalize_target_payload(
        self,
        payload: dict[str, np.ndarray],
        *,
        source: str,
    ) -> dict[str, np.ndarray]:
        target_payload: dict[str, np.ndarray] = {}
        for key, expected_last_dim in (
            ("global_orient", 3),
            ("body_pose", 69),
            ("betas", 10),
            ("transl", 3),
        ):
            if key not in payload:
                raise KeyError(f"Missing required GT SMPL field '{key}' in {source}")
            target_payload[key] = self._require_field(
                payload,
                key,
                expected_last_dim=expected_last_dim,
            )
        return target_payload

    @staticmethod
    def _normalize_frame_id(value) -> str:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, np.generic):
            value = value.item()
        text = str(value).strip()
        if text.endswith(".0"):
            text = text[:-2]
        if text.isdigit():
            return str(int(text))
        return text

    def _load_native_panoptic_gt3d(self, record: SampleRecord) -> np.ndarray | None:
        gt3d_path = self._resolve_native_panoptic_gt3d_path(record)
        if gt3d_path is None:
            return None
        payload = np.load(gt3d_path, allow_pickle=True)
        array = np.asarray(payload, dtype=np.float32)
        if array.shape != (19, 4):
            raise ValueError(
                f"Expected Panoptic gt3d file with shape (19, 4), got {array.shape}: {gt3d_path}"
            )
        return np.ascontiguousarray(array)

    def _resolve_native_panoptic_gt3d_path(self, record: SampleRecord) -> Path | None:
        normalized_frame_id = self._normalize_frame_id(record.frame_id)
        frame_stems = [record.frame_id]
        if normalized_frame_id.isdigit():
            frame_stems.insert(0, f"{int(normalized_frame_id):08d}")

        for frame_stem in dict.fromkeys(frame_stems):
            candidate = (
                self.cameras_dir / record.sequence_id / "gt3d" / f"{frame_stem}.npy"
            )
            if candidate.exists():
                return candidate
        return None
