"""Lightning DataModule for Stage 1 HuMMan experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from .collate import multiview_collate
from .datasets import HuMManStage1Dataset, HuMManStage2Dataset
from .splits import filter_records_by_split, load_sample_records, resolve_split_records


@dataclass(slots=True)
class Stage1DataConfig:
    """Configuration for the Stage 1 HuMMan LightningDataModule."""

    manifest_path: str
    gt_smpl_dir: str | None = None
    cameras_dir: str | None = None
    split_config_path: str | None = None
    split_name: str | None = None
    num_views: int = 2
    batch_size: int = 16
    num_workers: int = 0
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    seed: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    drop_last_train: bool = True


class Stage1HuMManDataModule(L.LightningDataModule):
    """DataModule for Stage 1 canonical-body fusion experiments."""

    def __init__(self, config: Stage1DataConfig) -> None:
        super().__init__()
        self.config = config
        self.train_dataset: HuMManStage1Dataset | None = None
        self.val_dataset: HuMManStage1Dataset | None = None
        self.test_dataset: HuMManStage1Dataset | None = None

    def prepare_data(self) -> None:
        manifest_path = Path(self.config.manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
        gt_smpl_dir = self._resolve_gt_smpl_dir()
        if not gt_smpl_dir.exists():
            raise FileNotFoundError(f"GT SMPL directory does not exist: {gt_smpl_dir}")
        cameras_dir = self._resolve_cameras_dir()
        if not cameras_dir.exists():
            raise FileNotFoundError(f"Cameras directory does not exist: {cameras_dir}")
        if self.config.split_config_path is not None:
            split_config_path = Path(self.config.split_config_path)
            if not split_config_path.exists():
                raise FileNotFoundError(f"Split config does not exist: {split_config_path}")

    def setup(self, stage: str | None = None) -> None:
        records = load_sample_records(self.config.manifest_path)
        selected_records = self._resolve_dataset_records(records)
        gt_smpl_dir = self._resolve_gt_smpl_dir()
        cameras_dir = self._resolve_cameras_dir()

        if stage in (None, "fit"):
            self.train_dataset = HuMManStage1Dataset(
                selected_records["train"],
                num_views=self.config.num_views,
                train=True,
                gt_smpl_dir=gt_smpl_dir,
                cameras_dir=cameras_dir,
                seed=self.config.seed,
            )
            self.val_dataset = HuMManStage1Dataset(
                selected_records["val"],
                num_views=self.config.num_views,
                train=False,
                gt_smpl_dir=gt_smpl_dir,
                cameras_dir=cameras_dir,
                seed=self.config.seed,
            )

        if stage in (None, "validate") and self.val_dataset is None:
            self.val_dataset = HuMManStage1Dataset(
                selected_records["val"],
                num_views=self.config.num_views,
                train=False,
                gt_smpl_dir=gt_smpl_dir,
                cameras_dir=cameras_dir,
                seed=self.config.seed,
            )

        if stage in (None, "test"):
            self.test_dataset = HuMManStage1Dataset(
                selected_records["test"],
                num_views=self.config.num_views,
                train=False,
                gt_smpl_dir=gt_smpl_dir,
                cameras_dir=cameras_dir,
                seed=self.config.seed,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not initialized; call setup('fit') first")
        return self._build_dataloader(
            self.train_dataset,
            shuffle=True,
            drop_last=self.config.drop_last_train,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is not initialized; call setup() first")
        return self._build_dataloader(self.val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is not initialized; call setup('test') first")
        return self._build_dataloader(self.test_dataset, shuffle=False, drop_last=False)

    def _build_dataloader(
        self,
        dataset: HuMManStage1Dataset,
        *,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=(
                self.config.persistent_workers if self.config.num_workers > 0 else False
            ),
            drop_last=drop_last,
            collate_fn=multiview_collate,
        )

    def _resolve_dataset_records(self, records: list) -> dict[str, list]:
        if self.config.split_config_path is not None:
            if self.config.split_name is None:
                raise ValueError("split_name must be set when split_config_path is provided")
            return resolve_split_records(
                records,
                split_config_path=self.config.split_config_path,
                split_name=self.config.split_name,
                num_views=self.config.num_views,
            )

        return {
            "train": filter_records_by_split(records, self.config.train_split),
            "val": filter_records_by_split(records, self.config.val_split),
            "test": filter_records_by_split(records, self.config.test_split),
        }

    def _resolve_gt_smpl_dir(self) -> Path:
        if self.config.gt_smpl_dir is not None:
            return Path(self.config.gt_smpl_dir).resolve()

        manifest_path = Path(self.config.manifest_path).resolve()
        return (manifest_path.parent / "smpl").resolve()

    def _resolve_cameras_dir(self) -> Path:
        if self.config.cameras_dir is not None:
            return Path(self.config.cameras_dir).resolve()

        manifest_path = Path(self.config.manifest_path).resolve()
        return (manifest_path.parent / "cameras").resolve()


@dataclass(slots=True)
class Stage2DataConfig:
    """Configuration for the Stage 2 HuMMan LightningDataModule."""

    manifest_path: str
    gt_smpl_dir: str | None = None
    cameras_dir: str | None = None
    input_smpl_cache_dir: str | None = None
    split_config_path: str | None = None
    split_name: str | None = None
    num_views: int = 2
    batch_size: int = 16
    num_workers: int = 0
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    seed: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    drop_last_train: bool = True


class Stage2HuMManDataModule(L.LightningDataModule):
    """DataModule for Stage 2 canonical-parameter fusion experiments."""

    def __init__(self, config: Stage2DataConfig) -> None:
        super().__init__()
        self.config = config
        self.train_dataset: HuMManStage2Dataset | None = None
        self.val_dataset: HuMManStage2Dataset | None = None
        self.test_dataset: HuMManStage2Dataset | None = None

    def prepare_data(self) -> None:
        manifest_path = Path(self.config.manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
        gt_smpl_dir = self._resolve_gt_smpl_dir()
        if not gt_smpl_dir.exists():
            raise FileNotFoundError(f"GT SMPL directory does not exist: {gt_smpl_dir}")
        cameras_dir = self._resolve_cameras_dir()
        if not cameras_dir.exists():
            raise FileNotFoundError(f"Cameras directory does not exist: {cameras_dir}")
        input_smpl_cache_dir = self._resolve_input_smpl_cache_dir()
        if not input_smpl_cache_dir.exists():
            raise FileNotFoundError(
                "Input SMPL cache directory does not exist: "
                f"{input_smpl_cache_dir}. Run scripts/precompute_input_smpl.py first."
            )
        if self.config.split_config_path is not None:
            split_config_path = Path(self.config.split_config_path)
            if not split_config_path.exists():
                raise FileNotFoundError(f"Split config does not exist: {split_config_path}")

    def setup(self, stage: str | None = None) -> None:
        records = load_sample_records(self.config.manifest_path)
        selected_records = self._resolve_dataset_records(records)
        gt_smpl_dir = self._resolve_gt_smpl_dir()
        cameras_dir = self._resolve_cameras_dir()
        input_smpl_cache_dir = self._resolve_input_smpl_cache_dir()

        if stage in (None, "fit"):
            self.train_dataset = HuMManStage2Dataset(
                selected_records["train"],
                num_views=self.config.num_views,
                train=True,
                gt_smpl_dir=gt_smpl_dir,
                cameras_dir=cameras_dir,
                input_smpl_cache_dir=input_smpl_cache_dir,
                seed=self.config.seed,
            )
            self.val_dataset = HuMManStage2Dataset(
                selected_records["val"],
                num_views=self.config.num_views,
                train=False,
                gt_smpl_dir=gt_smpl_dir,
                cameras_dir=cameras_dir,
                input_smpl_cache_dir=input_smpl_cache_dir,
                seed=self.config.seed,
            )

        if stage in (None, "validate") and self.val_dataset is None:
            self.val_dataset = HuMManStage2Dataset(
                selected_records["val"],
                num_views=self.config.num_views,
                train=False,
                gt_smpl_dir=gt_smpl_dir,
                cameras_dir=cameras_dir,
                input_smpl_cache_dir=input_smpl_cache_dir,
                seed=self.config.seed,
            )

        if stage in (None, "test"):
            self.test_dataset = HuMManStage2Dataset(
                selected_records["test"],
                num_views=self.config.num_views,
                train=False,
                gt_smpl_dir=gt_smpl_dir,
                cameras_dir=cameras_dir,
                input_smpl_cache_dir=input_smpl_cache_dir,
                seed=self.config.seed,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not initialized; call setup('fit') first")
        return self._build_dataloader(
            self.train_dataset,
            shuffle=True,
            drop_last=self.config.drop_last_train,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is not initialized; call setup() first")
        return self._build_dataloader(self.val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is not initialized; call setup('test') first")
        return self._build_dataloader(self.test_dataset, shuffle=False, drop_last=False)

    def _build_dataloader(
        self,
        dataset: HuMManStage2Dataset,
        *,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=(
                self.config.persistent_workers if self.config.num_workers > 0 else False
            ),
            drop_last=drop_last,
            collate_fn=multiview_collate,
        )

    def _resolve_dataset_records(self, records: list) -> dict[str, list]:
        if self.config.split_config_path is not None:
            if self.config.split_name is None:
                raise ValueError("split_name must be set when split_config_path is provided")
            return resolve_split_records(
                records,
                split_config_path=self.config.split_config_path,
                split_name=self.config.split_name,
                num_views=self.config.num_views,
            )

        return {
            "train": filter_records_by_split(records, self.config.train_split),
            "val": filter_records_by_split(records, self.config.val_split),
            "test": filter_records_by_split(records, self.config.test_split),
        }

    def _resolve_gt_smpl_dir(self) -> Path:
        if self.config.gt_smpl_dir is not None:
            return Path(self.config.gt_smpl_dir).resolve()

        manifest_path = Path(self.config.manifest_path).resolve()
        return (manifest_path.parent / "smpl").resolve()

    def _resolve_cameras_dir(self) -> Path:
        if self.config.cameras_dir is not None:
            return Path(self.config.cameras_dir).resolve()

        manifest_path = Path(self.config.manifest_path).resolve()
        return (manifest_path.parent / "cameras").resolve()

    def _resolve_input_smpl_cache_dir(self) -> Path:
        if self.config.input_smpl_cache_dir is not None:
            return Path(self.config.input_smpl_cache_dir).resolve()

        manifest_path = Path(self.config.manifest_path).resolve()
        return (manifest_path.parent / "sam3dbody_fitted_smpl").resolve()
