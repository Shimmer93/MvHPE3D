"""Lightning DataModule for Stage 1 HuMMan experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from .collate import multiview_collate
from .datasets import HuMManStage1Dataset
from .splits import filter_records_by_split, load_sample_records


@dataclass(slots=True)
class Stage1DataConfig:
    """Configuration for the Stage 1 HuMMan LightningDataModule."""

    manifest_path: str
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

    def setup(self, stage: str | None = None) -> None:
        records = load_sample_records(self.config.manifest_path)

        if stage in (None, "fit"):
            self.train_dataset = HuMManStage1Dataset(
                filter_records_by_split(records, self.config.train_split),
                num_views=self.config.num_views,
                train=True,
                seed=self.config.seed,
            )
            self.val_dataset = HuMManStage1Dataset(
                filter_records_by_split(records, self.config.val_split),
                num_views=self.config.num_views,
                train=False,
                seed=self.config.seed,
            )

        if stage in (None, "validate") and self.val_dataset is None:
            self.val_dataset = HuMManStage1Dataset(
                filter_records_by_split(records, self.config.val_split),
                num_views=self.config.num_views,
                train=False,
                seed=self.config.seed,
            )

        if stage in (None, "test"):
            self.test_dataset = HuMManStage1Dataset(
                filter_records_by_split(records, self.config.test_split),
                num_views=self.config.num_views,
                train=False,
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
