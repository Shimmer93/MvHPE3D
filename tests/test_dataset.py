from __future__ import annotations

from pathlib import Path

from mvhpe3d.data import Stage1DataConfig, Stage1HuMManDataModule
from mvhpe3d.data.datasets import HuMManStage1Dataset
from mvhpe3d.data.splits import filter_records_by_split, load_sample_records


def test_load_sample_records_and_filter_by_split(sample_manifest: Path) -> None:
    records = load_sample_records(sample_manifest)

    assert len(records) == 3
    assert len(filter_records_by_split(records, "train")) == 1
    assert len(filter_records_by_split(records, "val")) == 1
    assert len(filter_records_by_split(records, "test")) == 1


def test_humman_stage1_dataset_returns_expected_schema(sample_manifest: Path) -> None:
    records = filter_records_by_split(load_sample_records(sample_manifest), "train")
    dataset = HuMManStage1Dataset(records, num_views=2, train=False)

    sample = dataset[0]

    assert tuple(sample["views_input"].shape) == (2, 79)
    assert tuple(sample["target_betas"].shape) == (10,)
    assert tuple(sample["target_body_pose"].shape) == (69,)
    assert tuple(sample["view_aux"]["smpl_global_orient"].shape) == (2, 3)
    assert tuple(sample["view_aux"]["pred_cam_t"].shape) == (2, 3)
    assert tuple(sample["view_aux"]["cam_int"].shape) == (2, 3, 3)
    assert tuple(sample["view_aux"]["image_size"].shape) == (2, 2)
    assert sample["meta"]["sample_id"] == "sample_train"
    assert len(sample["meta"]["camera_ids"]) == 2


def test_stage1_datamodule_builds_train_val_and_test_batches(sample_manifest: Path) -> None:
    datamodule = Stage1HuMManDataModule(
        Stage1DataConfig(
            manifest_path=str(sample_manifest),
            num_views=2,
            batch_size=1,
            drop_last_train=False,
        )
    )

    datamodule.prepare_data()
    datamodule.setup(None)

    train_batch = next(iter(datamodule.train_dataloader()))
    val_batch = next(iter(datamodule.val_dataloader()))
    test_batch = next(iter(datamodule.test_dataloader()))

    assert tuple(train_batch["views_input"].shape) == (1, 2, 79)
    assert tuple(val_batch["views_input"].shape) == (1, 2, 79)
    assert tuple(test_batch["views_input"].shape) == (1, 2, 79)
    assert train_batch["meta"][0]["sample_id"] == "sample_train"
    assert val_batch["meta"][0]["sample_id"] == "sample_val"
    assert test_batch["meta"][0]["sample_id"] == "sample_test"
