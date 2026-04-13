from __future__ import annotations

import json
from pathlib import Path

from mvhpe3d.data import Stage1DataConfig, Stage1HuMManDataModule
from mvhpe3d.data.datasets import HuMManStage1Dataset
from mvhpe3d.data.splits import (
    filter_records_by_split,
    load_sample_records,
    resolve_split_records,
)


def test_load_sample_records_and_filter_by_split(sample_manifest: Path) -> None:
    records = load_sample_records(sample_manifest)

    assert len(records) == 3
    assert len(filter_records_by_split(records, "train")) == 1
    assert len(filter_records_by_split(records, "val")) == 1
    assert len(filter_records_by_split(records, "test")) == 1


def test_humman_stage1_dataset_returns_expected_schema(sample_manifest: Path) -> None:
    records = filter_records_by_split(load_sample_records(sample_manifest), "train")
    dataset = HuMManStage1Dataset(
        records,
        num_views=2,
        train=False,
        gt_smpl_dir=sample_manifest.parent / "smpl",
        cameras_dir=sample_manifest.parent / "cameras",
    )

    sample = dataset[0]

    assert tuple(sample["views_input"].shape) == (2, 249)
    assert tuple(sample["target_body_pose"].shape) == (69,)
    assert tuple(sample["target_betas"].shape) == (10,)
    assert tuple(sample["view_aux"]["pred_cam_t"].shape) == (2, 3)
    assert tuple(sample["view_aux"]["cam_int"].shape) == (2, 3, 3)
    assert tuple(sample["view_aux"]["image_size"].shape) == (2, 2)
    assert tuple(sample["target_aux"]["global_orient"].shape) == (3,)
    assert tuple(sample["target_aux"]["transl"].shape) == (3,)
    assert tuple(sample["target_aux"]["camera_rotation"].shape) == (2, 3, 3)
    assert tuple(sample["target_aux"]["camera_translation"].shape) == (2, 3)
    assert tuple(sample["target_aux"]["camera_global_orient"].shape) == (2, 3)
    assert tuple(sample["target_aux"]["camera_transl"].shape) == (2, 3)
    assert sample["meta"]["sample_id"] == "sample_train"
    assert len(sample["meta"]["camera_ids"]) == 2
    assert len(sample["meta"]["view_npz_paths"]) == 2


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

    assert tuple(train_batch["views_input"].shape) == (1, 2, 249)
    assert tuple(val_batch["views_input"].shape) == (1, 2, 249)
    assert tuple(test_batch["views_input"].shape) == (1, 2, 249)
    assert train_batch["meta"][0]["sample_id"] == "sample_train"
    assert val_batch["meta"][0]["sample_id"] == "sample_val"
    assert test_batch["meta"][0]["sample_id"] == "sample_test"
    assert len(test_batch["meta"][0]["view_npz_paths"]) == 2


def test_resolve_split_records_filters_views_by_named_policy(
    sample_inventory_manifest: Path,
    sample_split_config: Path,
) -> None:
    records = load_sample_records(sample_inventory_manifest)
    split_records = resolve_split_records(
        records,
        split_config_path=str(sample_split_config),
        split_name="cross_camera_split",
        num_views=2,
    )

    assert len(split_records["train"]) == 4
    assert len(split_records["val"]) == 4
    assert {view.camera_id for view in split_records["train"][0].views} == {
        "kinect_000",
        "kinect_001",
    }
    assert {view.camera_id for view in split_records["val"][0].views} == {
        "kinect_002",
        "kinect_003",
    }


def test_stage1_datamodule_supports_split_config_selection(
    sample_inventory_manifest: Path,
    sample_split_config: Path,
) -> None:
    datamodule = Stage1HuMManDataModule(
        Stage1DataConfig(
            manifest_path=str(sample_inventory_manifest),
            split_config_path=str(sample_split_config),
            split_name="cross_camera_split",
            num_views=2,
            batch_size=1,
            drop_last_train=False,
        )
    )

    datamodule.prepare_data()
    datamodule.setup(None)

    train_batch = next(iter(datamodule.train_dataloader()))
    val_batch = next(iter(datamodule.val_dataloader()))

    assert set(train_batch["meta"][0]["camera_ids"]) == {"kinect_000", "kinect_001"}
    assert set(val_batch["meta"][0]["camera_ids"]) == {"kinect_002", "kinect_003"}


def test_resolve_split_records_supports_random_partition(
    sample_inventory_manifest: Path,
    sample_split_config: Path,
) -> None:
    records = load_sample_records(sample_inventory_manifest)
    split_records = resolve_split_records(
        records,
        split_config_path=str(sample_split_config),
        split_name="random_training_only",
        num_views=2,
    )

    train_ids = {record.sample_id for record in split_records["train"]}
    val_ids = {record.sample_id for record in split_records["val"]}

    assert train_ids
    assert val_ids
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids <= {"sample_0", "sample_1", "sample_2"}


def test_resolve_split_records_random_partition_can_group_by_sequence(
    tmp_path: Path,
    sample_split_config: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    samples = [
        {
            "sample_id": "seq_a_frame_1",
            "sequence_id": "seq_a",
            "frame_id": "000001",
            "views": [],
        },
        {
            "sample_id": "seq_a_frame_2",
            "sequence_id": "seq_a",
            "frame_id": "000002",
            "views": [],
        },
        {
            "sample_id": "seq_b_frame_1",
            "sequence_id": "seq_b",
            "frame_id": "000001",
            "views": [],
        },
        {
            "sample_id": "seq_b_frame_2",
            "sequence_id": "seq_b",
            "frame_id": "000002",
            "views": [],
        },
    ]
    manifest_path.write_text(json.dumps({"samples": samples}), encoding="utf-8")

    split_config_path = tmp_path / "splits.yaml"
    split_config_path.write_text(
        """
random_by_sequence:
  ratio: 0.5
  random_seed: 0
  partition_by: sequence_id
  candidate_dataset:
    split: null
    cameras: null
    subjects: null
    actions: null
  train_dataset:
    split: null
    cameras: null
    subjects: null
    actions: null
  val_dataset:
    split: null
    cameras: null
    subjects: null
    actions: null
  test_dataset:
    split: null
    cameras: null
    subjects: null
    actions: null
        """.strip(),
        encoding="utf-8",
    )

    records = load_sample_records(manifest_path)
    split_records = resolve_split_records(
        records,
        split_config_path=str(split_config_path),
        split_name="random_by_sequence",
        num_views=0,
    )

    train_sequences = {record.sequence_id for record in split_records["train"]}
    val_sequences = {record.sequence_id for record in split_records["val"]}

    assert train_sequences
    assert val_sequences
    assert train_sequences.isdisjoint(val_sequences)
    for sequence_id in ("seq_a", "seq_b"):
        train_count = sum(record.sequence_id == sequence_id for record in split_records["train"])
        val_count = sum(record.sequence_id == sequence_id for record in split_records["val"])
        assert (train_count == 0) != (val_count == 0)
