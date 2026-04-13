"""Manifest and split helpers for HuMMan multiview experiments."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ViewRecord:
    """One per-view prediction file."""

    camera_id: str
    npz_path: Path


@dataclass(frozen=True)
class SampleRecord:
    """One training example consisting of multiple views plus metadata."""

    sample_id: str
    sequence_id: str
    frame_id: str
    target_path: Path | None
    views: tuple[ViewRecord, ...]
    split: str | None = None
    subject_id: str | None = None
    action_id: str | None = None


@dataclass(frozen=True)
class DatasetSelector:
    """One split selector for train/val/test dataset construction."""

    split: str | None = None
    cameras: tuple[str, ...] | None = None
    subjects: tuple[str, ...] | None = None
    actions: tuple[str, ...] | None = None


def _normalize_path(value: str, *, manifest_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def _load_yaml(path: str | Path) -> dict[str, Any]:
    file_path = Path(path).resolve()
    with file_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in YAML file {file_path}, got {type(payload)!r}")
    return payload


def load_sample_records(manifest_path: str | Path) -> list[SampleRecord]:
    """Load sample records from a JSON manifest.

    Expected JSON format:
    ``{"samples": [...records...]}`` or a top-level list of record dictionaries.
    Each record should contain:
    - ``sample_id``
    - ``sequence_id``
    - ``frame_id``
    - ``target_path`` (optional legacy field; Stage 1 GT now comes from HuMMan SMPL files)
    - ``split`` (optional source split tag, e.g. ``training``)
    - ``subject_id`` (optional)
    - ``action_id`` (optional)
    - ``views``: list of ``{"camera_id", "npz_path"}``
    """
    manifest_file = Path(manifest_path).resolve()
    with manifest_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    raw_records = payload["samples"] if isinstance(payload, dict) else payload
    manifest_dir = manifest_file.parent

    records: list[SampleRecord] = []
    for raw_record in raw_records:
        raw_views = raw_record.get("views", [])
        views = tuple(
            ViewRecord(
                camera_id=str(raw_view["camera_id"]),
                npz_path=_normalize_path(raw_view["npz_path"], manifest_dir=manifest_dir),
            )
            for raw_view in raw_views
        )
        records.append(
            SampleRecord(
                sample_id=str(raw_record["sample_id"]),
                sequence_id=str(raw_record["sequence_id"]),
                frame_id=str(raw_record["frame_id"]),
                target_path=(
                    _normalize_path(raw_record["target_path"], manifest_dir=manifest_dir)
                    if raw_record.get("target_path")
                    else None
                ),
                views=views,
                split=str(raw_record["split"]) if raw_record.get("split") is not None else None,
                subject_id=(
                    str(raw_record["subject_id"])
                    if raw_record.get("subject_id") is not None
                    else None
                ),
                action_id=(
                    str(raw_record["action_id"])
                    if raw_record.get("action_id") is not None
                    else None
                ),
            )
        )

    return records


def filter_records_by_split(
    records: list[SampleRecord], split: str | None
) -> list[SampleRecord]:
    """Select records matching a legacy split label stored in the manifest."""
    if split is None:
        return records
    return [record for record in records if record.split == split]


def load_named_split_config(
    split_config_path: str | Path, split_name: str
) -> dict[str, Any]:
    """Load one named dataset split policy from YAML."""
    split_config = _load_yaml(split_config_path)
    try:
        policy = split_config[split_name]
    except KeyError as exc:
        available = ", ".join(sorted(split_config))
        raise KeyError(
            f"Split '{split_name}' was not found in {Path(split_config_path).resolve()}. "
            f"Available splits: {available}"
        ) from exc

    if not isinstance(policy, dict):
        raise TypeError(
            f"Expected split policy '{split_name}' to be a mapping, got {type(policy)!r}"
        )
    return policy


def resolve_split_records(
    records: list[SampleRecord],
    *,
    split_config_path: str | Path,
    split_name: str,
    num_views: int,
) -> dict[str, list[SampleRecord]]:
    """Resolve train/val/test records from a named split policy."""
    policy = load_named_split_config(split_config_path, split_name)

    if "ratio" in policy:
        candidate_selector = _selector_from_payload(policy.get("candidate_dataset"))
        candidate_records = filter_records_by_selector(
            records,
            candidate_selector,
            min_views=0,
        )
        partitioned_records = _partition_records(
            candidate_records,
            train_ratio=float(policy["ratio"]),
            seed=int(policy.get("random_seed", 0)),
            partition_by=_normalize_optional_string(policy.get("partition_by")) or "sample_id",
        )
        return {
            "train": filter_records_by_selector(
                partitioned_records["train"],
                _selector_from_payload(policy.get("train_dataset")),
                min_views=num_views,
            ),
            "val": filter_records_by_selector(
                partitioned_records["val"],
                _selector_from_payload(policy.get("val_dataset")),
                min_views=num_views,
            ),
            "test": filter_records_by_selector(
                partitioned_records["test"],
                _selector_from_payload(policy.get("test_dataset") or policy.get("val_dataset")),
                min_views=num_views,
            ),
        }

    return {
        "train": filter_records_by_selector(
            records,
            _selector_from_payload(policy.get("train_dataset")),
            min_views=num_views,
        ),
        "val": filter_records_by_selector(
            records,
            _selector_from_payload(policy.get("val_dataset")),
            min_views=num_views,
        ),
        "test": filter_records_by_selector(
            records,
            _selector_from_payload(policy.get("test_dataset") or policy.get("val_dataset")),
            min_views=num_views,
        ),
    }


def filter_records_by_selector(
    records: list[SampleRecord],
    selector: DatasetSelector,
    *,
    min_views: int,
) -> list[SampleRecord]:
    """Apply metadata and camera filters to records, trimming views per policy."""
    filtered_records: list[SampleRecord] = []
    allowed_cameras = set(selector.cameras) if selector.cameras is not None else None
    allowed_subjects = set(selector.subjects) if selector.subjects is not None else None
    allowed_actions = set(selector.actions) if selector.actions is not None else None

    for record in records:
        if selector.split is not None and record.split != selector.split:
            continue
        if allowed_subjects is not None and record.subject_id not in allowed_subjects:
            continue
        if allowed_actions is not None and record.action_id not in allowed_actions:
            continue

        selected_views = tuple(
            view
            for view in record.views
            if allowed_cameras is None or view.camera_id in allowed_cameras
        )
        if len(selected_views) < min_views:
            continue

        filtered_records.append(replace(record, views=selected_views))

    return filtered_records


def _partition_records(
    records: list[SampleRecord], *, train_ratio: float, seed: int, partition_by: str
) -> dict[str, list[SampleRecord]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"ratio must be between 0 and 1, got {train_ratio}")
    if partition_by not in {"sample_id", "sequence_id", "subject_id", "action_id"}:
        raise ValueError(
            "partition_by must be one of "
            "{'sample_id', 'sequence_id', 'subject_id', 'action_id'}, "
            f"got {partition_by!r}"
        )

    sorted_records = sorted(
        records,
        key=lambda record: (record.sample_id, record.sequence_id, record.frame_id),
    )
    grouped_records: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in sorted_records:
        group_key = getattr(record, partition_by)
        normalized_group_key = group_key if group_key is not None else "<none>"
        grouped_records[str(normalized_group_key)].append(record)

    shuffled_group_keys = sorted(grouped_records)
    random.Random(seed).shuffle(shuffled_group_keys)

    train_group_count = int(len(shuffled_group_keys) * train_ratio)
    train_group_keys = set(shuffled_group_keys[:train_group_count])

    train_records: list[SampleRecord] = []
    val_records: list[SampleRecord] = []
    for group_key in shuffled_group_keys:
        destination = train_records if group_key in train_group_keys else val_records
        destination.extend(grouped_records[group_key])

    return {
        "train": train_records,
        "val": val_records,
        "test": list(val_records),
    }


def _selector_from_payload(payload: dict[str, Any] | None) -> DatasetSelector:
    if payload is None:
        return DatasetSelector()
    if not isinstance(payload, dict):
        raise TypeError(f"Expected selector mapping, got {type(payload)!r}")

    return DatasetSelector(
        split=_normalize_optional_string(payload.get("split")),
        cameras=_normalize_optional_sequence(payload.get("cameras")),
        subjects=_normalize_optional_sequence(payload.get("subjects")),
        actions=_normalize_optional_sequence(payload.get("actions")),
    )


def _normalize_optional_sequence(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError(f"Expected list or null, got {type(value)!r}")
    if len(value) == 0:
        return tuple()
    return tuple(str(item) for item in value)


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def summarize_records(records: list[SampleRecord]) -> dict[str, Any]:
    """Return a lightweight manifest summary for logging or debugging."""
    split_counts: dict[str, int] = {}
    for record in records:
        split_name = record.split if record.split is not None else "<none>"
        split_counts[split_name] = split_counts.get(split_name, 0) + 1

    return {
        "num_records": len(records),
        "split_counts": split_counts,
    }
