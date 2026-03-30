"""Manifest and split helpers for HuMMan multiview experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ViewRecord:
    """One per-view prediction file."""

    camera_id: str
    npz_path: Path


@dataclass(frozen=True)
class SampleRecord:
    """One training example consisting of multiple views and one canonical target."""

    sample_id: str
    sequence_id: str
    frame_id: str
    split: str
    target_path: Path | None
    views: tuple[ViewRecord, ...]


def _normalize_path(value: str, *, manifest_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def load_sample_records(manifest_path: str | Path) -> list[SampleRecord]:
    """Load sample records from a JSON manifest.

    Expected JSON format:
    ``{"samples": [...records...]}`` or a top-level list of record dictionaries.
    Each record should contain:
    - ``sample_id``
    - ``sequence_id``
    - ``frame_id``
    - ``split``
    - ``target_path`` (optional)
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
                split=str(raw_record["split"]),
                target_path=(
                    _normalize_path(raw_record["target_path"], manifest_dir=manifest_dir)
                    if raw_record.get("target_path")
                    else None
                ),
                views=views,
            )
        )

    return records


def filter_records_by_split(
    records: list[SampleRecord], split: str | None
) -> list[SampleRecord]:
    """Select records matching a split name."""
    if split is None:
        return records
    return [record for record in records if record.split == split]


def summarize_records(records: list[SampleRecord]) -> dict[str, Any]:
    """Return a lightweight manifest summary for logging or debugging."""
    split_counts: dict[str, int] = {}
    for record in records:
        split_counts[record.split] = split_counts.get(record.split, 0) + 1

    return {
        "num_records": len(records),
        "split_counts": split_counts,
    }
