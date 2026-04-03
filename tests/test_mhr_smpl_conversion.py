from __future__ import annotations

from pathlib import Path

import pytest

from mvhpe3d.utils import validate_mhr_asset_folder


def test_validate_mhr_asset_folder_accepts_complete_directory(tmp_path: Path) -> None:
    for file_name in (
        "compact_v6_1.model",
        "lod1.fbx",
        "corrective_blendshapes_lod1.npz",
        "corrective_activation.npz",
    ):
        (tmp_path / file_name).write_bytes(b"ok")

    resolved = validate_mhr_asset_folder(tmp_path)

    assert resolved == tmp_path.resolve()


def test_validate_mhr_asset_folder_reports_missing_files(tmp_path: Path) -> None:
    (tmp_path / "compact_v6_1.model").write_bytes(b"ok")

    with pytest.raises(FileNotFoundError, match="lod1.fbx"):
        validate_mhr_asset_folder(tmp_path)
