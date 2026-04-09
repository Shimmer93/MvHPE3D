from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mvhpe3d.utils import MHRToSMPLConverter, validate_mhr_asset_folder


def test_validate_mhr_asset_folder_accepts_complete_directory(tmp_path: Path) -> None:
    (tmp_path / "mhr_model.pt").write_bytes(b"ok")

    resolved = validate_mhr_asset_folder(tmp_path)

    assert resolved == tmp_path.resolve()


def test_validate_mhr_asset_folder_reports_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="mhr_model.pt"):
        validate_mhr_asset_folder(tmp_path)


def test_converter_caches_results_by_source_npz_path(tmp_path: Path, monkeypatch) -> None:
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    (asset_dir / "mhr_model.pt").write_bytes(b"ok")
    cache_dir = tmp_path / "cache"

    converter = MHRToSMPLConverter(
        mhr_assets_dir=str(asset_dir),
        cache_dir=str(cache_dir),
    )
    call_count = {"value": 0}

    def fake_fit_batch(**kwargs):
        call_count["value"] += 1
        batch_size = kwargs["mhr_model_params"].shape[0]
        device = kwargs["mhr_model_params"].device
        return {
            "betas": torch.full((batch_size, 10), 1.0, device=device),
            "body_pose": torch.full((batch_size, 69), 2.0, device=device),
            "global_orient": torch.full((batch_size, 3), 3.0, device=device),
            "transl": torch.full((batch_size, 3), 4.0, device=device),
        }

    monkeypatch.setattr(converter, "_fit_batch", fake_fit_batch)
    source_path = str(tmp_path / "sample.npz")

    first = converter.convert(
        mhr_model_params=torch.zeros(1, 204),
        shape_params=torch.zeros(1, 45),
        pred_cam_t=torch.zeros(1, 3),
        source_npz_paths=[source_path],
    )
    second = converter.convert(
        mhr_model_params=torch.ones(1, 204),
        shape_params=torch.ones(1, 45),
        pred_cam_t=torch.ones(1, 3),
        source_npz_paths=[source_path],
    )

    assert call_count["value"] == 1
    assert torch.equal(first["betas"], second["betas"])
    assert any(cache_dir.glob("sample.*.npz"))
