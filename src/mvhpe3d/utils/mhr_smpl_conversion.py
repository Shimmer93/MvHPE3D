"""Helpers for converting MHR parameters to SMPL parameters."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from .smpl import resolve_smpl_model_path

REQUIRED_MHR_ASSET_FILES = (
    "compact_v6_1.model",
    "lod1.fbx",
    "corrective_blendshapes_lod1.npz",
    "corrective_activation.npz",
)


def resolve_mhr_asset_folder(path_arg: str | None = None) -> Path:
    """Resolve the MHR asset directory from an explicit argument or known defaults."""
    if path_arg is not None:
        resolved = Path(path_arg).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"MHR asset directory does not exist: {resolved}")
        validate_mhr_asset_folder(resolved)
        return resolved

    candidates = [
        Path("/opt/data/assets"),
        Path(__file__).resolve().parents[3] / "external" / "MHR" / "assets",
    ]
    for candidate in candidates:
        if candidate.exists():
            resolved = candidate.resolve()
            validate_mhr_asset_folder(resolved)
            return resolved

    raise FileNotFoundError(
        "Could not resolve MHR assets. Pass --mhr-assets-dir explicitly."
    )


def validate_mhr_asset_folder(folder: str | Path) -> Path:
    """Validate that an MHR asset directory contains the required files."""
    resolved = Path(folder).resolve()
    missing_files = [
        file_name for file_name in REQUIRED_MHR_ASSET_FILES if not (resolved / file_name).exists()
    ]
    if missing_files:
        missing_text = ", ".join(missing_files)
        raise FileNotFoundError(
            f"MHR asset directory is missing required files: {missing_text}. "
            f"Expected them under {resolved}"
        )
    return resolved


class MHRToSMPLConverter:
    """Lazy wrapper around the external MHR-to-SMPL fitting tool."""

    def __init__(
        self,
        *,
        smpl_model_path: str | None = None,
        mhr_assets_dir: str | None = None,
        batch_size: int = 256,
    ) -> None:
        self.smpl_model_path = resolve_smpl_model_path(smpl_model_path)
        self.mhr_assets_dir = resolve_mhr_asset_folder(mhr_assets_dir)
        self.batch_size = batch_size
        self._converter = None

    def convert(
        self,
        *,
        mhr_model_params: torch.Tensor,
        shape_params: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Convert batched MHR parameters to fitted SMPL parameters."""
        if mhr_model_params.ndim != 2 or mhr_model_params.shape[1] != 204:
            raise ValueError(
                f"Expected mhr_model_params with shape [B, 204], got {tuple(mhr_model_params.shape)}"
            )
        if shape_params.ndim != 2 or shape_params.shape[1] != 45:
            raise ValueError(
                f"Expected shape_params with shape [B, 45], got {tuple(shape_params.shape)}"
            )

        converter = self._get_converter()
        expr_params = torch.zeros(
            (mhr_model_params.shape[0], 72),
            dtype=mhr_model_params.dtype,
            device=mhr_model_params.device,
        )
        results = converter.convert_mhr2smpl(
            mhr_parameters={
                "lbs_model_params": mhr_model_params,
                "identity_coeffs": shape_params,
                "face_expr_coeffs": expr_params,
            },
            single_identity=False,
            is_tracking=False,
            return_smpl_parameters=True,
            return_smpl_vertices=False,
            return_smpl_meshes=False,
            return_fitting_errors=False,
            batch_size=self.batch_size,
        )
        if results.result_parameters is None:
            raise RuntimeError("External MHR-to-SMPL conversion returned no parameters")
        return {
            key: value
            for key, value in results.result_parameters.items()
            if isinstance(value, torch.Tensor)
        }

    def _get_converter(self):
        if self._converter is not None:
            return self._converter

        repo_root = Path(__file__).resolve().parents[3]
        tool_dir = repo_root / "external" / "MHR" / "tools" / "mhr_smpl_conversion"
        mhr_root = repo_root / "external" / "MHR"
        for path in (tool_dir, mhr_root):
            path_str = str(path.resolve())
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

        from conversion import Conversion  # type: ignore
        from mhr.mhr import MHR  # type: ignore
        import smplx  # type: ignore

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mhr_model = MHR.from_files(folder=self.mhr_assets_dir, device=device, lod=1)
        smpl_model = smplx.SMPL(
            model_path=str(self.smpl_model_path),
            gender="neutral",
            batch_size=1,
        )
        self._converter = Conversion(
            mhr_model=mhr_model,
            smpl_model=smpl_model,
            method="pytorch",
            batch_size=self.batch_size,
        )
        return self._converter
