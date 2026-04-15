"""Helpers for converting compact MHR parameters to fitted SMPL parameters."""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import sys
from pathlib import Path
import warnings

import numpy as np
import torch
import trimesh

from .smpl import resolve_smpl_model_path

REQUIRED_MHR_ASSET_FILES = ("mhr_model.pt",)
REQUIRED_MAPPING_FILE = "mhr2smpl_mapping.npz"
NUM_FACE_EXPRESSION_COEFFS = 72
RESULT_PARAMETER_KEYS = ("betas", "body_pose", "global_orient", "transl")


def cache_path_for_source_npz(cache_dir: str | Path, source_npz_path: str | Path) -> Path:
    """Return the fitted-SMPL cache path corresponding to one source prediction file."""
    resolved_cache_dir = Path(cache_dir).resolve()
    resolved_source = Path(source_npz_path).resolve()
    digest = hashlib.sha1(str(resolved_source).encode("utf-8")).hexdigest()[:12]
    stem = resolved_source.stem
    return resolved_cache_dir / f"{stem}.{digest}.npz"


def resolve_mhr_asset_folder(path_arg: str | None = None) -> Path:
    """Resolve the MHR asset directory from an explicit argument or known defaults."""
    if path_arg is not None:
        resolved = Path(path_arg).resolve()
        if resolved.is_file():
            resolved = resolved.parent
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
    """Validate that an MHR asset directory contains the scripted MHR model."""
    resolved = Path(folder).resolve()
    missing_files = [
        file_name
        for file_name in REQUIRED_MHR_ASSET_FILES
        if not (resolved / file_name).exists()
    ]
    if missing_files:
        missing_text = ", ".join(missing_files)
        raise FileNotFoundError(
            f"MHR asset directory is missing required files: {missing_text}. "
            f"Expected them under {resolved}"
        )
    return resolved


class MHRToSMPLConverter:
    """Lazy wrapper around the direct compact-params -> SMPL fitting path."""

    def __init__(
        self,
        *,
        smpl_model_path: str | None = None,
        mhr_assets_dir: str | None = None,
        cache_dir: str | None = None,
        batch_size: int = 256,
    ) -> None:
        self.smpl_model_path = resolve_smpl_model_path(smpl_model_path)
        self.mhr_assets_dir = resolve_mhr_asset_folder(mhr_assets_dir)
        self.cache_dir = Path(cache_dir).resolve() if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self._exporters: dict[str, dict[str, object]] = {}

    def convert(
        self,
        *,
        mhr_model_params: torch.Tensor,
        shape_params: torch.Tensor,
        pred_cam_t: torch.Tensor | None = None,
        source_npz_paths: list[str] | None = None,
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
        if pred_cam_t is not None and (pred_cam_t.ndim != 2 or pred_cam_t.shape[1] != 3):
            raise ValueError(
                f"Expected pred_cam_t with shape [B, 3], got {tuple(pred_cam_t.shape)}"
            )
        if source_npz_paths is not None and len(source_npz_paths) != mhr_model_params.shape[0]:
            raise ValueError(
                "Expected source_npz_paths to have the same length as the batch size, "
                f"got {len(source_npz_paths)} paths for batch size {mhr_model_params.shape[0]}"
            )

        if self.cache_dir is not None and source_npz_paths is not None:
            return self._convert_with_cache(
                mhr_model_params=mhr_model_params,
                shape_params=shape_params,
                pred_cam_t=pred_cam_t,
                source_npz_paths=source_npz_paths,
            )
        return self._fit_batch(
            mhr_model_params=mhr_model_params,
            shape_params=shape_params,
            pred_cam_t=pred_cam_t,
        )

    def _convert_with_cache(
        self,
        *,
        mhr_model_params: torch.Tensor,
        shape_params: torch.Tensor,
        pred_cam_t: torch.Tensor | None,
        source_npz_paths: list[str],
    ) -> dict[str, torch.Tensor]:
        assert self.cache_dir is not None
        num_samples = mhr_model_params.shape[0]
        assembled: dict[str, list[torch.Tensor | None]] = {
            key: [None] * num_samples for key in RESULT_PARAMETER_KEYS
        }
        missing_indices: list[int] = []
        missing_paths: list[str] = []

        for index, source_npz_path in enumerate(source_npz_paths):
            cached = self._load_cached_result(source_npz_path)
            if cached is None:
                missing_indices.append(index)
                missing_paths.append(source_npz_path)
                continue
            for key in RESULT_PARAMETER_KEYS:
                assembled[key][index] = cached[key]

        if missing_indices:
            index_tensor = torch.as_tensor(
                missing_indices,
                device=mhr_model_params.device,
                dtype=torch.long,
            )
            missing_results = self._fit_batch(
                mhr_model_params=mhr_model_params.index_select(0, index_tensor),
                shape_params=shape_params.index_select(0, index_tensor),
                pred_cam_t=(
                    pred_cam_t.index_select(0, index_tensor) if pred_cam_t is not None else None
                ),
            )
            for offset, source_npz_path in enumerate(missing_paths):
                single_result = {
                    key: missing_results[key][offset].detach().cpu()
                    for key in RESULT_PARAMETER_KEYS
                }
                self._save_cached_result(source_npz_path, single_result)
                original_index = missing_indices[offset]
                for key in RESULT_PARAMETER_KEYS:
                    assembled[key][original_index] = single_result[key]

        stacked: dict[str, torch.Tensor] = {}
        for key, tensors in assembled.items():
            if any(tensor is None for tensor in tensors):
                raise RuntimeError(f"Incomplete cached conversion result for key '{key}'")
            stacked[key] = torch.stack(
                [tensor for tensor in tensors if tensor is not None],
                dim=0,
            ).to(device=mhr_model_params.device, dtype=torch.float32)
        return stacked

    def _fit_batch(
        self,
        *,
        mhr_model_params: torch.Tensor,
        shape_params: torch.Tensor,
        pred_cam_t: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if mhr_model_params.shape[0] == 0:
            return {
                key: torch.empty((0, 10 if key == "betas" else 69 if key == "body_pose" else 3))
                for key in RESULT_PARAMETER_KEYS
            }

        source_device = mhr_model_params.device
        runtime_device = self._resolve_runtime_device(source_device)
        exporter = self._get_exporter(runtime_device)

        runtime_mhr_params = mhr_model_params.to(device=runtime_device, dtype=torch.float32)
        runtime_shape_params = shape_params.to(device=runtime_device, dtype=torch.float32)
        runtime_pred_cam_t = (
            pred_cam_t.to(device=runtime_device, dtype=torch.float32)
            if pred_cam_t is not None
            else torch.zeros(
                (mhr_model_params.shape[0], 3),
                device=runtime_device,
                dtype=torch.float32,
            )
        )

        with torch.inference_mode(False):
            with torch.autocast(device_type=runtime_device.type, enabled=False):
                target_vertices = self._build_target_vertices(
                    exporter=exporter,
                    mhr_model_params=runtime_mhr_params,
                    shape_params=runtime_shape_params,
                    pred_cam_t=runtime_pred_cam_t,
                )
                with torch.enable_grad():
                    results = exporter["solver"].fit(
                        target_vertices=target_vertices,
                        single_identity=False,
                        is_tracking=False,
                    )
        return {
            key: value.to(source_device)
            for key, value in results.items()
            if isinstance(value, torch.Tensor)
        }

    def _cache_path_for_source(self, source_npz_path: str) -> Path:
        assert self.cache_dir is not None
        return cache_path_for_source_npz(self.cache_dir, source_npz_path)

    def _load_cached_result(self, source_npz_path: str) -> dict[str, torch.Tensor] | None:
        cache_path = self._cache_path_for_source(source_npz_path)
        if not cache_path.exists():
            return None
        with np.load(cache_path, allow_pickle=False) as payload:
            return {
                key: torch.from_numpy(np.asarray(payload[key], dtype=np.float32))
                for key in RESULT_PARAMETER_KEYS
            }

    def _save_cached_result(
        self,
        source_npz_path: str,
        result: dict[str, torch.Tensor],
    ) -> None:
        cache_path = self._cache_path_for_source(source_npz_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp_path,
            source_npz_path=str(Path(source_npz_path).resolve()),
            **{
                key: value.detach().cpu().numpy().astype(np.float32, copy=False)
                for key, value in result.items()
                if key in RESULT_PARAMETER_KEYS
            },
        )
        os.replace(tmp_path, cache_path)

    def _build_target_vertices(
        self,
        *,
        exporter: dict[str, object],
        mhr_model_params: torch.Tensor,
        shape_params: torch.Tensor,
        pred_cam_t: torch.Tensor,
    ) -> torch.Tensor:
        pred_vertices = self._reconstruct_pred_vertices(
            exporter=exporter,
            mhr_model_params=mhr_model_params,
            shape_params=shape_params,
        )
        camera_vertices = pred_vertices + pred_cam_t[:, None, :]
        mhr_faces = exporter["mhr_faces"]
        mapped_face_id = exporter["mapped_face_id"]
        baryc_coords = exporter["baryc_coords"][None, :, :, None]
        triangles = camera_vertices[:, mhr_faces[mapped_face_id], :]
        return (triangles * baryc_coords).sum(dim=2)

    @staticmethod
    def _reconstruct_pred_vertices(
        *,
        exporter: dict[str, object],
        mhr_model_params: torch.Tensor,
        shape_params: torch.Tensor,
    ) -> torch.Tensor:
        face_expr_coeffs = torch.zeros(
            (mhr_model_params.shape[0], NUM_FACE_EXPRESSION_COEFFS),
            device=mhr_model_params.device,
            dtype=mhr_model_params.dtype,
        )
        with torch.no_grad():
            mhr_vertices_cm, _ = exporter["scripted_mhr_model"](
                shape_params,
                mhr_model_params,
                face_expr_coeffs,
                True,
            )
        pred_vertices = mhr_vertices_cm * 0.01
        pred_vertices = pred_vertices.clone()
        pred_vertices[..., [1, 2]] *= -1.0
        return pred_vertices

    def _get_exporter(self, device: torch.device) -> dict[str, object]:
        cache_key = str(device)
        if cache_key in self._exporters:
            return self._exporters[cache_key]

        self._register_local_mhr_conversion_paths()

        import smplx  # type: ignore
        from pytorch_fitting import PyTorchSMPLFitting  # type: ignore

        scripted_mhr_model = torch.jit.load(
            str(self.mhr_assets_dir / "mhr_model.pt"),
            map_location="cpu",
        ).to(device)
        scripted_mhr_model.eval()
        mhr_faces = scripted_mhr_model.character_torch.mesh.faces.to(
            device=device,
            dtype=torch.long,
        )

        mapped_face_id_np, baryc_coords_np = self._load_surface_mapping()
        mapped_face_id = torch.from_numpy(mapped_face_id_np).to(
            device=device,
            dtype=torch.long,
        )
        baryc_coords = torch.from_numpy(baryc_coords_np).to(
            device=device,
            dtype=torch.float32,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"You are using a SMPL model, with only 10 shape coefficients\.",
            )
            warnings.filterwarnings(
                "ignore",
                category=np.exceptions.VisibleDeprecationWarning,
                module=r"smplx\.body_models",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                smpl_model = smplx.SMPL(
                    model_path=str(self.smpl_model_path),
                    gender="neutral",
                ).to(str(device))
        smpl_template_mesh = trimesh.Trimesh(
            smpl_model.v_template.detach().cpu().numpy(),
            smpl_model.faces,
            process=False,
        )
        smpl_edges = torch.from_numpy(smpl_template_mesh.edges_unique.copy()).long()
        solver = PyTorchSMPLFitting(
            smpl_model=smpl_model,
            smpl_edges=smpl_edges,
            smpl_model_type="smpl",
            hand_pose_dim=0,
            device=str(device),
            batch_size=self.batch_size,
        )
        exporter = {
            "scripted_mhr_model": scripted_mhr_model,
            "mhr_faces": mhr_faces,
            "mapped_face_id": mapped_face_id,
            "baryc_coords": baryc_coords,
            "solver": solver,
        }
        self._exporters[cache_key] = exporter
        return exporter

    @staticmethod
    def _register_local_mhr_conversion_paths() -> None:
        repo_root = Path(__file__).resolve().parents[3]
        candidate_paths = [
            repo_root / "external" / "MHR" / "tools" / "mhr_smpl_conversion",
            repo_root / "external" / "MHR",
        ]
        for candidate_path in candidate_paths:
            resolved = str(candidate_path.resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)

    @staticmethod
    def _load_surface_mapping() -> tuple[np.ndarray, np.ndarray]:
        repo_root = Path(__file__).resolve().parents[3]
        mapping_path = (
            repo_root
            / "external"
            / "MHR"
            / "tools"
            / "mhr_smpl_conversion"
            / "assets"
            / REQUIRED_MAPPING_FILE
        )
        if not mapping_path.exists():
            raise FileNotFoundError(f"MHR->SMPL mapping file not found: {mapping_path}")
        mapping = np.load(mapping_path)
        return (
            mapping["triangle_ids"].astype(np.int64, copy=False),
            mapping["baryc_coords"].astype(np.float32, copy=False),
        )

    @staticmethod
    def _resolve_runtime_device(source_device: torch.device) -> torch.device:
        requested_device = os.environ.get("MVHPE3D_MHR_CONVERTER_DEVICE", "").strip()
        if requested_device:
            device = torch.device(requested_device)
        elif source_device.type == "cuda":
            device = source_device
        else:
            device = torch.device("cpu")

        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"MVHPE3D_MHR_CONVERTER_DEVICE requested CUDA device '{device}', but CUDA is not available"
            )
        return device
