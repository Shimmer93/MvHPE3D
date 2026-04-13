#!/usr/bin/env python
"""Precompute cached input-view fitted SMPL parameters for all manifest samples."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mvhpe3d.data.splits import load_sample_records
from mvhpe3d.utils import MHRToSMPLConverter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute cached fitted SMPL parameters for every view in a manifest"
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        required=True,
        help="Path to the HuMMan manifest JSON",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional override for the fitted SMPL cache directory",
    )
    parser.add_argument(
        "--smpl-model-path",
        type=str,
        default=None,
        help="Optional override for the neutral SMPL model path",
    )
    parser.add_argument(
        "--mhr-assets-dir",
        type=str,
        default=None,
        help="Optional override for the MHR asset directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Runtime device for conversion: auto, cpu, cuda, or cuda:N",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for the MHR->SMPL fitting solver",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N batches",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_path).resolve()
    cache_dir = resolve_cache_dir(manifest_path, args.cache_dir)
    device = resolve_device(args.device)
    os.environ["MVHPE3D_MHR_CONVERTER_DEVICE"] = str(device)

    records = load_sample_records(manifest_path)
    npz_paths = collect_unique_view_npz_paths(records)

    print(f"Manifest: {manifest_path}")
    print(f"Samples: {len(records)}")
    print(f"Unique view files: {len(npz_paths)}")
    print(f"Cache dir: {cache_dir}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")

    if not npz_paths:
        print("No view files found in manifest. Nothing to precompute.")
        return

    converter = MHRToSMPLConverter(
        smpl_model_path=args.smpl_model_path,
        mhr_assets_dir=args.mhr_assets_dir,
        cache_dir=str(cache_dir),
        batch_size=args.batch_size,
    )

    num_batches = (len(npz_paths) + args.batch_size - 1) // args.batch_size
    for batch_index, batch_paths in enumerate(chunked(npz_paths, args.batch_size), start=1):
        batch_payload = [load_compact_view_fields(path) for path in batch_paths]
        mhr_model_params = torch.from_numpy(
            np.stack([payload["mhr_model_params"] for payload in batch_payload], axis=0)
        ).to(device=device, dtype=torch.float32)
        shape_params = torch.from_numpy(
            np.stack([payload["shape_params"] for payload in batch_payload], axis=0)
        ).to(device=device, dtype=torch.float32)
        pred_cam_t = torch.from_numpy(
            np.stack([payload["pred_cam_t"] for payload in batch_payload], axis=0)
        ).to(device=device, dtype=torch.float32)

        converter.convert(
            mhr_model_params=mhr_model_params,
            shape_params=shape_params,
            pred_cam_t=pred_cam_t,
            source_npz_paths=[str(path) for path in batch_paths],
        )

        if batch_index == 1 or batch_index == num_batches or batch_index % max(args.progress_every, 1) == 0:
            processed = min(batch_index * args.batch_size, len(npz_paths))
            print(f"Processed {processed}/{len(npz_paths)} view files")

    print("Finished precomputing fitted SMPL cache.")


def resolve_cache_dir(manifest_path: Path, cache_dir: str | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).resolve()
    return (manifest_path.parent / "sam3dbody_fitted_smpl").resolve()


def resolve_device(device_arg: str) -> torch.device:
    requested = device_arg.strip()
    if requested == "auto":
        env_requested = os.environ.get("MVHPE3D_MHR_CONVERTER_DEVICE", "").strip()
        if env_requested:
            return torch.device(env_requested)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def collect_unique_view_npz_paths(records: list) -> list[Path]:
    unique_paths = {view.npz_path.resolve() for record in records for view in record.views}
    return sorted(unique_paths)


def load_compact_view_fields(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as payload:
        return {
            "mhr_model_params": require_field(payload, "mhr_model_params", expected_last_dim=204),
            "shape_params": require_field(payload, "shape_params", expected_last_dim=45),
            "pred_cam_t": require_field(payload, "pred_cam_t", expected_last_dim=3),
        }


def require_field(
    payload: np.lib.npyio.NpzFile,
    key: str,
    *,
    expected_last_dim: int,
) -> np.ndarray:
    if key not in payload:
        raise KeyError(f"Missing required field '{key}' in {payload.zip.filename}")
    value = np.asarray(payload[key], dtype=np.float32)
    if value.ndim == 0:
        raise ValueError(f"Field '{key}' in {payload.zip.filename} must not be scalar")
    if value.ndim == 2 and value.shape[0] == 1:
        value = value[0]
    if value.shape[-1] != expected_last_dim:
        raise ValueError(
            f"Field '{key}' in {payload.zip.filename} has shape {value.shape}, "
            f"expected trailing dim {expected_last_dim}"
        )
    return np.ascontiguousarray(value)


def chunked(items: list[Path], chunk_size: int) -> list[list[Path]]:
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


if __name__ == "__main__":
    main()
