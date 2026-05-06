from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


def _load_infer_script_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "infer_multiview.py"
    spec = importlib.util.spec_from_file_location("mvhpe3d_infer_multiview_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load inference script from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_npz_frame_views_matches_common_stems(tmp_path: Path) -> None:
    infer = _load_infer_script_module()
    view_a = tmp_path / "view_a"
    view_b = tmp_path / "view_b"
    view_a.mkdir()
    view_b.mkdir()
    for stem in ("frame_000001", "frame_000002", "only_a"):
        np.savez(view_a / f"{stem}.npz", body_pose=np.zeros(69), betas=np.zeros(10))
    for stem in ("frame_000001", "frame_000002", "only_b"):
        np.savez(view_b / f"{stem}.npz", body_pose=np.zeros(69), betas=np.zeros(10))

    frame_views = infer.collect_npz_frame_views([view_a, view_b], mode="video")

    assert [[path.stem for path in frame] for frame in frame_views] == [
        ["frame_000001", "frame_000001"],
        ["frame_000002", "frame_000002"],
    ]


def test_temporal_window_indices_support_centered_and_causal() -> None:
    infer = _load_infer_script_module()

    assert infer.temporal_window_indices(
        frame_index=0,
        num_frames=4,
        window_size=3,
        causal=False,
    ) == [0, 0, 1]
    assert infer.temporal_window_indices(
        frame_index=2,
        num_frames=4,
        window_size=3,
        causal=False,
    ) == [1, 2, 3]
    assert infer.temporal_window_indices(
        frame_index=0,
        num_frames=4,
        window_size=3,
        causal=True,
    ) == [0, 0, 0]
    assert infer.temporal_window_indices(
        frame_index=2,
        num_frames=4,
        window_size=3,
        causal=True,
    ) == [0, 1, 2]


def test_build_stage2_input_from_direct_smpl_payload(tmp_path: Path) -> None:
    infer = _load_infer_script_module()
    npz_path = tmp_path / "view.npz"
    payload = {
        "body_pose": np.zeros((1, 69), dtype=np.float32),
        "betas": np.ones((1, 10), dtype=np.float32),
    }
    direct = infer.read_direct_smpl_parameters(payload, person_index=0, npz_path=npz_path)

    assert direct is not None
    stage2_input = infer.build_stage2_input(
        body_pose=direct["body_pose"],
        betas=direct["betas"],
    )

    assert stage2_input.shape == (148,)
    assert np.allclose(stage2_input[-10:], np.ones(10, dtype=np.float32))


def test_select_person_vector_rejects_missing_person(tmp_path: Path) -> None:
    infer = _load_infer_script_module()
    npz_path = tmp_path / "empty.npz"
    value = np.zeros((0, 204), dtype=np.float32)

    try:
        infer.select_person_vector(
            value,
            expected_dim=204,
            person_index=0,
            key="mhr_model_params",
            npz_path=npz_path,
        )
    except IndexError as exc:
        assert "has 0 people" in str(exc)
    else:
        raise AssertionError("Expected missing person to raise IndexError")


def test_resolve_device_accepts_explicit_cpu() -> None:
    infer = _load_infer_script_module()

    assert infer.resolve_device("cpu") == torch.device("cpu")
