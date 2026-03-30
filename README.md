# MvHPE3D

Calibration-free multi-view SMPL fusion from single-view SAM3DBody predictions.

## Installation

This repository uses `uv` for the main Python environment and tracks the
modified SAM3DBody code as a git submodule.

### 1. Initialize the submodule

```bash
git submodule update --init --recursive
```

### 2. Sync the main environment

```bash
uv sync
```

This will create or update the local `.venv` from the checked-in
`pyproject.toml` and `uv.lock`.

### 3. Optional sanity checks

```bash
uv run python --version
uv run pytest -q
```

### 4. Running commands

```bash
uv run <command>
source .venv/bin/activate
```

Notes:
- The current project metadata pins Python `>=3.12,<3.13`.
- `sam3` is not currently declared in this repository's main `pyproject.toml`.
  Install it separately only if you specifically need the optional SAM3-backed
  detector or segmentor paths in `external/sam-3d-body/`.

### 5. Exporter location

The compact-parameter exporter used by this project is tracked in the submodule:
- `external/sam-3d-body/demo_save_compact_params.py`

The submodule is pinned in git, so preprocessing code is part of the
reproducible repository state rather than an untracked local script.

## Problem Setting

**Input**: `N >= 2` single-view SAM3DBody predictions from different views of the same person.

**Stage 1 model input per view**:
- `smpl_betas`
- `smpl_body_pose`

**Available auxiliary per-view fields**:
- `smpl_global_orient`
- `pred_cam_t`
- `cam_int`
- `image_size`

**Output**: A fused canonical SMPL estimate that leverages multi-view consistency. In Stage 1, the prediction target is the canonical body (`smpl_betas`, `smpl_body_pose`) with root translation centered at the pelvis and root rotation removed.

**Canonical target space**: pelvis-centered, SMPL root rotation-removed canonical space. The fused supervision target is defined in this body-centered coordinate system rather than an external world frame.

**Key Constraint**: Calibration-free. No known camera extrinsics are provided. This project is framed as multi-view SMPL fusion from single-view perspective predictions.

## Dataset

### HuMMan (Single-Person)

**Preprocessing pipeline**:
1. `/home/zpengac/mmhpe/MmMvHPE/tools/data_preprocess.py` -> `/opt/data/humman_cropped`
   - Crops RGB/depth around detected person (224x224)
   - Extracts LiDAR point cloud from depth
   - Provides GT: 3D keypoints (`skl/`), SMPL params (`smpl/`), camera intrinsics (`cameras/`)

2. `external/sam-3d-body/demo_save_compact_params.py` with `--export_smpl_params`
   - Input: cropped RGB images from `/opt/data/humman_cropped/rgb/`
   - Output per image used in this project: `smpl_betas`, `smpl_body_pose`, `smpl_global_orient`, `pred_cam_t`, `cam_int`, `image_size`
   - In Stage 1, `smpl_global_orient + pred_cam_t` are kept for visualization only and are not used as model inputs or supervision targets

**Assumption**: Single person per sequence. No cross-view person association is needed.

## Training

**Input**: Fixed `N` views per sample.

**Sampling protocols**:
- `cross_camera_split` (main): train on `{kinect_000, 002, 003, 004, 005, 006, 008, iphone}`, val on `{kinect_001, 007, 009}`
- `random_split` (secondary sanity check): random 80/20 train/val split across all cameras

**Stage 1 (Current)**: Simple MLP-based baseline
- Per-view encoder: MLP over concatenated single-view features
- Fusion: permutation-invariant pooling across views (DeepSets-style)
- Decoder: MLP that predicts fused canonical body parameters
- Stage 1 input progression:
  1. per-view input uses only `smpl_betas + smpl_body_pose`
  2. fuse across views in canonical space
  3. predict fused `smpl_betas + smpl_body_pose`
- Supervision: GT SMPL parameters transformed to the pelvis-centered, root rotation-removed canonical space
- `smpl_global_orient + pred_cam_t` are excluded from Stage 1 model input and used only for qualitative visualization

**Stage 2 (Future)**:
- Add explicit per-view relative camera prediction for reprojection and image-space visualization
- Explore stronger set encoders or optimization-based fusion methods

## Evaluation

- MPJPE (Mean Per-Joint Position Error) in canonical space
- PA-MPJPE (Procrustes-Aligned MPJPE) in canonical space
- MVE (Mesh Vertex Error) in canonical space

## Visualization

- Side-by-side GT vs prediction in canonical space with arbitrary rotation
- Stage 1 image overlay is a visualization trick, not a learned output
- Rendering procedure for Stage 1 overlays:
  1. take the fused canonical `smpl_betas + smpl_body_pose`
  2. attach each view's exported `smpl_global_orient + pred_cam_t`
  3. render with that view's exported `cam_int`
- This means Stage 1 overlays reuse single-view camera/root predictions from SAM3DBody to place the fused body back into each image
- Therefore, image overlays are qualitative sanity checks only and should not be interpreted as evidence that Stage 1 has learned per-view camera geometry
- Future: replace the above with model-predicted per-view relative cameras
