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
- `mhr_model_params`
- `shape_params`

**Stage 2 model input per view**:
- cached fitted SMPL `body_pose`
- cached fitted SMPL `betas`
- internally converted to canonical `body_pose_6d + betas` for fusion/refinement

**Available auxiliary per-view fields**:
- `pred_cam_t`
- `cam_int`
- `image_size`

**Output**: A fused canonical SMPL estimate that leverages multi-view consistency. In both Stage 1 and Stage 2, the prediction target is the canonical body (`smpl_betas`, `smpl_body_pose`) with root translation centered at the pelvis and root rotation removed.

**Canonical target space**: pelvis-centered, SMPL root rotation-removed canonical space. The fused supervision target is defined in this body-centered coordinate system rather than an external world frame.

**Key Constraint**: Calibration-free. No known camera extrinsics are provided. This project is framed as multi-view SMPL fusion from single-view perspective predictions.

## Dataset

### HuMMan (Single-Person)

**Preprocessing pipeline**:
1. `/home/zpengac/mmhpe/MmMvHPE/tools/data_preprocess.py` -> `/opt/data/humman_cropped`
   - Crops RGB/depth around detected person (224x224)
   - Extracts LiDAR point cloud from depth
   - Provides GT: 3D keypoints (`skl/`), SMPL params (`smpl/`), camera intrinsics (`cameras/`)

2. `external/sam-3d-body/demo_save_compact_params.py`
   - Input: cropped RGB images from `/opt/data/humman_cropped/rgb/`
   - Output per image used as Stage 1 input: `mhr_model_params`, `shape_params`, `pred_cam_t`, `cam_int`, `image_size`
   - The training manifest is a split-agnostic sample inventory. Optional fields such as `split`, `subject_id`, and `action_id` are metadata used by split policies, not fixed train/val ownership.

3. Build the Stage 1 manifest from the exported `.npz` predictions:

```bash
uv run python scripts/build_humman_stage1_manifest.py \
  --predictions-dir /opt/data/humman_cropped/sam3dbody \
  --output-path /opt/data/humman_cropped/humman_stage1_manifest.json \
  --min-views 2
```

This keeps only valid single-person exports and groups them by
`sequence_id + frame_id`. Training supervision now always comes from the
HuMMan GT SMPL sequence files in `smpl/`, not from any exported prediction file.

4. Precompute cached fitted SMPL parameters for each exported view if you want
   to train or evaluate Stage 2:

```bash
uv run python scripts/precompute_input_smpl.py \
  --manifest-path /opt/data/humman_cropped/humman_stage1_manifest.json \
  --cache-dir /opt/data/humman_cropped/sam3dbody_fitted_smpl
```

This creates one cached fitted-SMPL file per source SAM3DBody prediction. Stage
2 uses these cached per-view fits as its primary model input representation.

**Assumption**: Single person per sequence. No cross-view person association is needed.

## Training

**Input**: Fixed `N` views per sample.

Run the Stage 1 fusion baseline with:

```bash
bash scripts/train_fusion.sh \
  --manifest-path /path/to/humman_stage1_manifest.json
```

This wrapper calls `scripts/train.py` with the default Stage 1 cross-camera
experiment config and writes each run under
`outputs/stage1/<experiment_name>/version_x/`, including `metrics.csv`,
optional W&B metadata, and `checkpoints/`. If `wandb` is importable, training
also enables a W&B logger automatically; set `WANDB_PROJECT` to override the
default project name `MvHPE3D`, and set `WANDB_MODE=disabled` to turn it off.
You can forward extra training flags such as `--max-epochs 50`,
`--fast-dev-run`, or a custom `--default-root-dir`. By default the datamodule
looks for GT SMPL under `$(dirname manifest)/smpl`; override that with
`--gt-smpl-dir /path/to/smpl` if your manifest lives elsewhere.

To run a test pass automatically right after training:

```bash
bash scripts/train_fusion.sh \
  --manifest-path /path/to/humman_stage1_manifest.json \
  --gt-smpl-dir /path/to/humman_cropped/smpl \
  --test-after-train \
  --test-ckpt best
```

`--test-ckpt` accepts:
- `best`: test the best validation checkpoint if available
- `last`: test the last saved checkpoint if available
- `current`: test the in-memory model weights directly

To switch split protocols without regenerating the manifest:

```bash
bash scripts/train_fusion.sh \
  --manifest-path /path/to/humman_stage1_manifest.json \
  --split-name random_split
```

The default split policy file is `configs/data/humman_stage1_splits.yaml`. It
contains named policies such as `cross_camera_split` and `random_split`.

Run the Stage 2 parameter-space fusion/refinement model with:

```bash
uv run python scripts/train.py \
  --config configs/experiment/stage2_cross_camera.yaml \
  --manifest-path /path/to/humman_stage1_manifest.json \
  --input-smpl-cache-dir /path/to/sam3dbody_fitted_smpl
```

Stage 2 still requires the manifest because it defines sample grouping, split
resolution, per-view source `.npz` paths, and default GT/camera lookup roots.
The fitted-SMPL cache replaces the Stage 1 compact per-view model input, not
the manifest itself.

Important:
- exported SAM3DBody `.npz` files must include `mhr_model_params` and `shape_params`
- supervision still comes from HuMMan GT SMPL under `smpl/`
- validation/test-time camera-frame metrics require the MHR conversion assets
  (default lookup: `/opt/data/assets`)
- Stage 2 training requires a precomputed fitted-SMPL cache directory such as
  `$(dirname manifest)/sam3dbody_fitted_smpl`

For multi-GPU training, forward Lightning trainer overrides from the CLI:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_fusion.sh \
  --manifest-path /path/to/humman_stage1_manifest.json \
  --accelerator gpu \
  --devices 2 \
  --strategy ddp
```

To evaluate a trained checkpoint on the test split:

```bash
uv run python scripts/test.py \
  --checkpoint-path /path/to/model.ckpt \
  --manifest-path /path/to/humman_stage1_manifest.json \
  --gt-smpl-dir /path/to/humman_cropped/smpl \
  --stage test
```

`scripts/test.py` now reports:
- loss terms on canonical SMPL parameters
- MPJPE from the first 24 SMPL joints in each selected view's camera frame
- PA-MPJPE from the same 24 joints after Procrustes alignment
- `input_avg_mpjpe` from the average of the per-view MHR inputs after MHR-to-SMPL conversion
- `input_avg_pa_mpjpe` from the same converted per-view inputs after Procrustes alignment

During validation and testing, GT SMPL is converted from HuMMan world space to
each selected camera frame using the HuMMan calibration. The fused prediction is
then placed into each view using the corresponding input-view MHR-to-SMPL
camera-frame fit, so MPJPE is no longer computed in pelvis-centered canonical
space.

If the MHR assets are not under `/opt/data/assets`, pass
`--mhr-assets-dir /path/to/mhr/assets`.

Converted input-view SMPL fits are cached by default under
`$(dirname manifest)/sam3dbody_fitted_smpl`. Override that with
`--input-smpl-cache-dir /path/to/cache` if you want a different location.

To save a few prediction-vs-GT comparison visualizations:

```bash
uv run python scripts/visualize.py \
  --checkpoint-path /path/to/model.ckpt \
  --manifest-path /path/to/humman_stage1_manifest.json \
  --gt-smpl-dir /path/to/humman_cropped/smpl \
  --stage test \
  --num-samples 8 \
  --output-dir outputs/stage1_visualizations
```

This renders predicted and GT SMPL meshes over the cropped RGB images for each
selected view and saves per-view overlays plus a contact sheet per sample. The
GT mesh uses HuMMan camera-frame SMPL, while the fused prediction uses the
corresponding input view's SAM3DBody-predicted camera-frame placement and is
projected with the saved per-view `cam_int`.

**Sampling protocols**:
- `cross_camera_split` (main): train on `{kinect_000, 002, 003, 004, 005, 006, 008, iphone}`, val on `{kinect_001, 007, 009}`
- `random_split` (secondary sanity check): random 80/20 train/val split by `sequence_id` across all cameras

**Stage 1**: Direct fusion from compact SAM3DBody parameters
- Per-view encoder: MLP over concatenated single-view features
- Fusion: permutation-invariant pooling across views (DeepSets-style)
- Decoder: MLP that predicts fused canonical SMPL parameters
- Stage 1 input progression:
  1. per-view input uses `mhr_model_params + shape_params`
  2. fuse across views in canonical space
  3. predict fused `smpl_betas + smpl_body_pose`
- Supervision: GT SMPL parameters transformed to the pelvis-centered, root rotation-removed canonical space
- `pred_cam_t` and `cam_int` are excluded from Stage 1 model input and kept only as auxiliary fields

**Stage 2 (Current)**: Parameter-space fusion from cached fitted SMPL
- Per-view source: offline cached MHR-to-SMPL fits from `scripts/precompute_input_smpl.py`
- Per-view training input: canonical `body_pose_6d + betas`
- Initialization: weighted permutation-invariant averaging across per-view canonical parameters
- Refinement: iterative residual updates in canonical parameter space
- Final output: fused canonical `smpl_betas + smpl_body_pose`
- Supervision: final parameter loss, initialization auxiliary loss, and SMPL joint loss in canonical space
- `global_orient`, `transl`, `pred_cam_t`, and `cam_int` remain auxiliary rather than direct Stage 2 model inputs

**Future beyond Stage 2**:
- Add explicit per-view relative camera prediction for reprojection and image-space visualization
- Explore stronger set encoders or optimization-based fusion methods after the parameter-space baseline is stable

## Evaluation

- MPJPE (Mean Per-Joint Position Error) on the first 24 SMPL joints in canonical space
- PA-MPJPE (Procrustes-Aligned MPJPE) on the same joints in canonical space
- MVE (Mesh Vertex Error) in canonical space

## Visualization

- `scripts/visualize.py` saves RGB, predicted overlay, GT overlay, and combined overlay for each selected camera view
- Overlay placement uses HuMMan GT `global_orient`, `transl`, and camera extrinsics for both meshes
- This is a qualitative comparison of fused body shape/pose, not evidence that Stage 1 or Stage 2 predicts cameras or root placement
- Future: replace the above with model-predicted per-view relative cameras
