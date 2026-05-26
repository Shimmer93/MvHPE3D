# BEHAVE Stage 2 HeatFormer Protocol

## Purpose

This note tracks the design for adding BEHAVE Stage 2 training and evaluation to MvHPE3D. The goal is not just to run on BEHAVE; the goal is to make the result directly comparable to HeatFormer.

## Dataset Layout

Current root:

`/dysData/shimmer/datasets/behave`

Observed structure:

- `<sequence>/smpl_fit_all.npz`
- `<sequence>/object_fit_all.npz`
- `<sequence>/info.json`
- `<sequence>/<frame_time>/k0.color.jpg` through `k3.color.jpg`
- `<sequence>/<frame_time>/k*.color.json`
- `<sequence>/<frame_time>/person/person_J3d.json`
- `<sequence>/<frame_time>/person/fit02/person_fit.pkl`

`smpl_fit_all.npz` contains dense arrays:

- `poses`: `[T, 156]`
- `betas`: `[T, 10]`
- `trans`: `[T, 3]`
- `frame_times`: `[T]`
- `gender`: scalar string

The first 72 pose values are sufficient for SMPL body supervision:

- global orientation: `poses[:, :3]`
- body pose: `poses[:, 3:72]`

## HeatFormer Evaluation Contract

The BEHAVE evaluation path must match HeatFormer:

- Views: `k0`, `k1`, `k2`, `k3`
- Target joints: OpenPose/BEHAVE Body25 truncated to the first 15 joints
- Valid joint mask: OpenPose confidence greater than or equal to `0.3`
- MPJPE: millimeters, root aligned at joint index `8`
- PA-MPJPE: millimeters, after similarity alignment
- Split: exactly the HeatFormer train/valid/test split

HeatFormer implementation details checked from `eval_BEHAVE.py` and `lib/dataset/multiview_BEHAVE.py`:

- eval loader uses `batch_size=8` and `drop_last=True`
- OpenPose validity uses `>= score` with `score=0.3`
- BEHAVE MPJPE multiplies predicted and target joints by the visibility mask, then averages over all 15 joints
- BEHAVE PA-MPJPE applies unweighted similarity alignment before the same visibility-mask error

Do not use MPI-INF-3DHP H36M-17 evaluation for BEHAVE. Do not use `J_regressor_h36m_correct.npy` for BEHAVE metrics.

## Reproducible Preparation Script

Added:

`scripts/prepare_behave_stage2.py`

The script prepares BEHAVE for this repository without modifying the raw dataset:

- creates image links under `data/behave/frames`
- exports GT SMPL arrays under `data/behave/gt_smpl`
- writes a Stage 2 manifest
- writes a JSON coverage report

By default it follows the HeatFormer DB files. This matters because the DB is the split/protocol authority, and some DB timestamps are not present in the local dense `smpl_fit_all.npz` archive.

For `--protocol heatformer`, the script now:

- loads `BEHAVE_train_db.pt` and `BEHAVE_valid_db.pt`
- assigns compact sparse frame indices per sequence in DB order
- creates image links using those sparse indices
- exports GT SMPL targets directly from DB `pose`, `betas`, and `trans`
- stores `heatformer_db_split` and `heatformer_db_index` in each manifest sample

Default command:

```bash
uv run python scripts/prepare_behave_stage2.py --dataset-root /dysData/shimmer/datasets/behave --heatformer-db-root /dysData/shimmer/datasets/behave/preprocessed_data_z --output-root data/behave --manifest-path data/behave/behave_stage2_manifest.json --sam3dbody-root data/behave/sam3dbody --min-views 0
```

The default `--min-views 0` preserves the full sampling grid. Once SAM3DBody outputs exist, use the data loader to filter by `num_views`, or rebuild with `--min-views 4` only when a strict four-view manifest is explicitly needed.

Current manifest status:

- path: `data/behave/behave_stage2_manifest.json`
- samples: `15901`
- train samples: `11347`
- validation/test samples: `4554`
- split source: `heatformer_db`
- split verification: manifest records have 11,347 `heatformer_db_split=train` entries from `BEHAVE_train_db.pt`, 4,554 `heatformer_db_split=valid` entries from `BEHAVE_valid_db.pt`, 0 missing DB metadata, and 0 split mismatches
- frame links: `63604`
- post-SAM view count distribution: `{0: 15, 1: 34, 2: 450, 3: 2538, 4: 12864}`
- strict 4-view samples with the current `num_views: 4` loader: `8519` train, `4345` validation/test

SAM3DBody status:

- launcher: `scripts/run_behave_sam3dbody_multigpu.py`
- PID file: `logs/behave_sam3dbody_multigpu_20260526_0117.log.pid`
- log file: `logs/behave_sam3dbody_multigpu_20260526_0117.log`
- launched with all eight GPUs: `0,1,2,3,4,5,6,7`
- child jobs use the active venv Python directly and set `MOMENTUM_ENABLED=0`
- completed successfully with `63604` compact `.npz` outputs, no failed jobs, no retries, and no tracebacks
- resume behavior: the launcher now skips only complete sequence-camera folders, so partial folders can be rerun safely after interruption
- optimized run used `--workers-per-gpu 3` on all eight GPUs
- the previous one-worker log is preserved as `logs/behave_sam3dbody_multigpu_20260526_0117.oneworker.log`

Important filtering rule:

- A SAM3DBody compact file counts as a valid view only if `mhr_model_params`, `shape_params`, and `pred_cam_t` contain exactly one person.
- Empty detector outputs and multi-person detector outputs are excluded when rebuilding the manifest.
- `scripts/precompute_input_smpl.py` also checks this condition and reports a clear error if an invalid file reaches precompute.

Input SMPL precompute:

- launcher: `scripts/run_precompute_input_smpl_multigpu.py`
- child jobs use the active venv Python directly and set `MOMENTUM_ENABLED=0`
- completed successfully with `60004` cache `.npz` files
- all 8 shards completed without failure

Automatic continuation:

- script: `scripts/run_behave_stage2_after_sam.sh`
- PID file: `logs/behave_stage2_after_sam_20260526_0129.log.pid`
- log file: `logs/behave_stage2_after_sam_20260526_0129.log`
- waits for SAM3DBody, rebuilds the manifest, validates four-view coverage, precomputes input SMPL, runs a fast-dev smoke test, and launches full BEHAVE Stage 2 training
- successfully passed SAM completion, manifest rebuild, precompute, and fast-dev smoke gates
- launched full training at `outputs/stage2/behave_stage2_joint_residual/version_1`
- first full-training validation point after epoch 0: `val/mpjpe=42.964`, `val/pa_mpjpe=18.529`, `val/input_mpjpe=43.894`, `val/input_pa_mpjpe=24.493`
- 100-epoch fit completed and `--test-after-train` ran successfully
- final epoch-99 validation row: `val/mpjpe=50.486`, `val/pa_mpjpe=24.687`, `val/input_mpjpe=43.894`, `val/input_pa_mpjpe=24.493`, `val/auc=0.750`
- best validation PA-MPJPE so far is epoch 15: `val/mpjpe=39.304`, `val/pa_mpjpe=16.130`, `val/auc=0.832`
- best validation MPJPE so far is epoch 17: `val/mpjpe=39.169`, `val/pa_mpjpe=16.273`, `val/auc=0.831`
- final training epoch row is epoch 99 with `train/loss_epoch=0.001689` and `train/loss_joints_epoch=0.001360`
- current checkpoint artifacts include `outputs/stage2/behave_stage2_joint_residual/version_1/checkpoints/epoch=015-step=000128.ckpt` and `last.ckpt`
- post-train test used best checkpoint `outputs/stage2/behave_stage2_joint_residual/version_1/checkpoints/epoch=015-step=000128.ckpt`
- test metrics: `test/mpjpe=39.304`, `test/pa_mpjpe=16.130`, `test/pck_150=0.999948`, `test/auc=0.832286`
- test input SAM3DBody metrics: `test/input_mpjpe=43.894`, `test/input_pa_mpjpe=24.493`
- continuation finished at 2026-05-26 06:28:35 CST with one finish marker and no hard error markers

## Completion Record

Completed artifacts:

- Config: `configs/experiment/behave_stage2_joint_residual.yaml`
- Data config: `configs/data/behave_stage2.yaml`
- Manifest: `data/behave/behave_stage2_manifest.json`
- Manifest report: `data/behave/behave_stage2_manifest.report.json`
- Metrics: `outputs/stage2/behave_stage2_joint_residual/version_1/metrics.csv`
- Best checkpoint: `outputs/stage2/behave_stage2_joint_residual/version_1/checkpoints/epoch=015-step=000128.ckpt`
- Last checkpoint: `outputs/stage2/behave_stage2_joint_residual/version_1/checkpoints/last.ckpt`
- Continuation log: `logs/behave_stage2_after_sam_20260526_0129.log`

Optional follow-up after baseline completion: decide whether BEHAVE needs image feature variants or projection losses.
