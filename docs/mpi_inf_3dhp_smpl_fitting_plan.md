# MPI-INF-3DHP Pseudo-SMPL Fitting Improvement Plan

## Motivation

The current MPI-INF-3DHP stage2 pipeline trains in a canonical/input-root SMPL
space, but the pseudo-SMPL target is fitted from sparse MPI 3D joints. Recent
diagnostics show that stage2 can get close to the fitted pseudo-SMPL target, so
the fitted target quality is now a likely bottleneck.

Current validation reference points:

- SAM3DBody input fused baseline: about 65 mm MPJPE.
- Current input-root pseudo-SMPL target: about 42 mm MPJPE against MPI joints.
- Model versus fitted pseudo-SMPL PA-MPJPE: about 32 mm.

This suggests that improving pseudo-SMPL quality is more useful than only
tuning the stage2 residual model.

## Current Limitations

The current fitter in `scripts/fit_mpi_inf_3dhp_gt_smpl.py` has several
structural limitations:

- It fits only the 17 HeatFormer/H36M evaluation joints.
- It maps MPI/H36M joints directly to SMPL24 joints.
- `head` and `headtop` both map to the same SMPL head joint, which creates an
  unavoidable mismatch.
- Betas are optimized per chunk, not consistently per subject or sequence.
- The objective is a simple joint MSE plus weak pose, shape, bone-length, and
  optional temporal regularization.
- The fit is initialized from zeros, with no coarse-to-fine schedule.

## Design Goals

1. Keep the final evaluation protocol unchanged.
2. Fit pseudo-SMPL targets in the same canonical/input-root frame used by
   stage2.
3. Reduce structural joint-regressor mismatch before increasing model capacity.
4. Make the fitting pipeline resumable and compatible with the existing
   multi-GPU sharding.
5. Add diagnostics that separate pseudo-label quality from stage2 learning
   quality.

## Proposed Design

### 1. Replace Direct SMPL24 Indexing With an H36M17 Regressor

Use a proper SMPL-to-H36M17 joint regressor instead of direct indexing into
SMPL24 joints.

Expected benefits:

- Removes the duplicated `head`/`headtop -> SMPL head` mapping.
- Better matches the evaluation skeleton used by HeatFormer-style protocols.
- Reduces irreducible fitting error from skeleton-definition mismatch.

Implementation plan:

- Add a configurable `--joint-regressor-path`.
- Load a regressor shaped `[17, V]` or `[17, J]`.
- If the regressor maps vertices, compute joints from `smpl_output.vertices`.
- If it maps joints, compute from the selected SMPL joint tensor.
- Keep the current direct SMPL24 mapping as a fallback for reproducibility.

Validation:

- Report pseudo-SMPL MPJPE/PA-MPJPE against MPI joints before and after.
- Check per-joint error, especially `head`, `headtop`, wrists, and ankles.

### 2. Add a Better Headtop Proxy if No Regressor Is Available

If a valid H36M regressor is unavailable, define a better `headtop` proxy from
SMPL vertices or a small fixed vertex set rather than mapping it to the SMPL
head joint.

Expected benefits:

- Reduces one known structural mismatch without changing the rest of the
  fitting code.

Implementation plan:

- Add a `--headtop-vertex-ids` option or internal default vertex set.
- Build a 17-joint fitted skeleton where `headtop` is the mean of those
  vertices.
- Keep the old mapping behind an explicit compatibility option.

Validation:

- Compare per-joint errors for `head` and `headtop`.
- Ensure overall MPJPE improves without worsening limbs.

### 3. Fit With More MPI Joints, Evaluate on the Same 17 Joints

MPI-INF-3DHP provides more than the 17 evaluation joints. Use the extra joints
as fitting constraints, but keep evaluation on the HeatFormer 17-joint set.

Expected benefits:

- Hands, feet, toes, and clavicles can reduce pose ambiguity.
- More constraints should produce better full-body SMPL pose labels.

Implementation plan:

- Add `--fit-joint-set eval17|mpi28`.
- Add an MPI28-to-SMPL/regressor mapping or regressor.
- Use per-joint weights so unreliable or poorly matched joints can be
  downweighted.
- Keep `target_joints` and stage2 evaluation unchanged.

Validation:

- Report eval17 MPJPE even when fitting uses MPI28.
- Compare canonical pseudo-SMPL MPJPE and downstream stage2 MPJPE.

### 4. Improve the Optimization Schedule

Use a coarse-to-fine fitting schedule instead of one Adam phase.

Proposed schedule:

1. Shape warmup: optimize betas and simple body pose with strong pose prior.
2. Pose phase: optimize body pose with moderate pose prior and full joint loss.
3. Fine phase: lower pose prior, increase joint weight, optionally use LBFGS.
4. Temporal phase: optional sequence/chunk smoothness for body pose.

Expected benefits:

- More stable convergence.
- Less local-minimum behavior from zero initialization.
- More consistent body shape across frames.

Implementation plan:

- Add repeated schedule blocks to the CLI, or hard-code a conservative default
  schedule with override flags.
- Add `--lbfgs-iters`.
- Save per-frame and per-sequence fitting reports with final joint MPJPE.

Validation:

- Track fitting loss curves on a small fixed validation subset.
- Compare full validation pseudo-label MPJPE after each schedule change.

### 5. Use Subject- or Sequence-Level Betas

The current chunk-level betas can vary across chunks. MPI subjects should have
stable body shape.

Expected benefits:

- More consistent pseudo-SMPL labels.
- Less noise in stage2 pose targets.

Implementation plan:

- Stage A: fit a shared beta per sequence or subject using sampled frames.
- Stage B: freeze or lightly regularize betas while fitting per-frame pose.
- Save the shared beta in each sequence `.npz`.

Validation:

- Compare beta variance across frames before and after.
- Check whether pseudo-SMPL MPJPE stays the same or improves.
- Check whether stage2 training becomes more stable.

## Diagnostics to Add

Add or keep the following metrics:

- `val/mpjpe`: predicted SMPL joints versus raw MPI joints in input-root space.
- `val/canonical_pseudo_mpjpe`: predicted SMPL versus fitted pseudo-SMPL.
- `pseudo_smpl/mpjpe`: fitted pseudo-SMPL versus raw MPI joints.
- Per-joint pseudo-SMPL MPJPE report.
- Per-sequence pseudo-SMPL MPJPE report.

These diagnostics answer three different questions:

- Is the pseudo target good?
- Is stage2 learning the pseudo target?
- Does learning the pseudo target improve the official MPI joint metric?

## Recommended Implementation Order

1. Add per-joint and per-sequence fitting diagnostics.
2. Add a proper H36M17 regressor path or better headtop proxy.
3. Refit pseudo-SMPL labels and compare pseudo-label MPJPE.
4. Add MPI28 fitting constraints with per-joint weights.
5. Add the coarse-to-fine optimizer schedule.
6. Add subject/sequence-level betas.
7. Retrain stage2 only after the pseudo-label metrics improve.

## Implementation Status

Implemented first:

- Per-frame, per-sequence, and per-joint pseudo-SMPL fitting diagnostics.
- `--fit-joint-source smpl24_headtop_proxy`, now the default, which replaces
  the duplicated `headtop -> SMPL head` target with a neutral-template vertex
  proxy.
- `--fit-joint-source regressor` with `--joint-regressor-path` for a future
  H36M17 joint regressor.
- `--fit-joint-source regressor_headtop_proxy`, which uses the H36M17 regressor
  for body joints and keeps the neutral-template vertex proxy for `headtop`.
- Multi-GPU passthrough for the new fitter options.
- `scripts/evaluate_mpi_inf_3dhp_gt_smpl_fit.py` for checking pseudo-label
  quality before retraining stage2.

Still pending:

- MPI28 fitting constraints.
- Coarse-to-fine optimizer schedule.
- Subject- or sequence-level shared betas.

## Success Criteria

Short term:

- Reduce pseudo-SMPL target MPJPE from about 42 mm to below 38 mm.
- Reduce head/headtop per-joint error without increasing limb errors.

Medium term:

- Reduce stage2 validation MPJPE below the current 61 mm range.
- Keep `canonical_pseudo_mpjpe` and raw MPI `mpjpe` moving in the same
  direction.

Long term:

- Make the pseudo-SMPL target good enough that stage2 improvements are no
  longer capped by fitting artifacts.
