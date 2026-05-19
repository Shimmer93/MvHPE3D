# Stage 2R RGB-Guided Residual Refiner

This note documents the Stage 2R model implemented in [src/mvhpe3d/models/stage2/stage2r_rgb_guided_residual_refiner.py](/home/zpengac/mmhpe/MvHPE3D/src/mvhpe3d/models/stage2/stage2r_rgb_guided_residual_refiner.py:1).

## Position In The Pipeline

Stage 2R is a per-frame extension of Stage 2.

It is not Stage 4, because it does not consume Stage 3 temporal predictions. It still operates on one synchronized multi-view frame and predicts one canonical SMPL body.

The intended naming is:

- Stage 2: SMPL-only multi-view parameter fusion
- Stage 2R: RGB-guided residual refinement on top of Stage 2
- Stage 3: temporal refinement

## Goal

Stage 2R keeps the successful Stage 2 joint-graph refiner as the backbone, then adds a small RGB-conditioned residual head.

The design goal is narrow:

- preserve the strong SMPL-only Stage 2 prediction
- use frozen RGB features only where they can explain remaining per-joint errors
- avoid replacing the Stage 2 fusion mechanism with a slower RGB-heavy model

## Inputs

Stage 2R uses the normal Stage 2 input per view:

- canonical fitted SMPL `body_pose_6d`
- canonical fitted SMPL `betas`

Each view also has a frozen RGB feature vector loaded from an offline cache:

```text
view_rgb_feature: [B, V, rgb_feature_dim]
```

The default RGB feature dimension is `384`, matching `vit_small_patch16_224.dino`.

Stage 2R also requires a Stage 2 checkpoint. The checkpoint supplies the backbone prediction and the learned Stage 2 view weights.

## Model

Stage 2R first runs the Stage 2 backbone and reads:

- `pred_pose_6d`
- `pred_betas`
- `init_pose_6d`
- `init_betas`
- `pose_view_weights` when available

For each view, Stage 2R encodes:

- the Stage 2 parameter input
- the frozen RGB feature

The per-view features are pooled with the Stage 2 learned pose weights. This means the RGB residual head follows the same joint-wise view preference that Stage 2 already learned.

The residual head predicts only a bounded pose correction:

```text
pred_pose_6d = stage2_pred_pose_6d + stage2r_pose_residual_6d
```

The residual is passed through `tanh` and scaled by `pose_residual_scale`. The last layer is zero-initialized, so the model starts as an exact Stage 2 predictor.

The current Stage 2R model leaves betas unchanged:

```text
pred_betas = stage2_pred_betas
```

## Frozen And Joint-Train Variants

Two experiment configs are kept:

- `configs/experiment/stage2r_rgb_guided_residual_refiner.yaml`
- `configs/experiment/stage2r_rgb_guided_residual_refiner_joint_train.yaml`

The frozen variant keeps the Stage 2 backbone fixed and trains only the RGB residual head.

The joint-train variant sets `freeze_backbone: false` and trains the Stage 2 backbone with a smaller LR through `stage2_backbone_lr_scale`. This variant has performed better in current experiments.

## Precompute RGB Features

RGB features are cached offline so training does not run the image encoder in the loop.

```bash
uv run python scripts/precompute_rgb_features.py --manifest-path /path/to/humman_stage1_manifest.json --rgb-dir /path/to/humman_cropped/rgb --output-dir /path/to/humman_cropped/rgb_features
```

## Train

Frozen Stage 2R:

```bash
uv run python scripts/train.py --config configs/experiment/stage2r_rgb_guided_residual_refiner.yaml --manifest-path /path/to/humman_stage1_manifest.json --input-smpl-cache-dir /path/to/sam3dbody_fitted_smpl --rgb-feature-cache-dir /path/to/rgb_features --stage2-checkpoint-path /path/to/stage2.ckpt
```

Joint-train Stage 2R:

```bash
uv run python scripts/train.py --config configs/experiment/stage2r_rgb_guided_residual_refiner_joint_train.yaml --manifest-path /path/to/humman_stage1_manifest.json --input-smpl-cache-dir /path/to/sam3dbody_fitted_smpl --rgb-feature-cache-dir /path/to/rgb_features --stage2-checkpoint-path /path/to/stage2.ckpt
```

## Evaluation

Evaluation uses the same data inputs and Stage 2 checkpoint path:

```bash
uv run python scripts/test.py --config configs/experiment/stage2r_rgb_guided_residual_refiner_joint_train.yaml --checkpoint-path /path/to/stage2r.ckpt --manifest-path /path/to/humman_stage1_manifest.json --input-smpl-cache-dir /path/to/sam3dbody_fitted_smpl --rgb-feature-cache-dir /path/to/rgb_features --stage2-checkpoint-path /path/to/stage2.ckpt
```

The Stage 2 checkpoint path is needed when it is not already stored in the Stage 2R checkpoint hyperparameters.
