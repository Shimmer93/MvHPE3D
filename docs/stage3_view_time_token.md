# Stage 3 View-Time Token Fusion

This note describes the Stage 3 v2 design implemented in [src/mvhpe3d/models/stage3/view_time_token.py](/home/zpengac/mmhpe/MvHPE3D/src/mvhpe3d/models/stage3/view_time_token.py:1), [src/mvhpe3d/data/datasets/humman_stage3_tokens.py](/home/zpengac/mmhpe/MvHPE3D/src/mvhpe3d/data/datasets/humman_stage3_tokens.py:1), and [src/mvhpe3d/lightning/stage3_module.py](/home/zpengac/mmhpe/MvHPE3D/src/mvhpe3d/lightning/stage3_module.py:1).

## Goal

The current Stage 3 temporal model uses a dense synchronized window:

```text
[B, T, V, 148]
```

It assumes the same selected views are available for every frame in the temporal window. That is useful as a controlled baseline, but it does not directly model sparse or asynchronous multiview evidence.

The view-time token model changes the temporal representation:

```text
token = one fitted SMPL input from one view at one time
```

The goal is to refine the target-frame Stage 2 prediction using a set of view-time tokens that may be dense, sparse, interleaved, or randomly missing.

This is inspired by DenseWarper's sparse interleaved multiview setting, but adapted to this repo's calibration-free SMPL-parameter pipeline. We borrow the idea of fusing view-time evidence over a temporal window, not the heatmap or epipolar components.

Reference:

```text
https://openreview.net/pdf?id=MLs6ThXmcz
```

## Core Principle

Stage 3 v2 keeps the conservative Stage 3 assumption:

- Stage 2 remains the target-frame multiview fusion backbone.
- Stage 3 predicts only a small residual correction.
- Stage 3 does not replace Stage 2 or learn a full SMPL predictor from scratch.

The difference is where temporal context comes from.

The original Stage 3 Conv1D model builds one feature per frame after running Stage 2 on each frame.

The token model builds one feature per view-time observation:

```text
frame t-2, camera 0 -> token
frame t-2, camera 1 -> token
frame t-1, camera 0 -> token
frame t,   camera 1 -> token
frame t+1, camera 0 -> token
...
```

Then a target query attends to all valid tokens and predicts a residual for the target frame.

## Dataset Schema

The token dataset returns both target-frame Stage 2 input and temporal token context.

Target-frame input:

```text
views_input: [V, 148]
```

This is the usual Stage 2 input for the target frame:

```text
body_pose_6d: 138
betas:        10
```

Token context:

```text
view_time_tokens:     [K, 148]
token_time_offsets:   [K]
token_camera_indices: [K]
token_valid_mask:     [K]
```

where:

```text
K = window_size * num_views
```

Invalid or missing tokens are zero-filled and marked false in `token_valid_mask`.

The target frame still uses real selected camera views through `views_input`, because the Stage 2 backbone needs a normal multiview target-frame input to produce the base prediction.

## Sampling Modes

The dataset supports several token sampling modes.

### Dense Sync

```text
token_sampling: "dense_sync"
```

Every selected camera contributes a token at every frame in the window.

This is the closest token equivalent to the existing dense Stage 3 setup.

### Sparse Interleaved

```text
token_sampling: "sparse_interleaved"
```

Only one selected camera contributes at each time step, rotating across the selected cameras.

For two cameras and a five-frame centered window, the pattern is:

```text
t-2: camera 0
t-1: camera 1
t:   camera 0
t+1: camera 1
t+2: camera 0
```

This is the most direct analogue of sparse interleaved multiview input.

### Random View-Time

```text
token_sampling: "random_view_time"
token_drop_prob: 0.5
```

Each view-time token is independently dropped with probability `token_drop_prob`.

This is useful for robustness experiments under missing views.

### Causal Sparse

```text
token_sampling: "causal_sparse"
causal: true
```

This uses the same interleaved pattern, but with a past-only temporal window.

## Model Architecture

The model is implemented as `Stage3ViewTimeTokenModel`.

Input tokens:

$$
x_k = [p_k, b_k]
$$

where:

- $p_k \in \mathbb{R}^{138}$ is `body_pose_6d`
- $b_k \in \mathbb{R}^{10}$ is `betas`

Each token is encoded with an MLP:

$$
h_k = E(x_k).
$$

The model adds a learned camera embedding and a continuous time-offset embedding:

$$
\tilde{h}_k = h_k + e_{\text{camera}(k)} + e_{\Delta t_k}.
$$

Invalid tokens are masked.

A learned target query is prepended:

$$
q_0, \tilde{h}_1, \dots, \tilde{h}_K.
$$

The sequence is processed by a Transformer encoder. The output target-query feature is used for prediction:

$$
u_t = \text{Transformer}(q_0, \tilde{h}_1, \dots, \tilde{h}_K)_0.
$$

The model predicts a bounded pose residual:

$$
\Delta p_t = s_p \cdot \tanh(R_p(u_t)).
$$

The final prediction is:

$$
\hat{p}_t = p_t^{\text{stage2}} + \Delta p_t.
$$

If beta learning is enabled, the same pattern is used for shape:

$$
\hat{b}_t = b_t^{\text{stage2}} + s_b \cdot \tanh(R_b(u_t)).
$$

By default, beta learning is disabled and the target-frame Stage 2 betas are reused.

## Why Use The Target-Frame Stage 2 Prediction As Base

The target-frame Stage 2 prediction is still the strongest single estimate. The token model should not relearn multiview SMPL fusion from scratch before we know whether the temporal-token idea helps.

The model therefore predicts a residual over:

```text
stage2_pred_pose_6d
stage2_pred_betas
```

The residual heads are zero-initialized. At initialization:

```text
Stage 3 token output == target-frame Stage 2 output
```

This makes the experiment easier to interpret. Any improvement or degradation comes from the learned token residual.

## Losses

The token model uses the same Stage 3 losses as the Conv1D temporal model:

- pose 6D loss
- optional beta loss
- canonical joint loss
- articulation loss
- pose residual regularization
- optional beta residual regularization

No projection loss is used in this design.

The objective stays in canonical SMPL parameter space, consistent with the Stage 1 and Stage 2 scope.

## Configs

Main experiment:

```text
configs/experiment/stage3_view_time_token.yaml
```

Data config:

```text
configs/data/humman_stage3_view_time_token.yaml
```

Model config:

```text
configs/model/stage3_view_time_token.yaml
```

The default data config uses:

```yaml
representation: "view_time_tokens"
token_sampling: "sparse_interleaved"
window_size: 5
num_views: 2
```

## Training

Example:

```bash
uv run python scripts/train.py \
  --config configs/experiment/stage3_view_time_token.yaml \
  --stage2-checkpoint-path outputs/stage2/stage2_cross_camera_joint_graph_refiner/version_1/checkpoints/epoch=080-step=015390.ckpt
```

The Stage 2 backbone is frozen by default.

## Evaluation Plan

The first comparison should be:

```text
Stage 2
Stage 3 Conv1D dense window
Stage 3 token dense_sync
Stage 3 token sparse_interleaved
Stage 3 token random_view_time
```

Useful metrics:

- MPJPE
- PA-MPJPE
- camera-space test MPJPE
- residual magnitude
- robustness under missing views
- temporal acceleration error, if added later

The most important ablation is whether the token model improves over Stage 2 and the Conv1D Stage 3 when views are sparse or missing.

## Current Limitations

The first implementation is intentionally simple:

- camera identity is represented by a learned index embedding
- no explicit image evidence is used
- no visibility or confidence feature is included yet
- no learned deformable temporal offsets are implemented yet
- token count is fixed at `window_size * num_views`

These choices keep the baseline easy to train and debug.

## Next Extensions

Good next steps:

- add per-token confidence features from SAM3DBody fitting
- add per-token view visibility estimates
- add a learned per-joint token attention head
- add a deformable temporal token attention module
- support variable token counts with packed token batches
- evaluate causal sparse mode for online inference

The most paper-relevant extension is per-joint deformable token attention:

```text
target joint query
  -> attend to sparse neighboring view-time tokens
  -> aggregate pose evidence for that joint
  -> predict joint-specific residual
```

That would be the closest parameter-space analogue of DenseWarper's temporal warping idea.
