# Stage 2 Joint-Residual Fusion

This note explains the design of the new Stage 2 model implemented in [src/mvhpe3d/models/stage2/joint_residual.py](/home/zpengac/mmhpe/MvHPE3D/src/mvhpe3d/models/stage2/joint_residual.py:1).

## Goal

The model takes several single-view SMPL estimates and predicts one fused canonical SMPL body.

Compared with the earlier Stage 2 model, the main change is:

- do not trust every view equally
- do not trust a whole view equally for the whole body
- let different joints prefer different views

This is mainly meant to help `PA-MPJPE`, because `PA-MPJPE` cares a lot about fine articulation after alignment, not just coarse whole-body consistency.

## Intuition

Imagine two cameras looking at a person:

- one camera sees the left arm clearly but the right arm poorly
- the other camera sees the right arm clearly but the left arm poorly

A global view weight is too coarse for this case. If one view gets a high score, it may dominate the entire body, even though it is only better for a few joints.

The new model instead asks:

- which view is more reliable for the left elbow?
- which view is more reliable for the right knee?
- after combining views joint by joint, what small correction is still needed?

So the model works in two steps:

1. build an initial fused pose by choosing views separately for each joint
2. refine each joint locally with a residual correction

## Input And Output

For each view, the input is:

- `body_pose` in 6D rotation form: `23 x 6 = 138`
- `betas`: `10`

So each view contributes a `148`-dimensional vector.

For `V` input views, the tensor shape is:

```text
[B, V, 148]
```

The output is one fused canonical SMPL body:

- final body pose: `23 x 6`
- final betas: `10`

and also an intermediate initialization.

## High-Level Pipeline

### 1. Encode each view

Each view parameter vector is passed through an MLP:

$$
h_v = E(x_v)
$$

where:

- $x_v \in \mathbb{R}^{148}$ is the input for view $v$
- $h_v \in \mathbb{R}^{d}$ is the learned feature for that view

This gives one latent feature per view.

### 2. Predict a pose proposal from each view

Instead of averaging the raw input pose directly, the model first predicts a cleaned-up pose proposal:

$$
p_v = P(h_v)
$$

where:

- $p_v \in \mathbb{R}^{23 \times 6}$

This matters because the raw fitted SMPL from one view may contain view-specific errors. The proposal head gives the network a chance to denoise each view before fusion.

### 3. Predict joint-wise view weights

For each view, the model predicts one score per joint:

$$
\ell_{v,j} = W(h_v)_j
$$

Then it normalizes scores across views with softmax:

$$
\alpha_{v,j} = \frac{\exp(\ell_{v,j})}{\sum_{v'} \exp(\ell_{v',j})}
$$

Here:

- $j$ is the joint index
- $\alpha_{v,j}$ says how much joint $j$ should trust view $v$

This is the central idea of the model.

### 4. Build the initial fused pose

The initial fused pose is a weighted average of the per-view pose proposals:

$$
p^{(0)}_j = \sum_v \alpha_{v,j} \, p_{v,j}
$$

This is done joint by joint, not with one global weight for the whole body.

### 5. Measure disagreement across views

The model also measures how much the views disagree for each joint:

$$
d_j = \sum_v \alpha_{v,j} \, |p_{v,j} - p^{(0)}_j|
$$

If disagreement is large, that usually means the joint is ambiguous or hard to observe. This disagreement feature is given to the refinement network.

### 6. Refine each joint with a residual

For each joint, the model forms a local feature using:

- pooled per-view feature for that joint
- initial joint pose
- joint disagreement

Then it predicts a residual correction:

$$
\Delta p_j = R\big(g_j, p^{(0)}_j, d_j\big)
$$

and the final pose is:

$$
p_j = p^{(0)}_j + \Delta p_j
$$

This is why the model is called `joint_residual`: fusion happens at the joint level, and the final prediction is a residual correction over the initialized pose.

## Betas Branch

The shape parameters `betas` are handled separately.

The model predicts:

- a beta proposal from each view
- one scalar view weight per view for shape fusion

Then it computes:

$$
b^{(0)} = \sum_v \beta_v \, s_v
$$

where:

- $s_v$ is the beta proposal from view $v$
- $\beta_v$ is the normalized scalar weight for that view

It also computes beta disagreement and predicts a beta residual:

$$
b = b^{(0)} + \Delta b
$$

If `learn_betas=false`, the model skips learned beta refinement and simply carries forward the average input betas.

## Why This Can Help PA-MPJPE

`MPJPE` and `PA-MPJPE` reward slightly different things.

- `MPJPE` cares about raw absolute joint error
- `PA-MPJPE` removes a similarity transform first, so it cares more about articulation quality after alignment

The older Stage 2 model was stronger at improving coarse whole-body consensus, so it often helped `MPJPE` more than `PA-MPJPE`.

This new model is more targeted at articulation because:

- it predicts per-view pose proposals instead of averaging raw inputs
- it chooses views per joint, not per body
- it uses disagreement as a cue for uncertain joints
- it refines joints locally instead of only updating one global parameter vector

That does not guarantee better `PA-MPJPE`, but it is a more direct design for that goal.

## Mapping To Code

The main code blocks correspond to:

- `view_encoder`: per-view feature extraction
- `pose_proposal_head`: per-view pose proposals
- `pose_weight_head`: per-joint view weighting
- `pose_refinement_encoder` and `pose_refinement_head`: joint-local residual refinement
- `betas_proposal_head`, `beta_weight_head`, `betas_refinement_head`: shape fusion/refinement

The core forward flow is:

```text
views_input
  -> view_encoder
  -> per-view pose proposals
  -> per-joint view weights
  -> initial fused pose
  -> disagreement features
  -> joint-wise residual refinement
  -> final fused pose
```

## Practical Summary

In plain language, the model says:

> Do not ask which camera is best for the whole person.
> Ask which camera is best for each body part, combine them, then clean up the result.

That is the entire design motivation.
