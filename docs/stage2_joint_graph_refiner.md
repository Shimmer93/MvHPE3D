# Stage 2 Joint-Graph Refiner

This note explains the design of the new Stage 2 graph-refiner model implemented in [src/mvhpe3d/models/stage2/joint_graph_refiner.py](/home/zpengac/mmhpe/MvHPE3D/src/mvhpe3d/models/stage2/joint_graph_refiner.py:1).

## Goal

The model takes several single-view SMPL estimates and predicts one fused canonical SMPL body.

This model is intentionally conservative.

Its purpose is not to replace the stable `stage2_joint_residual` design. Its purpose is to keep that stable design almost unchanged, and then test one small idea:

- after the usual joint-wise fusion and local refinement,
- can a very small graph-based correction improve articulation consistency?

So this model should be understood as:

- `stage2_joint_residual`
- plus one lightweight graph refinement residual

## Intuition

The baseline `stage2_joint_residual` already does two important things well:

- it lets different joints trust different views
- it refines each joint with a local MLP residual

What it does not explicitly model is interaction between neighboring joints.

For example:

- if the shoulder rotates, the elbow and wrist should usually move in a compatible way
- if the hip and knee are uncertain, the ankle should not move arbitrarily
- if the torso twists, the collar and shoulder configuration should stay coherent

A graph block gives the model a small way to pass information along the body kinematic structure.

The key design decision here is:

- do not let the graph block control the whole model
- only let it add a small correction after the stable local refinement path

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

### 2. Predict a pose proposal from each view

As in `stage2_joint_residual`, the model predicts a per-view pose proposal:

$$
p_v = P(h_v)
$$

where:

- $p_v \in \mathbb{R}^{23 \times 6}$

### 3. Predict joint-wise view weights

For each joint and each view, the model predicts a score:

$$
\ell_{v,j} = W(h_v)_j
$$

Then it normalizes scores across views:

$$
\alpha_{v,j} = \frac{\exp(\ell_{v,j})}{\sum_{v'} \exp(\ell_{v',j})}
$$

This gives one learned weight for each joint-view pair.

### 4. Build the initial fused pose

The initial fused pose is:

$$
p^{(0)}_j = \sum_v \alpha_{v,j} \, p_{v,j}
$$

This is identical in spirit to `stage2_joint_residual`.

### 5. Build local refinement features

The model measures view disagreement per joint:

$$
d_j = \sum_v \alpha_{v,j} \, |p_{v,j} - p^{(0)}_j|
$$

It also pools the per-view latent features into one feature per joint:

$$
g_j = \sum_v \alpha_{v,j} \, h_v
$$

Then it forms the local refinement input:

$$
u_j = [g_j, p^{(0)}_j, d_j]
$$

and encodes it with an MLP.

### 6. Predict the usual local pose residual

Just like `stage2_joint_residual`, the model predicts a local residual:

$$
\Delta p^{\text{local}}_j = R_{\text{local}}(u_j)
$$

### 7. Apply one small graph refinement

Now comes the only new part.

The local refinement features are treated as node features on the 23-joint SMPL body graph. A small graph block passes information between connected joints:

$$
\tilde{u}_j = G(u_1, \dots, u_{23})
$$

Then a separate head predicts a graph-based pose residual:

$$
\Delta p^{\text{graph}}_j = R_{\text{graph}}(\tilde{u}_j)
$$

This graph residual is intentionally bounded:

$$
\Delta p^{\text{graph}}_j
= s \cdot \tanh\!\left(R_{\text{graph}}(\tilde{u}_j)\right)
$$

where $s$ is a small scale factor, `graph_delta_scale`.

The graph head is also zero-initialized, so at the start of training:

$$
\Delta p^{\text{graph}}_j \approx 0
$$

That means the model starts very close to `stage2_joint_residual`.

### 8. Final pose prediction

The final pose is:

$$
p_j = p^{(0)}_j + \Delta p^{\text{local}}_j + \Delta p^{\text{graph}}_j
$$

So the graph branch is an add-on correction, not the main prediction path.

## Betas Branch

The shape branch is unchanged from `stage2_joint_residual`.

The model predicts:

- one beta proposal per view
- one scalar weight per view
- one residual correction after fusion

The fused shape initialization is:

$$
b^{(0)} = \sum_v \beta_v \, s_v
$$

and the final shape is:

$$
b = b^{(0)} + \Delta b
$$

There is no graph refinement in the beta branch.

## Why This Design Is Safer Than The Earlier Graph Attempts

Earlier graph variants in this repo let the graph branch influence the main pose prediction too strongly. That made optimization unstable and hurt performance.

This version is safer because:

- the stable fusion path is left intact
- the graph branch acts only after local refinement features are already built
- the graph adjacency is fixed by default
- the graph residual is small and bounded
- the graph head starts at zero

So this model is really asking a narrow question:

> If the stable baseline already works, does one small graph correction help joint consistency?

That is a much cleaner experiment than making the graph path responsible for the whole pose.

## Mapping To Code

The main code blocks correspond to:

- `view_encoder`: per-view feature extraction
- `pose_proposal_head`: per-view pose proposals
- `pose_weight_head`: per-joint view weighting
- `pose_refinement_encoder`: local refinement feature builder
- `pose_refinement_head`: local pose residual
- `pose_graph_blocks`: graph message passing over the 23-joint SMPL body graph
- `pose_graph_head`: small graph-based pose residual
- `betas_proposal_head`, `beta_weight_head`, `betas_refinement_head`: unchanged shape branch

The forward flow is:

```text
views_input
  -> view_encoder
  -> per-view pose proposals
  -> per-joint view weights
  -> initial fused pose
  -> local refinement features
  -> local pose residual
  -> one graph refinement block
  -> small graph pose residual
  -> final fused pose
```

## Difference From Stage 2 Joint-Residual

The difference from `stage2_joint_residual` is small by design.

`stage2_joint_residual`:

$$
p_j = p^{(0)}_j + \Delta p^{\text{local}}_j
$$

`stage2_joint_graph_refiner`:

$$
p_j = p^{(0)}_j + \Delta p^{\text{local}}_j + \Delta p^{\text{graph}}_j
$$

So the only new term is the graph residual.

## Practical Summary

In plain language, the model says:

> First do the same stable fusion and local refinement as before.
> Then let neighboring joints talk to each other once.
> Use that only for a small extra correction.

That is the whole design idea.
