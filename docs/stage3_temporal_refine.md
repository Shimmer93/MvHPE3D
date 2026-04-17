# Stage 3 Temporal Refine

This note explains the Stage 3 temporal model implemented in [src/mvhpe3d/models/stage3/temporal_refine.py](/home/zpengac/mmhpe/MvHPE3D/src/mvhpe3d/models/stage3/temporal_refine.py:1) and [src/mvhpe3d/lightning/stage3_module.py](/home/zpengac/mmhpe/MvHPE3D/src/mvhpe3d/lightning/stage3_module.py:1).

## Goal

The Stage 2 models in this repo are still frame-by-frame models.

That means they can fuse multiple views at one instant, but they do not use motion context from nearby frames.

Stage 3 adds that missing temporal context.

The design is intentionally conservative:

- keep the Stage 2 multiview fusion backbone
- run it independently on each frame in a short window
- use a small temporal model to refine only the center frame

So Stage 3 should be understood as:

- Stage 2 gives a strong per-frame estimate
- Stage 3 cleans that estimate up using nearby frames

## Intuition

Imagine the center frame has an ambiguous elbow pose:

- one frame earlier, the arm was clearly bending
- one frame later, the wrist keeps moving in the same direction

Even if the center frame alone is ambiguous, the short motion pattern around it is often enough to recover a cleaner pose.

That is the core idea of Stage 3:

- let Stage 2 solve the multiview fusion problem
- let Stage 3 solve the short-range temporal ambiguity problem

This is a cleaner division of labor than trying to make one model handle everything at once.

## Input And Output

The Stage 3 dataset returns a temporal window of Stage 2-style per-view inputs:

- per frame: `V` views
- per view: `148` values = `body_pose_6d (138) + betas (10)`

So the input shape is:

```text
[B, T, V, 148]
```

where:

- $B$ is batch size
- $T$ is temporal window length
- $V$ is number of views

The output is only for the center frame:

- refined body pose: `23 x 6`
- refined betas: `10`

and the usual axis-angle SMPL pose used elsewhere in the repo.

## Temporal Window Construction

Stage 3 uses a short centered window around each frame.

For a window radius $r$, the model sees:

$$
\{x_{t-r}, x_{t-r+1}, \dots, x_t, \dots, x_{t+r}\}.
$$

At sequence boundaries, the dataset clamps indices to the valid range, so the window stays the same size.

One important implementation detail is that Stage 3 uses only cameras that are present across the whole window.

That means the temporal model always sees a consistent set of view identities inside one sample.

## High-Level Pipeline

### 1. Run the Stage 2 backbone on every frame

Each frame in the window is processed independently by a Stage 2 backbone:

$$
h_\tau = F_{\text{stage2}}(x_\tau)
$$

for each frame $\tau$ in the window.

In practice, the Stage 2 backbone produces:

- initial fused pose `init_pose_6d`
- refined fused pose `pred_pose_6d`
- fused `pred_betas`

This backbone is typically frozen.

So Stage 3 does not relearn multiview fusion from scratch. It builds on top of an existing Stage 2 model.

### 2. Build one temporal feature vector per frame

For each frame, Stage 3 concatenates:

- Stage 2 initialization
- Stage 2 final pose
- Stage 2 final betas

So the temporal feature for frame $\tau$ is:

$$
z_\tau =
\left[
p^{(0)}_\tau,\;
p^{\text{stage2}}_\tau,\;
b^{\text{stage2}}_\tau
\right].
$$

This gives a temporal feature sequence:

$$
\{z_{t-r}, \dots, z_t, \dots, z_{t+r}\}.
$$

The reason for including both the initialization and the final Stage 2 pose is simple:

- the final Stage 2 pose says what the backbone believes
- the initialization says what the fused evidence looked like before refinement

Together they give the temporal model a little more context than the final pose alone.

### 3. Run a small temporal network

The temporal feature sequence is projected into a hidden space and then processed by a stack of residual 1D temporal convolution blocks:

$$
u_{t-r:t+r} = G(z_{t-r:t+r}).
$$

This is a local temporal model, not a transformer.

That choice is intentional:

- easier to train
- lower memory
- enough for short-range motion smoothing and refinement

### 4. Use the center-frame feature to predict a residual

Only the center-frame hidden feature is used for the final output:

$$
u_t = G(z_{t-r:t+r})_t.
$$

Then the model predicts a small pose residual:

$$
\Delta p_t = R_p(u_t)
$$

and optionally a beta residual:

$$
\Delta b_t = R_b(u_t).
$$

These residuals are bounded with `tanh` and small scale factors:

$$
\hat{p}_t = p^{\text{stage2}}_t + s_p \cdot \tanh(\Delta p_t)
$$

$$
\hat{b}_t = b^{\text{stage2}}_t + s_b \cdot \tanh(\Delta b_t).
$$

This keeps Stage 3 in a refinement regime instead of letting it drift far away from the Stage 2 solution.

## Why The Model Predicts Only The Center Frame

This is a common choice in temporal refinement models.

The idea is:

- use the full window as context
- supervise only the center frame

That avoids edge artifacts inside the training target and keeps the problem simple.

So Stage 3 is not trying to be a sequence-to-sequence generator.

It is a sequence-to-center-frame refiner.

## Why This Design Is Conservative

This model avoids a full end-to-end temporal multiview architecture for a reason.

A full temporal-view model would change too many things at once:

- multiview fusion
- temporal modeling
- optimization behavior
- memory usage

The current Stage 3 is more controlled:

- Stage 2 still handles multiview fusion
- Stage 3 only handles temporal refinement

That makes it easier to answer the actual research question:

> Does short-range temporal context improve the per-frame Stage 2 output?

## Losses

The Stage 3 objective is intentionally similar to Stage 2.

It uses:

- pose 6D loss
- optional beta loss
- canonical joint loss
- articulation-oriented aligned joint loss

At a high level:

$$
\mathcal{L}_{\text{stage3}}
=
\lambda_{\text{pose}} \mathcal{L}_{\text{pose}}
+
\lambda_{\text{betas}} \mathcal{L}_{\text{betas}}
+
\lambda_{\text{joint}} \mathcal{L}_{\text{joint}}
+
\lambda_{\text{art}} \mathcal{L}_{\text{art}}.
$$

There is currently no explicit temporal smoothness loss.

That is deliberate for the first version. The model is first asked to use temporal context directly, without extra smoothing terms.

## Mapping To Code

The main parts correspond to:

- [humman_stage3_sequence.py](/home/zpengac/mmhpe/MvHPE3D/src/mvhpe3d/data/datasets/humman_stage3_sequence.py:1): builds temporal windows with shared camera identities
- [stage3_module.py](/home/zpengac/mmhpe/MvHPE3D/src/mvhpe3d/lightning/stage3_module.py:1): runs the Stage 2 backbone per frame, builds temporal features, and applies the Stage 3 losses
- [temporal_refine.py](/home/zpengac/mmhpe/MvHPE3D/src/mvhpe3d/models/stage3/temporal_refine.py:1): temporal conv stack and center-frame residual heads

The forward flow is:

```text
window of Stage 2 inputs
  -> Stage 2 backbone on each frame
  -> temporal feature sequence
  -> temporal convolution blocks
  -> center-frame hidden feature
  -> small residual correction
  -> refined center-frame pose
```

## Practical Summary

In plain language, the model says:

> First solve the hard multiview problem frame by frame.
> Then look at a few neighboring frames and ask whether the center-frame pose should be nudged to better match the local motion pattern.

That is the whole Stage 3 design motivation.
