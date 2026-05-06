# Stage 4 Multi-Human Cross-View Association

This note describes a candidate next stage for extending the current single-actor multiview HPE pipeline to multi-human scenes.

The current Panoptic and HuMMan Stage 1/2/3 loaders assume that each multiview sample already corresponds to one actor. That assumption is reasonable for single-actor cropped data, but it breaks for full images with multiple people. SAM3DBody can output multiple detections per image, so the missing piece is cross-view identity association before fusion.

## Goal

Given synchronized multiview images, build per-person multiview samples:

- one person identity
- one timestamp
- multiple camera views
- one selected SAM3DBody row per view

The output should be a multi-human manifest that can feed the existing fusion models with minimal changes.

## Core Principle

Use a track-first, match-second pipeline.

Frame-by-frame cross-view matching is fragile because people can overlap, leave some views, or look similar in one frame. Per-camera temporal tracks provide stronger identity cues before cross-view association is attempted.

The intended pipeline is:

```text
per-view SAM3DBody detections
  -> per-view short tracks
  -> cross-view association per timestamp or short window
  -> multi-human multiview manifest
  -> existing Stage 1/2/3 fusion models per associated person
```

## SAM3DBody Multi-Person Output

The local compact SAM3DBody exporter stores a person dimension `N` in each `.npz`:

```text
mhr_model_params: [N, 204]
shape_params:     [N, 45]
pred_cam_t:       [N, 3]
cam_int:          [3, 3]
image_size:       [2]
```

For multi-human association, the exporter should also preserve detection-level metadata:

```text
bbox:              [N, 4]
det_score:         [N]
pred_keypoints_2d: [N, J, 2]
keypoint_score:    [N, J] optional
reid_embedding:    [N, D] optional
```

The existing Panoptic export path defaults to single-actor cropped full-image inference. Multi-human scenes should use detector-enabled per-image inference, not the batched single-person crop path.

## Manifest Schema

The key schema change is that each view must identify both the `.npz` file and the person row inside that file.

Example:

```json
{
  "sample_id": "seq001_frame000123_person0004",
  "sequence_id": "seq001",
  "frame_id": "000123",
  "person_id": "person0004",
  "views": [
    {
      "camera_id": "kinect_001",
      "npz_path": "/data/sam3dbody/seq001_kinect_001_000123.npz",
      "person_index": 2,
      "track_id": "kinect_001_track0017"
    },
    {
      "camera_id": "kinect_007",
      "npz_path": "/data/sam3dbody/seq001_kinect_007_000123.npz",
      "person_index": 0,
      "track_id": "kinect_007_track0009"
    }
  ]
}
```

The current dataset code reads one person implicitly from each `.npz`. Stage 4 would make `person_index` explicit and use it to slice `mhr_model_params`, `shape_params`, `pred_cam_t`, and any auxiliary fields.

## Per-View Tracking

Before matching cameras, build tracks independently in each camera.

Recommended baseline:

- detect people per frame
- associate detections over time using bbox IoU, 2D keypoints, appearance, and optional SMPL pose similarity
- keep short tracklets even when confidence drops briefly
- split or terminate tracks when identities become ambiguous

Per-view tracking output:

```json
{
  "camera_id": "kinect_001",
  "track_id": "kinect_001_track0017",
  "frames": [
    {"frame_id": "000120", "npz_path": "...", "person_index": 1},
    {"frame_id": "000121", "npz_path": "...", "person_index": 1}
  ]
}
```

This keeps local identity stable before cross-camera matching.

## Cross-View Matching Features

For each timestamp or short temporal window, create candidate edges between detections or tracks from different cameras.

Use a weighted cost:

$$
C = w_a C_a + w_p C_p + w_s C_s + w_t C_t + w_g C_g
$$

where:

- $C_a$: appearance/ReID distance
- $C_p$: canonical pose distance
- $C_s$: shape distance
- $C_t$: temporal identity consistency cost
- $C_g$: optional calibrated geometry cost

### Appearance Cost

Use a person ReID embedding extracted from the RGB crop.

This should be the backbone signal for calibration-free association, because it is independent of camera calibration and does not rely on SAM3DBody depth.

### Canonical Pose Cost

Use root-removed canonical pose, not camera translation.

Possible features:

```text
body_pose_6d
SMPL root-centered joints
2D keypoint layout normalized by bbox
```

This helps reject visually similar people with different poses.

### Shape Cost

Use a conservative shape cue:

```text
shape_params or fitted betas
```

Shape should be low-weight. It can help across a short sequence, but single-image shape estimates are noisy.

### Temporal Consistency Cost

Once a cross-view identity is assigned, prefer keeping it stable:

```text
same camera track should keep the same global person_id
identity switches should be penalized
short gaps should be allowed
```

This cost is essential in crowded scenes.

### Geometry Cost

If calibration is available, use it as a strong debug or evaluation signal.

Useful calibrated costs:

- epipolar distance between 2D keypoints
- triangulated 3D keypoint reprojection error
- consistency of root-centered 3D pose after triangulation
- visibility-aware camera support

For this project, geometry should be optional because the core research direction is calibration-free multiview fusion. Panoptic can still use geometry for validation and ablations.

Avoid using SAM3DBody `pred_cam_t` as a primary matching cue. It is monocular and can be badly biased in absolute camera space.

## Assignment Strategy

### MVP Baseline

For the first implementation, use anchor-camera Hungarian matching:

1. Pick an anchor camera at each timestamp, preferably the camera with the most confident detections.
2. Match anchor detections to each other camera using Hungarian assignment.
3. Reject matches above a cost threshold.
4. Merge accepted matches into one cross-view person group.
5. Keep unmatched detections as single-view or low-confidence groups.

This is simple, inspectable, and good enough for early debugging.

### Better Version

Use graph clustering or min-cost flow over a temporal window.

Graph definition:

```text
node = one per-camera tracklet segment
edge = same-person likelihood across cameras
```

Constraints:

- one global person can have at most one detection per camera at one timestamp
- one camera tracklet should not belong to multiple global identities
- identity assignments should be smooth over time

This is more robust than pairwise matching when there are more than two cameras or when some views miss a person.

## Fusion-Based Consistency Check

After a tentative cross-view group is formed, use fusion consistency to reject bad groups.

A good group should have:

- low canonical pose variance across views
- low shape variance across views
- stable identity across neighboring frames
- plausible fused output

A bad association often has one view whose canonical SMPL pose strongly disagrees with the others.

This check should not be the only matcher, but it is useful as a validation step after appearance and track-based association.

## Output Dataset Contract

The Stage 4 association script should produce:

```text
multi_human_manifest.json
association_report.json
optional per-frame visualization overlays
```

The manifest should contain:

- `sample_id`
- `sequence_id`
- `frame_id`
- `person_id`
- `views`
- per-view `camera_id`
- per-view `npz_path`
- per-view `person_index`
- optional `track_id`
- optional matching diagnostics

Diagnostics should include:

```text
num_detections
num_tracklets
num_associated_people
num_single_view_people
match_costs
rejected_edges
identity_switch_candidates
```

## Evaluation

For calibrated datasets such as Panoptic:

- cross-view association accuracy if GT identity labels are available
- percentage of correctly grouped multiview detections
- identity-switch count over time
- triangulation/reprojection consistency
- downstream fused MPJPE per associated person

For calibration-free or weakly supervised data:

- temporal identity stability
- view-count distribution per person
- fusion consistency score
- manual visualization audit

## Implementation Plan

### Step 1: Export Multi-Person Metadata

Extend the SAM3DBody compact exporter to save:

```text
bbox
det_score
pred_keypoints_2d
```

If ReID is available, save:

```text
reid_embedding
```

### Step 2: Add Person-Indexed Dataset Reads

Update dataset view records to support:

```json
{"npz_path": "...", "person_index": 2}
```

When `person_index` is absent, preserve current single-person behavior for backward compatibility.

### Step 3: Build Per-View Tracks

Create:

```text
scripts/build_per_view_tracks.py
```

Inputs:

```text
SAM3DBody npz directory
RGB directory optional
camera list
```

Outputs:

```text
per_view_tracks.json
track_debug/
```

### Step 4: Build Cross-View Association

Create:

```text
scripts/build_multihuman_manifest.py
```

Inputs:

```text
per_view_tracks.json
SAM3DBody npz directory
optional cameras/calibration
```

Outputs:

```text
multi_human_stage1_manifest.json
association_report.json
```

### Step 5: Add Visual Debugging

Add overlays for:

- per-view detections and track ids
- cross-view associated person ids
- rejected high-cost edges
- missing-view cases

Association bugs are easier to diagnose visually than from metrics alone.

## Recommended First Baseline

The first practical baseline should be:

```text
per-view tracker
+ ReID appearance matching
+ canonical pose/shape consistency
+ temporal identity smoothing
+ optional Panoptic geometry diagnostics
```

This keeps the core method usable without calibration while still allowing calibrated datasets to expose association failures.

## Non-Goals For The First Version

- Do not train a learned cross-view matcher immediately.
- Do not depend on SAM3DBody absolute `pred_cam_t`.
- Do not require every person to appear in every view.
- Do not solve severe long-term re-identification after long occlusion in the first pass.

The first version should prioritize a transparent manifest builder and strong visual debugging over an opaque end-to-end learned association model.
