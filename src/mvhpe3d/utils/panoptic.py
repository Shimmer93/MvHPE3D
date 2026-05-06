"""Panoptic/Kinoptic joint constants shared by data and metrics."""

from __future__ import annotations

PANOPTIC_GT_UNIT_SCALE = 0.01
PANOPTIC_BODY_CENTER_INDEX = 2

# Panoptic coco19 joint order:
# 0 Neck, 1 Nose, 2 BodyCenter, 3 lShoulder, 4 lElbow, 5 lWrist,
# 6 lHip, 7 lKnee, 8 lAnkle, 9 rShoulder, 10 rElbow, 11 rWrist,
# 12 rHip, 13 rKnee, 14 rAnkle, 15 lEye, 16 lEar, 17 rEye, 18 rEar.
PANOPTIC_TO_SMPL24 = {
    2: 0,  # BodyCenter -> pelvis
    6: 1,  # lHip
    12: 2,  # rHip
    7: 4,  # lKnee
    13: 5,  # rKnee
    8: 7,  # lAnkle
    14: 8,  # rAnkle
    0: 12,  # Neck
    3: 16,  # lShoulder
    9: 17,  # rShoulder
    4: 18,  # lElbow
    10: 19,  # rElbow
    5: 20,  # lWrist
    11: 21,  # rWrist
}

PANOPTIC_EVAL_JOINT_INDICES = tuple(sorted(PANOPTIC_TO_SMPL24))
PANOPTIC_EVAL_SMPL24_INDICES = tuple(
    PANOPTIC_TO_SMPL24[index] for index in PANOPTIC_EVAL_JOINT_INDICES
)
PANOPTIC_EVAL_ROOT_COLUMN = PANOPTIC_EVAL_JOINT_INDICES.index(
    PANOPTIC_BODY_CENTER_INDEX
)
