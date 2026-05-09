#!/usr/bin/env bash

# uv run python scripts/train.py \
#     --config configs/experiment/stage3_temporal_refine_e2e_scratch.yaml \
#     --manifest-path /opt/data/humman_cropped/humman_stage1_manifest.json \
#     --gt-smpl-dir /opt/data/humman_cropped/smpl \
#     --cameras-dir /opt/data/humman_cropped/cameras \
#     --mhr-assets-dir /opt/data/assets \
#     --input-smpl-cache-dir /opt/data/humman_cropped/sam3dbody_fitted_smpl

uv run python scripts/train.py \
    --config configs/experiment/stage3_temporal_refine_joint_train.yaml \
    --stage2-checkpoint-path /home/zpengac/mmhpe/MvHPE3D/outputs/stage2/stage2_cross_camera_joint_graph_refiner/version_1/checkpoints/epoch=080-step=015390.ckpt \
    --manifest-path /opt/data/humman_cropped/humman_stage1_manifest.json \
    --gt-smpl-dir /opt/data/humman_cropped/smpl \
    --cameras-dir /opt/data/humman_cropped/cameras \
    --mhr-assets-dir /opt/data/assets \
    --input-smpl-cache-dir /opt/data/humman_cropped/sam3dbody_fitted_smpl