#!/usr/bin/env bash

bash scripts/test_fusion.sh \
    --config configs/experiment/stage2_cross_camera_joint_graph_refiner.yaml \
    --checkpoint-path /home/zpengac/mmhpe/MvHPE3D/outputs/stage2/stage2_cross_camera_joint_graph_refiner/version_1/checkpoints/epoch=080-step=015390.ckpt \
    --manifest-path /opt/data/humman_cropped/humman_stage1_manifest.json \
    --gt-smpl-dir /opt/data/humman_cropped/smpl \
    --cameras-dir /opt/data/humman_cropped/cameras \
    --mhr-assets-dir /opt/data/assets \
    --input-smpl-cache-dir /opt/data/humman_cropped/sam3dbody_fitted_smpl \
    --pred-camera-mode input_corrected \
    --stage test