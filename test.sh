#!/usr/bin/env bash

bash scripts/test_fusion.sh \
    --checkpoint-path /home/zpengac/mmhpe/MvHPE3D/outputs/stage1/stage1_cross_camera/version_3/checkpoints/last.ckpt \
    --manifest-path /opt/data/humman_cropped/humman_stage1_manifest.json \
    --gt-smpl-dir /opt/data/humman_cropped/smpl \
    --cameras-dir /opt/data/humman_cropped/cameras \
    --mhr-assets-dir /opt/data/assets \
    --input-smpl-cache-dir /opt/data/humman_cropped/sam3dbody_fitted_smpl \
    --stage test