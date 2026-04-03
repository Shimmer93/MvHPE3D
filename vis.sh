#!/usr/bin/env bash

uv run python scripts/visualize.py \
    --checkpoint-path /home/zpengac/mmhpe/MvHPE3D/outputs/stage1/checkpoints/stage1_cross_camera/epochepoch=067-stepstep=206788.ckpt \
    --manifest-path /opt/data/humman_cropped/humman_stage1_manifest.json \
    --gt-smpl-dir /opt/data/humman_cropped/smpl \
    --stage test \
    --num-samples 8 \
    --output-dir outputs/stage1_visualizations