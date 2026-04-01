#!/bin/bash

CUDA_VISIBLE_DEVICES=1 MOMENTUM_ENABLED=0 uv run python external/sam-3d-body/demo_save_compact_params.py \
    --image_folder /opt/data/humman_cropped/rgb \
    --output_folder /opt/data/humman_cropped/sam3dbody \
    --checkpoint_path /opt/data/SAM_3dbody_checkpoints/model.ckpt \
    --mhr_path /opt/data/SAM_3dbody_checkpoints/assets/mhr_model.pt \
    --detector_name "" \
    --inference_type body \
    --batch_size 16