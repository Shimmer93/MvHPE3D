uv run python scripts/visualize_video.py \
  --config configs/experiment/stage1_cross_camera_residual_fusion.yaml \
  --checkpoint-path /home/zpengac/mmhpe/MvHPE3D/outputs/stage1/stage1_cross_camera_residual_fusion/version_0/checkpoints/epoch=092-step=017670.ckpt \
  --manifest-path /opt/data/humman_cropped/humman_stage1_manifest.json \
  --gt-smpl-dir /opt/data/humman_cropped/smpl \
  --cameras-dir /opt/data/humman_cropped/cameras \
  --mhr-assets-dir /opt/data/assets \
  --input-smpl-cache-dir /opt/data/humman_cropped/sam3dbody_fitted_smpl \
  --output-dir outputs/stage1_videos \
  --stage test \
  --max-sequences 100 \
  --pred-camera-mode gt \
  --selection-mode best_improvement \
  --selection-metric mpjpe \
  "$@"
