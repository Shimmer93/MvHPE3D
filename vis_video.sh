# uv run python scripts/visualize_video.py \
#   --config configs/experiment/stage2_cross_camera.yaml \
#   --checkpoint-path /home/zpengac/mmhpe/MvHPE3D/outputs/stage1/stage2_cross_camera/version_0/checkpoints/epoch=090-step=017290.ckpt \
#   --manifest-path /opt/data/humman_cropped/humman_stage1_manifest.json \
#   --gt-smpl-dir /opt/data/humman_cropped/smpl \
#   --cameras-dir /opt/data/humman_cropped/cameras \
#   --mhr-assets-dir /opt/data/assets \
#   --input-smpl-cache-dir /opt/data/humman_cropped/sam3dbody_fitted_smpl \
#   --output-dir outputs/stage2_videos_new \
#   --stage test \
#   --max-sequences 10 \
#   --pred-camera-mode gt \
#   # --selection-mode best_improvement \
#   # --selection-metric pa_mpjpe \
#   "$@"

# uv run python scripts/visualize_video_new.py \
#     --config configs/experiment/stage2_cross_camera_joint_graph_refiner.yaml \
#     --checkpoint-path /home/zpengac/mmhpe/MvHPE3D/outputs/stage2/stage2_cross_camera_joint_graph_refiner/version_1/checkpoints/epoch=080-step=015390.ckpt \
#     --manifest-path /opt/data/humman_cropped/humman_stage1_manifest.json \
#     --cameras-dir /opt/data/humman_cropped/cameras \
#     --gt-smpl-dir /opt/data/humman_cropped/smpl \
#     --input-smpl-cache-dir /opt/data/humman_cropped/sam3dbody_fitted_smpl \
#     --output-dir outputs/stage2_videos_refiner \
#     --stage test \
#     --pred-camera-mode input_corrected \
#     --max-sequences 10 \
#     # --selection-mode best_improvement \
#     # --selection-metric pa_mpjpe \
#     "$@"

# PYTHONPATH=src uv run python scripts/demo_video_2x2_mesh_compare.py \
#     --checkpoint-path /home/zpengac/mmhpe/MvHPE3D/outputs/stage2/stage2_cross_camera_joint_graph_refiner/version_1/checkpoints/epoch=080-step=015390.ckpt \
#     --config configs/experiment/stage2_cross_camera_joint_graph_refiner.yaml \
#     --manifest-path /opt/data/humman_cropped/humman_stage1_manifest.json \
#     --input-smpl-cache-dir /opt/data/humman_cropped/sam3dbody_fitted_smpl \
#     --sequence-a p000571_a000182 \
#     --view-a kinect_007 \
#     --reference-frame-a index \
#     --reference-frame-index-a 20 \
#     --sequence-b p001221_a000182 \
#     --view-b kinect_001 \
#     --reference-frame-b index \
#     --reference-frame-index-b 20 \
#     --output-dir outputs/demo
    
# PYTHONPATH=src uv run python scripts/demo_video_2x2_mesh_compare.py \
#     --checkpoint-path /home/zpengac/mmhpe/MvHPE3D/outputs/stage2/stage2_cross_camera_joint_graph_refiner/version_1/checkpoints/epoch=080-step=015390.ckpt \
#     --config configs/experiment/stage2_cross_camera_joint_graph_refiner.yaml \
#     --manifest-path /opt/data/humman_cropped/humman_stage1_manifest.json \
#     --input-smpl-cache-dir /opt/data/humman_cropped/sam3dbody_fitted_smpl \
#     --sequence-a p000438_a000040 \
#     --view-a kinect_007 \
#     --reference-frame-a index \
#     --reference-frame-index-a 65 \
#     --sequence-b p000488_a000040 \
#     --view-b kinect_001 \
#     --reference-frame-b index \
#     --reference-frame-index-b 50 \
#     --output-dir outputs/demo

# PYTHONPATH=src uv run python scripts/demo_video_2x2_mesh_compare.py \
#     --checkpoint-path /home/zpengac/mmhpe/MvHPE3D/outputs/stage2/stage2_cross_camera_joint_graph_refiner/version_1/checkpoints/epoch=080-step=015390.ckpt \
#     --config configs/experiment/stage2_cross_camera_joint_graph_refiner.yaml \
#     --manifest-path /opt/data/humman_cropped/humman_stage1_manifest.json \
#     --input-smpl-cache-dir /opt/data/humman_cropped/sam3dbody_fitted_smpl \
#     --sequence-a p000454_a000071 \
#     --view-a kinect_007 \
#     --reference-frame-a index \
#     --reference-frame-index-a 55 \
#     --sequence-b p000447_a000071 \
#     --view-b kinect_001 \
#     --reference-frame-b index \
#     --reference-frame-index-b 50 \
#     --output-dir outputs/demo

PYTHONPATH=src uv run python scripts/render_smpl_normal_image.py \
    --config configs/experiment/stage2_cross_camera_joint_graph_refiner.yaml \
    --manifest-path /opt/data/humman_cropped/humman_stage1_manifest.json \
    --stage test \
    --max-sequences 100 \
    --random-selection \
    --reference-frame random \
    --output-dir /home/zpengac/mmhpe/WildHuman/input_humman