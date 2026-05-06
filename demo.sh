PYTHONPATH=src uv run python scripts/demo_video_2x2_mesh_compare.py \
    --checkpoint-path /home/zpengac/mmhpe/MvHPE3D/outputs/stage2/stage2_cross_camera_joint_graph_refiner/version_1/checkpoints/epoch=080-step=015390.ckpt \
    --config configs/experiment/stage2_cross_camera_joint_graph_refiner.yaml \
    --manifest-path /opt/data/humman_cropped/humman_stage1_manifest.json \
    --input-smpl-cache-dir /opt/data/humman_cropped/sam3dbody_fitted_smpl \
    --sequence-a p000571_a000182 \
    --view-a kinect_007 \
    --reference-frame-a index \
    --reference-frame-index-a 20 \
    --sequence-b p001221_a000182 \
    --view-b kinect_001 \
    --reference-frame-b index \
    --reference-frame-index-b 20 \
    --output-dir outputs/demo
    
PYTHONPATH=src uv run python scripts/demo_video_2x2_mesh_compare.py \
    --checkpoint-path /home/zpengac/mmhpe/MvHPE3D/outputs/stage2/stage2_cross_camera_joint_graph_refiner/version_1/checkpoints/epoch=080-step=015390.ckpt \
    --config configs/experiment/stage2_cross_camera_joint_graph_refiner.yaml \
    --manifest-path /opt/data/humman_cropped/humman_stage1_manifest.json \
    --input-smpl-cache-dir /opt/data/humman_cropped/sam3dbody_fitted_smpl \
    --sequence-a p000438_a000040 \
    --view-a kinect_007 \
    --reference-frame-a index \
    --reference-frame-index-a 65 \
    --sequence-b p000488_a000040 \
    --view-b kinect_001 \
    --reference-frame-b index \
    --reference-frame-index-b 50 \
    --output-dir outputs/demo

PYTHONPATH=src uv run python scripts/demo_video_2x2_mesh_compare.py \
    --checkpoint-path /home/zpengac/mmhpe/MvHPE3D/outputs/stage2/stage2_cross_camera_joint_graph_refiner/version_1/checkpoints/epoch=080-step=015390.ckpt \
    --config configs/experiment/stage2_cross_camera_joint_graph_refiner.yaml \
    --manifest-path /opt/data/humman_cropped/humman_stage1_manifest.json \
    --input-smpl-cache-dir /opt/data/humman_cropped/sam3dbody_fitted_smpl \
    --sequence-a p000454_a000071 \
    --view-a kinect_007 \
    --reference-frame-a index \
    --reference-frame-index-a 55 \
    --sequence-b p000447_a000071 \
    --view-b kinect_001 \
    --reference-frame-b index \
    --reference-frame-index-b 50 \
    --output-dir outputs/demo