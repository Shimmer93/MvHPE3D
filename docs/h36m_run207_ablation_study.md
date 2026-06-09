# H36M Run207 Complete Ablation Study

Generated: 2026-06-10 06:22:00 remote local time

Purpose: complete ablation around Run207. All runs keep the Run207 lineage: Stage2 checkpoint loaded through `--stage2-checkpoint-path`, `resume_unified_checkpoint=None`, frozen Stage2, same train script, same fair SMPL -> H36M evaluation path.

Dashboard: `http://127.0.0.1:8877/` through `ssh -L 8877:127.0.0.1:8877 tbw`. Use the `run207_ablation` family filter.

Dashboard API: `http://127.0.0.1:8877/api/runs?family=run207_ablation`

Manifest: `outputs/stage2_unified/h36m_hparam_tuning/run207_ablation_manifest_20260609.json`

## Completion Evidence

- Planned ablation runs: 29 (`run231`-`run259`).
- Rows indexed in dashboard DB: 29.
- Completed rows at `latest_epoch >= 69`: 29.
- Dashboard API `run207_ablation` rows visible: 29.
- Missing from DB: [].
- Missing from API: [].
- Not complete: [].

## Baseline

| Run | Latest Epoch | Best MPJPE | PA @ Best MPJPE | Best PA | Best MPJPE Epoch | Best PA Epoch |
|---|---|---|---|---|---|---|
| 207 | 69 | 31.875 | 22.258 | 22.242 | 42 | 48 |

## Final Ablation Table

| Run | Group | Target | Latest Epoch | Best MPJPE | Delta vs Run207 | PA @ Best MPJPE | Best PA | Best MPJPE Epoch | Best PA Epoch | Description |
|---|---|---|---|---|---|---|---|---|---|---|
| 231 | input | `no_root_input_global_orient` | 69 | 32.375 | +0.500 | 22.287 | 22.276 | 67 | 66 | remove root-head input_global_orient feature |
| 232 | input | `no_root_input_transl` | 69 | 32.127 | +0.252 | 22.289 | 22.251 | 37 | 54 | remove root-head input_transl feature |
| 233 | input | `no_root_measurement_residual` | 69 | 32.285 | +0.410 | 22.274 | 22.221 | 35 | 58 | remove root-head 2D measurement residual feature |
| 234 | input | `no_root_measurement_confidence` | 69 | 32.074 | +0.199 | 22.339 | 22.234 | 33 | 61 | remove root-head 2D measurement confidence feature |
| 235 | input | `no_root_measurement_valid` | 69 | 31.860 | -0.015 | 22.362 | 22.282 | 26 | 56 | ignore root-head 2D measurement valid mask |
| 236 | input | `no_root_image_size` | 69 | 31.922 | +0.047 | 22.332 | 22.237 | 57 | 39 | remove root-head image-size normalization |
| 237 | input | `no_body_stage2_pose` | 69 | 31.913 | +0.038 | 22.221 | 22.221 | 69 | 69 | remove frozen Stage2 body-pose feature from body head |
| 238 | input | `no_body_input_pose_mean` | 69 | 32.229 | +0.354 | 22.392 | 22.282 | 64 | 59 | remove decoded views_input pose mean from body head |
| 239 | input | `no_body_input_pose_dispersion` | 69 | 31.953 | +0.078 | 22.327 | 22.279 | 57 | 53 | remove decoded views_input pose dispersion from body head |
| 240 | input | `no_body_input_betas` | 69 | 32.155 | +0.281 | 22.345 | 22.289 | 41 | 61 | remove decoded views_input betas mean from body head |
| 241 | input | `no_body_image_residual` | 69 | 32.138 | +0.263 | 22.251 | 22.238 | 42 | 66 | remove body-head root-corrected 2D residual feature |
| 242 | input | `no_body_image_confidence` | 69 | 32.007 | +0.132 | 22.266 | 22.239 | 56 | 33 | remove body-head 2D measurement confidence feature |
| 243 | input | `no_body_image_valid` | 69 | 31.937 | +0.062 | 22.303 | 22.294 | 53 | 44 | ignore body-head 2D measurement valid mask |
| 244 | input | `no_body_image_size` | 69 | 32.138 | +0.263 | 22.382 | 22.281 | 36 | 50 | remove body-head image-size normalization |
| 245 | loss | `no_camera_joint_loss` | 69 | 34.088 | +2.213 | 22.279 | 22.266 | 68 | 60 | remove main camera-joint training loss |
| 246 | loss | `no_pa_joint_loss` | 69 | 31.857 | -0.018 | 22.307 | 22.294 | 60 | 55 | remove PA-joint training loss |
| 247 | loss | `no_gt_projection_loss` | 69 | 32.007 | +0.132 | 22.321 | 22.197 | 30 | 65 | remove GT 2D projection loss |
| 248 | loss | `no_preserve_joint_loss` | 69 | 31.990 | +0.115 | 22.306 | 22.245 | 57 | 59 | remove canonical preserve loss |
| 249 | loss | `no_do_no_harm_loss` | 69 | 32.076 | +0.201 | 22.284 | 22.255 | 66 | 54 | remove do-no-harm loss |
| 250 | loss | `no_gate_sparsity_loss` | 69 | 31.995 | +0.120 | 22.270 | 22.270 | 38 | 38 | remove body gate sparsity loss |
| 251 | loss | `no_body_delta_loss` | 69 | 31.956 | +0.082 | 22.325 | 22.300 | 69 | 60 | remove body-pose delta L2 loss |
| 252 | loss | `no_betas_delta_loss` | 69 | 32.085 | +0.210 | 22.337 | 22.305 | 54 | 40 | remove betas delta L2 loss |
| 253 | loss | `no_global_orient_delta_loss` | 69 | 32.078 | +0.203 | 22.330 | 22.271 | 49 | 56 | remove global-orient delta L2 loss |
| 254 | loss_schedule | `no_limb_curriculum` | 69 | 31.974 | +0.099 | 22.241 | 22.241 | 44 | 44 | remove limb2 curriculum weights from camera and PA losses |
| 255 | update_path | `no_body_update` | 69 | 32.553 | +0.678 | 22.459 | 22.458 | 33 | 24 | disable body update for all epochs by starting body after training |
| 256 | update_path | `no_root_update` | 69 | 35.809 | +3.934 | 22.254 | 22.242 | 19 | 29 | disable root/global-orient update by setting delta scale to zero |
| 257 | update_path | `no_compose_global_orient_delta` | 69 | 32.097 | +0.223 | 22.349 | 22.255 | 58 | 27 | use axis-angle addition instead of SO(3) composition for root update |
| 258 | compound_input | `views_input_only_external_removed` | 69 | 33.030 | +1.155 | 22.336 | 22.281 | 53 | 28 | keep views_input-derived evidence and remove external camera/image/Stage2 body inputs |
| 259 | compound_input | `no_all_image_measurement_inputs` | 69 | 32.275 | +0.400 | 22.327 | 22.249 | 57 | 54 | remove all root/body 2D measurement residual/confidence/valid/image-size inputs |

## Impact Sorted By MPJPE Delta

| Run | Target | Group | Best MPJPE | Delta vs Run207 | Best PA | Description |
|---|---|---|---|---|---|---|
| 256 | `no_root_update` | update_path | 35.809 | +3.934 | 22.242 | disable root/global-orient update by setting delta scale to zero |
| 245 | `no_camera_joint_loss` | loss | 34.088 | +2.213 | 22.266 | remove main camera-joint training loss |
| 258 | `views_input_only_external_removed` | compound_input | 33.030 | +1.155 | 22.281 | keep views_input-derived evidence and remove external camera/image/Stage2 body inputs |
| 255 | `no_body_update` | update_path | 32.553 | +0.678 | 22.458 | disable body update for all epochs by starting body after training |
| 231 | `no_root_input_global_orient` | input | 32.375 | +0.500 | 22.276 | remove root-head input_global_orient feature |
| 233 | `no_root_measurement_residual` | input | 32.285 | +0.410 | 22.221 | remove root-head 2D measurement residual feature |
| 259 | `no_all_image_measurement_inputs` | compound_input | 32.275 | +0.400 | 22.249 | remove all root/body 2D measurement residual/confidence/valid/image-size inputs |
| 238 | `no_body_input_pose_mean` | input | 32.229 | +0.354 | 22.282 | remove decoded views_input pose mean from body head |
| 240 | `no_body_input_betas` | input | 32.155 | +0.281 | 22.289 | remove decoded views_input betas mean from body head |
| 241 | `no_body_image_residual` | input | 32.138 | +0.263 | 22.238 | remove body-head root-corrected 2D residual feature |
| 244 | `no_body_image_size` | input | 32.138 | +0.263 | 22.281 | remove body-head image-size normalization |
| 232 | `no_root_input_transl` | input | 32.127 | +0.252 | 22.251 | remove root-head input_transl feature |
| 257 | `no_compose_global_orient_delta` | update_path | 32.097 | +0.223 | 22.255 | use axis-angle addition instead of SO(3) composition for root update |
| 252 | `no_betas_delta_loss` | loss | 32.085 | +0.210 | 22.305 | remove betas delta L2 loss |
| 253 | `no_global_orient_delta_loss` | loss | 32.078 | +0.203 | 22.271 | remove global-orient delta L2 loss |
| 249 | `no_do_no_harm_loss` | loss | 32.076 | +0.201 | 22.255 | remove do-no-harm loss |
| 234 | `no_root_measurement_confidence` | input | 32.074 | +0.199 | 22.234 | remove root-head 2D measurement confidence feature |
| 247 | `no_gt_projection_loss` | loss | 32.007 | +0.132 | 22.197 | remove GT 2D projection loss |
| 242 | `no_body_image_confidence` | input | 32.007 | +0.132 | 22.239 | remove body-head 2D measurement confidence feature |
| 250 | `no_gate_sparsity_loss` | loss | 31.995 | +0.120 | 22.270 | remove body gate sparsity loss |
| 248 | `no_preserve_joint_loss` | loss | 31.990 | +0.115 | 22.245 | remove canonical preserve loss |
| 254 | `no_limb_curriculum` | loss_schedule | 31.974 | +0.099 | 22.241 | remove limb2 curriculum weights from camera and PA losses |
| 251 | `no_body_delta_loss` | loss | 31.956 | +0.082 | 22.300 | remove body-pose delta L2 loss |
| 239 | `no_body_input_pose_dispersion` | input | 31.953 | +0.078 | 22.279 | remove decoded views_input pose dispersion from body head |
| 243 | `no_body_image_valid` | input | 31.937 | +0.062 | 22.294 | ignore body-head 2D measurement valid mask |
| 236 | `no_root_image_size` | input | 31.922 | +0.047 | 22.237 | remove root-head image-size normalization |
| 237 | `no_body_stage2_pose` | input | 31.913 | +0.038 | 22.221 | remove frozen Stage2 body-pose feature from body head |
| 235 | `no_root_measurement_valid` | input | 31.860 | -0.015 | 22.282 | ignore root-head 2D measurement valid mask |
| 246 | `no_pa_joint_loss` | loss | 31.857 | -0.018 | 22.294 | remove PA-joint training loss |

## Reading

- Best ablation by MPJPE is `run246` (`no_pa_joint_loss`) with 31.857 mm, delta -0.018 mm vs Run207. This is too small to treat as a real improvement without repeat seeds, and its best PA is 22.294 mm vs Run207 best PA 22.242 mm.
- Best ablation by PA-MPJPE is `run247` (`no_gt_projection_loss`) with 22.197 mm, but its MPJPE is 32.007 mm, delta +0.132 mm vs Run207.
- Worst ablation by MPJPE is `run256` (`no_root_update`) with 35.809 mm, delta +3.934 mm.
- Large MPJPE regressions identify required components: root/global-orient update, camera-joint loss, external camera/image/Stage2 body evidence beyond views_input-only, body update, root global-orient input, root 2D residual, and image-measurement inputs.
- Sub-0.1 mm deltas should be treated as noise unless repeated; they are not enough to justify dropping a component by themselves.

Large MPJPE regressions (`Delta vs Run207 >= +0.250`):

- `run256` `no_root_update`: +3.934 mm, disable root/global-orient update by setting delta scale to zero.
- `run245` `no_camera_joint_loss`: +2.213 mm, remove main camera-joint training loss.
- `run258` `views_input_only_external_removed`: +1.155 mm, keep views_input-derived evidence and remove external camera/image/Stage2 body inputs.
- `run255` `no_body_update`: +0.678 mm, disable body update for all epochs by starting body after training.
- `run231` `no_root_input_global_orient`: +0.500 mm, remove root-head input_global_orient feature.
- `run233` `no_root_measurement_residual`: +0.410 mm, remove root-head 2D measurement residual feature.
- `run259` `no_all_image_measurement_inputs`: +0.400 mm, remove all root/body 2D measurement residual/confidence/valid/image-size inputs.
- `run238` `no_body_input_pose_mean`: +0.354 mm, remove decoded views_input pose mean from body head.
- `run240` `no_body_input_betas`: +0.281 mm, remove decoded views_input betas mean from body head.
- `run241` `no_body_image_residual`: +0.263 mm, remove body-head root-corrected 2D residual feature.
- `run244` `no_body_image_size`: +0.263 mm, remove body-head image-size normalization.
- `run232` `no_root_input_transl`: +0.252 mm, remove root-head input_transl feature.

Near-noise MPJPE deltas (`abs(delta) < 0.100`):

- `run235` `no_root_measurement_valid`: -0.015 mm.
- `run246` `no_pa_joint_loss`: -0.018 mm.
- `run237` `no_body_stage2_pose`: +0.038 mm.
- `run236` `no_root_image_size`: +0.047 mm.
- `run243` `no_body_image_valid`: +0.062 mm.
- `run239` `no_body_input_pose_dispersion`: +0.078 mm.
- `run251` `no_body_delta_loss`: +0.082 mm.
- `run254` `no_limb_curriculum`: +0.099 mm.

## Dashboard API Snapshot

- Visible run numbers: 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259
