#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/dysData/shimmer/MvHPE3D"
UV="/dysData/shimmer/.local/bin/uv"
DATASET_ROOT="/dysData/shimmer/datasets/behave"
DB_ROOT="/dysData/shimmer/datasets/behave/preprocessed_data_z"
MANIFEST_PATH="data/behave/behave_stage2_manifest.json"
SAM_ROOT="data/behave/sam3dbody"
INPUT_SMPL_CACHE="data/behave/sam3dbody_fitted_smpl"
SAM_LOG="logs/behave_sam3dbody_multigpu_20260526_0117.log"
SAM_PID_FILE="${SAM_LOG}.pid"
GPUS="0,1,2,3,4,5,6,7"

cd "$REPO_ROOT"
mkdir -p logs

echo "[INFO] started behave stage2 continuation at $(date)"
echo "[INFO] repo: $REPO_ROOT"
echo "[INFO] manifest: $MANIFEST_PATH"

if [[ -f "$SAM_PID_FILE" ]]; then
    SAM_PID="$(cat "$SAM_PID_FILE")"
    echo "[INFO] monitoring SAM3DBody pid $SAM_PID"
    while ps -p "$SAM_PID" >/dev/null 2>&1; do
        OUTPUTS="$(find "$SAM_ROOT" -name "*.npz" 2>/dev/null | wc -l)"
        OK_JOBS="$(grep -c "^\[OK gpu=" "$SAM_LOG" 2>/dev/null || true)"
        FAIL_JOBS="$(grep -c "^\[FAIL gpu=" "$SAM_LOG" 2>/dev/null || true)"
        RETRY_JOBS="$(grep -c "^\[RETRY gpu=" "$SAM_LOG" 2>/dev/null || true)"
        echo "[INFO] $(date) SAM running: outputs=$OUTPUTS ok=$OK_JOBS fail=$FAIL_JOBS retry=$RETRY_JOBS"
        sleep 120
    done
else
    echo "[WARN] SAM pid file not found: $SAM_PID_FILE"
fi

echo "[INFO] SAM3DBody process no longer running at $(date)"
if ! grep -q "All BEHAVE SAM3DBody jobs finished." "$SAM_LOG"; then
    echo "[ERROR] SAM3DBody log does not show successful completion: $SAM_LOG" >&2
    tail -120 "$SAM_LOG" >&2 || true
    exit 1
fi
if grep -q "BEHAVE SAM3DBody jobs failed" "$SAM_LOG"; then
    echo "[ERROR] SAM3DBody log contains failed jobs: $SAM_LOG" >&2
    tail -120 "$SAM_LOG" >&2 || true
    exit 1
fi

echo "[INFO] rebuilding BEHAVE manifest with valid SAM views"
"$UV" run python scripts/prepare_behave_stage2.py \
    --dataset-root "$DATASET_ROOT" \
    --heatformer-db-root "$DB_ROOT" \
    --output-root data/behave \
    --manifest-path "$MANIFEST_PATH" \
    --sam3dbody-root "$SAM_ROOT" \
    --min-views 0

echo "[INFO] manifest coverage summary"
"$UV" run python - <<'PY'
import collections
import json
from pathlib import Path

manifest = json.loads(Path("data/behave/behave_stage2_manifest.json").read_text())
report = json.loads(Path("data/behave/behave_stage2_manifest.report.json").read_text())
samples = manifest["samples"]
view_counts = collections.Counter(len(sample["views"]) for sample in samples)
split_view_counts = collections.defaultdict(collections.Counter)
for sample in samples:
    split_view_counts[sample["split"]][len(sample["views"])] += 1
print("samples", len(samples))
print("view_counts", dict(sorted(view_counts.items())))
print("split_view_counts", {key: dict(sorted(value.items())) for key, value in sorted(split_view_counts.items())})
print("summary", report.get("summary"))
train4 = sum(1 for sample in samples if sample["split"] == "train" and len(sample["views"]) >= 4)
val4 = sum(1 for sample in samples if sample["split"] == "val" and len(sample["views"]) >= 4)
print("train_min4", train4)
print("val_min4", val4)
if train4 == 0 or val4 == 0:
    raise SystemExit("No train/val samples with four valid views.")
PY

echo "[INFO] precomputing input SMPL cache"
"$UV" run python scripts/run_precompute_input_smpl_multigpu.py \
    --manifest-path "$MANIFEST_PATH" \
    --cache-dir "$INPUT_SMPL_CACHE" \
    --smpl-model-path data/weights/SMPL_NEUTRAL.pkl \
    --mhr-assets-dir data/assets \
    --gpus "$GPUS" \
    --batch-size 256 \
    --skip-existing

echo "[INFO] running BEHAVE Stage 2 fast-dev smoke"
"$UV" run python scripts/train.py \
    --config configs/experiment/behave_stage2_joint_residual.yaml \
    --smpl-model-path data/weights/SMPL_NEUTRAL.pkl \
    --mhr-assets-dir data/assets \
    --input-smpl-cache-dir "$INPUT_SMPL_CACHE" \
    --devices 1 \
    --strategy auto \
    --fast-dev-run

echo "[INFO] launching full BEHAVE Stage 2 training"
"$UV" run python scripts/train.py \
    --config configs/experiment/behave_stage2_joint_residual.yaml \
    --smpl-model-path data/weights/SMPL_NEUTRAL.pkl \
    --mhr-assets-dir data/assets \
    --input-smpl-cache-dir "$INPUT_SMPL_CACHE" \
    --devices "$GPUS" \
    --strategy ddp_find_unused_parameters_true \
    --max-epochs 100 \
    --test-after-train

echo "[INFO] BEHAVE Stage 2 continuation finished at $(date)"
