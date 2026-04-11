#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_CONFIG="configs/experiment/stage1_cross_camera.yaml"
DEFAULT_OUTPUT_ROOT="outputs/stage1"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/train_fusion.sh --manifest-path /path/to/humman_stage1_manifest.json [extra train.py args]

Required:
  --manifest-path PATH    Path to the Stage 1 manifest JSON used by scripts/train.py.

Optional:
  --config PATH           Experiment config to use.
                          Default: configs/experiment/stage1_cross_camera.yaml
  --gt-smpl-dir PATH      Optional HuMMan GT SMPL directory override.
  --split-config-path PATH
                          Optional split policy YAML override.
  --split-name NAME       Optional named split policy override.
  --accelerator VALUE     Forwarded to scripts/train.py.
  --devices VALUE         Forwarded to scripts/train.py.
  --strategy VALUE        Forwarded to scripts/train.py.
  --num-nodes VALUE       Forwarded to scripts/train.py.
  --test-after-train      Forwarded to scripts/train.py.
  --test-ckpt VALUE       Forwarded to scripts/train.py. One of: best, last, current.
  --mhr-assets-dir PATH   Forwarded to scripts/train.py for test-time input conversion.
  --input-smpl-cache-dir PATH
                          Optional cache directory for fitted input-view SMPL parameters.
  --disable-learn-betas   Forwarded to scripts/train.py. Disables learned beta prediction.
  --default-root-dir DIR  Output root for checkpoints and logs.
                          Default: outputs/stage1
  -h, --help              Show this help message.

Examples:
  bash scripts/train_fusion.sh \
    --manifest-path /opt/data/humman_cropped/humman_stage1_manifest.json

  bash scripts/train_fusion.sh \
    --manifest-path /opt/data/humman_cropped/humman_stage1_manifest.json \
    --gt-smpl-dir /opt/data/humman_cropped/smpl

  CUDA_VISIBLE_DEVICES=0 bash scripts/train_fusion.sh \
    --manifest-path /opt/data/humman_cropped/humman_stage1_manifest.json \
    --split-name random_split \
    --max-epochs 50

  CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_fusion.sh \
    --manifest-path /opt/data/humman_cropped/humman_stage1_manifest.json \
    --accelerator gpu \
    --devices 2 \
    --strategy ddp

  bash scripts/train_fusion.sh \
    --manifest-path /opt/data/humman_cropped/humman_stage1_manifest.json \
    --gt-smpl-dir /opt/data/humman_cropped/smpl \
    --disable-learn-betas \
    --test-after-train \
    --test-ckpt best
EOF
}

MANIFEST_PATH=""
CONFIG_PATH="${DEFAULT_CONFIG}"
SPLIT_CONFIG_PATH=""
SPLIT_NAME=""
GT_SMPL_DIR=""
OUTPUT_ROOT="${DEFAULT_OUTPUT_ROOT}"
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest-path)
            if [[ $# -lt 2 ]]; then
                echo "error: --manifest-path requires a value" >&2
                exit 1
            fi
            MANIFEST_PATH="$2"
            shift 2
            ;;
        --config)
            if [[ $# -lt 2 ]]; then
                echo "error: --config requires a value" >&2
                exit 1
            fi
            CONFIG_PATH="$2"
            shift 2
            ;;
        --split-config-path)
            if [[ $# -lt 2 ]]; then
                echo "error: --split-config-path requires a value" >&2
                exit 1
            fi
            SPLIT_CONFIG_PATH="$2"
            shift 2
            ;;
        --gt-smpl-dir)
            if [[ $# -lt 2 ]]; then
                echo "error: --gt-smpl-dir requires a value" >&2
                exit 1
            fi
            GT_SMPL_DIR="$2"
            shift 2
            ;;
        --split-name)
            if [[ $# -lt 2 ]]; then
                echo "error: --split-name requires a value" >&2
                exit 1
            fi
            SPLIT_NAME="$2"
            shift 2
            ;;
        --default-root-dir)
            if [[ $# -lt 2 ]]; then
                echo "error: --default-root-dir requires a value" >&2
                exit 1
            fi
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "${MANIFEST_PATH}" ]]; then
    usage
    echo "error: --manifest-path is required" >&2
    exit 1
fi

if [[ ! -f "${REPO_ROOT}/${CONFIG_PATH}" && ! -f "${CONFIG_PATH}" ]]; then
    echo "error: config not found: ${CONFIG_PATH}" >&2
    exit 1
fi

if [[ ! -f "${REPO_ROOT}/${MANIFEST_PATH}" && ! -f "${MANIFEST_PATH}" ]]; then
    echo "error: manifest not found: ${MANIFEST_PATH}" >&2
    exit 1
fi

cd "${REPO_ROOT}"

if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH}"
else
    export PYTHONPATH="${REPO_ROOT}/src"
fi

CMD=(
    uv run python scripts/train.py
    --config "${CONFIG_PATH}"
    --manifest-path "${MANIFEST_PATH}"
    --default-root-dir "${OUTPUT_ROOT}"
)

if [[ -n "${SPLIT_CONFIG_PATH}" ]]; then
    CMD+=(--split-config-path "${SPLIT_CONFIG_PATH}")
fi

if [[ -n "${GT_SMPL_DIR}" ]]; then
    CMD+=(--gt-smpl-dir "${GT_SMPL_DIR}")
fi

if [[ -n "${SPLIT_NAME}" ]]; then
    CMD+=(--split-name "${SPLIT_NAME}")
fi

if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
    CMD+=("${PASSTHROUGH_ARGS[@]}")
fi

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
