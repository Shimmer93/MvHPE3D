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
  --split-config-path PATH
                          Optional split policy YAML override.
  --split-name NAME       Optional named split policy override.
  --default-root-dir DIR  Output root for checkpoints and logs.
                          Default: outputs/stage1
  -h, --help              Show this help message.

Examples:
  bash scripts/train_fusion.sh \
    --manifest-path /opt/data/humman_cropped/humman_stage1_manifest.json

  CUDA_VISIBLE_DEVICES=0 bash scripts/train_fusion.sh \
    --manifest-path /opt/data/humman_cropped/humman_stage1_manifest.json \
    --split-name random_split \
    --max-epochs 50
EOF
}

MANIFEST_PATH=""
CONFIG_PATH="${DEFAULT_CONFIG}"
SPLIT_CONFIG_PATH=""
SPLIT_NAME=""
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

CMD=(
    uv run python scripts/train.py
    --config "${CONFIG_PATH}"
    --manifest-path "${MANIFEST_PATH}"
    --default-root-dir "${OUTPUT_ROOT}"
)

if [[ -n "${SPLIT_CONFIG_PATH}" ]]; then
    CMD+=(--split-config-path "${SPLIT_CONFIG_PATH}")
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
