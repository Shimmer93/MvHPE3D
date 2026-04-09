#!/usr/bin/env bash

set -Eeuo pipefail

on_error() {
    local exit_code="$?"
    echo "error: scripts/test_fusion.sh failed at line ${BASH_LINENO[0]}: ${BASH_COMMAND}" >&2
    exit "${exit_code}"
}

trap on_error ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_CONFIG="configs/experiment/stage1_cross_camera.yaml"
DEFAULT_OUTPUT_ROOT="outputs/stage1_eval"
DEFAULT_STAGE="test"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/test_fusion.sh --checkpoint-path /path/to/model.ckpt --manifest-path /path/to/humman_stage1_manifest.json [extra test.py args]

Required:
  --checkpoint-path PATH  Path to the Stage 1 checkpoint used by scripts/test.py.
  --manifest-path PATH    Path to the Stage 1 manifest JSON used by scripts/test.py.

Optional:
  --config PATH           Experiment config to use.
                          Default: configs/experiment/stage1_cross_camera.yaml
  --gt-smpl-dir PATH      Optional HuMMan GT SMPL directory override.
  --cameras-dir PATH      Optional HuMMan camera JSON directory override.
  --split-config-path PATH
                          Optional split policy YAML override.
  --split-name NAME       Optional named split policy override.
  --stage VALUE           Evaluation split: test or val.
                          Default: test
  --default-root-dir DIR  Output root used by scripts/test.py.
                          Default: outputs/stage1_eval
  --output-path PATH      Optional JSON output path.
  --mhr-assets-dir PATH   Forwarded to scripts/test.py for MHR-to-SMPL conversion.
  --input-smpl-cache-dir PATH
                          Optional cache directory for fitted input-view SMPL parameters.
  -h, --help              Show this help message.

Examples:
  bash scripts/test_fusion.sh \
    --checkpoint-path /home/zpengac/mmhpe/MvHPE3D/outputs/stage1/checkpoints/stage1_cross_camera/epochepoch=067-stepstep=206788.ckpt \
    --manifest-path /opt/data/humman_cropped/humman_stage1_manifest.json \
    --gt-smpl-dir /opt/data/humman_cropped/smpl

  bash scripts/test_fusion.sh \
    --checkpoint-path /path/to/model.ckpt \
    --manifest-path /path/to/humman_stage1_manifest.json \
    --gt-smpl-dir /path/to/smpl \
    --cameras-dir /path/to/cameras \
    --output-path outputs/stage1_eval/test_metrics.json
EOF
}

CHECKPOINT_PATH=""
MANIFEST_PATH=""
CONFIG_PATH="${DEFAULT_CONFIG}"
GT_SMPL_DIR=""
CAMERAS_DIR=""
SPLIT_CONFIG_PATH=""
SPLIT_NAME=""
STAGE="${DEFAULT_STAGE}"
OUTPUT_ROOT="${DEFAULT_OUTPUT_ROOT}"
OUTPUT_PATH=""
MHR_ASSETS_DIR=""
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-path)
            if [[ $# -lt 2 ]]; then
                echo "error: --checkpoint-path requires a value" >&2
                exit 1
            fi
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
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
        --gt-smpl-dir)
            if [[ $# -lt 2 ]]; then
                echo "error: --gt-smpl-dir requires a value" >&2
                exit 1
            fi
            GT_SMPL_DIR="$2"
            shift 2
            ;;
        --cameras-dir)
            if [[ $# -lt 2 ]]; then
                echo "error: --cameras-dir requires a value" >&2
                exit 1
            fi
            CAMERAS_DIR="$2"
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
        --stage)
            if [[ $# -lt 2 ]]; then
                echo "error: --stage requires a value" >&2
                exit 1
            fi
            STAGE="$2"
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
        --output-path)
            if [[ $# -lt 2 ]]; then
                echo "error: --output-path requires a value" >&2
                exit 1
            fi
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --mhr-assets-dir)
            if [[ $# -lt 2 ]]; then
                echo "error: --mhr-assets-dir requires a value" >&2
                exit 1
            fi
            MHR_ASSETS_DIR="$2"
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

if [[ -z "${CHECKPOINT_PATH}" || -z "${MANIFEST_PATH}" ]]; then
    usage
    echo "error: --checkpoint-path and --manifest-path are required" >&2
    exit 1
fi

if [[ ! -f "${REPO_ROOT}/${CONFIG_PATH}" && ! -f "${CONFIG_PATH}" ]]; then
    echo "error: config not found: ${CONFIG_PATH}" >&2
    exit 1
fi

if [[ ! -f "${REPO_ROOT}/${CHECKPOINT_PATH}" && ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "error: checkpoint not found: ${CHECKPOINT_PATH}" >&2
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
    uv run python scripts/test.py
    --checkpoint-path "${CHECKPOINT_PATH}"
    --config "${CONFIG_PATH}"
    --manifest-path "${MANIFEST_PATH}"
    --stage "${STAGE}"
    --default-root-dir "${OUTPUT_ROOT}"
)

if [[ -n "${GT_SMPL_DIR}" ]]; then
    CMD+=(--gt-smpl-dir "${GT_SMPL_DIR}")
fi

if [[ -n "${CAMERAS_DIR}" ]]; then
    CMD+=(--cameras-dir "${CAMERAS_DIR}")
fi

if [[ -n "${SPLIT_CONFIG_PATH}" ]]; then
    CMD+=(--split-config-path "${SPLIT_CONFIG_PATH}")
fi

if [[ -n "${SPLIT_NAME}" ]]; then
    CMD+=(--split-name "${SPLIT_NAME}")
fi

if [[ -n "${OUTPUT_PATH}" ]]; then
    CMD+=(--output-path "${OUTPUT_PATH}")
fi

if [[ -n "${MHR_ASSETS_DIR}" ]]; then
    CMD+=(--mhr-assets-dir "${MHR_ASSETS_DIR}")
fi

if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
    CMD+=("${PASSTHROUGH_ARGS[@]}")
fi

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
