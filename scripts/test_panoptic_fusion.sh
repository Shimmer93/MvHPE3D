#!/usr/bin/env bash

set -Eeuo pipefail

on_error() {
    local exit_code="$?"
    echo "error: scripts/test_panoptic_fusion.sh failed at line ${BASH_LINENO[0]}: ${BASH_COMMAND}" >&2
    exit "${exit_code}"
}

trap on_error ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_DATASET_ROOT="/opt/data/panoptic_kinoptic_single_actor_cropped"
DEFAULT_CONFIG="configs/experiment/stage1_panoptic_cross_camera.yaml"
DEFAULT_MANIFEST_PATH="${DEFAULT_DATASET_ROOT}/panoptic_stage1_manifest_visible.json"
DEFAULT_GT_SMPL_DIR="${DEFAULT_DATASET_ROOT}/smpl"
DEFAULT_CAMERAS_DIR="${DEFAULT_DATASET_ROOT}"
DEFAULT_SPLIT_CONFIG_PATH="configs/data/panoptic_stage1_splits.yaml"
DEFAULT_SPLIT_NAME="cross_camera_split"
DEFAULT_OUTPUT_ROOT="outputs/stage1_panoptic_eval"
DEFAULT_OUTPUT_PATH="${DEFAULT_OUTPUT_ROOT}/test_metrics.json"
DEFAULT_INPUT_SMPL_CACHE_DIR="${DEFAULT_DATASET_ROOT}/sam3dbody_fitted_smpl"
DEFAULT_MHR_ASSETS_DIR="/opt/data/assets"
DEFAULT_STAGE="test"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/test_panoptic_fusion.sh --checkpoint-path /path/to/model.ckpt [extra test.py args]

Required:
  --checkpoint-path PATH  Path to the Stage 1 Panoptic checkpoint.

Optional:
  --config PATH           Experiment config to use.
                          Default: configs/experiment/stage1_panoptic_cross_camera.yaml
  --manifest-path PATH    Panoptic Stage 1 manifest JSON.
                          Default: /opt/data/panoptic_kinoptic_single_actor_cropped/panoptic_stage1_manifest_visible.json
  --gt-smpl-dir PATH      Panoptic sequence SMPL target cache directory.
                          Default: /opt/data/panoptic_kinoptic_single_actor_cropped/smpl
  --cameras-dir PATH      Panoptic dataset root containing sequence camera metadata.
                          Default: /opt/data/panoptic_kinoptic_single_actor_cropped
  --split-config-path PATH
                          Split policy YAML.
                          Default: configs/data/panoptic_stage1_splits.yaml
  --split-name NAME       Named split policy.
                          Default: cross_camera_split
  --stage VALUE           Evaluation split: test or val.
                          Default: test
  --default-root-dir DIR  Output root used by scripts/test.py.
                          Default: outputs/stage1_panoptic_eval
  --output-path PATH      JSON output path.
                          Default: outputs/stage1_panoptic_eval/test_metrics.json
  --mhr-assets-dir PATH   MHR asset directory for input-view SMPL conversion.
                          Default: /opt/data/assets
  --input-smpl-cache-dir PATH
                          Cache directory for fitted input-view SMPL parameters.
                          Default: /opt/data/panoptic_kinoptic_single_actor_cropped/sam3dbody_fitted_smpl
  -h, --help              Show this help message.

Examples:
  bash scripts/test_panoptic_fusion.sh \
    --checkpoint-path outputs/stage1_panoptic_cross_camera/checkpoints/best.ckpt

  bash scripts/test_panoptic_fusion.sh \
    --checkpoint-path /path/to/model.ckpt \
    --stage val \
    --devices 1 \
    --output-path outputs/stage1_panoptic_eval/val_metrics.json
EOF
}

CHECKPOINT_PATH=""
CONFIG_PATH="${DEFAULT_CONFIG}"
MANIFEST_PATH="${DEFAULT_MANIFEST_PATH}"
GT_SMPL_DIR="${DEFAULT_GT_SMPL_DIR}"
CAMERAS_DIR="${DEFAULT_CAMERAS_DIR}"
SPLIT_CONFIG_PATH="${DEFAULT_SPLIT_CONFIG_PATH}"
SPLIT_NAME="${DEFAULT_SPLIT_NAME}"
STAGE="${DEFAULT_STAGE}"
OUTPUT_ROOT="${DEFAULT_OUTPUT_ROOT}"
OUTPUT_PATH="${DEFAULT_OUTPUT_PATH}"
OUTPUT_PATH_WAS_SET=0
MHR_ASSETS_DIR="${DEFAULT_MHR_ASSETS_DIR}"
INPUT_SMPL_CACHE_DIR="${DEFAULT_INPUT_SMPL_CACHE_DIR}"
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
        --config)
            if [[ $# -lt 2 ]]; then
                echo "error: --config requires a value" >&2
                exit 1
            fi
            CONFIG_PATH="$2"
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
            OUTPUT_PATH_WAS_SET=1
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
        --input-smpl-cache-dir)
            if [[ $# -lt 2 ]]; then
                echo "error: --input-smpl-cache-dir requires a value" >&2
                exit 1
            fi
            INPUT_SMPL_CACHE_DIR="$2"
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

if [[ -z "${CHECKPOINT_PATH}" ]]; then
    usage
    echo "error: --checkpoint-path is required" >&2
    exit 1
fi

if [[ "${STAGE}" != "test" && "${STAGE}" != "val" ]]; then
    echo "error: --stage must be test or val, got: ${STAGE}" >&2
    exit 1
fi

if [[ "${OUTPUT_PATH_WAS_SET}" -eq 0 && "${STAGE}" == "val" ]]; then
    OUTPUT_PATH="${OUTPUT_ROOT}/val_metrics.json"
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

if [[ ! -d "${REPO_ROOT}/${CAMERAS_DIR}" && ! -d "${CAMERAS_DIR}" ]]; then
    echo "error: cameras dir not found: ${CAMERAS_DIR}" >&2
    exit 1
fi

if [[ ! -f "${REPO_ROOT}/${SPLIT_CONFIG_PATH}" && ! -f "${SPLIT_CONFIG_PATH}" ]]; then
    echo "error: split config not found: ${SPLIT_CONFIG_PATH}" >&2
    exit 1
fi

if [[ ! -d "${REPO_ROOT}/${GT_SMPL_DIR}" && ! -d "${GT_SMPL_DIR}" ]]; then
    cat >&2 <<EOF
error: Panoptic SMPL target cache not found: ${GT_SMPL_DIR}

Create it before testing:
  CUDA_VISIBLE_DEVICES=1 uv run python scripts/precompute_panoptic_gt_smpl.py \\
    --dataset-root ${CAMERAS_DIR} \\
    --output-dir ${GT_SMPL_DIR} \\
    --manifest-path ${MANIFEST_PATH} \\
    --device cuda:0 \\
    --batch-size 512 \\
    --num-iters 120
EOF
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
    --gt-smpl-dir "${GT_SMPL_DIR}"
    --cameras-dir "${CAMERAS_DIR}"
    --split-config-path "${SPLIT_CONFIG_PATH}"
    --split-name "${SPLIT_NAME}"
    --stage "${STAGE}"
    --default-root-dir "${OUTPUT_ROOT}"
    --output-path "${OUTPUT_PATH}"
    --mhr-assets-dir "${MHR_ASSETS_DIR}"
    --input-smpl-cache-dir "${INPUT_SMPL_CACHE_DIR}"
)

if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
    CMD+=("${PASSTHROUGH_ARGS[@]}")
fi

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
