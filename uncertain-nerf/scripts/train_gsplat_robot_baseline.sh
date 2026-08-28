#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 || "$#" -gt 5 ]]; then
  echo "Usage: bash scripts/train_gsplat_robot_baseline.sh GSPLAT_DIR DATA_DIR RESULT_DIR GPU_ID [MAX_STEPS]"
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GSPLAT_DIR="$(realpath "$1")"
DATA_DIR="$(realpath "$2")"
mkdir -p "$3"
RESULT_DIR="$(realpath "$3")"
GPU_ID="$4"
MAX_STEPS="${5:-1000}"
PYTHON_INPUT="${PURI_GSPLAT_PYTHON:-python}"
if [[ "${PYTHON_INPUT}" == */* ]]; then
  PYTHON_BIN="$(realpath "${PYTHON_INPUT}")"
else
  PYTHON_BIN="${PYTHON_INPUT}"
fi
PATCH_PATH="${ROOT_DIR}/patches/gsplat_v1.5.3_robot_screen.patch"
EXPECTED_COMMIT="937e29912570c372bed6747a5c9bf85fed877bae"

if [[ "$(git -C "${GSPLAT_DIR}" rev-parse HEAD)" != "${EXPECTED_COMMIT}" ]]; then
  echo "GSPLAT_DIR is not the pinned v1.5.3 checkout"
  exit 2
fi
if ! git -C "${GSPLAT_DIR}" apply --reverse --check --ignore-whitespace \
    "${PATCH_PATH}" >/dev/null 2>&1; then
  echo "Required robot screen split patch is not applied"
  exit 2
fi
if [[ ! -d "${DATA_DIR}/sparse/0" || ! -d "${DATA_DIR}/images_4" ]]; then
  echo "Dataset must contain sparse/0 and images_4"
  exit 2
fi
if ! [[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_STEPS must be a positive integer"
  exit 2
fi
if ! [[ "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be one non-negative integer"
  exit 2
fi

cd "${GSPLAT_DIR}/examples"
PYTHONPATH="${ROOT_DIR}" \
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" simple_trainer.py default \
  --disable_viewer \
  --disable_video \
  --data_dir "${DATA_DIR}" \
  --data_factor 4 \
  --result_dir "${RESULT_DIR}" \
  --test_every 8 \
  --val_every 0 \
  --eval_split test \
  --max_steps "${MAX_STEPS}" \
  --eval_steps -1 \
  --save_steps "${MAX_STEPS}" \
  --sh_degree 3 \
  --tb_every 0
