#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "Usage: bash scripts/evaluate_gsplat_robot_baseline.sh GSPLAT_DIR DATA_DIR RESULT_DIR GPU_ID CHECKPOINT"
  exit 2
fi

GSPLAT_DIR="$(realpath "$1")"
DATA_DIR="$(realpath "$2")"
mkdir -p "$3"
RESULT_DIR="$(realpath "$3")"
GPU_ID="$4"
CHECKPOINT="$(realpath "$5")"
PYTHON_INPUT="${PURI_GSPLAT_PYTHON:-python}"
if [[ "${PYTHON_INPUT}" == */* ]]; then
  PYTHON_BIN="$(realpath "${PYTHON_INPUT}")"
else
  PYTHON_BIN="${PYTHON_INPUT}"
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint does not exist: ${CHECKPOINT}"
  exit 2
fi
if ! [[ "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be one non-negative integer"
  exit 2
fi

cd "${GSPLAT_DIR}/examples"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" simple_trainer.py default \
  --disable_viewer \
  --disable_video \
  --data_dir "${DATA_DIR}" \
  --data_factor 2 \
  --result_dir "${RESULT_DIR}" \
  --test_every 8 \
  --val_every 8 \
  --eval_split test \
  --sh_degree 3 \
  --ckpt "${CHECKPOINT}"
