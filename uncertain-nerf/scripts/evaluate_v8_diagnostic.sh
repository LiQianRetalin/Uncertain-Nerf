#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: bash scripts/evaluate_v8_diagnostic.sh RENDER_DIR METHOD SEED GPU_ID [TRAINING_SUMMARY] [--compute-lpips]"
  exit 2
fi

RENDER_DIR="$1"
METHOD="$2"
SEED="$3"
GPU_ID="$4"
TRAINING_SUMMARY="${5:-}"
LPIPS_FLAG="${6:-}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv-v7}"

cd "${ROOT_DIR}"
source "${VENV_DIR}/bin/activate"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

ARGS=(--pred_dir "${RENDER_DIR}" --method "${METHOD}" --seed "${SEED}"
      --device cuda --output "${RENDER_DIR}/metrics.json")
if [[ -n "${TRAINING_SUMMARY}" ]]; then
  ARGS+=(--training_summary "${TRAINING_SUMMARY}")
fi
if [[ "${LPIPS_FLAG}" == "--compute-lpips" ]]; then
  ARGS+=(--compute_lpips)
fi
python -m v6.evaluation "${ARGS[@]}"
