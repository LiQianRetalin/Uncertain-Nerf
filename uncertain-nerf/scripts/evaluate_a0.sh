#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: bash scripts/evaluate_a0.sh RENDER_DIR SEED GPU_ID [TRAINING_SUMMARY]"
  exit 2
fi

RENDER_DIR="$1"
SEED="$2"
GPU_ID="$3"
TRAINING_SUMMARY="${4:-}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv-v7}"

cd "${ROOT_DIR}"
source "${VENV_DIR}/bin/activate"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
ARGS=(
  --pred_dir "${RENDER_DIR}"
  --method puri-nerf-a0-original
  --seed "${SEED}"
  --device cuda
  --compute_lpips
  --output "${RENDER_DIR}/metrics.json"
)
if [[ -n "${TRAINING_SUMMARY}" ]]; then
  ARGS+=(--training_summary "${TRAINING_SUMMARY}")
fi
python -m v6.evaluation "${ARGS[@]}"
