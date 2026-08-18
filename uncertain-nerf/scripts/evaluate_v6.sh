#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: bash scripts/evaluate_v6.sh RENDER_DIR EXPERIMENT_DIR SEED DEVICE [GT_DEPTH_DIR] [GT_POINTCLOUD] [PRED_POINTCLOUD]"
  exit 2
fi

RENDER_DIR="$1"
EXPERIMENT_DIR="$2"
SEED="$3"
DEVICE="$4"
GT_DEPTH_DIR="${5:-}"
GT_POINTCLOUD="${6:-}"
PRED_POINTCLOUD="${7:-}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv-v6}"

cd "${ROOT_DIR}"
source "${VENV_DIR}/bin/activate"

ARGS=(
  --pred_dir "${RENDER_DIR}"
  --method uncertain-nerf-v6
  --seed "${SEED}"
  --device "${DEVICE}"
  --compute_lpips
  --training_summary "${EXPERIMENT_DIR}/training_summary.json"
)
if [[ -n "${GT_DEPTH_DIR}" ]]; then ARGS+=(--gt_depth_dir "${GT_DEPTH_DIR}"); fi
if [[ -n "${GT_POINTCLOUD}" ]]; then ARGS+=(--gt_pointcloud "${GT_POINTCLOUD}"); fi
if [[ -n "${PRED_POINTCLOUD}" ]]; then ARGS+=(--pred_pointcloud "${PRED_POINTCLOUD}"); fi

python -m v6.evaluation "${ARGS[@]}"
