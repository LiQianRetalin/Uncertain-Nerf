#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: bash scripts/train_v6.sh DATA_DIR SCENE_NAME SEED GPU_ID [CONFIG]"
  exit 2
fi

DATA_DIR="$1"
SCENE_NAME="$2"
SEED="$3"
GPU_ID="$4"
CONFIG="${5:-configs/llff_colmap_v6.txt}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv-v6}"

cd "${ROOT_DIR}"
source "${VENV_DIR}/bin/activate"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python run_nerf_v6.py \
  --config "${CONFIG}" \
  --datadir "${DATA_DIR}" \
  --expname "${SCENE_NAME}_v6_seed${SEED}" \
  --seed "${SEED}" \
  --no_reload
