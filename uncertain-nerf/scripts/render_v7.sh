#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: bash scripts/render_v7.sh DATA_DIR SCENE_NAME SEED GPU_ID SPLIT[test|path] [CONFIG]"
  exit 2
fi
DATA_DIR="$1"; SCENE_NAME="$2"; SEED="$3"; GPU_ID="$4"; SPLIT="$5"
CONFIG="${6:-configs/llff_colmap_v7.txt}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv-v7}"
EXP_NAME="${SCENE_NAME}_v7_seed${SEED}"
cd "${ROOT_DIR}"
source "${VENV_DIR}/bin/activate"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
python run_nerf_v7.py --config "${CONFIG}" --datadir "${DATA_DIR}" \
  --expname "${EXP_NAME}" --seed "${SEED}" --render_only --render_split "${SPLIT}" \
  --ft_path "./logs-v7/${EXP_NAME}/checkpoints/latest.pt"
