#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: bash scripts/render_a0.sh DATA_DIR SCENE_NAME SEED GPU_ID SPLIT[train|val|test|path] [CONFIG]"
  exit 2
fi

DATA_DIR="$1"
SCENE_NAME="$2"
SEED="$3"
GPU_ID="$4"
SPLIT="$5"
CONFIG="${6:-configs/a0_fern.txt}"
if [[ "${SPLIT}" != "train" && "${SPLIT}" != "val" && \
      "${SPLIT}" != "test" && "${SPLIT}" != "path" ]]; then
  echo "SPLIT must be train, val, test or path"
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv-v7}"
EXP_NAME="${SCENE_NAME}_a0_original_seed${SEED}"
CHECKPOINT="${ROOT_DIR}/logs-a0/${EXP_NAME}/checkpoints/latest.pt"

cd "${ROOT_DIR}"
source "${VENV_DIR}/bin/activate"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "STOP_MISSING_CHECKPOINT: ${CHECKPOINT}"
  exit 4
fi
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
python run_nerf_a0.py \
  --config "${CONFIG}" \
  --datadir "${DATA_DIR}" \
  --expname "${EXP_NAME}" \
  --seed "${SEED}" \
  --render_only \
  --render_split "${SPLIT}" \
  --ft_path "${CHECKPOINT}"
