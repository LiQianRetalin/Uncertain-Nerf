#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: bash scripts/train_a0.sh DATA_DIR SCENE_NAME SEED GPU_ID [CONFIG]"
  exit 2
fi

DATA_DIR="$1"
SCENE_NAME="$2"
SEED="$3"
GPU_ID="$4"
CONFIG="${5:-configs/a0_fern.txt}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv-v7}"
EXP_NAME="${SCENE_NAME}_a0_original_seed${SEED}"
OUTPUT_DIR="${ROOT_DIR}/logs-a0/${EXP_NAME}"
CONSOLE_LOG="${OUTPUT_DIR}/console_$(date +%Y%m%d_%H%M%S).log"

cd "${ROOT_DIR}"
bash "${ROOT_DIR}/scripts/check_v8_environment.sh" "${GPU_ID}"
source "${VENV_DIR}/bin/activate"
if [[ -e "${OUTPUT_DIR}/checkpoints/latest.pt" ]]; then
  echo "STOP_EXISTING_RUN: ${OUTPUT_DIR} already contains latest.pt"
  echo "Resume explicitly or use a new experiment name; do not overwrite it."
  exit 4
fi
mkdir -p "${OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python run_nerf_a0.py \
  --config "${CONFIG}" \
  --datadir "${DATA_DIR}" \
  --expname "${EXP_NAME}" \
  --seed "${SEED}" \
  --no_reload 2>&1 | tee -a "${CONSOLE_LOG}"
