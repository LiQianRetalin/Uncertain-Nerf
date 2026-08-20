#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: bash scripts/train_v8_diagnostic.sh MODE[baseline|v7_5] DATA_DIR SCENE_NAME SEED GPU_ID [CONFIG]"
  exit 2
fi

MODE="$1"
DATA_DIR="$2"
SCENE_NAME="$3"
SEED="$4"
GPU_ID="$5"
if [[ "${MODE}" != "baseline" && "${MODE}" != "v7_5" ]]; then
  echo "MODE must be baseline or v7_5"
  exit 2
fi

if [[ $# -ge 6 ]]; then
  CONFIG="$6"
elif [[ "${MODE}" == "baseline" ]]; then
  CONFIG="configs/v8_baseline.txt"
else
  CONFIG="configs/v8_v7_5.txt"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv-v7}"
EXP_NAME="${SCENE_NAME}_v8_${MODE}_seed${SEED}"
OUTPUT_DIR="${ROOT_DIR}/logs-v8/${EXP_NAME}"
CONSOLE_LOG="${OUTPUT_DIR}/console_$(date +%Y%m%d_%H%M%S).log"

cd "${ROOT_DIR}"
bash "${ROOT_DIR}/scripts/check_v8_environment.sh" "${GPU_ID}"
source "${VENV_DIR}/bin/activate"
if [[ -e "${OUTPUT_DIR}/checkpoints/latest.pt" ]]; then
  echo "STOP_EXISTING_RUN: ${OUTPUT_DIR} already contains latest.pt"
  echo "Resume explicitly or choose a new experiment name; do not overwrite it."
  exit 4
fi
mkdir -p "${OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python run_nerf_v8.py \
  --config "${CONFIG}" \
  --mode "${MODE}" \
  --datadir "${DATA_DIR}" \
  --expname "${EXP_NAME}" \
  --seed "${SEED}" \
  --no_reload 2>&1 | tee -a "${CONSOLE_LOG}"
