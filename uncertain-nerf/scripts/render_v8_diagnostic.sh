#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: bash scripts/render_v8_diagnostic.sh MODE DATA_DIR SCENE_NAME SEED GPU_ID SPLIT[train|val|test|path] [CONFIG]"
  exit 2
fi

MODE="$1"
DATA_DIR="$2"
SCENE_NAME="$3"
SEED="$4"
GPU_ID="$5"
SPLIT="$6"
if [[ "${MODE}" != "baseline" && "${MODE}" != "v7_5" ]]; then
  echo "MODE must be baseline or v7_5"
  exit 2
fi
if [[ "${SPLIT}" != "train" && "${SPLIT}" != "val" && \
      "${SPLIT}" != "test" && "${SPLIT}" != "path" ]]; then
  echo "SPLIT must be train, val, test or path"
  exit 2
fi

if [[ $# -ge 7 ]]; then
  CONFIG="$7"
elif [[ "${MODE}" == "baseline" ]]; then
  CONFIG="configs/v8_baseline.txt"
else
  CONFIG="configs/v8_v7_5.txt"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv-v7}"
EXP_NAME="${SCENE_NAME}_v8_${MODE}_seed${SEED}"
CHECKPOINT="${ROOT_DIR}/logs-v8/${EXP_NAME}/checkpoints/latest.pt"

cd "${ROOT_DIR}"
source "${VENV_DIR}/bin/activate"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
python run_nerf_v8.py \
  --config "${CONFIG}" \
  --mode "${MODE}" \
  --datadir "${DATA_DIR}" \
  --expname "${EXP_NAME}" \
  --seed "${SEED}" \
  --render_only \
  --render_split "${SPLIT}" \
  --ft_path "${CHECKPOINT}"
