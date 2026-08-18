#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: bash scripts/train_v6_three_seeds.sh DATA_DIR SCENE_NAME GPU_ID [CONFIG]"
  exit 2
fi

DATA_DIR="$1"
SCENE_NAME="$2"
GPU_ID="$3"
CONFIG="${4:-configs/llff_colmap_v6.txt}"

for SEED in 0 1 2; do
  bash scripts/train_v6.sh "${DATA_DIR}" "${SCENE_NAME}" "${SEED}" "${GPU_ID}" "${CONFIG}"
done
