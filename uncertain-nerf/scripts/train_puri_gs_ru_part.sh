#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -ne 9 ]]; then
  echo "Usage: bash scripts/train_puri_gs_ru_part.sh GSPLAT_DIR GARDEN_DIR RESULT_DIR GPU_ID DINO_REPO DINO_WEIGHT FEATURE_CACHE TRACK_CACHE full|smoke"
  exit 2
fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PURI_GSPLAT_PYTHON:-${ROOT_DIR}/.venv-gsplat153/bin/python}"
MODE="$9"
if [[ "${MODE}" != "full" && "${MODE}" != "smoke" ]]; then
  echo "mode must be full or smoke"
  exit 2
fi
STEPS=30000
if [[ "${MODE}" == "smoke" ]]; then
  STEPS=10001
fi
ARGS=(
  --config "${ROOT_DIR}/configs/puri_gs_ru_part_garden30k.yaml"
  --gsplat-dir "$1" --data-dir "$2" --result-dir "$3" --gpu "$4"
  --dino-repo-dir "$5" --dino-weight-path "$6" --feature-cache-dir "$7"
  --track-cache "$8" --max-steps "${STEPS}"
)
if [[ "${MODE}" == "smoke" ]]; then
  ARGS+=(--non-scientific-smoke)
fi
cd "${ROOT_DIR}"
"${PYTHON_BIN}" run_puri_gs.py "${ARGS[@]}"
