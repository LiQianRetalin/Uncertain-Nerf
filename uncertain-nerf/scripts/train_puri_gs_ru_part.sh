#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -lt 9 || "$#" -gt 10 ]]; then
  echo "Usage: bash scripts/train_puri_gs_ru_part.sh GSPLAT_DIR GARDEN_DIR RESULT_DIR GPU_ID DINO_REPO DINO_WEIGHT FEATURE_CACHE TRACK_CACHE full|smoke [parent|noop|current]"
  exit 2
fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PURI_GSPLAT_PYTHON:-${ROOT_DIR}/.venv-gsplat153/bin/python}"
MODE="$9"
CONTROL_MODE="${10:-current}"
if [[ "${MODE}" != "full" && "${MODE}" != "smoke" ]]; then
  echo "mode must be full or smoke"
  exit 2
fi
if [[ "${CONTROL_MODE}" != "parent" && "${CONTROL_MODE}" != "noop" && "${CONTROL_MODE}" != "current" ]]; then
  echo "control mode must be parent, noop, or current"
  exit 2
fi
case "${CONTROL_MODE}" in
  parent) CONFIG="puri_gs_ru_part_parent_garden30k.yaml" ;;
  noop) CONFIG="puri_gs_ru_part_noop_garden30k.yaml" ;;
  current) CONFIG="puri_gs_ru_part_garden30k.yaml" ;;
esac
STEPS=30000
if [[ "${MODE}" == "smoke" ]]; then
  STEPS=10001
fi
ARGS=(
  --config "${ROOT_DIR}/configs/${CONFIG}"
  --gsplat-dir "$1" --data-dir "$2" --result-dir "$3" --gpu "$4"
  --dino-repo-dir "$5" --dino-weight-path "$6" --feature-cache-dir "$7"
  --max-steps "${STEPS}"
)
if [[ "${CONTROL_MODE}" != "parent" ]]; then
  ARGS+=(--track-cache "$8")
fi
if [[ -n "${PURI_RU_PART_REPLAY_CHECKPOINT:-}" ]]; then
  ARGS+=(--replay-checkpoint "${PURI_RU_PART_REPLAY_CHECKPOINT}")
fi
if [[ "${MODE}" == "smoke" ]]; then
  ARGS+=(--non-scientific-smoke)
fi
cd "${ROOT_DIR}"
"${PYTHON_BIN}" run_puri_gs.py "${ARGS[@]}"
