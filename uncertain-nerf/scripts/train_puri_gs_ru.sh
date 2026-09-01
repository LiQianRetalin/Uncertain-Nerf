#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 8 || "$#" -gt 11 ]]; then
  echo "Usage: bash scripts/train_puri_gs_ru.sh GSPLAT_DIR DATA_DIR RESULT_DIR GPU_ID DINO_REPO DINO_WEIGHT FEATURE_CACHE MAX_STEPS [DATA_FACTOR] [TRAIN_KEYWORD] [TEST_KEYWORD]"
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PURI_GSPLAT_PYTHON:-${ROOT_DIR}/.venv-gsplat153/bin/python}"

ARGS=(
  --config "${ROOT_DIR}/configs/puri_gs_ru_full30k.yaml"
  --gsplat-dir "$1"
  --data-dir "$2"
  --result-dir "$3"
  --gpu "$4"
  --dino-repo-dir "$5"
  --dino-weight-path "$6"
  --feature-cache-dir "$7"
  --max-steps "$8"
)
if [[ "$#" -ge 9 ]]; then
  ARGS+=(--data-factor "$9")
fi
if [[ "$#" -ge 10 ]]; then
  if [[ "$#" -ne 11 ]]; then
    echo "TRAIN_KEYWORD and TEST_KEYWORD must be supplied together"
    exit 2
  fi
  ARGS+=(--train-keyword "${10}" --test-keyword "${11}")
fi

cd "${ROOT_DIR}"
"${PYTHON_BIN}" run_puri_gs.py "${ARGS[@]}"

