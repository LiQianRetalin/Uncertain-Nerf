#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 5 || "$#" -gt 9 ]]; then
  echo "Usage: bash scripts/train_puri_gs.sh CONFIG GSPLAT_DIR DATA_DIR RESULT_DIR GPU_ID [MAX_STEPS] [DATA_FACTOR] [TRAIN_KEYWORD] [TEST_KEYWORD]"
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_INPUT="${PURI_GSPLAT_PYTHON:-python}"
MAX_STEPS="${6:-10000}"

ARGS=(
  --config "$1"
  --gsplat-dir "$2"
  --data-dir "$3"
  --result-dir "$4"
  --gpu "$5"
  --max-steps "${MAX_STEPS}"
)
if [[ "$#" -ge 7 ]]; then
  ARGS+=(--data-factor "$7")
fi
if [[ "$#" -ge 8 ]]; then
  if [[ "$#" -ne 9 ]]; then
    echo "TRAIN_KEYWORD and TEST_KEYWORD must be supplied together"
    exit 2
  fi
  ARGS+=(--train-keyword "$8" --test-keyword "$9")
fi

cd "${ROOT_DIR}"
"${PYTHON_INPUT}" run_puri_gs.py "${ARGS[@]}"
