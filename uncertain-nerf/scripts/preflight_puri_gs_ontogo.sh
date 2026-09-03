#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv-gsplat153/bin/python"
GSPLAT_DIR="${ROOT_DIR}/external/gsplat-v1.5.3-ru"
TRANSFER_FILE="$(dirname "${ROOT_DIR}")/data-transfer/puri_gs_ontogo_patio_high_v1.tar"
DATA_PARENT="${ROOT_DIR}/data/nerf_on-the-go"
DATA_DIR="${DATA_PARENT}/patio_high_v1"
EXTRACT_ROOT="${DATA_PARENT}/.patio_high_v1.extracting"
EXPECTED_ARCHIVE_SHA="4b4d6a6a1e03328b1ff377e9994ab8dfc2c2b5d9d45abb2c8595d52be791abd9"

cd "${ROOT_DIR}"
BRANCH="$(git branch --show-current)"
COMMIT="$(git rev-parse HEAD)"
SHORT_COMMIT="${COMMIT:0:8}"
REPORT_DIR="${ROOT_DIR}/reports/ru-generalization-ontogo-${SHORT_COMMIT}"
CONSOLE_DIR="${ROOT_DIR}/logs-puri/ru-generalization-ontogo-${SHORT_COMMIT}/console"

printf 'BRANCH=%s\n' "${BRANCH}"
printf 'COMMIT=%s\n' "${COMMIT}"
test "${BRANCH}" = "dev"
test -z "$(git status --short --untracked-files=no)"
test -x "${PYTHON_BIN}"
test -d "${GSPLAT_DIR}/.git"
test "$(git -C "${GSPLAT_DIR}" rev-parse HEAD)" = \
  "937e29912570c372bed6747a5c9bf85fed877bae"
test -f "${TRANSFER_FILE}"
printf '%s  %s\n' "${EXPECTED_ARCHIVE_SHA}" "${TRANSFER_FILE}" | sha256sum -c -
test ! -e "${DATA_DIR}"
test ! -e "${EXTRACT_ROOT}"
test ! -e "${REPORT_DIR}"
test ! -e "${CONSOLE_DIR}"

bash "${ROOT_DIR}/scripts/prepare_puri_gs_ontogo.sh" "${GSPLAT_DIR}"

mkdir -p "${CONSOLE_DIR}"
"${PYTHON_BIN}" -m pytest -q \
  tests/test_ontogo_patio_high.py \
  tests/test_ru_generalization_summary.py \
  tests/test_ru_efficiency_audit.py \
  tests/test_puri_gs_configs.py \
  tests/test_dino_features.py \
  tests/test_ru_inference_path.py \
  2>&1 | tee "${CONSOLE_DIR}/pytest_ontogo_preflight.log"

mkdir -p "${DATA_PARENT}" "${EXTRACT_ROOT}"
tar -xf "${TRANSFER_FILE}" -C "${EXTRACT_ROOT}"
test "$(find "${EXTRACT_ROOT}/patio_high_v1/images_4" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 267

"${PYTHON_BIN}" tools/audit_ontogo_patio_high.py \
  --data-dir "${EXTRACT_ROOT}/patio_high_v1" \
  --gsplat-dir "${GSPLAT_DIR}" \
  --verify-image-hashes \
  2>&1 | tee "${CONSOLE_DIR}/patio_high_archive_audit.log"

mv "${EXTRACT_ROOT}/patio_high_v1" "${DATA_DIR}"
rmdir "${EXTRACT_ROOT}"
mkdir -p "${REPORT_DIR}"
"${PYTHON_BIN}" tools/audit_ontogo_patio_high.py \
  --data-dir "${DATA_DIR}" \
  --gsplat-dir "${GSPLAT_DIR}" \
  --output-json "${REPORT_DIR}/patio_high_protocol_audit.json" \
  2>&1 | tee "${CONSOLE_DIR}/patio_high_final_audit.log"

printf 'PATIO_HIGH_DATA_DIR=%s\n' "${DATA_DIR}"
printf 'PATIO_HIGH_REPORT_DIR=%s\n' "${REPORT_DIR}"
printf 'PATIO_HIGH_CONSOLE_DIR=%s\n' "${CONSOLE_DIR}"
printf 'PATIO_HIGH_PREFLIGHT_COMMIT=%s\n' "${COMMIT}"
printf 'PATIO_HIGH_SERVER_PREFLIGHT=PASS\n'
