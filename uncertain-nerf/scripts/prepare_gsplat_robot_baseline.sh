#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: bash scripts/prepare_gsplat_robot_baseline.sh TARGET_DIR"
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$1"
PATCH_PATH="${ROOT_DIR}/patches/gsplat_v1.5.3_robot_screen.patch"
A1_PATCH_PATH="${ROOT_DIR}/patches/gsplat_v1.5.3_puri_gs_a1.patch"
EXPECTED_COMMIT="937e29912570c372bed6747a5c9bf85fed877bae"
REPOSITORY="${PURI_GSPLAT_REPOSITORY:-https://github.com/nerfstudio-project/gsplat.git}"

if [[ ! -e "${TARGET_DIR}" ]]; then
  git clone --depth 1 --branch v1.5.3 \
    "${REPOSITORY}" "${TARGET_DIR}"
elif [[ ! -d "${TARGET_DIR}/.git" ]]; then
  echo "TARGET_DIR exists but is not a git checkout: ${TARGET_DIR}"
  exit 2
fi

ACTUAL_COMMIT="$(git -C "${TARGET_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  echo "Expected gsplat ${EXPECTED_COMMIT}, got ${ACTUAL_COMMIT}"
  exit 2
fi

if git -C "${TARGET_DIR}" apply --reverse --check --ignore-whitespace \
    "${PATCH_PATH}" >/dev/null 2>&1; then
  echo "PURI-GS trainer patch is already applied."
elif git -C "${TARGET_DIR}" apply --check --unidiff-zero --ignore-whitespace \
    "${A1_PATCH_PATH}" >/dev/null 2>&1; then
  git -C "${TARGET_DIR}" apply --unidiff-zero --ignore-whitespace \
    "${A1_PATCH_PATH}"
  if ! git -C "${TARGET_DIR}" apply --reverse --check --ignore-whitespace \
      "${PATCH_PATH}" >/dev/null 2>&1; then
    git -C "${TARGET_DIR}" apply --reverse --unidiff-zero --ignore-whitespace \
      "${A1_PATCH_PATH}"
    echo "Incremental A1 patch did not produce the expected source; rolled back."
    exit 3
  fi
  echo "Upgraded the existing robot screen source with the PURI-GS A1 patch."
else
  git -C "${TARGET_DIR}" apply --check --ignore-whitespace "${PATCH_PATH}"
  git -C "${TARGET_DIR}" apply --ignore-whitespace "${PATCH_PATH}"
  echo "Applied the complete PURI-GS trainer patch."
fi

echo "Prepared gsplat v1.5.3 at ${TARGET_DIR}"
