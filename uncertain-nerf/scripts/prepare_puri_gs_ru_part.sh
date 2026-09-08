#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -ne 1 ]]; then
  echo "Usage: bash scripts/prepare_puri_gs_ru_part.sh EXISTING_GSPLAT_DIR"
  exit 2
fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$1"
PATCH="${ROOT_DIR}/patches/gsplat_v1.5.3_puri_gs_ru_part.patch"
EXPECTED="937e29912570c372bed6747a5c9bf85fed877bae"
if grep '^+++ b/' "${PATCH}" | grep -E '/(cuda|csrc|third_party)/' >/dev/null; then
  echo "RU-PART patch contains CUDA/kernel sources"
  exit 3
fi
if [[ -d "${TARGET_DIR}/.git" ]] && \
  [[ "$(git -C "${TARGET_DIR}" rev-parse HEAD)" == "${EXPECTED}" ]] && \
  git -C "${TARGET_DIR}" apply --reverse --check --unidiff-zero --ignore-whitespace "${PATCH}" >/dev/null 2>&1; then
  echo "PURI-GS-RU-PART-PATCH-READY"
  exit 0
fi
bash "${ROOT_DIR}/scripts/prepare_puri_gs_ru.sh" "${TARGET_DIR}"
test -d "${TARGET_DIR}/.git"
test "$(git -C "${TARGET_DIR}" rev-parse HEAD)" = "${EXPECTED}"
git -C "${TARGET_DIR}" apply --check --unidiff-zero --ignore-whitespace "${PATCH}"
git -C "${TARGET_DIR}" apply --unidiff-zero --ignore-whitespace "${PATCH}"
git -C "${TARGET_DIR}" apply --reverse --check --unidiff-zero --ignore-whitespace "${PATCH}"
echo "PURI-GS-RU-PART-PATCH-READY"
