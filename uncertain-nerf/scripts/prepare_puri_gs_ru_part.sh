#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -ne 1 ]]; then
  echo "Usage: bash scripts/prepare_puri_gs_ru_part.sh EXISTING_GSPLAT_DIR"
  exit 2
fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$1"
PATCH="${ROOT_DIR}/patches/gsplat_v1.5.3_puri_gs_ru_part.patch"
DIAGNOSTIC_PATCH="${ROOT_DIR}/patches/gsplat_v1.5.3_ru_part_mechanism_diagnostic.patch"
EXPECTED="937e29912570c372bed6747a5c9bf85fed877bae"
if grep -h '^+++ b/' "${PATCH}" "${DIAGNOSTIC_PATCH}" | \
  grep -E '/(cuda|csrc|third_party)/' >/dev/null; then
  echo "RU-PART patch contains CUDA/kernel sources"
  exit 3
fi
if [[ -d "${TARGET_DIR}/.git" ]] && \
  [[ "$(git -C "${TARGET_DIR}" rev-parse HEAD)" == "${EXPECTED}" ]] && \
  grep -q 'from puri_gs.ru_part import RUPARTStrategy' "${TARGET_DIR}/examples/simple_trainer.py" && \
  grep -q 'ru_part_replay_ckpt' "${TARGET_DIR}/examples/simple_trainer.py" && \
  git -C "${TARGET_DIR}" apply --reverse --check --ignore-whitespace "${DIAGNOSTIC_PATCH}" >/dev/null 2>&1; then
  echo "PURI-GS-RU-PART-PATCH-READY"
  exit 0
fi
bash "${ROOT_DIR}/scripts/prepare_puri_gs_ru.sh" "${TARGET_DIR}"
test -d "${TARGET_DIR}/.git"
test "$(git -C "${TARGET_DIR}" rev-parse HEAD)" = "${EXPECTED}"
if ! grep -q 'from puri_gs.ru_part import RUPARTStrategy' "${TARGET_DIR}/examples/simple_trainer.py" || \
  ! grep -q 'ru_part_replay_ckpt' "${TARGET_DIR}/examples/simple_trainer.py"; then
  git -C "${TARGET_DIR}" apply --check --unidiff-zero --ignore-whitespace "${PATCH}"
  git -C "${TARGET_DIR}" apply --unidiff-zero --ignore-whitespace "${PATCH}"
fi
if ! git -C "${TARGET_DIR}" apply --reverse --check --ignore-whitespace "${DIAGNOSTIC_PATCH}" >/dev/null 2>&1; then
  git -C "${TARGET_DIR}" apply --check --ignore-whitespace "${DIAGNOSTIC_PATCH}"
  git -C "${TARGET_DIR}" apply --ignore-whitespace "${DIAGNOSTIC_PATCH}"
fi
grep -q 'from puri_gs.ru_part import RUPARTStrategy' "${TARGET_DIR}/examples/simple_trainer.py"
grep -q 'ru_part_replay_ckpt' "${TARGET_DIR}/examples/simple_trainer.py"
git -C "${TARGET_DIR}" apply --reverse --check --ignore-whitespace "${DIAGNOSTIC_PATCH}"
echo "PURI-GS-RU-PART-PATCH-READY"
